"""Document tamper detection - chống giả mạo CCCD (photo substitution + inpainting).

Hướng nghiên cứu A: phát hiện ảnh chân dung bị thay thế / vùng chữ bị inpaint
trên ảnh thẻ. Hai tầng tín hiệu:

1. **Neural (chủ lực)**: ``TamperNet`` (xem ``tamper_net.py``) - multi-stream
   RGB + SRM forensic + block-DCT, 2 head cls/mask. Load từ
   ``AI_TAMPER_MODEL`` (``.pth`` torch hoặc ``.onnx``). Train bằng
   ``training/train_tamper.py`` trên FantasyID (DeepID@ICCV 2025) hoặc dữ liệu
   CCCD tự tạo.

2. **Heuristic forensic (fallback, cv2/numpy thuần)**: 4 tín hiệu clássic -
   * ``boundary_energy``   - năng lượng gradient dọc mép vùng chân dung
                            (splice để lại đường cắt bất thường);
   * ``noise_step``        - chênh lệch độ nhiễu (hp-filter sigma) trong/ngoài
                            mép chân dung (lịch sử nén/dao động nhiễu khác nhau);
   * ``blockiness``        - cường độ lưới JPEG 8x8 (double-compression signature);
   * ``low_hf_fraction``   - tỉ lệ block "phẳng" năng lượng tần số cao
                            (vùng bị inpaint/smoothing).

   **Nguyên tắc fallback (khác với liveness blur-fallback):** engine heuristic
   KHÔNG BAO GIỜ trả status ``TAMPERED`` (tối đa ``SUSPICIOUS`` -> chuyển sang
   review) và result luôn ghi rõ ``model_loaded=false``, để verify_service
   không âm thầm hạ chuẩn bảo mật khi model chưa được train.

Chuẩn hoá input model: RGB float32 [0, 1], letterbox về ``INPUT_SIZE`` (256).

Env (đọc lúc khởi tạo service)::

    AI_TAMPER_ENABLED      true    bật/tắt module
    AI_TAMPER_MODEL        models/tamper_net.pth   (.pth hoặc .onnx)
    AI_TAMPER_THRESHOLD    0.55    >= đây -> TAMPERED (chỉ khi model loaded)
    AI_TAMPER_SUSPECT      0.35    >= đây -> SUSPICIOUS
    AI_TAMPER_MODEL_WEIGHT 0.70    trọng số fusion: w*model + (1-w)*signals
"""
from typing import Any, Dict, List, Optional, Tuple
import os

import cv2
import numpy as np

# Trạng thái trả về
OK, SUSPICIOUS, TAMPERED, DISABLED = "OK", "SUSPICIOUS", "TAMPERED", "DISABLED"

# Trọng số fusion các tín hiệu heuristic (tổng = 1.0). Khi thiếu tín hiệu
# (ảnh mặt sau thẻ không có chân dung) thì trọng số được chuẩn hoá lại.
SIGNAL_WEIGHTS = {
    "boundary_energy": 0.35,
    "noise_step": 0.25,
    "blockiness": 0.15,
    "low_hf_fraction": 0.25,
}

# TODO(calibrate): ngưỡng chuẩn hoá thô -> [0, 1], phải fit lại trên bộ dữ liệu
# thẻ thật (in + scan) trước khi dùng cho quyết định reject.
SIGNAL_NORM = {
    "boundary_energy": lambda v: float(np.clip((v - 1.5) / 2.5, 0.0, 1.0)),
    "noise_step": lambda v: float(np.clip(v / 1.2, 0.0, 1.0)),
    "blockiness": lambda v: float(np.clip((v - 0.15) / 0.35, 0.0, 1.0)),
    "low_hf_fraction": lambda v: float(np.clip((v - 0.15) / 0.50, 0.0, 1.0)),
}


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("false", "0", "no", "off")


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _letterbox(img_rgb: np.ndarray, size: int) -> Tuple[np.ndarray, Tuple[float, float, float, int, int]]:
    """Resize giữ tỉ lệ, pad về size x size. Trả về (canvas, (lx, ly, scale, W, H))."""
    h, w = img_rgb.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    lx, ly = (size - nw) // 2, (size - nh) // 2
    canvas = np.empty((size, size, 3), dtype=img_rgb.dtype)
    canvas[:] = resized.reshape(-1, 3).mean(axis=0).astype(img_rgb.dtype)
    canvas[ly:ly + nh, lx:lx + nw] = resized
    return canvas, (float(lx), float(ly), float(scale), w, h)


def _unletterbox_box(box, meta) -> List[int]:
    """Đưa box từ toạ độ canvas 256 về toạ độ ảnh gốc."""
    lx, ly, scale, W, H = meta
    x = int(round((box[0] - lx) / scale))
    y = int(round((box[1] - ly) / scale))
    w = int(round(box[2] / scale))
    h = int(round(box[3] / scale))
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    w, h = max(1, min(w, W - x)), max(1, min(h, H - y))
    return [x, y, w, h]


def mask_to_regions(mask01: np.ndarray, meta, min_frac: float = 0.01, max_regions: int = 3) -> List[Dict[str, Any]]:
    """Mask nhị phân (S x S) -> danh sách vùng nghi giả mạo toạ độ ảnh gốc."""
    m = (mask01 >= 0.5).astype(np.uint8)
    if m.size == 0 or int(m.sum()) == 0:
        return []
    num, _labels, stats, _cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    total = int(m.size)
    regions = []
    for i in range(1, num):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_frac * total:
            continue
        regions.append((area, [x, y, w, h]))
    regions.sort(key=lambda r: r[0], reverse=True)
    return [
        {"box": _unletterbox_box(b, meta), "area_frac": round(area / total, 4)}
        for area, b in regions[:max_regions]
    ]


# --------------------------------------------------------------------------- #
# Heuristic forensic signals (chỉ cv2/numpy, không cần torch/model)
# --------------------------------------------------------------------------- #
def _take(arr: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> Optional[np.ndarray]:
    h, w = arr.shape[:2]
    r0, c0, r1, c1 = max(0, r0), max(0, c0), min(h, r1), min(w, c1)
    if r1 <= r0 or c1 <= c0:
        return None
    return arr[r0:r1, c0:c1]


def _ring_stats(hp: np.ndarray, box, d: int = 4, t: int = 10) -> Tuple[Optional[float], Optional[float]]:
    """Sigma nhiễu (hp-filter) của dải mỏng trong mép và ngoài mép vùng chân dung."""
    x, y, w, h = (int(v) for v in box)
    x0, y0, x1, y1 = x, y, x + w, y + h
    if w < 2 * (d + t) or h < 2 * (d + t):
        return None, None

    inner_parts = [
        _take(hp, y0 + d, x0 + d, y0 + d + t, x1 - d),
        _take(hp, y1 - d - t, x0 + d, y1 - d, x1 - d),
        _take(hp, y0 + d, x0 + d, y1 - d, x0 + d + t),
        _take(hp, y0 + d, x1 - d - t, y1 - d, x1 - d),
    ]
    outer_parts = [
        _take(hp, y0 - d - t, x0, y0 - d, x1),
        _take(hp, y1 + d, x0, y1 + d + t, x1),
        _take(hp, y0, x0 - d - t, y1, x0 - d),
        _take(hp, y0, x1 + d, y1, x1 + d + t),
    ]
    inner = [p.ravel() for p in inner_parts if p is not None and p.size]
    outer = [p.ravel() for p in outer_parts if p is not None and p.size]
    if not inner or not outer:
        return None, None
    s_in = float(np.std(np.concatenate(inner))) + 1e-6
    s_out = float(np.std(np.concatenate(outer))) + 1e-6
    return s_in, s_out


def _boundary_signals(gray: np.ndarray, photo_box) -> Tuple[Optional[float], Optional[float]]:
    """(boundary_energy, noise_step) quanh mép vùng chân dung - None nếu không có box."""
    if photo_box is None:
        return None, None
    x, y, w, h = (int(v) for v in photo_box)
    if w < 24 or h < 24:
        return None, None
    x0, y0, x1, y1 = max(0, x), max(0, y), min(gray.shape[1], x + w), min(gray.shape[0], y + h)
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None, None

    g = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    t = 2
    strips = [g[y0:y0 + t, x0:x1], g[y1 - t:y1, x0:x1], g[y0:y1, x0:x0 + t], g[y0:y1, x1 - t:x1]]
    strips = [s for s in strips if s.size]
    if not strips:
        return None, None
    boundary = float(np.mean([float(s.mean()) for s in strips])) / (float(g.mean()) + 1e-3)

    hp = gray - cv2.GaussianBlur(gray, (3, 3), 0)
    s_in, s_out = _ring_stats(hp.astype(np.float32), (x0, y0, x1 - x0, y1 - y0))
    noise_step = abs(float(np.log(s_in / s_out))) if (s_in and s_out) else None
    return boundary, noise_step


def _blockiness(gray: np.ndarray) -> float:
    """Chênh lệch biên độ tại đường lưới JPEG (8px) vs đường giữa block."""
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return 0.0
    dv = np.abs(np.diff(gray, axis=1))  # dv[:, j] = |col j+1 - col j| -> lưới tại j%8 == 7
    dh = np.abs(np.diff(gray, axis=0))
    grid_v, mid_v = float(dv[:, 7::8].mean()), float(dv[:, 3::8].mean())
    grid_h, mid_h = float(dh[7::8, :].mean()), float(dh[3::8, :].mean())
    r_v = (grid_v + 1e-6) / (mid_v + 1e-6) - 1.0
    r_h = (grid_h + 1e-6) / (mid_h + 1e-6) - 1.0
    return float(max(0.0, (r_v + r_h) / 2.0))


def _low_hf_fraction(gray: np.ndarray, block: int = 16) -> float:
    """Tỉ lệ block có năng lượng tần số cao < 25% trung vị (vùng phẳng bất thường)."""
    hp = np.abs(gray - cv2.GaussianBlur(gray, (5, 5), 0)).astype(np.float32)
    h, w = hp.shape
    hh, ww = (h // block) * block, (w // block) * block
    if hh == 0 or ww == 0:
        return 0.0
    bmap = hp[:hh, :ww].reshape(hh // block, block, ww // block, block).mean(axis=(1, 3))
    med = float(np.median(bmap))
    return float(np.mean(bmap < 0.25 * med + 1e-6))


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class TamperService:
    """Phát hiện giả mạo thẻ: TamperNet (nếu có) + tín hiệu forensic heuristic."""

    def __init__(self):
        self.enabled = _env_flag("AI_TAMPER_ENABLED", True)
        self.threshold = float(os.environ.get("AI_TAMPER_THRESHOLD", "0.55"))
        self.suspect = float(os.environ.get("AI_TAMPER_SUSPECT", "0.35"))
        self.model_weight = float(os.environ.get("AI_TAMPER_MODEL_WEIGHT", "0.7"))
        default_model = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models", "tamper_net.pth",
        )
        self.model_path = os.environ.get("AI_TAMPER_MODEL") or default_model
        self._model = None          # torch nn.Module HOẶC dict {"session": ort} cho onnx
        self._backend = None        # "torch" | "onnx" | None
        self._load_attempted = False
        self._face_service = None

    # ------------------------------------------------------------ face box
    def _get_face_service(self):
        if self._face_service is None:
            from app.services.face_service import FaceService
            self._face_service = FaceService()
        return self._face_service

    def _photo_box(self, img_rgb: np.ndarray) -> Optional[List[int]]:
        """Vùng chân dung trên thẻ (dùng detector có sẵn) - ảnh mặt sau -> None."""
        try:
            boxes = self._get_face_service().detect_face_boxes(img_rgb)
        except Exception:
            return None
        if not boxes:
            return None
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        return [int(x), int(y), int(w), int(h)]

    # --------------------------------------------------------------- model
    def available(self) -> bool:
        return self.enabled and self._load_backend() is not None

    def _load_backend(self) -> Optional[str]:
        if self._load_attempted:
            return self._backend
        self._load_attempted = True
        if not self.enabled or not os.path.exists(self.model_path):
            return None
        try:
            if self.model_path.lower().endswith(".onnx"):
                import onnxruntime as ort
                self._model = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                self._backend = "onnx"
            else:
                import torch
                from app.services.tamper_net import TamperNet, INPUT_SIZE

                model = TamperNet(input_size=INPUT_SIZE)
                state = torch.load(self.model_path, map_location="cpu")
                if next(iter(state)).startswith("module."):
                    state = {k[7:]: v for k, v in state.items()}
                model.load_state_dict(state)
                model.eval()
                self._model = model
                self._backend = "torch"
        except Exception:
            self._model, self._backend = None, None
        return self._backend

    def _predict(self, img_rgb: np.ndarray) -> Tuple[Optional[float], List[Dict[str, Any]]]:
        """ -> (prob_p forged từ model, vùng nghi giả mạo từ mask head)."""
        backend = self._load_backend()
        if backend is None:
            return None, []
        from app.services.tamper_net import INPUT_SIZE

        canvas, meta = _letterbox(img_rgb, INPUT_SIZE)
        x = canvas.astype(np.float32) / 255.0
        blob = np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...])
        try:
            if backend == "torch":
                import torch
                with torch.no_grad():
                    out = self._model(torch.from_numpy(blob))
                prob = float(torch.sigmoid(out["logit"])[0])
                mask = torch.sigmoid(out["mask_logits"])[0, 0].cpu().numpy()
            else:
                names = [o.name for o in self._model.get_outputs()]
                raw = self._model.run(names, {self._model.get_inputs()[0].name: blob})
                outs = dict(zip(names, raw))
                prob = float(1.0 / (1.0 + np.exp(-float(np.asarray(outs["logit"]).reshape(-1)[0]))))
                mask = 1.0 / (1.0 + np.exp(-np.asarray(outs["mask_logits"], dtype=np.float32)[0, 0]))
        except Exception:
            return None, []
        return prob, mask_to_regions(mask, meta)

    # ------------------------------------------------------------ heuristic
    @staticmethod
    def forensic_signals(img_rgb: np.ndarray, photo_box: Optional[List[int]]) -> Dict[str, Any]:
        """4 tín hiệu forensic thô + đã chuẩn hoá [0,1] (không cần model)."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        boundary, noise_step = _boundary_signals(gray, photo_box)
        raw = {
            "boundary_energy": boundary,
            "noise_step": noise_step,
            "blockiness": _blockiness(gray),
            "low_hf_fraction": _low_hf_fraction(gray),
        }
        norm = {k: (SIGNAL_NORM[k](v) if v is not None else None) for k, v in raw.items()}
        return {"raw": raw, "norm": norm}

    @staticmethod
    def _signal_score(norm: Dict[str, Optional[float]]) -> float:
        """Fusion có chuẩn hoá trọng số trên các tín hiệu thực sự có mặt."""
        present = {k: v for k, v in norm.items() if v is not None}
        if not present:
            return 0.0
        wsum = sum(SIGNAL_WEIGHTS[k] for k in present)
        return float(sum(SIGNAL_WEIGHTS[k] * v for k, v in present.items()) / wsum)

    @staticmethod
    def _classify(score: float, model_loaded: bool, threshold: float, suspect: float) -> str:
        """Heuristic-only bị cap ở SUSPICIOUS - không bao giờ reject một mình."""
        if score >= threshold and model_loaded:
            return TAMPERED
        if score >= suspect:
            return SUSPICIOUS
        return OK

    # --------------------------------------------------------------- main
    def analyze(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "status": DISABLED, "tamper_score": 0.0, "engine": None,
                    "model_loaded": False, "model_prob": None, "photo_box": None,
                    "signals": None, "regions": [], "threshold": {}}

        photo_box = self._photo_box(img_rgb)
        signals = self.forensic_signals(img_rgb, photo_box)
        signal_score = self._signal_score(signals["norm"])

        model_prob, regions = self._predict(img_rgb)
        model_loaded = model_prob is not None

        if model_loaded:
            w = self.model_weight
            score = w * model_prob + (1.0 - w) * signal_score
            engine = "TamperNet"
        else:
            score, engine = signal_score, "heuristic"

        status = self._classify(score, model_loaded, self.threshold, self.suspect)
        return {
            "enabled": True,
            "status": status,
            "tamper_score": round(float(score), 4),
            "engine": engine,
            "model_loaded": model_loaded,
            "model_prob": None if model_prob is None else round(model_prob, 4),
            "photo_box": photo_box,
            "signals": {
                "raw": {k: (None if v is None else round(float(v), 4)) for k, v in signals["raw"].items()},
                "norm": {k: (None if v is None else round(float(v), 4)) for k, v in signals["norm"].items()},
                "signal_score": round(signal_score, 4),
            },
            "regions": regions,
            "threshold": {"tampered": self.threshold, "suspect": self.suspect},
        }
