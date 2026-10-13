from typing import Any, Dict, List, Optional, Tuple
import io
import os
import numpy as np
from PIL import Image, ImageOps
import cv2
import onnxruntime as ort

# Standard ArcFace 5-point template for a 112x112 aligned face.
# Order: left eye, right eye, nose tip, left mouth corner, right mouth corner.
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# MediaPipe FaceMesh indices matching the 5 template points (image coordinates).
LM_ORDER = [33, 263, 1, 61, 291]

# Cosine-similarity thresholds on L2-normalized ArcFace embeddings.
# Higher score = more similar. Measured on real data: genuine selfie-vs-ID
# face ~0.31..1.0, different people < 0.15. Overridable via env to calibrate.
MATCH_THRESHOLD = float(os.environ.get("AI_MATCH_SCORE", 0.30))
PARTIAL_THRESHOLD = float(os.environ.get("AI_PARTIAL_SCORE", 0.20))


class FaceService:
    def __init__(self):
        self._det_session = None
        self._det_input_name = None
        self._det_output_names = None
        # version-RFB-640 expects a fixed 480x640 (HxW) input; cv2.resize takes
        # (width, height), so store it that way.  Feeding anything else makes
        # onnxruntime raise and the detector silently falls back to Haar.
        self._det_size = (640, 480)

        self._rec_session = None
        self._rec_input_name = None
        self._rec_output_name = None

        self._landmarker = None
        self._mp = None

    # ------------------------------------------------------------- detector
    def _init_detector(self):
        if self._det_session is not None:
            return
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "version-RFB-640.onnx")
        if not os.path.exists(model_path):
            raise RuntimeError(f"Detector model not found: {model_path}")
        providers = ["CPUExecutionProvider"]
        self._det_session = ort.InferenceSession(model_path, providers=providers)
        self._det_input_name = self._det_session.get_inputs()[0].name
        self._det_output_names = [o.name for o in self._det_session.get_outputs()]

    # ----------------------------------------------------------- recognizer
    def _init_recognizer(self):
        if self._rec_session is not None:
            return
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "recognition.onnx")
        if not os.path.exists(model_path):
            raise RuntimeError(f"Recognition model not found: {model_path}")
        providers = ["CPUExecutionProvider"]
        self._rec_session = ort.InferenceSession(model_path, providers=providers)
        self._rec_input_name = self._rec_session.get_inputs()[0].name
        self._rec_output_name = self._rec_session.get_outputs()[0].name

    # ----------------------------------------------------------- landmarks
    def _get_landmarker(self):
        if self._landmarker is not None:
            return self._landmarker
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models",
            "face_landmarker.task",
        )
        if not os.path.exists(model_path):
            raise RuntimeError(f"Face landmarker model not found: {model_path}")
        options = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        return self._landmarker

    def landmarks5(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        try:
            landmarker = self._get_landmarker()
            mp = self._mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
            result = landmarker.detect(mp_image)
        except Exception:
            return None
        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        h, w = img_rgb.shape[:2]
        pts = [[lms[i].x * w, lms[i].y * h] for i in LM_ORDER]
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def load_image(content: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(content))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.array(img)

    # ----------------------------------------------------------- alignment
    @staticmethod
    def align_face(img_rgb: np.ndarray, pts: np.ndarray) -> np.ndarray:
        tform, _ = cv2.estimateAffinePartial2D(pts, ARCFACE_DST, method=cv2.LMEDS)
        if tform is None:
            tform = cv2.getAffineTransform(pts[:3], ARCFACE_DST[:3])
        return cv2.warpAffine(img_rgb, tform, (112, 112), borderMode=cv2.BORDER_REPLICATE)

    # ---------------------------------------------------------- embedding
    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        self._init_recognizer()
        img = aligned_rgb.astype(np.float32)
        img = (img - 127.5) / 127.5
        blob = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
        out = self._rec_session.run([self._rec_output_name], {self._rec_input_name: blob})[0][0]
        norm = np.linalg.norm(out) + 1e-6
        return out / norm

    @staticmethod
    def _tile_images(img_rgb: np.ndarray, tile: int = 800, overlap: float = 0.5) -> List[np.ndarray]:
        h, w = img_rgb.shape[:2]
        if max(h, w) <= tile + 200 or h <= tile or w <= tile:
            return [img_rgb]
        step = max(1, int(tile * (1 - overlap)))
        tiles: List[np.ndarray] = []
        for y in range(0, max(h - tile, 0) + 1, step):
            y2 = min(y + tile, h)
            if y2 - y < 64:
                continue
            for x in range(0, max(w - tile, 0) + 1, step):
                x2 = min(x + tile, w)
                if x2 - x < 64:
                    continue
                tiles.append(img_rgb[y:y2, x:x2])
        return tiles if tiles else [img_rgb]

    @staticmethod
    def _crop_square(img_rgb: np.ndarray, box) -> Optional[np.ndarray]:
        x, y, wd, ht = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        side = max(wd, ht)
        mx, my = int(side * 0.30), int(side * 0.30)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(img_rgb.shape[1], x + wd + mx), min(img_rgb.shape[0], y + ht + my)
        if x1 <= x0 or y1 <= y0:
            return None
        return img_rgb[y0:y1, x0:x1]

    def _aligned_embedding(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        # 1) landmark-based alignment over whole image or overlapping tiles
        #    (high-res photos of a card make the printed face too small for a
        #    single pass, so tile the image when it is large).
        for cand in self._tile_images(img_rgb):
            pts = self.landmarks5(cand)
            if pts is not None:
                aligned = self.align_face(cand, pts)
                return self.embed(aligned)
        # 2) fallback: any detected face box, crop (with margin) and embed
        for cand in self._tile_images(img_rgb):
            for box in self.detect_faces(cand):
                crop = self._crop_square(cand, box)
                if crop is None:
                    continue
                crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
                return self.embed(crop)
        return None

    # --------------------------------------------------- UltraFace helpers
    @staticmethod
    def _preprocess_ultraface(img: np.ndarray, size=(640, 480)) -> np.ndarray:
        """Resize to the model's fixed input and apply the reference scaling.

        The reference detector feeds BGR with ``(x - 127.5) / 127.5`` (i.e. roughly
        [-1, 1]); this must match or the network produces garbage detections.
        """
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32)
        x = (x - 127.5) / 127.5
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...])

    @staticmethod
    def _iou(a, b) -> float:
        ax, ay, aw, ah = a[0], a[1], a[2], a[3]
        bx, by, bw, bh = b[0], b[1], b[2], b[3]
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _nms(cls, dets: list, iou_thres: float = 0.4) -> list:
        """Greedy non-maximum suppression on (x, y, w, h, score) tuples."""
        dets = sorted(dets, key=lambda d: d[4], reverse=True)
        keep = []
        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [d for d in dets if cls._iou(best, d) < iou_thres]
        return keep

    @classmethod
    def _postprocess_ultraface(cls, outputs: dict, img_wh, conf_thres: float = 0.6,
                               iou_thres: float = 0.4) -> list:
        """Decode UltraFace output into (x, y, w, h) boxes in original pixels.

        ``version-RFB-640`` emits normalised (x1, y1, x2, y2) boxes and per-anchor
        scores, so we rescale by the source size and dedupe with NMS.
        """
        scores = outputs.get("scores")
        boxes = outputs.get("boxes")
        if scores is None or boxes is None or boxes.size == 0 or scores.size == 0:
            return []
        sc = np.squeeze(scores)
        if sc.ndim == 2 and sc.shape[1] >= 2:
            sc = sc[:, 1]
        else:
            sc = sc.reshape(-1)
        scores = sc
        boxes = np.squeeze(boxes).reshape(-1, 4)
        w, h = img_wh
        dets = []
        for i in range(boxes.shape[0]):
            s = float(scores[i])
            if s < conf_thres:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            x1 *= w; x2 *= w; y1 *= h; y2 *= h
            x1i = int(max(0, x1)); y1i = int(max(0, y1))
            x2i = int(min(w, x2)); y2i = int(min(h, y2))
            wd, ht = x2i - x1i, y2i - y1i
            if wd < 24 or ht < 24:
                continue
            dets.append((x1i, y1i, wd, ht, s))
        return [(d[0], d[1], d[2], d[3]) for d in cls._nms(dets, iou_thres)]

    def detect_face_boxes(self, img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            self._init_detector()
            inp = self._preprocess_ultraface(img_rgb, self._det_size)
            raw = self._det_session.run(self._det_output_names, {self._det_input_name: inp})
            outputs = {name: val for name, val in zip(self._det_output_names, raw)}
            boxes = self._postprocess_ultraface(outputs, (img_rgb.shape[1], img_rgb.shape[0]))
        except Exception:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            boxes = [(x, y, w, h) for (x, y, w, h) in faces]
        return [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes]

    def detect_faces(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, ...]:
        crops = []
        h, w, _ = img_rgb.shape
        for (x, y, wd, ht) in self.detect_face_boxes(img_rgb):
            x2 = min(w, x + wd)
            y2 = min(h, y + ht)
            x0 = max(0, x)
            y0 = max(0, y)
            if x2 <= x0 or y2 <= y0:
                continue
            crops.append(img_rgb[y0:y2, x0:x2])
        return tuple(crops)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None or a.size == 0 or b.size == 0:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        return float(np.dot(a, b) / denom)

    # ------------------------------------------------------------ matching
    def face_match(self, selfie_b: Optional[bytes], id_b: Optional[bytes]) -> Dict[str, Any]:
        if not selfie_b or not id_b:
            return {"match_score": 0.0, "status": "MISSING", "face_count_selfie": 0, "face_count_id": 0}
        selfie = self.load_image(selfie_b)
        id_img = self.load_image(id_b)

        selfie_emb = self._aligned_embedding(selfie)
        id_emb = self._aligned_embedding(id_img)

        if selfie_emb is None or id_emb is None:
            return {
                "match_score": 0.0,
                "status": "NO_FACE",
                "face_count_selfie": 0 if selfie_emb is None else 1,
                "face_count_id": 0 if id_emb is None else 1,
            }

        score = self.cosine_similarity(selfie_emb, id_emb)
        if score >= MATCH_THRESHOLD:
            status = "MATCH"
        elif score >= PARTIAL_THRESHOLD:
            status = "PARTIAL"
        else:
            status = "NO_MATCH"
        return {
            "match_score": round(score, 4),
            "status": status,
            "face_count_selfie": 1,
            "face_count_id": 1,
        }
