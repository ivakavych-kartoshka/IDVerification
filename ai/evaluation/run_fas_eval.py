"""Đánh giá chống giả mạo hiện diện (PAD/FAS) — APCER / BPCER / EER.

Theo ISO/IEC 30107-3:

* **APCER** (Attack Presentation Classification Error Rate) = tỉ lệ *tấn công*
  bị nhận nhầm là thật;
* **BPCER** (Bona fide Presentation Classification Error Rate) = tỉ lệ *người
  thật* bị từ chối;
* **ACER** = trung bình hai chỉ số trên; **EER** = điểm hai đường cắt nhau.

Chấm bằng ``MiniFASNetV2`` (``real_prob`` là điểm "thật"), KHÔNG qua
``LivenessService`` để tránh lẫn blur-fallback.

Bố cục dữ liệu::

    data/replayattack/
      real/            <- ảnh/video bona fide (đệ quy)
      attack/print/    <- tên thư mục con = loại tấn công
      attack/replay/

hoặc ``--real``/``--attack`` trỏ thẳng vào hai thư mục bất kỳ. Ảnh không phát
hiện được khuôn mặt bị **loại khỏi metric** và báo riêng (``no_face``) — với
bộ dữ liệu tấn công in/replay, ảnh mất mặt thường là lỗi detector chứ không
phải mô hình PAD.

Replay-Attack / MSU-MFSD / CASIA-FASD đều là **video**: file video được lấy
đều ``--frames-per-video`` frame (mặc định 5) và mỗi frame tính như một mẫu.

Ví dụ (sau khi đã tải Replay-Attack)::

    cd ai
    $env:PYTHONIOENCODING="utf-8"
    python evaluation/run_fas_eval.py `
        --real data/replayattack/real `
        --attack data/replayattack/attack `
        --out reports/fas_replayattack.json

Chỉ chạy ``--limit-real 60 --limit-attack 60`` để smoke test code; số liệu
smoke **không** dùng để báo cáo.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.face_service import FaceService  # noqa: E402
from app.services.fasnet import FASNetAntiSpoof  # noqa: E402


def _images(root: str, limit: int) -> List[Tuple[str, str]]:
    """(path, nhãn con) — nhãn con là tên thư mục cấp 1 dưới ``root``."""
    files: List[Tuple[str, str]] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.mp4", "*.avi", "*.mov", "*.mkv"):
        for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
            rel = os.path.relpath(p, root)
            sub = rel.split(os.sep)[0] if os.sep in rel else "attack"
            files.append((p, sub))
    files.sort()
    if limit and len(files) > limit:
        idx = np.linspace(0, len(files) - 1, limit).astype(int)
        files = [files[i] for i in sorted(set(idx.tolist()))]
    return files


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _iter_frames(path: str, frames_per_video: int):
    """Yield (frame BGR, tag) — ảnh: 1 frame; video: lấy đều N frame."""
    if os.path.splitext(path)[1].lower() not in VIDEO_EXTS:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            yield img, path
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    n = max(1, frames_per_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        got = 0
        while got < n:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, f"{path}#f{got}"
            got += 1
        cap.release()
        return
    for fi in np.linspace(0, total - 1, min(n, total)).astype(int).tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if ok and frame is not None:
            yield frame, f"{path}#f{fi}"
    cap.release()


def _stats(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {"n": int(arr.size), "median": round(float(np.median(arr)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2)}


def _rates(real: np.ndarray, attack: np.ndarray, thr: float) -> Dict[str, float]:
    apcer = float((attack >= thr).mean()) if attack.size else 0.0
    bpcer = float((real < thr).mean()) if real.size else 0.0
    return {"threshold": round(float(thr), 3), "apcer": round(apcer, 4),
            "bpcer": round(bpcer, 4), "acer": round((apcer + bpcer) / 2.0, 4)}


def _eer(real: np.ndarray, attack: np.ndarray) -> Dict[str, float]:
    """Điểm EER bằng cách quét hợp các ngưỡng ứng viên."""
    if real.size == 0 or attack.size == 0:
        return {"eer": 0.0, "threshold": 0.0}
    cand = np.unique(np.concatenate([real, attack]))
    best: Optional[Tuple[float, float, float]] = None
    for thr in cand:
        apcer = float((attack >= thr).mean())
        bpcer = float((real < thr).mean())
        gap = abs(apcer - bpcer)
        if best is None or gap < best[0]:
            best = (gap, thr, (apcer + bpcer) / 2.0)
    assert best is not None
    return {"eer": round(best[2], 4), "threshold": round(float(best[1]), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description="APCER/BPCER/EER cho MiniFASNetV2")
    ap.add_argument("--real", required=True, help="thư mục ảnh bona fide")
    ap.add_argument("--attack", required=True, help="thư mục ảnh tấn công (subfolder = loại)")
    ap.add_argument("--limit-real", type=int, default=0)
    ap.add_argument("--limit-attack", type=int, default=0)
    ap.add_argument("--frames-per-video", type=int, default=5,
                    help="số frame lấy đều từ mỗi video (Replay-Attack/MSU-MFSD)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="ngưỡng real_prob (mặc định = AI_FAS_REAL_THRESHOLD)")
    ap.add_argument("--no-detect", action="store_true",
                    help="coi cả ảnh là khuôn mặt (dùng cho dataset đã crop sẵn, vd CASIA-FASD crops)")
    ap.add_argument("--tag", default="", help="nhãn phiên chạy, vd 'smoke'")
    ap.add_argument("--scores-csv", default="")
    ap.add_argument("--out", default="reports/fas_eval.json")
    args = ap.parse_args()

    real_files = _images(args.real, args.limit_real)
    attack_files = _images(args.attack, args.limit_attack)
    if not real_files or not attack_files:
        print("[!] Thiếu ảnh ở --real hoặc --attack -> dừng.")
        return 2
    print(f"Bona fide: {len(real_files)} ảnh | Tấn công: {len(attack_files)} ảnh | tag={args.tag or '-'}")

    face = FaceService()
    fas = FASNetAntiSpoof()

    real_scores: List[float] = []
    attack_scores: List[Tuple[float, str]] = []
    lat: List[float] = []
    rows: List[Tuple[str, str, str, Optional[float], str, str]] = []
    n_no_face = 0

    def score_one(img_rgb: np.ndarray) -> Tuple[Optional[float], str, float]:
        if args.no_detect:
            h, w = img_rgb.shape[:2]
            boxes = [(0, 0, w, h)]
        else:
            boxes = face.detect_face_boxes(img_rgb)
        if not boxes:
            return None, "(no-face)", 0.0
        bbox = max(boxes, key=lambda b: b[2] * b[3])
        t0 = time.perf_counter()
        res = fas.predict(img_rgb, bbox)
        ms = (time.perf_counter() - t0) * 1000.0
        return float(res["real_prob"]), res.get("label", ""), ms

    def run(files: List[Tuple[str, str]], kind: str) -> None:
        nonlocal n_no_face
        for i, (path, sub) in enumerate(files, 1):
            for frame, tag in _iter_frames(path, args.frames_per_video):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                score, label, ms = score_one(rgb)
                if score is None:
                    n_no_face += 1
                    rows.append((tag, kind, sub, None, label, "no_face"))
                else:
                    lat.append(ms)
                    rows.append((tag, kind, sub, round(score, 4), label, ""))
                    if kind == "real":
                        real_scores.append(score)
                    else:
                        attack_scores.append((score, sub))
            if i % 50 == 0:
                print(f"  ... {kind} {i}/{len(files)}")

    run(real_files, "real")
    run(attack_files, "attack")

    real_arr = np.asarray(real_scores, dtype=np.float64)
    atk_arr = np.asarray([s for s, _ in attack_scores], dtype=np.float64)
    if real_arr.size == 0 or atk_arr.size == 0:
        print("[!] Không chấm được đủ hai lớp (thiếu mặt) -> dừng.")
        return 2

    at_point = _rates(real_arr, atk_arr, args.threshold)
    eer = _eer(real_arr, atk_arr)

    per_attack: Dict[str, Dict[str, object]] = {}
    for sub in sorted({s for _, s in attack_scores}):
        arr = np.asarray([s for s, ss in attack_scores if ss == sub], dtype=np.float64)
        per_attack[sub] = {"n": int(arr.size),
                           "apcer": round(float((arr >= args.threshold).mean()), 4),
                           "median_real_prob": round(float(np.median(arr)), 4)}
    # bonus: ngưỡng tại FPR (BPCER) = 5% / 10% để chọn vận hành
    ops: Dict[str, Dict[str, float]] = {}
    for bp in (0.05, 0.10):
        thr = float(np.percentile(real_arr, bp * 100))
        r = _rates(real_arr, atk_arr, thr)
        ops[f"bpcer_{int(bp * 100)}pct"] = r

    out = {
        "tag": args.tag,
        "real_dir": args.real, "attack_dir": args.attack,
        "n_real": int(real_arr.size), "n_attack": int(atk_arr.size),
        "n_no_face": n_no_face,
        "no_detect": bool(args.no_detect),
        "engine": "MiniFASNetV2 (real_prob)",
        "at_threshold": at_point,
        "eer": eer,
        "operating_points": ops,
        "per_attack": per_attack,
        "latency_ms": _stats(lat),
        "scores": {
            "real": {"mean": round(float(real_arr.mean()), 4),
                     "p10": round(float(np.percentile(real_arr, 10)), 4),
                     "median": round(float(np.median(real_arr)), 4)},
            "attack": {"mean": round(float(atk_arr.mean()), 4),
                       "p90": round(float(np.percentile(atk_arr, 90)), 4),
                       "median": round(float(np.median(atk_arr)), 4)},
        },
        "warning": "SMOKE - không dùng số liệu này để báo cáo" if args.tag == "smoke" else "",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    if args.scores_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.scores_csv)), exist_ok=True)
        with open(args.scores_csv, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["path", "class", "attack_type", "real_prob", "label", "note"])
            wr.writerows(rows)

    print(f"\nn_real={out['n_real']} n_attack={out['n_attack']} no_face={n_no_face}")
    print(f"@thr={args.threshold}: APCER={at_point['apcer']:.4f} "
          f"BPCER={at_point['bpcer']:.4f} ACER={at_point['acer']:.4f}")
    print(f"EER={eer['eer']:.4f} @ thr={eer['threshold']}")
    for name, r in ops.items():
        print(f"  {name}: thr={r['threshold']:.3f} APCER={r['apcer']:.4f} BPCER={r['bpcer']:.4f}")
    for sub, m in per_attack.items():
        print(f"  [{sub}] n={m['n']} APCER={m['apcer']:.4f} median_real_prob={m['median_real_prob']}")
    print(f"\nĐã ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
