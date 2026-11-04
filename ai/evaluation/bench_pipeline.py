"""Benchmark độ trễ từng module trên ảnh thật — số liệu cho Chương 4.

Đo tuần tự trên N ảnh lấy đều từ split test (mặc định 200):

* ``io``       : ``cv2.imread`` + chuyển BGR→RGB;
* ``detect``   : UltraFace ``version-RFB-640.onnx``;
* ``landmark`` : MediaPipe FaceLandmarker (5 điểm);
* ``embed``    : ArcFace ``recognition.onnx`` (căn chỉnh + suy luận 112x112);
* ``fas``      : MiniFASNetV2 (crop 80x80, scale 2.7, có TTA lật ngang);
* ``ocr``      : RapidOCR trên ảnh JPEG encode sẵn (đo riêng phần suy luận);
* ``qr``       : QRCodeDetector trên fixture ``tests/fixtures/cccd_back_qr.png``.

Module tamper **không** đo lại ở đây — đã có cột ``ms`` cho cả 1385 ảnh test
trong ``reports/tamper_test_scores.csv``; script đọc luôn file đó để in kèm
median/p95 nhằm đủ bảng độ trễ đầu-cuối.

Lưu ý phương pháp: đây là độ trễ *xử lý ảnh* trên máy đo, không gồm upload
mạng và hàng đợi. Lần chạy đầu mỗi module gồm cold-start (load model) nên
script chạy warm-up trước khi tính thống kê.

Ví dụ::

    cd ai
    $env:PYTHONIOENCODING="utf-8"
    python evaluation/bench_pipeline.py `
        --data data/fantasyid/prepared/test `
        --limit 200 `
        --out reports/latency.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.face_service import FaceService  # noqa: E402
from app.services.liveness_service import LivenessService  # noqa: E402
from app.services.ocr_service import OCRService  # noqa: E402
from app.services.qr_service import QRService  # noqa: E402


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "max": round(float(arr.max()), 2),
    }


def _sample_images(root: str, limit: int) -> List[str]:
    files: List[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    files = sorted(set(files))
    if not files:
        return []
    if limit and len(files) > limit:
        idx = np.linspace(0, len(files) - 1, limit).astype(int)
        files = [files[i] for i in sorted(set(idx.tolist()))]
    return files


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000.0, out


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark độ trễ pipeline")
    ap.add_argument("--data", default="data/fantasyid/prepared/test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--qr-fixture", default="tests/fixtures/cccd_back_qr.png")
    ap.add_argument("--qr-repeats", type=int, default=20)
    ap.add_argument("--tamper-csv", default="reports/tamper_test_scores.csv")
    ap.add_argument("--out", default="reports/latency.json")
    args = ap.parse_args()

    files = _sample_images(args.data, args.limit)
    if not files:
        print(f"[!] Không tìm thấy ảnh trong {args.data}")
        return 2
    print(f"Benchmark {len(files)} ảnh từ {args.data}")

    face = FaceService()
    live = LivenessService()
    ocr = OCRService()
    qr = QRService()

    # Warm-up: load model + JIT cho từng module trước khi đo.
    wimg = cv2.imread(files[0], cv2.IMREAD_COLOR)
    wrgb = cv2.cvtColor(wimg, cv2.COLOR_BGR2RGB)
    face.detect_face_boxes(wrgb)
    face.landmarks5(wrgb)
    live.passive_liveness(wrgb)
    ok, wbuf = cv2.imencode(".jpg", wimg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        ocr.extract(wbuf.tobytes())
    if os.path.exists(args.qr_fixture):
        with open(args.qr_fixture, "rb") as fh:
            qr.decode(fh.read())
    print("Warm-up xong\n")

    acc: Dict[str, List[float]] = {k: [] for k in
                                   ("io", "detect", "landmark", "embed", "fas", "ocr")}
    n_no_face = 0
    n_embed = 0
    n_ocr_ok = 0
    for i, path in enumerate(files, 1):
        ms, img = _timed(lambda p=path: cv2.imread(p, cv2.IMREAD_COLOR))
        if img is None:
            continue
        acc["io"].append(ms)
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        acc["io"].append((time.perf_counter() - t0) * 1000.0)

        ms, boxes = _timed(lambda: face.detect_face_boxes(rgb))
        acc["detect"].append(ms)

        ms, lms = _timed(lambda: face.landmarks5(rgb))
        acc["landmark"].append(ms)

        if lms is not None and lms.shape == (5, 2):
            aligned = face.align_face(rgb, lms)
            ms, _emb = _timed(lambda: face.embed(aligned))
            acc["embed"].append(ms)
            n_embed += 1

        if boxes:
            bbox = max(boxes, key=lambda b: b[2] * b[3])
            ms, _res = _timed(lambda: live.passive_liveness(rgb, bbox))
            acc["fas"].append(ms)
        else:
            n_no_face += 1

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            try:
                ms, _txt = _timed(lambda: ocr.extract(buf.tobytes()))
                acc["ocr"].append(ms)
                n_ocr_ok += 1
            except Exception as exc:  # OCR lỗi -> bỏ ảnh đó, không làm hỏng bench
                print(f"  [ocr] lỗi ở {os.path.basename(path)}: {exc}")

        if i % 25 == 0:
            print(f"  ... {i}/{len(files)}")

    qr_ms: List[float] = []
    if os.path.exists(args.qr_fixture):
        with open(args.qr_fixture, "rb") as fh:
            qr_bytes = fh.read()
        for _ in range(args.qr_repeats):
            ms, _r = _timed(lambda: qr.decode(qr_bytes))
            qr_ms.append(ms)

    tamper_ms: List[float] = []
    if os.path.exists(args.tamper_csv):
        with open(args.tamper_csv, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                v = row.get("ms") or ""
                if v:
                    tamper_ms.append(float(v))

    out = {
        "data": args.data,
        "n_images": len(files),
        "n_with_face": len(acc["fas"]),
        "n_no_face": n_no_face,
        "n_embed": n_embed,
        "n_ocr_ok": n_ocr_ok,
        "components": {k: _stats(v) for k, v in acc.items()},
        "qr": {**_stats(qr_ms), "n_repeats": args.qr_repeats, "fixture": args.qr_fixture},
        "tamper_ref": {**_stats(tamper_ms), "source": args.tamper_csv},
    }
    med = {k: out["components"][k]["median"] for k in out["components"]}
    med["qr"] = out["qr"]["median"]
    med["tamper"] = out["tamper_ref"]["median"]
    out["estimated_sequential_total_ms"] = round(sum(med.values()), 1)
    out["estimated_sequential_breakdown"] = med

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"\n{'module':<10} {'n':>5} {'median':>8} {'p95':>8} {'mean':>8} {'max':>9}")
    print("-" * 54)
    for k in ("io", "detect", "landmark", "embed", "fas", "ocr"):
        s = out["components"][k]
        print(f"{k:<10} {s['n']:>5} {s['median']:>8.1f} {s['p95']:>8.1f} "
              f"{s['mean']:>8.1f} {s['max']:>9.1f}")
    s = out["qr"]
    print(f"{'qr':<10} {s['n']:>5} {s['median']:>8.1f} {s['p95']:>8.1f} "
          f"{s['mean']:>8.1f} {s['max']:>9.1f}")
    s = out["tamper_ref"]
    print(f"{'tamper*':<10} {s['n']:>5} {s['median']:>8.1f} {s['p95']:>8.1f} "
          f"{s['mean']:>8.1f} {s['max']:>9.1f}")
    print("-" * 54)
    print(f"Tổng tuần tự (median): {out['estimated_sequential_total_ms']} ms")
    print(f"(* tamper đọc từ {args.tamper_csv}; QR đo trên fixture)")
    print(f"\nĐã ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
