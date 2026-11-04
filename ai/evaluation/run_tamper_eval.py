"""Đánh giá module tamper trên split giữ lại (held-out) - dùng cho Chương 4.

Khác ``training/calibrate_tamper.py`` (quét ngưỡng trên **val** để *chọn* ngưỡng),
script này **áp ngưỡng đã chọn** lên split test và báo cáo số liệu trung thực:

* image-level: Accuracy / Precision / Recall / F1 / FPR / FNR tại mỗi ngưỡng,
  cộng AUC + AP (không phụ thuộc ngưỡng) - metric DeepID Track 1;
* theo loại tấn công (``face`` / ``text`` / ``face_text`` ...) qua ``manifest.csv``
  của ``prepare_fantasyid.py``;
* localization: IoU giữa vùng model báo (union các ``regions``) và mask GT
  ``region_provenance == "altered"``;
* độ trễ ``ms/ảnh`` của riêng module tamper (để phân tích độ trễ đầu-cuối).

Ví dụ::

    cd ai
    $env:PYTHONIOENCODING="utf-8"
    python evaluation/run_tamper_eval.py `
        --data data/fantasyid/prepared/test `
        --weights models/tamper_net.pth `
        --thresholds 0.55,0.57,0.675 `
        --manifest-csv data/fantasyid/prepared/test/manifest.csv `
        --out reports/tamper_test.json

Quy tắc phương pháp luận: ngưỡng lấy từ ``models/tamper_calibration.json``
(val), **không** chọn lại trên test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app.services.tamper_service import TamperService  # noqa: E402
from training.train_tamper import CardTamperDataset  # noqa: E402


def _prf(scores: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, float]:
    """Precision/Recall/F1 + FPR/FNR tại một ngưỡng cố định."""
    pred = (scores >= thr).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": round(float(thr), 3),
        "accuracy": round((tp + tn) / max(1, scores.size), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),          # = TPR = 1 - FNR
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(1.0 - recall, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """AUC phi ngưỡng bằng thống hạng (Mann-Whitney), có xử lý hoà điểm."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    s_sorted = scores[order]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # trung bình hạng cho hoà điểm
        i = j + 1
    sum_pos = float(ranks[labels == 1].sum())
    return round((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), 4)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """AP (area under precision-recall) - nhạy với tập mất cân bằng hơn AUC."""
    if int((labels == 1).sum()) == 0 or int((labels == 0).sum()) == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y == 1)
    precision = tp / np.arange(1, y.size + 1, dtype=np.float64)
    recall = tp / float((labels == 1).sum())
    # lấy precision tại mỗi mức recall tăng dần (điều hoà hoà điểm)
    ap = float(np.sum((recall - np.concatenate(([0.0], recall[:-1]))) * precision))
    return round(ap, 4)


def _iou_regions_vs_gt(regions: List[Dict[str, object]], gt_mask: np.ndarray) -> Optional[float]:
    """IoU union các vùng dự đoán (box ảnh gốc) với mask GT ``altered``."""
    h, w = gt_mask.shape[:2]
    if h == 0 or w == 0:
        return None
    pred = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        x, y, bw, bh = r["box"]
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(w, int(round(x + bw)))
        y1 = min(h, int(round(y + bh)))
        if x1 > x0 and y1 > y0:
            pred[y0:y1, x0:x1] = 1
    gt = (gt_mask > 0).astype(np.uint8)
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    if union == 0:
        return None
    return round(inter / union, 4)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate tamper module on a held-out split")
    ap.add_argument("--data", required=True, help="root split test (layout bona_fide/ forged/ masks/)")
    ap.add_argument("--weights", default="models/tamper_net.pth")
    ap.add_argument("--thresholds", default="0.55,0.57,0.675",
                    help="danh sách ngưỡng áp dụng (nguồn: models/tamper_calibration.json)")
    ap.add_argument("--model-weight", type=float, default=0.7)
    ap.add_argument("--manifest-csv", default="", help="manifest.csv để tách theo loại tấn công")
    ap.add_argument("--max-images", type=int, default=0, help="0 = dùng hết")
    ap.add_argument("--skip-localization", action="store_true", help="không tính IoU (nhanh hơn)")
    ap.add_argument("--scores-csv", default="",
                    help="ghi điểm từng ảnh ra CSV (path,stem,label,attack_type,score,model_prob,n_regions,ms)")
    ap.add_argument("--out", default="reports/tamper_eval.json")
    args = ap.parse_args()

    if os.path.exists(args.weights):
        os.environ["AI_TAMPER_MODEL"] = os.path.abspath(args.weights)
    else:
        print(f"[!] Không thấy {args.weights} -> chỉ chấm heuristic (model_loaded=false).")
    os.environ["AI_TAMPER_MODEL_WEIGHT"] = str(args.model_weight)

    dataset = CardTamperDataset(args.data, train=False)
    print(f"[split] {args.data}: {len(dataset)} ảnh")

    attack_of: Dict[str, str] = {}
    if args.manifest_csv and os.path.exists(args.manifest_csv):
        import csv as _csv
        with open(args.manifest_csv, "r", encoding="utf-8-sig") as fh:
            for row in _csv.DictReader(fh):
                attack_of[row["stem"]] = row.get("attack_type", "none")
        print(f"[attack] đọc {len(attack_of)} dòng manifest")

    svc = TamperService()
    engine = svc._load_backend()
    print(f"engine={engine} | model_weight={args.model_weight} | "
          f"deployed thr={svc.threshold} suspect={svc.suspect}")

    scores: List[float] = []
    model_probs: List[float] = []
    labels: List[int] = []
    attack_types: List[str] = []
    ious: List[float] = []
    lat_ms: List[float] = []
    rows: List[Dict[str, object]] = []   # per-image, tuỳ chọn ghi ra CSV

    limit = args.max_images or len(dataset)
    n = min(limit, len(dataset))
    for i in range(n):
        path, label, mask_path = dataset.samples[i]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        res = svc.analyze(img_rgb)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

        scores.append(float(res["tamper_score"]))
        labels.append(int(label))
        stem = _stem(path)
        attack_types.append(attack_of.get(stem, "unknown") if label == 1 else "bonafide")
        if res["model_prob"] is not None:
            model_probs.append(float(res["model_prob"]))
        rows.append({
            "path": path, "stem": stem, "label": int(label),
            "attack_type": attack_types[-1], "tamper_score": float(res["tamper_score"]),
            "model_prob": res["model_prob"], "n_regions": len(res.get("regions") or []),
            "photo_box": 1 if res.get("photo_box") else 0,
            "signal_score": (res.get("signals") or {}).get("signal_score"),
            "blockiness": ((res.get("signals") or {}).get("norm") or {}).get("blockiness"),
            "ms": round(lat_ms[-1], 2),
        })

        if not args.skip_localization and label == 1 and mask_path and os.path.isfile(mask_path):
            gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                if gt.shape[:2] != img.shape[:2]:
                    gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                v = _iou_regions_vs_gt(res.get("regions") or [], gt)
                if v is not None:
                    ious.append(v)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{n}")

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    if s.size == 0:
        print("[!] Không chấm được ảnh nào - kiểm tra --data và layout thư mục.")
        return 2

    print(f"\nĐã chấm {s.size} ảnh ({int((y == 1).sum())} forged / {int((y == 0).sum())} bona fide)")
    thr_list = [float(t) for t in str(args.thresholds).split(",") if t.strip()]

    print("\n=== Image-level (áp ngưỡng đã hiệu chuẩn trên val) ===")
    per_thr = {}
    for thr in thr_list:
        m = _prf(s, y, thr)
        per_thr[f"{thr:g}"] = m
        print(f"  thr={m['threshold']:<6} F1={m['f1']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  ACC={m['accuracy']:.4f}  FPR={m['fpr']:.4f}  FNR={m['fnr']:.4f}")

    auc = _auc(s, y)
    ap_all = _average_precision(s, y)
    print(f"\nAUC={auc}  AP={ap_all}  (không phụ thuộc ngưỡng)")

    per_attack: Dict[str, Dict[str, float]] = {}
    if any(t not in ("unknown", "bonafide") for t in attack_types):
        print("\n=== Theo loại tấn công (tại ngưỡng deploy của service) ===")
        atk_arr = np.asarray(attack_types)
        for atk in sorted({t for t in attack_types if t != "bonafide"}):
            m = atk_arr == atk
            sub_y = y[m]
            if int(sub_y.sum()) == 0:
                continue
            rec = _prf(s[m], sub_y, svc.threshold)
            per_attack[atk] = {"n": int(m.sum()), "recall": rec["recall"],
                               "precision_vs_bona": rec["precision"]}
            print(f"  {atk:12s} n={int(m.sum()):5d}  recall@{svc.threshold:g}={rec['recall']:.4f}")

    if ious:
        arr = np.asarray(ious, dtype=np.float64)
        print(f"\nLocalization IoU: mean={arr.mean():.4f} median={np.median(arr):.4f} "
              f"p90={np.percentile(arr, 90):.4f}  (n={arr.size})")
    else:
        print("\nLocalization IoU: không có mask GT (bỏ qua)")

    lat = np.asarray(lat_ms, dtype=np.float64)
    print(f"\nĐộ trễ module tamper: median={np.median(lat):.1f} ms  "
          f"p95={np.percentile(lat, 95):.1f} ms  (engine={engine})")

    payload = {
        "data": args.data,
        "weights": args.weights,
        "engine": engine,
        "model_weight": args.model_weight,
        "deployed_threshold": svc.threshold,
        "n_images": int(s.size),
        "n_forged": int((y == 1).sum()),
        "n_bona_fide": int((y == 0).sum()),
        "auc": auc,
        "average_precision": ap_all,
        "per_threshold": per_thr,
        "per_attack_type": per_attack,
        "localization_iou": None if not ious else {
            "n": len(ious), "mean": round(float(np.mean(ious)), 4),
            "median": round(float(np.median(ious)), 4),
        },
        "latency_ms": {"median": round(float(np.median(lat)), 2),
                       "p95": round(float(np.percentile(lat, 95)), 2)},
        "model_only_auc": _auc(np.asarray(model_probs, dtype=np.float64), y) if model_probs else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nĐã ghi {args.out}")

    if args.scores_csv and rows:
        import csv as _csv
        os.makedirs(os.path.dirname(os.path.abspath(args.scores_csv)), exist_ok=True)
        with open(args.scores_csv, "w", encoding="utf-8", newline="") as fh:
            wr = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"Đã ghi {args.scores_csv} ({len(rows)} dòng) -> phân tích device/attack không cần suy luận lại")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
