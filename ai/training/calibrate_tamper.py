"""Calibrate ngưỡng cho module tamper (hướng A).

Hai việc, chạy trên split val (KHÔNG dùng test):

1. **Ngưỡng quyết định** ``AI_TAMPER_THRESHOLD`` / ``AI_TAMPER_SUSPECT``:
   - quét ngưỡng trên score fusion (``w*model + (1-w)*signals``) đúng như
     ``tamper_service.analyze`` tính, tìm ngưỡng cho F1 image-level cao nhất
     (metric DeepID Track 1);
   - in thêm operating point ở FPR = 10% như cách BTC DeepID báo cáo
     (threshold đặt sao cho FPR val = 10%) để so với con số FNR ~50% của
     baseline TruFor trong paper.

2. **Hằng số chuẩn hoá tín hiệu heuristic** ``SIGNAL_NORM``:
   - tính phân bố raw của 4 tín hiệu (boundary_energy, noise_step, blockiness,
     low_hf_fraction) tách theo lớp bona fide / forged;
   - in median + p90/p10 để thấy tín hiệu nào thực sự phân biệt được lớp
     (tín hiệu nào overlap nhiều thì nên hạ trọng số trong SIGNAL_WEIGHTS).

Ví dụ::

    cd ai
    python training/calibrate_tamper.py ^
        --data D:/datasets/fantasyid/FantasyID/val ^
        --weights models/tamper_net.pth

Output: JSON ở ``--out`` (mặc định ``models/tamper_calibration.json``) để commit
vào repo làm hồ sơ cho ngưỡng đang dùng.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import tamper_service as ts  # noqa: E402
from app.services.tamper_service import SIGNAL_WEIGHTS, TamperService  # noqa: E402
from training.train_tamper import CardTamperDataset  # noqa: E402


def _f1_at(scores: np.ndarray, labels: np.ndarray, thr: float) -> Tuple[float, float, float]:
    pred = (scores >= thr).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def _threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float = 0.10) -> float:
    """Ngưỡng sao cho FPR trên bona fide val = ``fpr`` (cách BTC DeepID báo cáo)."""
    neg = np.sort(scores[labels == 0])
    if neg.size == 0:
        return 0.5
    idx = int(np.clip(np.ceil((1.0 - fpr) * neg.size) - 1, 0, neg.size - 1))
    return float(neg[idx])


def _percentiles(values: List[float]) -> Optional[Dict[str, float]]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    return {"n": int(arr.size), "median": round(float(np.median(arr)), 4),
            "p10": round(float(np.percentile(arr, 10)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate ngưỡng tamper")
    ap.add_argument("--data", required=True, help="root split val (tự nhận layout)")
    ap.add_argument("--weights", default="models/tamper_net.pth", help="checkpoint hoặc .onnx")
    ap.add_argument("--out", default="models/tamper_calibration.json")
    ap.add_argument("--model-weight", type=float, default=0.7)
    ap.add_argument("--max-images", type=int, default=0, help="0 = dùng hết")
    ap.add_argument("--val-manifest", default="", help="chỉ chấm các ảnh trong file này (đúng split val lúc train)")
    ap.add_argument("--manifest-csv", default="", help="manifest.csv của prepare_fantasyid (stem,label,attack_type) để báo cáo theo loại tấn công")
    args = ap.parse_args()

    if os.path.exists(args.weights):
        os.environ["AI_TAMPER_MODEL"] = os.path.abspath(args.weights)
    else:
        print(f"[!] Không thấy {args.weights} -> calibrate ở chế độ heuristic-only.")
    os.environ["AI_TAMPER_MODEL_WEIGHT"] = str(args.model_weight)

    dataset = CardTamperDataset(args.data, train=False)
    if args.val_manifest and os.path.exists(args.val_manifest):
        with open(args.val_manifest, "r", encoding="utf-8") as fh:
            wanted = {ln.strip() for ln in fh if ln.strip()}
        dataset.samples = [s for s in dataset.samples if s[0] in wanted]
        print(f"[split] lọc theo {args.val_manifest}: {len(dataset.samples)} ảnh val")

    attack_of: Dict[str, str] = {}
    if args.manifest_csv and os.path.exists(args.manifest_csv):
        import csv as _csv
        with open(args.manifest_csv, "r", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                attack_of[row["stem"]] = row.get("attack_type", "none")
        print(f"[attack] đọc {len(attack_of)} dòng manifest {args.manifest_csv}")
    svc = TamperService()
    print(f"engine={svc._load_backend()} | model_weight={args.model_weight}")

    scores: List[float] = []
    labels: List[int] = []
    prob_scores: List[float] = []
    attack_types: List[str] = []
    signals: Dict[str, Dict[str, List[float]]] = {k: {"bona": [], "forged": []} for k in SIGNAL_WEIGHTS}
    raw: Dict[str, Dict[str, List[float]]] = {k: {"bona": [], "forged": []} for k in SIGNAL_WEIGHTS}

    limit = args.max_images or len(dataset)
    import cv2

    for i in range(min(limit, len(dataset))):
        path, label, _mask = dataset.samples[i]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = svc.analyze(img)
        scores.append(float(res["tamper_score"]))
        labels.append(int(label))
        stem = os.path.splitext(os.path.basename(path))[0]
        attack_types.append(attack_of.get(stem, "unknown") if label == 1 else "bonafide")
        if res["model_prob"] is not None:
            prob_scores.append(float(res["model_prob"]))
        key = "forged" if label == 1 else "bona"
        for name in SIGNAL_WEIGHTS:
            nv = res["signals"]["norm"].get(name)
            rv = res["signals"]["raw"].get(name)
            if nv is not None:
                signals[name][key].append(float(nv))
            if rv is not None:
                raw[name][key].append(float(rv))
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{min(limit, len(dataset))}")

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    print(f"\nĐã chấm {len(s)} ảnh ({int((y == 1).sum())} forged / {int((y == 0).sum())} bona fide)")

    # --- quét ngưỡng tối ưu F1 ---
    grid = np.linspace(0.05, 0.95, 91)
    rows = [(_f1_at(s, y, float(t)), float(t)) for t in grid]
    (best_f1, best_p, best_r), best_thr = max(rows, key=lambda r: r[0][0])

    thr_fpr10 = _threshold_at_fpr(s, y, 0.10)
    f1_fpr10 = _f1_at(s, y, thr_fpr10)

    print("\n=== Ngưỡng quyết định ===")
    print(f"best F1   = {best_f1:.4f}  @ thr={best_thr:.3f}  (P={best_p:.3f} R={best_r:.3f})")
    print(f"FPR=10%   -> thr={thr_fpr10:.3f}  F1={f1_fpr10[0]:.4f} "
          f"(P={f1_fpr10[1]:.3f} R={f1_fpr10[2]:.3f}, FNR={1 - f1_fpr10[2]:.3f})")
    print("Gợi ý đặt:  AI_TAMPER_THRESHOLD=<best thr>  AI_TAMPER_SUSPECT="
          f"{max(0.05, round(best_thr - 0.15, 2))}")

    # --- phân bố tín hiệu heuristic ---
    print("\n=== Tín hiệu heuristic (norm) — tín hiệu nào tách được lớp? ===")
    sep = {}
    for name in SIGNAL_WEIGHTS:
        b, f = signals[name]["bona"], signals[name]["forged"]
        pb, pf = _percentiles(b), _percentiles(f)
        if pb and pf:
            delta = abs(pf["median"] - pb["median"])
            sep[name] = round(float(delta), 4)
        print(f"  {name:16s} bona p50={pb['median'] if pb else None} "
              f"| forged p50={pf['median'] if pf else None} | |Δmedian|={sep.get(name)}")
    useful = sorted(sep.items(), key=lambda kv: kv[1], reverse=True)
    print("  Xếp hạng khả năng phân biệt:", [f"{k}={v}" for k, v in useful])
    print("  -> tín hiệu |Δmedian| ~ 0 có thể hạ/huỷ trọng số trong SIGNAL_WEIGHTS.")

    if prob_scores:
        ps = np.asarray(prob_scores, dtype=np.float64)
        m = _f1_at(ps, y, 0.5)
        print(f"\n[model-only] F1@0.5 = {m[0]:.4f} (P={m[1]:.3f} R={m[2]:.3f})")
        print(f"[heuristic-only] fusion score F1@best = {best_f1:.4f}")

    # --- F1 theo loại tấn công (face / text / face_text) ---
    per_attack: Dict[str, Dict[str, float]] = {}
    if any(t != "unknown" for t in attack_types):
        print(f"\n=== F1 theo loại tấn công (thr={best_thr:.3f}, recall trong lớp forged) ===")
        for atk in sorted(set(t for t in attack_types if t != "bonafide")):
            m = np.array([t == atk for t in attack_types])
            if m.sum() == 0:
                continue
            f1a, pa, ra = _f1_at(s[m], y[m], best_thr)
            per_attack[atk] = {"n": int(m.sum()), "f1": round(f1a, 4),
                               "precision": round(pa, 4), "recall": round(ra, 4)}
            print(f"  {atk:12s} n={int(m.sum()):5d}  F1={f1a:.4f}  P={pa:.4f}  R={ra:.4f}")

    payload = {
        "data": args.data,
        "weights": args.weights,
        "model_weight": args.model_weight,
        "n_images": int(len(s)),
        "n_forged": int((y == 1).sum()),
        "n_bona_fide": int((y == 0).sum()),
        "engine": svc._load_backend(),
        "recommended": {
            "AI_TAMPER_THRESHOLD": round(best_thr, 3),
            "AI_TAMPER_SUSPECT": round(max(0.05, best_thr - 0.15), 3),
            "best_f1": round(best_f1, 4),
            "precision": round(best_p, 4),
            "recall": round(best_r, 4),
        },
        "operating_point_fpr10": {
            "threshold": round(thr_fpr10, 3),
            "f1": round(f1_fpr10[0], 4),
            "precision": round(f1_fpr10[1], 4),
            "recall": round(f1_fpr10[2], 4),
            "fnr": round(1 - f1_fpr10[2], 4),
        },
        "signal_separation": sep,
        "per_attack_type": per_attack,
        "signal_norm_percentiles": {
            name: {"bona_fide": _percentiles(signals[name]["bona"]),
                   "forged": _percentiles(signals[name]["forged"])}
            for name in SIGNAL_WEIGHTS
        },
        "signal_raw_percentiles": {
            name: {"bona_fide": _percentiles(raw[name]["bona"]),
                   "forged": _percentiles(raw[name]["forged"])}
            for name in SIGNAL_WEIGHTS
        },
        "module_constants": {
            "SIGNAL_WEIGHTS": SIGNAL_WEIGHTS,
            "SIGNAL_NORM": {k: "xem tamper_service.SIGNAL_NORM" for k in ts.SIGNAL_NORM},
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nĐã ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
