"""Ablation trọng số fusion của module tamper — bảng cho Chương 4.

Không chạy lại mạng: đọc điểm per-image đã dump ở ``--scores-csv`` của
``training/calibrate_tamper.py`` (val) và ``evaluation/run_tamper_eval.py``
(test). Mỗi dòng CSV chứa ``tamper_score`` (fusion tại w=0.7), ``model_prob``
và ``signal_score``; vì ``score(w) = w*model + (1-w)*signal`` là tuyến tính nên
tái tạo được điểm cho mọi w bằng một phép biến đổi.

Quy tắc phương pháp luận giữ nguyên: **ngưỡng chọn trên val, áp lên test**.
Với mỗi w, quét ngưỡng tối đa F1 trên val rồi báo F1/AUC/AP/FPR trên test.

Ví dụ::

    cd ai
    $env:PYTHONIOENCODING="utf-8"
    python evaluation/ablate_tamper_weights.py `
        --val-csv reports/tamper_val_scores.csv `
        --test-csv reports/tamper_test_scores.csv `
        --out reports/tamper_ablation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _read_scores(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Trả về (model, signal, label, fused@w0, attack_types)."""
    model: List[float] = []
    signal: List[float] = []
    labels: List[int] = []
    fused: List[float] = []
    attacks: List[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            mp = row.get("model_prob") or ""
            sg = row.get("signal_score") or ""
            # Ảnh thiếu model_prob (chế độ heuristic-only) -> coi model=signal
            # để không làm lệch phép nội suy tuyến tính.
            m = float(mp) if mp else (float(sg) if sg else 0.0)
            s = float(sg) if sg else 0.0
            model.append(m)
            signal.append(s)
            labels.append(int(row["label"]))
            fused.append(float(row["tamper_score"]))
            attacks.append(row.get("attack_type") or "unknown")
    return (np.asarray(model, dtype=np.float64), np.asarray(signal, dtype=np.float64),
            np.asarray(labels, dtype=np.int32), np.asarray(fused, dtype=np.float64),
            attacks)


def _prf(scores: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (scores >= thr).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(float(thr), 3),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "fnr": round(1.0 - recall, 4),
    }


def _auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos_sum = float(ranks[labels == 1].sum())
    return (pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _ap(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Average Precision (diện tích dưới đường precision-recall)."""
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        if r > prev_r:
            ap += float(p) * float(r - prev_r)
            prev_r = r
    return ap


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_thr, best = 0.5, {"f1": -1.0}
    for thr in np.linspace(0.05, 0.95, 181):
        m = _prf(scores, labels, float(thr))
        if m["f1"] > best["f1"]:
            best_thr, best = float(thr), m
    return best_thr, best


def main() -> int:
    ap = argparse.ArgumentParser(description="Ablation trọng số fusion tamper")
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--weights", default="0,0.3,0.5,0.7,1.0",
                    help="danh sách w (0 = chỉ heuristic, 1 = chỉ model)")
    ap.add_argument("--out", default="reports/tamper_ablation.json")
    args = ap.parse_args()

    vm, vs, vy, _vf, _va = _read_scores(args.val_csv)
    tm, ts, ty, tf, tatt = _read_scores(args.test_csv)
    if vm.size == 0 or tm.size == 0:
        print("[!] CSV rỗng -> dừng.")
        return 2

    grid = [float(w) for w in args.weights.split(",") if w.strip()]
    rows: List[Dict[str, object]] = []
    for w in grid:
        v = w * vm + (1.0 - w) * vs
        t = w * tm + (1.0 - w) * ts
        thr, val_best = _best_threshold(v, vy)
        test_at = _prf(t, ty, thr)
        rows.append({
            "w_model": round(w, 3),
            "val_best_f1": val_best["f1"],
            "val_thr": round(thr, 3),
            "val_auc": None if _auc(v, vy) is None else round(float(_auc(v, vy)), 4),
            "test_auc": None if _auc(t, ty) is None else round(float(_auc(t, ty)), 4),
            "test_ap": None if _ap(t, ty) is None else round(float(_ap(t, ty)), 4),
            "test": test_at,
            "test_f1_at_deployed_055": _prf(t, ty, 0.55)["f1"],
        })

    # Tách theo loại tấn công ở cấu hình đang triển khai (w=0.7, thr=0.55)
    deployed = next((r for r in rows if abs(float(r["w_model"]) - 0.7) < 1e-9), None)
    per_attack: Dict[str, Dict[str, float]] = {}
    for atk in sorted(set(tatt)):
        idx = np.asarray([a == atk for a in tatt])
        if not idx.any():
            continue
        m = _prf(tf[idx], ty[idx], 0.55)
        per_attack[atk] = {"n": int(idx.sum()), **m}
    if deployed is not None:
        deployed["test_per_attack_at_055"] = per_attack

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "val_csv": args.val_csv, "test_csv": args.test_csv,
        "n_val": int(vm.size), "n_test": int(tm.size),
        "note": "score(w) = w*model + (1-w)*signal; ngưỡng chọn trên val, áp lên test",
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"\nAblation fusion (val n={vm.size}, test n={tm.size})")
    print(f"{'w_model':>7} | {'valF1':>6} {'valThr':>6} {'valAUC':>7} | "
          f"{'testF1':>6} {'testAUC':>7} {'testAP':>6} {'F1@.55':>6} {'FPR@thr':>7}")
    print("-" * 78)
    for r in rows:
        t = r["test"]  # type: ignore[assignment]
        print(f"{r['w_model']:>7} | {r['val_best_f1']:>6.4f} {r['val_thr']:>6.3f} "
              f"{(r['val_auc'] if r['val_auc'] is not None else float('nan')):>7.4f} | "
              f"{t['f1']:>6.4f} "
              f"{(r['test_auc'] if r['test_auc'] is not None else float('nan')):>7.4f} "
              f"{(r['test_ap'] if r['test_ap'] is not None else float('nan')):>6.4f} "
              f"{r['test_f1_at_deployed_055']:>6.4f} {t['fpr']:>7.4f}")
    print(f"\nĐã ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
