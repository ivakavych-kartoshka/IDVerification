"""Phân tích CSV điểm từng ảnh từ ``run_tamper_eval.py`` theo device / attack_type.

Dùng để trả lời 2 câu hỏi mà con số tổng hợp (F1, AUC) không trả lời được:
1. Mô hình **lật chiều** ở nhóm nào? (AUC < 0.5 thường do device shift, không phải
   do nhãn đảo -> soi bảng device x label trước khi kết luận);
2. Mask head có thực sự sinh vùng nào không (``n_regions``) hay IoU=0 là do
   không dự đoán được vùng nào?

Ví dụ::

    cd ai
    $env:PYTHONIOENCODING="utf-8"
    python evaluation/analyze_tamper_scores.py --csv reports/tamper_test_scores.csv --thr 0.55
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
from typing import Dict, List, Optional


def _f(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _device(stem: str) -> str:
    return stem.split("__")[0] if "__" in stem else "(no-device)"


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0}
    vals = sorted(vals)
    n = len(vals)
    return {"n": n, "mean": sum(vals) / n, "median": vals[n // 2],
            "min": vals[0], "max": vals[-1]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Breakdown of per-image tamper scores")
    ap.add_argument("--csv", required=True, help="scores CSV từ run_tamper_eval.py")
    ap.add_argument("--thr", type=float, default=0.55, help="ngưỡng deploy để tính recall/FPR")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[!] Không thấy {args.csv}")
        return 2

    with open(args.csv, "r", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    print(f"{len(rows)} dòng từ {args.csv} | ngưỡng = {args.thr}")

    # ---- 1. device x label: điểm trung bình theo nhóm
    buckets: Dict[tuple, List[Dict[str, str]]] = collections.defaultdict(list)
    for r in rows:
        buckets[((_device(r["stem"])), r["label"])].append(r)

    def _score(r: Dict[str, str]) -> Optional[float]:
        return _f(r.get("tamper_score"))

    print("\n=== tamper_score theo device x nhãn ===")
    print(f"  {'device':14s} {'label':7s} {'n':>5s} {'mean':>7s} {'median':>7s} "
          f"{'pos@thr':>8s} {'model_prob':>11s}")
    for (dev, label), grp in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        sc = _stats([v for v in (_score(r) for r in grp) if v is not None])
        mp = _stats([v for v in (_f(r.get("model_prob")) for r in grp) if v is not None])
        pos = sum(1 for r in grp if (_score(r) or 0.0) >= args.thr)
        pct = 100.0 * pos / max(1, sc.get("n", 0))
        lab = "bonafide" if label == "0" else "forged"
        print(f"  {dev:14s} {lab:7s} {sc.get('n', 0):5d} {sc.get('mean', 0):7.3f} "
              f"{sc.get('median', 0):7.3f} {pct:7.1f}% {mp.get('mean', float('nan')):11.3f}")

    # ---- 2. attack_type
    print("\n=== theo attack_type (chỉ forged) ===")
    by_atk: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    for r in rows:
        if r["label"] == "1":
            by_atk[r.get("attack_type", "unknown")].append(r)
    print(f"  {'attack_type':22s} {'n':>5s} {'mean':>7s} {'recall@thr':>11s} "
          f"{'mean_model':>11s}")
    for atk, grp in sorted(by_atk.items(), key=lambda kv: -len(kv[1])):
        sc = _stats([v for v in (_score(r) for r in grp) if v is not None])
        mp = _stats([v for v in (_f(r.get("model_prob")) for r in grp) if v is not None])
        rec = sum(1 for r in grp if (_score(r) or 0.0) >= args.thr) / max(1, len(grp))
        print(f"  {atk:22s} {len(grp):5d} {sc.get('mean', 0):7.3f} {rec * 100:10.1f}% "
              f"{mp.get('mean', float('nan')):11.3f}")

    # ---- 3. heuristic signals & mask head
    sig = _stats([v for v in (_f(r.get("signal_score")) for r in rows) if v is not None])
    blk1 = _stats([_f(r.get("blockiness")) or 0.0 for r in rows if r["label"] == "0"])
    blk2 = _stats([_f(r.get("blockiness")) or 0.0 for r in rows if r["label"] == "1"])
    nreg = [int(_f(r.get("n_regions")) or 0) for r in rows]
    zero = sum(1 for v in nreg if v == 0)
    print("\n=== tín hiệu phụ ===")
    print(f"  heuristic signal_score: mean={sig.get('mean', 0):.3f} (n={sig.get('n', 0)})")
    print(f"  blockiness(norm) bona={blk1.get('mean', 0):.3f} forged={blk2.get('mean', 0):.3f}")
    print(f"  mask head: {zero}/{len(rows)} ảnh KHÔNG sinh vùng nào "
          f"({100.0 * zero / max(1, len(rows)):.1f}%) -> IoU=0 chủ yếu do không dự đoán")

    # ---- 4. photo box (dùng cho boundary/noise signals)
    pb = sum(1 for r in rows if (_f(r.get("photo_box")) or 0) >= 1)
    print(f"  tìm thấy vùng chân dung (photo_box): {pb}/{len(rows)} "
          f"({100.0 * pb / max(1, len(rows)):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
