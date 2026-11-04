"""Phân tích thành phần dataset (device / attack_type / nhãn) - không cần suy luận.

Mục đích: phát hiện **shortcut confound** trước khi tin vào số đo accuracy.
Nếu thiết bị quay/chụp tương gần như tuyến tính với nhãn (ví dụ forged chủ yếu
từ ``scan`` còn bona fide chủ yếu từ ``iphone``), mô hình có thể học dấu vết
thiết bị thay vì dấu vết giả mạo -> điểm trên val cao nhưng lật chiều trên test.

Ví dụ::

    cd ai
    $env:PYTHONIOENCODING="utf-8"
    python evaluation/diag_dataset_composition.py `
        --prepared data/fantasyid/prepared

In bảng device x label, attack_type x label, và %forged theo device cho train/test.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
from typing import Dict, List


def _load(manifest: str) -> List[Dict[str, str]]:
    if not os.path.isfile(manifest):
        return []
    with open(manifest, "r", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _device(stem: str) -> str:
    return stem.split("__")[0] if "__" in stem else "(no-device)"


def _report(rows: List[Dict[str, str]], name: str) -> None:
    if not rows:
        print(f"=== {name}: không có manifest ===")
        return
    tab: Dict[tuple, int] = collections.Counter((_device(r["stem"]), r["label"]) for r in rows)
    atk: Dict[tuple, int] = collections.Counter(
        (r.get("attack_type", "none") if r["label"] == "1" else "bonafide", r["label"]) for r in rows
    )
    devs = sorted({d for d, _ in tab}, key=lambda d: -sum(tab[(d, l)] for l in ("0", "1")))
    n0 = sum(1 for r in rows if r["label"] == "0")
    n1 = sum(1 for r in rows if r["label"] == "1")
    print(f"=== {name}: {len(rows)} ảnh ({n0} bona fide / {n1} forged) ===")
    print(f"  {'device':16s} {'bona':>6s} {'forged':>7s} {'%forged':>8s}")
    for d in devs:
        b, f = tab[(d, "0")], tab[(d, "1")]
        pct = 100.0 * f / max(1, b + f)
        flag = "  <- lệch nhãn theo device" if (b == 0 or f == 0) or pct > 85 or pct < 15 else ""
        print(f"  {d:16s} {b:6d} {f:7d} {pct:7.1f}%{flag}")
    print("  attack_type:", {k: v for (k, _), v in sorted(atk.items(), key=lambda kv: -kv[1])})


def main() -> int:
    ap = argparse.ArgumentParser(description="FantasyID composition diagnostic")
    ap.add_argument("--prepared", required=True, help="gốc chứa train/ test/ (có manifest.csv)")
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        _report(_load(os.path.join(args.prepared, split, "manifest.csv")), split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
