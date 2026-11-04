"""Chuyển ``models/fantasyid_val.txt`` sang stem mới sau khi đổi quy tắc đặt tên.

Bối cảnh: ``prepare_fantasyid.py`` trước đây chỉ gắn tiền tố device khi trùng
stem (``<device>__<stem>``), khiến manifest val trộn lẫn tên có/không có device.
Sau khi sửa thành *luôn* prefix device, toàn bộ stem đổi -> manifest cũ trỏ sai.

Script này giữ nguyên **đúng cùng tập ảnh val** (để số liệu calibration còn so
sánh được) bằng cách:

1. Replay thuật toán đặt tên CŨ trên ``train.csv`` — cùng thứ tự ``sorted`` và
   cùng bộ lọc ``os.path.isfile`` như ``prepare_fantasyid.main()`` — để dựng
   bảng ``old_stem -> rel_path``;
2. Tính stem MỚI (luôn có device) cho từng ``rel_path`` bằng hàm hiện hành;
3. Ghi lại manifest theo đường dẫn tương đối so với ``prepared/train``.

Ví dụ::

    cd ai
    $env:PYTHONIOENCODING="utf-8"
    python training/migrate_fantasyid_val_manifest.py ^
        --csv data/fantasyid/extract/FantasyID/train.csv ^
        --old models/fantasyid_val.old.txt ^
        --out models/fantasyid_val.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.prepare_fantasyid import _csv_rows, _unique_stem  # noqa: E402


def _old_unique_stem(rel: str, used: Dict[str, int]) -> str:
    """Replica thuật toán đặt tên CŨ (chỉ prefix khi trùng stem)."""
    stem = os.path.splitext(os.path.basename(rel))[0]
    if stem not in used:
        used[stem] = 0
        return stem
    used[stem] += 1
    parent = os.path.basename(os.path.dirname(rel))
    cand = f"{parent}__{stem}"
    n = 1
    while cand in used:
        cand = f"{parent}__{stem}__{n}"
        n += 1
    used[cand] = 0
    return cand


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate FantasyID val manifest stems")
    ap.add_argument("--csv", required=True, help="train.csv của FantasyID")
    ap.add_argument("--old", required=True, help="manifest cũ (đường dẫn tuyệt đối, stem cũ)")
    ap.add_argument("--out", required=True, help="manifest mới (đường dẫn tương đối, stem mới)")
    args = ap.parse_args()

    # root chứa train/ val/ test/ để khớp path trong csv — chính là thư mục cha
    # của csv (vd ``.../FantasyID``), tương đương ``dirname(--src)`` bên prepare.
    root = os.path.dirname(os.path.abspath(args.csv))
    rows = _csv_rows(args.csv)

    # 1) Replay thuật toán cũ + tính stem mới trong cùng một vòng lặp.
    used_old: Dict[str, int] = {}
    used_new: Dict[str, int] = {}
    old_to_rel: Dict[str, Tuple[str, bool]] = {}
    new_of_rel: Dict[str, str] = {}
    for rel_path, (is_attack, _attack_type) in sorted(rows.items()):
        if not os.path.isfile(os.path.join(root, rel_path)):
            continue
        old_stem = _old_unique_stem(rel_path, used_old)
        new_stem = _unique_stem(rel_path, used_new)
        old_to_rel[old_stem] = (rel_path, is_attack)
        new_of_rel[rel_path] = new_stem
    print(f"[replay] {len(old_to_rel)} ảnh train -> bảng stem cũ/mới")

    # 2) Đọc manifest cũ, lấy (subdir nhãn, stem).
    entries: List[Tuple[str, str]] = []
    with open(args.old, "r", encoding="utf-8-sig") as fh:
        for ln in fh:
            p = ln.strip().replace("\\", "/")
            if not p:
                continue
            parts = p.split("/")
            label_dir = parts[-2] if len(parts) >= 2 else ""
            stem = os.path.splitext(parts[-1])[0]
            entries.append((label_dir, stem))
    print(f"[manifest] {len(entries)} dòng cũ ({args.old})")

    # 3) Map sang stem mới, giữ thứ tự cũ.
    out_lines: List[str] = []
    missing: List[str] = []
    for label_dir, old_stem in entries:
        hit = old_to_rel.get(old_stem)
        if hit is None:
            missing.append(old_stem)
            continue
        rel_path, is_attack = hit
        sub = "forged" if is_attack else "bona_fide"
        if label_dir and label_dir != sub:
            print(f"[!] lệch nhãn: {old_stem} nằm ở {label_dir} nhưng csv nói {sub}")
        new_stem = new_of_rel[rel_path]
        out_lines.append(f"{sub}\\{new_stem}.jpg")

    if missing:
        print(f"[!] {len(missing)} stem cũ không map được, ví dụ: {missing[:5]}")
        print("    -> manifest cũ có thể đã sinh từ bản csv/dataset khác; dừng để tránh mất mẫu.")
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines) + "\n")
    n_attack = sum(1 for ln in out_lines if ln.startswith("forged\\"))
    print(f"[ok] ghi {args.out}: {len(out_lines)} ảnh "
          f"({len(out_lines) - n_attack} bona fide / {n_attack} forged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
