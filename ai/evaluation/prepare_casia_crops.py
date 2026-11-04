"""Chuẩn hoá bản CASIA-FASD đã cắt khuôn mặt (256x256) thành layout FAS eval.

Bản pack trên mạng (Kaggle/HF) có 2 vấn đề đã kiểm chứng:

1. ``train_img`` và ``test_img`` **rò rỉ**: 20/20 subject train đều nằm trong
   test và 1.403 frame trùng khoá ``(subject, quality, video, frame)``;
2. nhãn chỉ có ``real``/``fake`` (không tách print/replay).

Vì vậy script này gộp hai split, **dedup theo khoá frame**, rồi tạo layout
``<out>/real`` + ``<out>/attack`` bằng hardlink (không tốn thêm dung lượng) kèm
``manifest.csv`` để phân tích theo video/người.

Ta **không huấn luyện** trên dữ liệu này (chỉ chấm MiniFASNet có sẵn), nên việc
dùng hợp hai split sau khi dedup là hợp lệ; điều bắt buộc là không gọi nó là
"split test chính thức" trong bài báo.

Ví dụ::

    cd ai
    python evaluation/prepare_casia_crops.py `
        --src "data/archive (1)" `
        --out data/fas_casia
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
from typing import Dict, List, Tuple

PAT = re.compile(r"^(\d+)_(?:([A-Za-z]+)_)?(\d+)\.avi_(\d+)_(real|fake)\.jpg$")


def _scan(split_dir: str, origin: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for p in sorted(glob.glob(os.path.join(split_dir, "*.jpg"))):
        name = os.path.basename(p)
        m = PAT.match(name)
        if not m:
            continue
        subj, qual, vid, frame, label = m.groups()
        rows.append({
            "path": p, "name": name,
            "subject": int(subj), "quality": qual or "none", "video": int(vid),
            "frame": int(frame), "label": "real" if label == "real" else "fake",
            "origin": origin,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Chuẩn hoá CASIA-FASD crops -> layout FAS eval")
    ap.add_argument("--src", default="data/archive (1)",
                    help="thư mục chứa train_img/test_img (mỗi cái có <split>/<split>/color)")
    ap.add_argument("--out", default="data/fas_casia")
    args = ap.parse_args()

    rows: List[Dict[str, object]] = []
    for split in ("train_img", "test_img"):
        color = os.path.join(args.src, split, split, "color")
        if not os.path.isdir(color):
            print(f"[!] thiếu {color}")
            return 2
        rows.extend(_scan(color, split))

    # Dedup theo khoá frame; giữ bản gặp đầu (train trước test).
    seen: Dict[Tuple[int, str, int, int], str] = {}
    unique: List[Dict[str, object]] = []
    for r in rows:
        key = (int(r["subject"]), str(r["quality"]), int(r["video"]), int(r["frame"]))
        if key in seen:
            if seen[key] != r["label"]:
                print(f"[!] nhãn xung đột tại {key}: {seen[key]} vs {r['label']} -> bỏ qua bản sau")
            continue
        seen[key] = str(r["label"])
        unique.append(r)

    real_dir = os.path.join(args.out, "real")
    atk_dir = os.path.join(args.out, "attack")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(atk_dir, exist_ok=True)

    linked = copied = 0
    for r in unique:
        dst = os.path.join(real_dir if r["label"] == "real" else atk_dir, str(r["name"]))
        if os.path.exists(dst):
            continue
        try:
            os.link(str(r["path"]), dst)
            linked += 1
        except OSError:
            shutil.copy2(str(r["path"]), dst)
            copied += 1

    man = os.path.join(args.out, "manifest.csv")
    with open(man, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "label", "subject", "quality", "video", "frame", "origin"])
        for r in unique:
            wr.writerow([r["name"], r["label"], r["subject"], r["quality"],
                         r["video"], r["frame"], r["origin"]])

    n_real = sum(1 for r in unique if r["label"] == "real")
    n_fake = len(unique) - n_real
    subj = {int(r["subject"]) for r in unique}
    vids = {(int(r["subject"]), str(r["quality"]), int(r["video"])) for r in unique}
    print(f"gốc: {len(rows)} ảnh | sau dedup: {len(unique)} ảnh "
          f"({n_real} real / {n_fake} fake) | {len(subj)} người | {len(vids)} video")
    print(f"hardlink={linked} copy={copied} | manifest: {man}")
    print(f"layout: {real_dir} | {atk_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
