"""Chuyển FantasyID (Zenodo 17063366) sang layout train được của TamperNet.

FantasyID gốc::

    FantasyID/
      train.csv, test.csv                 # path,is_attack,attack_type
      train/bonafide/<device>/*.jpg|.json
      train/attack/<digital_1|digital_2>/<device>/*.jpg|.json
      test/...

Mỗi ảnh .jpg có sidecar .json chứa ``regions[]``; region có
``region_attributes.region_provenance == "altered"`` là vùng bị can thiệp
(face swap / text inpaint) -> đây chính là mask localization GT.
Toạ độ box trong JSON ở hệ **ảnh gốc chưa crop**; ``cropping_info.
transformation_matrix_into_cropped`` (3x3 homography) đưa về hệ ảnh .jpg thật,
nên phải biến đổi trước khi rasterize mask.

Output (đúng layout ``CardTamperDataset``)::

    <out>/bona_fide/*.jpg
    <out>/forged/*.jpg
    <out>/masks/<stem>.png        # trắng = vùng altered (chỉ ảnh forged)

Ảnh được giữ nguyên tên gốc; nếu trùng tên giữa các device thì thêm tiền tố
đường dẫn để không ghi đè.

Ví dụ::

    cd ai
    python training/prepare_fantasyid.py ^
        --src D:/datasets/fantasyid/extract/FantasyID/train ^
        --csv D:/datasets/fantasyid/extract/FantasyID/train.csv ^
        --out D:/datasets/fantasyid/prepared/train

    python training/prepare_fantasyid.py ^
        --src .../test --csv .../test.csv --out .../prepared/test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _csv_rows(csv_path: str) -> Dict[str, Tuple[bool, str]]:
    """path (posix, tương đối gốc) -> (is_attack, attack_type)."""
    out: Dict[str, Tuple[bool, str]] = {}
    with open(csv_path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            p = row["path"].replace("\\", "/")
            out[p] = (row["is_attack"].strip().lower() == "true", row["attack_type"].strip())
    return out


def _apply_h(h: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    v = h @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(v[2]) < 1e-12:
        return float("nan"), float("nan")
    return float(v[0] / v[2]), float(v[1] / v[2])


def _altered_boxes(json_path: str) -> Tuple[List[Tuple[float, float, float, float]], Optional[np.ndarray]]:
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    boxes: List[Tuple[float, float, float, float]] = []
    for region in data.get("regions", []):
        attrs = region.get("region_attributes", {})
        if attrs.get("region_provenance") != "altered":
            continue
        sa = region.get("shape_attributes", {})
        if sa.get("name") != "rect":
            continue
        boxes.append((float(sa["x"]), float(sa["y"]), float(sa["width"]), float(sa["height"])))
    crop = data.get("cropping_info") or {}
    mat = crop.get("transformation_matrix_into_cropped")
    h = np.asarray(mat, dtype=np.float64) if mat else None
    return boxes, h


def _mask_from_boxes(json_path: str, w: int, h_img: int, dilate: int) -> Optional[np.ndarray]:
    boxes, h = _altered_boxes(json_path)
    if not boxes:
        return None
    mask = np.zeros((h_img, w), dtype=np.uint8)
    for (x, y, bw, bh) in boxes:
        if h is not None:
            # 4 góc -> homography -> hệ ảnh cropped (chính là ảnh .jpg).
            corners = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
            pts = [_apply_h(h, cx, cy) for (cx, cy) in corners]
            if any(np.isnan(px) or np.isnan(py) for px, py in pts):
                continue
            poly = np.asarray(pts, dtype=np.float32)
            pts_i = np.round(poly).astype(np.int32)
            cv2.fillPoly(mask, [pts_i], 255)
        else:
            xi, yi = max(0, int(round(x))), max(0, int(round(y)))
            x2, y2 = min(w, int(round(x + bw))), min(h_img, int(round(y + bh)))
            if x2 > xi and y2 > yi:
                mask[yi:y2, xi:x2] = 255
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        mask = cv2.dilate(mask, k)
    if int(mask.sum()) == 0:
        return None
    return mask


def _unique_stem(rel: str, used: Dict[str, int]) -> str:
    stem = os.path.splitext(os.path.basename(rel))[0]
    if stem not in used:
        used[stem] = 0
        return stem
    used[stem] += 1
    # gắn thêm tên thư mục cha để tránh trùng
    parent = os.path.basename(os.path.dirname(rel))
    cand = f"{parent}__{stem}"
    n = 1
    while cand in used:
        cand = f"{parent}__{stem}__{n}"
        n += 1
    used[cand] = 0
    return cand


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare FantasyID -> TamperNet layout")
    ap.add_argument("--src", required=True, help="thư mục split (vd .../FantasyID/train)")
    ap.add_argument("--csv", required=True, help="train.csv / test.csv / val.csv tương ứng")
    ap.add_argument("--out", required=True, help="thư mục đích (bona_fide/ forged/ masks/)")
    ap.add_argument("--dilate", type=int, default=8, help="nới box mask (px) để bao trùm viền blend")
    ap.add_argument("--copy", action="store_true", help="copy ảnh (mặc định: hardlink, rơi về copy nếu fail)")
    ap.add_argument("--limit", type=int, default=0, help="0 = hết")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    root = os.path.dirname(src)  # gốc chứa train/ val/ test/ để khớp path trong csv
    rows = _csv_rows(args.csv)

    out_bona = os.path.join(args.out, "bona_fide")
    out_forged = os.path.join(args.out, "forged")
    out_masks = os.path.join(args.out, "masks")
    for d in (out_bona, out_forged, out_masks):
        os.makedirs(d, exist_ok=True)

    used: Dict[str, int] = {}
    n_bona = n_forged = n_mask = n_missing_json = n_no_box = 0
    processed = 0
    manifest: List[Tuple[str, int, str]] = []

    for rel_path, (is_attack, attack_type) in sorted(rows.items()):
        img_path = os.path.join(root, rel_path)
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[!] đọc lỗi: {img_path}")
            continue
        h_img, w_img = img.shape[:2]

        stem = _unique_stem(rel_path, used)
        dst_label = out_forged if is_attack else out_bona
        dst_img = os.path.join(dst_label, stem + os.path.splitext(img_path)[1])
        manifest.append((stem, 1 if is_attack else 0, attack_type))

        if args.copy:
            if not os.path.exists(dst_img):
                shutil.copy2(img_path, dst_img)
        else:
            if not os.path.exists(dst_img):
                try:
                    os.link(img_path, dst_img)
                except OSError:
                    shutil.copy2(img_path, dst_img)

        if is_attack:
            n_forged += 1
            json_path = os.path.splitext(img_path)[0] + ".json"
            if not os.path.isfile(json_path):
                n_missing_json += 1
            else:
                mask = _mask_from_boxes(json_path, w_img, h_img, args.dilate)
                if mask is None:
                    n_no_box += 1
                else:
                    cv2.imwrite(os.path.join(out_masks, stem + ".png"), mask)
                    n_mask += 1
        else:
            n_bona += 1

        processed += 1
        if (processed % 200) == 0:
            print(f"  ... {processed} ảnh (bona={n_bona} forged={n_forged} mask={n_mask})")
        if args.limit and processed >= args.limit:
            break

    print("\n=== Kết quả ===")
    print(f"bona fide : {n_bona}")
    print(f"forged    : {n_forged}")
    print(f"mask GT   : {n_mask}")
    print(f"thiếu json: {n_missing_json} | json không có box altered: {n_no_box}")
    print(f"-> {args.out}")

    manifest_path = os.path.join(args.out, "manifest.csv")
    with open(manifest_path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["stem", "label", "attack_type"])
        wr.writerows(manifest)
    print(f"manifest  : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
