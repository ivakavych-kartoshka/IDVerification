"""Training harness cho TamperNet (document tamper detection - hướng A).

Dataset (tự nhận layout, xem ``CardTamperDataset``)::

    root/bona_fide/*.jpg + root/forged/*.jpg     (folder convention)
    root/*.jpg                                   (nhãn suy từ tên file:
                                                  *_live.* = bona fide,
                                                  *_attack*/*_forged.* = forged)
    root/<split>/...                             (truyền --data root/<split>)

Mask GT localization (tuỳ chọn): đặt trong ``root/masks`` (hoặc ``mask``,
``gt_masks``, ``localization``) cùng tên stem với ảnh. Không có mask -> chỉ
image-level cls (nên đặt ``--mask-weight 0``).

**FantasyID** (DeepID@ICCV 2025, https://zenodo.org/records/17063366) là bộ
chuẩn: 786 bona fide + 1572 forged cho train-val, 300 + 1085 cho test. Metric
tham chiếu là F1 @ 0.5 (DeepID Track 1); baseline TruFor mà BTC cung cấp bị
dataset này thách thức (FNR ~50% ở FPR 10%).

Ví dụ::

    cd ai
    python training/train_tamper.py --data D:/datasets/fantasyid/FantasyID/train_val ^
        --epochs 40 --batch-size 16 --out models/tamper_net.pth

    # chỉ để kiểm tra pipeline chạy được:
    python training/train_tamper.py --data <root> --epochs 1 --dry-run

Đánh giá: ``--eval-only`` in F1 (như metric DeepID Track 1) trên split val.
Export ONNX để service dùng: ``python training/export_tamper_onnx.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.tamper_net import INPUT_SIZE, TamperLoss, TamperNet, count_params  # noqa: E402
from app.services.tamper_service import _letterbox  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Nhãn suy ra từ tên file khi dataset không chia thư mục (layout FantasyID gốc
# đặt tên kiểu ``<id>_live.jpg`` / ``<id>_attack_<n>.jpg`` hoặc
# ``<id>_bona_fide.*`` / ``<id>_forged.*``).
_FORGED_MARKERS = ("attack", "forged", "fake", "tamper", "manip", "synthetic")
_BONA_MARKERS = ("live", "bona", "bona_fide", "bonafide", "genuine", "pristine")

# Thư mục chứa mask GT localization (nhiều tên gọi khác nhau tuỳ bản release).
_MASK_DIR_NAMES = ("masks", "mask", "gt_masks", "localization", "annotation_masks")


def _label_from_name(name: str) -> Optional[int]:
    """Suy nhãn từ tên file; None nếu không nhận ra marker nào."""
    low = name.lower()
    if any(m in low for m in _FORGED_MARKERS):
        return 1
    if any(m in low for m in _BONA_MARKERS):
        return 0
    return None


def _find_mask(mask_dirs, stem: str) -> Optional[str]:
    for d in mask_dirs:
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            cand = os.path.join(d, stem + ext)
            if os.path.exists(cand):
                return cand
    return None


def _iter_images(d: str):
    if not os.path.isdir(d):
        return
    for name in sorted(os.listdir(d)):
        if name.lower().endswith(IMG_EXTS):
            yield name


class CardTamperDataset(Dataset):
    """Dataset 2 lớp bonafide/forged, tự nhận layout.

    Hỗ trợ cả 3 kiểu tổ chức phổ biến::

        root/bona_fide/*.jpg + root/forged/*.jpg            (folder convention)
        root/*.jpg                                          (nhãn theo tên file)
        root/<split>/...                                    (truyền root=<split>)
    """

    def __init__(self, root: str, size: int = INPUT_SIZE, train: bool = True):
        self.root = root
        self.size = size
        self.train = train
        self.samples: List[Tuple[str, int, Optional[str]]] = []

        mask_dirs = [os.path.join(root, n) for n in _MASK_DIR_NAMES]
        mask_dirs = [d for d in mask_dirs if os.path.isdir(d)]

        # 1) Layout có thư mục theo lớp.
        for label, sub in ((0, "bona_fide"), (1, "forged")):
            d = os.path.join(root, sub)
            for name in _iter_images(d):
                stem = os.path.splitext(name)[0]
                self.samples.append((os.path.join(d, name), label, _find_mask(mask_dirs, stem)))

        # 2) Layout phẳng: nhãn suy từ tên file.
        if not self.samples:
            for name in _iter_images(root):
                label = _label_from_name(name)
                if label is None:
                    continue
                stem = os.path.splitext(name)[0]
                self.samples.append((os.path.join(root, name), label, _find_mask(mask_dirs, stem)))

        if not self.samples:
            raise RuntimeError(
                f"Không tìm thấy ảnh có nhãn trong {root} "
                "(cần bona_fide/ + forged/ hoặc tên file chứa live/attack)"
            )
        n_forged = sum(1 for _, y, _ in self.samples if y == 1)
        self.n_masks = sum(1 for _, _, m in self.samples if m)
        print(
            f"[CardTamperDataset] {root}: {len(self.samples)} ảnh "
            f"({len(self.samples) - n_forged} bona fide / {n_forged} forged), "
            f"mask GT: {self.n_masks}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import cv2

        path, label, mask_path = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Không đọc được ảnh: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        canvas, _ = _letterbox(img, self.size)

        x = canvas.astype(np.float32) / 255.0
        if self.train:
            # Augmentation nhẹ, tránh phá tín hiệu forensic (không blur/nén mạnh).
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]
            x = np.clip(x * np.random.uniform(0.9, 1.1), 0.0, 1.0)

        item = {"image": torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), "label": torch.tensor(label, dtype=torch.float32)}

        if mask_path:
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m_canvas, _ = _letterbox(cv2.cvtColor(m, cv2.COLOR_GRAY2RGB), self.size)
                item["mask"] = torch.from_numpy((m_canvas[:, :, 0] > 127).astype(np.float32))
        return item


def _collate(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch]) if all("mask" in b for b in batch) else None
    return images, labels, masks


@torch.no_grad()
def evaluate(model, loader, threshold: float = 0.5) -> dict:
    model.eval()
    device = next(model.parameters()).device
    tp = fp = fn = tn = 0
    for images, labels, _ in loader:
        images = images.to(device)
        prob = torch.sigmoid(model(images)["logit"])
        pred = (prob >= threshold).float()
        for p, y in zip(pred.tolist(), labels.tolist()):
            if p == 1 and y == 1:
                tp += 1
            elif p == 1 and y == 0:
                fp += 1
            elif p == 0 and y == 1:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train TamperNet")
    ap.add_argument("--data", required=True, help="root dataset (tự nhận layout)")
    ap.add_argument("--out", default="models/tamper_net.pth")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=INPUT_SIZE)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--val-manifest", default="", help="file danh sách ảnh val (ghi lần đầu, đọc lại để tái lập split)")
    ap.add_argument("--mask-weight", type=float, default=0.5)
    ap.add_argument("--no-masks", action="store_true", help="bỏ qua mask GT (chỉ train detection)")
    ap.add_argument("--pos-weight", type=float, default=3.0, help="BCE pos_weight cho lớp forged")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="1 epoch, batch nhỏ, không lưu")
    args = ap.parse_args()

    if args.dry_run:
        args.epochs, args.batch_size = 1, min(args.batch_size, 2)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                         ("cpu" if args.device == "auto" else args.device))

    dataset = CardTamperDataset(args.data, size=args.size, train=not args.eval_only)
    if args.no_masks:
        for i, (p, y, _m) in enumerate(dataset.samples):
            dataset.samples[i] = (p, y, None)
    if dataset.n_masks == 0 and args.mask_weight > 0:
        print("Không có mask GT -> tự đặt mask-weight = 0 (chỉ train detection).")
        args.mask_weight = 0.0
    if args.no_masks:
        args.mask_weight = 0.0

    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val

    if args.val_manifest:
        # Ghi/đọc danh sách file val để calibrate dùng ĐÚNG split đã dùng khi
        # train (tránh leak: calibrate quét ngưỡng trên chính val đã chọn
        # checkpoint là OK, nhưng phải là cùng một tập).
        if os.path.exists(args.val_manifest):
            with open(args.val_manifest, "r", encoding="utf-8") as fh:
                wanted = {ln.strip() for ln in fh if ln.strip()}
            val_idx = [i for i, (p, _y, _m) in enumerate(dataset.samples) if p in wanted]
            train_idx = [i for i in range(len(dataset)) if i not in set(val_idx)]
            print(f"[split] đọc manifest {args.val_manifest}: val={len(val_idx)} train={len(train_idx)}")
        else:
            gen = torch.Generator().manual_seed(42)
            perm = torch.randperm(len(dataset), generator=gen).tolist()
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            os.makedirs(os.path.dirname(os.path.abspath(args.val_manifest)), exist_ok=True)
            with open(args.val_manifest, "w", encoding="utf-8") as fh:
                for i in val_idx:
                    fh.write(dataset.samples[i][0] + "\n")
            print(f"[split] ghi manifest {args.val_manifest}: val={len(val_idx)} train={len(train_idx)}")
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
    else:
        gen = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=_collate)

    model = TamperNet(input_size=args.size).to(device)
    print(f"TamperNet params={count_params(model):,} | train={n_train} val={n_val} | device={device}")

    if args.eval_only:
        if not os.path.exists(args.out):
            print(f"Không thấy checkpoint {args.out}")
            return 1
        state = torch.load(args.out, map_location="cpu")
        model.load_state_dict(state)
        print("eval:", evaluate(model, val_loader))
        return 0

    criterion = TamperLoss(mask_weight=args.mask_weight, pos_weight=args.pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for images, labels, masks in train_loader:
            images, labels = images.to(device), labels.to(device)
            masks = masks.to(device) if masks is not None else None
            optimizer.zero_grad()
            out = model(images)
            loss, _parts = criterion(out, labels, masks)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        metrics = evaluate(model, val_loader)
        print(f"epoch {epoch:>3}/{args.epochs} loss={running / max(seen, 1):.4f} val_f1={metrics['f1']:.4f} "
              f"(P={metrics['precision']:.3f} R={metrics['recall']:.3f})", flush=True)
        if not args.dry_run and metrics["f1"] >= best_f1:
            best_f1 = metrics["f1"]
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  -> lưu {args.out} (f1={best_f1:.4f})")

    if args.dry_run:
        print("dry-run xong (không lưu checkpoint).")
    else:
        print(f"Xong. Best val F1 = {best_f1:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
