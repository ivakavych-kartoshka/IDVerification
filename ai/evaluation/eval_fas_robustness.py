"""Đánh giá độ bền của FAS trước suy giảm chất lượng ảnh.

Bộ CASIA-FASD crops đã được đánh giá ở điều kiện sạch (§6 của
``PAPER_EVIDENCE.vi.md``). Trong triển khai thật, ảnh selfie/CCCD thường bị
nén JPEG, mờ do rung tay, hạ độ phân giải khi gửi qua mạng, hoặc thiếu sáng.
Script này giữ nguyên ``MiniFASNetV2`` và ngưỡng 0,5, chỉ thay đổi ảnh đầu
vào theo từng điều kiện, rồi báo APCER/BPCER/EER ở cấp frame và video.

Mỗi ảnh được đọc **một lần**, sau đó áp lần lượt các điều kiện và chấm điểm.
Không suy luận lại từ CSV cũ vì điểm phải được tính trên ảnh đã biến đổi.

Ví dụ::

    cd ai
    $env:PYTHONIOENCODING="utf-8"
    python evaluation/eval_fas_robustness.py `
        --manifest data/fas_casia/manifest.csv `
        --real data/fas_casia/real `
        --attack data/fas_casia/attack `
        --out reports/fas_robustness.json `
        --scores-csv reports/fas_robustness_scores.csv

Mặc định TTA tắt (khuyến nghị từ §6.4); thêm ``--tta`` để chạy lại với TTA.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.fasnet import FASNetAntiSpoof  # noqa: E402


DEFAULT_CONDITIONS = [
    "clean",
    "jpeg_q70",
    "jpeg_q50",
    "jpeg_q30",
    "blur_s1.5",
    "blur_s3.0",
    "down_128",
    "down_64",
    "dark_0.5",
]


def _read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sample(rows: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    if not limit or len(rows) <= limit:
        return rows
    idx = np.linspace(0, len(rows) - 1, limit).astype(int)
    return [rows[i] for i in sorted(set(idx.tolist()))]


def _apply(condition: str, bgr: np.ndarray) -> np.ndarray:
    """Trả ảnh BGR đã biến đổi. Mọi điều kiện giữ nguyên kích thước gốc."""
    if condition == "clean":
        return bgr
    if condition.startswith("jpeg_q"):
        quality = int(condition.split("q", 1)[1])
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return bgr
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if condition.startswith("blur_s"):
        sigma = float(condition.split("s", 1)[1])
        return cv2.GaussianBlur(bgr, (0, 0), sigma)
    if condition.startswith("down_"):
        side = int(condition.split("_", 1)[1])
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (side, side), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if condition.startswith("dark_"):
        alpha = float(condition.split("_", 1)[1])
        return cv2.convertScaleAbs(bgr, alpha=alpha, beta=0)
    raise ValueError(f"Điều kiện không hỗ trợ: {condition}")


def _rates(real: np.ndarray, attack: np.ndarray, thr: float) -> Dict[str, float]:
    apcer = float((attack >= thr).mean()) if attack.size else 0.0
    bpcer = float((real < thr).mean()) if real.size else 0.0
    return {
        "threshold": round(float(thr), 3),
        "apcer": round(apcer, 4),
        "bpcer": round(bpcer, 4),
        "acer": round((apcer + bpcer) / 2.0, 4),
    }


def _eer(real: np.ndarray, attack: np.ndarray) -> Dict[str, float]:
    if not real.size or not attack.size:
        return {"eer": 0.0, "threshold": 0.0}
    best = None
    for thr in np.unique(np.concatenate([real, attack])):
        apcer = float((attack >= thr).mean())
        bpcer = float((real < thr).mean())
        gap = abs(apcer - bpcer)
        if best is None or gap < best[0]:
            best = (gap, float(thr), (apcer + bpcer) / 2.0)
    assert best is not None
    return {"eer": round(best[2], 4), "threshold": round(best[1], 4)}


def _auc(real: np.ndarray, attack: np.ndarray) -> float:
    if not real.size or not attack.size:
        return float("nan")
    scores = np.concatenate([real, attack])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_real = ranks[: real.size].sum()
    return float((r_real - real.size * (real.size + 1) / 2.0) / (real.size * attack.size))


def _metrics(rows: List[Dict[str, object]], thr: float) -> Dict[str, object]:
    real_f = np.asarray([r["real_prob"] for r in rows if r["label"] == "real"], dtype=float)
    atk_f = np.asarray([r["real_prob"] for r in rows if r["label"] == "fake"], dtype=float)

    vids: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for r in rows:
        key = (str(r["subject"]), str(r["quality"]), str(r["video"]))
        v = vids.setdefault(key, {"label": r["label"], "scores": []})
        v["scores"].append(float(r["real_prob"]))  # type: ignore[union-attr]

    real_v = np.asarray([np.mean(v["scores"]) for v in vids.values() if v["label"] == "real"])  # type: ignore[arg-type]
    atk_v = np.asarray([np.mean(v["scores"]) for v in vids.values() if v["label"] == "fake"])  # type: ignore[arg-type]

    return {
        "frame": {
            "n_real": int(real_f.size),
            "n_attack": int(atk_f.size),
            "at_threshold": _rates(real_f, atk_f, thr),
            "eer": _eer(real_f, atk_f),
            "auc": round(_auc(real_f, atk_f), 4),
        },
        "video": {
            "n_real": int(real_v.size),
            "n_attack": int(atk_v.size),
            "at_threshold": _rates(real_v, atk_v, thr),
            "eer": _eer(real_v, atk_v),
            "auc": round(_auc(real_v, atk_v), 4),
            "median_real_prob": {
                "real": round(float(np.median(real_v)), 4),
                "attack": round(float(np.median(atk_v)), 4),
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Độ bền FAS trước suy giảm ảnh")
    ap.add_argument("--manifest", default="data/fas_casia/manifest.csv")
    ap.add_argument("--real", default="data/fas_casia/real")
    ap.add_argument("--attack", default="data/fas_casia/attack")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit-real", type=int, default=0)
    ap.add_argument("--limit-attack", type=int, default=0)
    ap.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    ap.add_argument("--tta", action="store_true", help="bật flip-TTA (mặc định tắt)")
    ap.add_argument("--out", default="reports/fas_robustness.json")
    ap.add_argument("--scores-csv", default="reports/fas_robustness_scores.csv")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in DEFAULT_CONDITIONS]
    if unknown:
        print(f"[!] Điều kiện lạ: {unknown}")
        return 2

    rows = _read_manifest(args.manifest)
    real_rows = _sample([r for r in rows if r["label"] == "real"], args.limit_real)
    atk_rows = _sample([r for r in rows if r["label"] == "fake"], args.limit_attack)
    selected = real_rows + atk_rows
    if not selected:
        print("[!] Manifest rỗng -> dừng.")
        return 2

    fas = FASNetAntiSpoof()
    fas.tta = bool(args.tta)

    scores: Dict[str, List[Dict[str, object]]] = {c: [] for c in conditions}
    missing = 0
    total = len(selected)

    for i, r in enumerate(selected, 1):
        root = args.real if r["label"] == "real" else args.attack
        path = os.path.join(root, r["file"])
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            missing += 1
            continue
        h, w = bgr.shape[:2]
        for cond in conditions:
            img = _apply(cond, bgr)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = fas.predict(rgb, (0, 0, w, h))
            scores[cond].append(
                {
                    "file": r["file"],
                    "label": r["label"],
                    "subject": r["subject"],
                    "quality": r["quality"],
                    "video": r["video"],
                    "real_prob": float(res["real_prob"]),
                    "pred": res.get("label", ""),
                }
            )
        if i % 100 == 0 or i == total:
            print(f"  ... {i}/{total} ảnh")

    out: Dict[str, object] = {
        "manifest": args.manifest,
        "real_dir": args.real,
        "attack_dir": args.attack,
        "engine": "MiniFASNetV2 (real_prob)",
        "tta": bool(args.tta),
        "threshold": args.threshold,
        "n_images": total - missing,
        "n_missing": missing,
        "conditions": {},
        "deltas_vs_clean": {},
    }
    cond_out: Dict[str, Dict[str, object]] = {}
    for cond in conditions:
        cond_out[cond] = _metrics(scores[cond], args.threshold)
    out["conditions"] = cond_out

    if "clean" in cond_out:
        base = cond_out["clean"]["video"]  # type: ignore[index]
        for cond in conditions:
            if cond == "clean":
                continue
            cur = cond_out[cond]["video"]  # type: ignore[index]
            out["deltas_vs_clean"][cond] = {  # type: ignore[index]
                "apcer": round(cur["at_threshold"]["apcer"] - base["at_threshold"]["apcer"], 4),
                "bpcer": round(cur["at_threshold"]["bpcer"] - base["at_threshold"]["bpcer"], 4),
                "eer": round(cur["eer"]["eer"] - base["eer"]["eer"], 4),
                "auc": round(cur["auc"] - base["auc"], 4),
            }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    if args.scores_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.scores_csv)), exist_ok=True)
        with open(args.scores_csv, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["condition", "file", "label", "subject", "quality", "video",
                         "real_prob", "pred"])
            for cond in conditions:
                for r in scores[cond]:
                    wr.writerow([cond, r["file"], r["label"], r["subject"], r["quality"],
                                 r["video"], round(float(r["real_prob"]), 4), r["pred"]])

    print(f"\nẢnh: {total - missing} (thiếu file: {missing}) | TTA={args.tta} | thr={args.threshold}")
    for cond in conditions:
        m = cond_out[cond]
        v = m["video"]  # type: ignore[index]
        f = m["frame"]  # type: ignore[index]
        print(f"[{cond:11s}] video APCER={v['at_threshold']['apcer']:.4f} "
              f"BPCER={v['at_threshold']['bpcer']:.4f} EER={v['eer']['eer']:.4f}"
              f"@{v['eer']['threshold']:.3f} AUC={v['auc']:.4f} | "
              f"frame EER={f['eer']['eer']:.4f}")
    print(f"\nĐã ghi {args.out}")
    if args.scores_csv:
        print(f"Đã ghi {args.scores_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
