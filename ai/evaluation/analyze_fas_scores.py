"""Tổng hợp kết quả FAS theo video/người từ CSV điểm từng frame.

``run_fas_eval.py`` chấm từng ảnh/frame và ghi ``path,class,attack_type,real_prob,
label,note``. Với bản CASIA-FASD crops, các frame rất tương quan trong cùng một
video, nên báo cáo chỉ theo frame sẽ thổi phồng độ chính xác. Script này gộp
điểm theo ``(subject, quality, video)`` bằng trung bình, rồi tính APCER/BPCER/
ACER/EER ở cấp video — đơn vị mà tài liệu FAS thường công bố.

Ví dụ::

    cd ai
    python evaluation/analyze_fas_scores.py `
        --scores-csv reports/fas_casia_crops_scores.csv `
        --manifest data/fas_casia/manifest.csv `
        --out reports/fas_casia_crops_video.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def _rates(real: np.ndarray, attack: np.ndarray, thr: float) -> Dict[str, float]:
    apcer = float((attack >= thr).mean()) if attack.size else 0.0
    bpcer = float((real < thr).mean()) if real.size else 0.0
    return {"threshold": round(float(thr), 3), "apcer": round(apcer, 4),
            "bpcer": round(bpcer, 4), "acer": round((apcer + bpcer) / 2, 4)}


def _eer(real: np.ndarray, attack: np.ndarray) -> Dict[str, float]:
    if not real.size or not attack.size:
        return {"eer": 0.0, "threshold": 0.0}
    cands = np.unique(np.concatenate([real, attack]))
    best = None
    for thr in cands:
        apcer = float((attack >= thr).mean())
        bpcer = float((real < thr).mean())
        gap = abs(apcer - bpcer)
        if best is None or gap < best[0]:
            best = (gap, float(thr), (apcer + bpcer) / 2.0)
    assert best is not None
    return {"eer": round(best[2], 4), "threshold": round(best[1], 3)}


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Gộp điểm FAS theo video/người")
    ap.add_argument("--scores-csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="reports/fas_video_metrics.json")
    args = ap.parse_args()

    meta: Dict[str, Dict[str, str]] = {}
    with open(args.manifest, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            meta[row["file"]] = row

    vids: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    frame_scores: Dict[str, List[float]] = collections.defaultdict(list)
    no_face = 0
    missing = 0
    with open(args.scores_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = os.path.basename(row["path"])
            m = meta.get(name)
            if m is None:
                missing += 1
                continue
            if row.get("note") == "no_face" or not row.get("real_prob"):
                no_face += 1
                continue
            key = (m["subject"], m["quality"], m["video"])
            v = vids.setdefault(key, {"label": m["label"], "scores": []})
            v["scores"].append(float(row["real_prob"]))  # type: ignore[union-attr]
            frame_scores[m["label"]].append(float(row["real_prob"]))

    real_v = np.array([np.mean(v["scores"]) for v in vids.values() if v["label"] == "real"])  # type: ignore[arg-type]
    atk_v = np.array([np.mean(v["scores"]) for v in vids.values() if v["label"] == "fake"])  # type: ignore[arg-type]
    real_f = np.array(frame_scores["real"])
    atk_f = np.array(frame_scores["fake"])

    out = {
        "scores_csv": args.scores_csv,
        "manifest": args.manifest,
        "n_no_face": no_face,
        "n_unmatched": missing,
        "frame": {
            "n_real": int(real_f.size), "n_attack": int(atk_f.size),
            "at_threshold": _rates(real_f, atk_f, args.threshold),
            "eer": _eer(real_f, atk_f), "auc": round(_auc(real_f, atk_f), 4),
        },
        "video": {
            "n_real": int(real_v.size), "n_attack": int(atk_v.size),
            "at_threshold": _rates(real_v, atk_v, args.threshold),
            "eer": _eer(real_v, atk_v), "auc": round(_auc(real_v, atk_v), 4),
            "median_real_prob": {"real": round(float(np.median(real_v)), 4),
                                 "attack": round(float(np.median(atk_v)), 4)},
        },
        "by_quality": {},
    }

    for qual in sorted({k[1] for k in vids}):
        r = np.array([np.mean(v["scores"]) for k, v in vids.items()
                      if k[1] == qual and v["label"] == "real"])  # type: ignore[arg-type]
        a = np.array([np.mean(v["scores"]) for k, v in vids.items()
                      if k[1] == qual and v["label"] == "fake"])  # type: ignore[arg-type]
        out["by_quality"][qual] = {
            "n_real": int(r.size), "n_attack": int(a.size),
            "at_threshold": _rates(r, a, args.threshold),
            "eer": _eer(r, a),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"frame: real={real_f.size} attack={atk_f.size} | video: real={real_v.size} attack={atk_v.size}"
          f" | no_face={no_face} unmatched={missing}")
    for level in ("frame", "video"):
        d = out[level]
        print(f"[{level}] APCER={d['at_threshold']['apcer']:.4f} "
              f"BPCER={d['at_threshold']['bpcer']:.4f} ACER={d['at_threshold']['acer']:.4f} "
              f"EER={d['eer']['eer']:.4f}@{d['eer']['threshold']:.3f} AUC={d['auc']:.4f}")
    for qual, d in out["by_quality"].items():
        print(f"[video/{qual}] n={d['n_real']}/{d['n_attack']} "
              f"APCER={d['at_threshold']['apcer']:.4f} BPCER={d['at_threshold']['bpcer']:.4f} "
              f"EER={d['eer']['eer']:.4f}")
    print(f"Đã ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
