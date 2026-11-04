"""Đánh giá face verification trên LFW bằng chính ``FaceService``.

Bộ LFW deep-funneled (View 1) có:

- ``matchpairsDevTrain.csv`` + ``mismatchpairsDevTrain.csv``: 1.100 + 1.100
  cặp dùng để chọn ngưỡng (validation).
- ``matchpairsDevTest.csv`` + ``mismatchpairsDevTest.csv``: 500 + 500 cặp
  dùng để báo cáo (test), chỉ chấm một lần.

Mỗi ảnh được trích embedding **một lần** và cache theo đường dẫn; một cặp có
thể tái sử dụng ảnh của cặp khác. Điểm cặp là cosine similarity giữa hai
embedding ArcFace. Ảnh không phát hiện được mặt nhận điểm 0 và được tính là
false reject nếu là cặp genuine; báo cáo thêm ``detected_only`` để tách riêng
chất lượng nhận dạng khỏi tỷ lệ detect.

Ví dụ::

    cd ai
    $env:PYTHONIOENCODING="utf-8"
    python evaluation/run_face_eval.py `
        --root data/lfw-deepfunneled/lfw-deepfunneled `
        --pairs-dir data `
        --out reports/face_lfw.json `
        --scores-csv reports/face_lfw_scores.csv

Thêm ``--limit-per-class 20`` để smoke test nhanh.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.face_service import FaceService, MATCH_THRESHOLD  # noqa: E402


def _read_pairs(path: Path, kind: str) -> List[Tuple[str, int, str, int]]:
    """Đọc CSV pair của LFW View 1, trả (name1, num1, name2, num2)."""
    rows: List[Tuple[str, int, str, int]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for raw in reader:
            vals = [v.strip() for v in raw]
            if not any(vals):
                continue
            if kind == "match":
                if len(vals) < 3 or not vals[0]:
                    raise ValueError(f"Dòng match không hợp lệ trong {path}: {raw}")
                name, n1, n2 = vals[:3]
                rows.append((name, int(n1), name, int(n2)))
            else:
                if len(vals) < 4 or not vals[0] or not vals[2]:
                    raise ValueError(f"Dòng mismatch không hợp lệ trong {path}: {raw}")
                name1, n1, name2, n2 = vals[:4]
                rows.append((name1, int(n1), name2, int(n2)))
    return rows


def _image_path(root: Path, name: str, num: int) -> Path:
    return root / name / f"{name}_{num:04d}.jpg"


def _rates(genuine: np.ndarray, impostor: np.ndarray, thr: float) -> Dict[str, float]:
    far = float((impostor >= thr).mean()) if impostor.size else 0.0
    frr = float((genuine < thr).mean()) if genuine.size else 0.0
    acc = (
        float(((genuine >= thr).sum() + (impostor < thr).sum())
              / (genuine.size + impostor.size))
        if (genuine.size + impostor.size)
        else 0.0
    )
    return {
        "threshold": round(float(thr), 4),
        "far": round(far, 4),
        "frr": round(frr, 4),
        "tar": round(1.0 - frr, 4),
        "accuracy": round(acc, 4),
    }


def _eer(genuine: np.ndarray, impostor: np.ndarray) -> Dict[str, float]:
    if not genuine.size or not impostor.size:
        return {"eer": 0.0, "threshold": 0.0}
    best: Optional[Tuple[float, float, float]] = None
    for thr in np.unique(np.concatenate([genuine, impostor])):
        far = float((impostor >= thr).mean())
        frr = float((genuine < thr).mean())
        gap = abs(far - frr)
        if best is None or gap < best[0]:
            best = (gap, float(thr), (far + frr) / 2.0)
    assert best is not None
    return {"eer": round(best[2], 4), "threshold": round(best[1], 4)}


def _auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """AUC Mann-Whitney có xử lý hoà điểm (giống evaluator FAS)."""
    if not genuine.size or not impostor.size:
        return float("nan")
    scores = np.concatenate([genuine, impostor])
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
    r_genuine = ranks[: genuine.size].sum()
    return float(
        (r_genuine - genuine.size * (genuine.size + 1) / 2.0)
        / (genuine.size * impostor.size)
    )


def _threshold_for_far(impostor: np.ndarray, target: float) -> float:
    """Ngưỡng thấp nhất đạt FAR <= target trên tập impostor."""
    for thr in np.unique(impostor):
        if float((impostor >= thr).mean()) <= target:
            return float(thr)
    return float(np.nextafter(float(impostor.max()), np.inf))


def _split_metrics(
    rows: List[Dict[str, Any]], threshold: float
) -> Dict[str, Any]:
    genuine = np.asarray(
        [r["score"] for r in rows if r["label"] == "genuine"], dtype=float
    )
    impostor = np.asarray(
        [r["score"] for r in rows if r["label"] == "impostor"], dtype=float
    )
    no_face = [r for r in rows if r["no_face"]]
    detected = [r for r in rows if not r["no_face"]]
    genuine_d = np.asarray(
        [r["score"] for r in detected if r["label"] == "genuine"], dtype=float
    )
    impostor_d = np.asarray(
        [r["score"] for r in detected if r["label"] == "impostor"], dtype=float
    )
    return {
        "n_pairs": len(rows),
        "n_genuine": int(genuine.size),
        "n_impostor": int(impostor.size),
        "n_no_face_pairs": len(no_face),
        "coverage": round(1.0 - len(no_face) / len(rows), 4) if rows else 0.0,
        "at_threshold": _rates(genuine, impostor, threshold),
        "eer": _eer(genuine, impostor),
        "auc": round(_auc(genuine, impostor), 4),
        "detected_only": {
            "n_pairs": len(detected),
            "n_genuine": int(genuine_d.size),
            "n_impostor": int(impostor_d.size),
            "at_threshold": _rates(genuine_d, impostor_d, threshold),
            "eer": _eer(genuine_d, impostor_d),
            "auc": round(_auc(genuine_d, impostor_d), 4),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="FAR/FRR/EER cho FaceService trên LFW")
    ap.add_argument(
        "--root",
        default="data/lfw-deepfunneled/lfw-deepfunneled",
        help="thư mục chứa các folder tên người của LFW deep-funneled",
    )
    ap.add_argument("--pairs-dir", default="data", help="thư mục chứa các file pair CSV")
    ap.add_argument("--out", default="reports/face_lfw.json")
    ap.add_argument("--scores-csv", default="reports/face_lfw_scores.csv")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="ngưỡng MATCH để báo cáo; mặc định lấy MATCH_THRESHOLD (0,30)",
    )
    ap.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="giới hạn số cặp mỗi lớp cho smoke test (0 = toàn bộ)",
    )
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    root = Path(args.root)
    pairs_dir = Path(args.pairs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục LFW: {root}")

    val_rows: List[Dict[str, Any]] = []
    for kind, fname in (("match", "matchpairsDevTrain.csv"),
                        ("mismatch", "mismatchpairsDevTrain.csv")):
        for name1, num1, name2, num2 in _read_pairs(pairs_dir / fname, kind):
            if args.limit_per_class and len(
                [r for r in val_rows if r["label"] == ("genuine" if kind == "match" else "impostor")]
            ) >= args.limit_per_class:
                continue
            p1 = _image_path(root, name1, num1)
            p2 = _image_path(root, name2, num2)
            val_rows.append(
                {
                    "split": "val",
                    "label": "genuine" if kind == "match" else "impostor",
                    "pair_id": f"val-{kind}-{num1}-{num2}-{name1}-{name2}",
                    "name1": name1,
                    "num1": num1,
                    "name2": name2,
                    "num2": num2,
                    "path1": p1,
                    "path2": p2,
                }
            )

    test_rows: List[Dict[str, Any]] = []
    for kind, fname in (("match", "matchpairsDevTest.csv"),
                        ("mismatch", "mismatchpairsDevTest.csv")):
        for name1, num1, name2, num2 in _read_pairs(pairs_dir / fname, kind):
            if args.limit_per_class and len(
                [r for r in test_rows if r["label"] == ("genuine" if kind == "match" else "impostor")]
            ) >= args.limit_per_class:
                continue
            test_rows.append(
                {
                    "split": "test",
                    "label": "genuine" if kind == "match" else "impostor",
                    "pair_id": f"test-{kind}-{num1}-{num2}-{name1}-{name2}",
                    "name1": name1,
                    "num1": num1,
                    "name2": name2,
                    "num2": num2,
                    "path1": _image_path(root, name1, num1),
                    "path2": _image_path(root, name2, num2),
                }
            )

    missing = sorted(
        {str(r[k]) for r in val_rows + test_rows for k in ("path1", "path2")
         if not r[k].exists()}
    )
    if missing:
        print(f"Thiếu {len(missing)} ảnh, ví dụ: {missing[:5]}")
        return 2

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(MATCH_THRESHOLD)
    )
    face = FaceService()
    cache: Dict[str, Optional[np.ndarray]] = {}
    cache_hits = 0

    def embedding(path: Path) -> Optional[np.ndarray]:
        nonlocal cache_hits
        key = str(path)
        if key in cache:
            cache_hits += 1
            return cache[key]
        img = FaceService.load_image(path.read_bytes())
        emb = face._aligned_embedding(img)
        cache[key] = emb
        return emb

    def score_rows(rows: List[Dict[str, Any]]) -> None:
        for i, r in enumerate(rows, 1):
            t0 = time.perf_counter()
            e1 = embedding(r["path1"])
            e2 = embedding(r["path2"])
            ok = e1 is not None and e2 is not None
            r["score"] = (
                float(FaceService.cosine_similarity(e1, e2)) if ok else 0.0
            )
            r["no_face"] = not ok
            r["face1"] = e1 is not None
            r["face2"] = e2 is not None
            r["ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            if args.progress_every and (i % args.progress_every == 0 or i == len(rows)):
                print(f"  ... {r['split']} {i}/{len(rows)} cặp", flush=True)

    t0 = time.perf_counter()
    score_rows(val_rows)
    score_rows(test_rows)
    elapsed = time.perf_counter() - t0

    val_metrics = _split_metrics(val_rows, threshold)
    test_metrics = _split_metrics(test_rows, threshold)
    val_eer_thr = float(val_metrics["eer"]["threshold"])
    test_at_val = _split_metrics(test_rows, val_eer_thr)

    # TAR@FAR: chọn ngưỡng trên val để đạt FAR mục tiêu, áp nguyên sang test.
    val_impostor = np.asarray(
        [r["score"] for r in val_rows if r["label"] == "impostor"], dtype=float
    )
    test_genuine = np.asarray(
        [r["score"] for r in test_rows if r["label"] == "genuine"], dtype=float
    )
    test_impostor = np.asarray(
        [r["score"] for r in test_rows if r["label"] == "impostor"], dtype=float
    )
    tar_at_far: Dict[str, Dict[str, float]] = {}
    for target in (0.01, 0.001):
        thr_t = _threshold_for_far(val_impostor, target)
        key = f"far_{target:g}"
        tar_at_far[key] = {
            "target_far": target,
            "val_threshold": round(thr_t, 4),
            "val_far": round(float((val_impostor >= thr_t).mean()), 4),
            "test_far": round(float((test_impostor >= thr_t).mean()), 4),
            "test_tar": round(float((test_genuine >= thr_t).mean()), 4),
        }

    out: Dict[str, Any] = {
        "dataset": {
            "name": "LFW View 1 (deep-funneled)",
            "root": str(root),
            "pairs_dir": str(pairs_dir),
            "n_unique_images_used": len(cache),
            "n_no_face_images": sum(1 for v in cache.values() if v is None),
            "cache_hits": cache_hits,
        },
        "protocol": {
            "val": ["matchpairsDevTrain.csv", "mismatchpairsDevTrain.csv"],
            "test": ["matchpairsDevTest.csv", "mismatchpairsDevTest.csv"],
            "threshold_policy": (
                "Chọn ngưỡng EER trên DevTrain, chỉ chấm DevTest một lần; "
                "ngưỡng production được báo cáo riêng."
            ),
            "no_face_policy": (
                "Cặp có ít nhất một ảnh không detect được mặt nhận score 0.0 "
                "và được tính vào FAR/FRR chính; báo cáo detected_only để tách "
                "riêng chất lượng nhận dạng."
            ),
            "match_threshold": round(threshold, 4),
            "limit_per_class": args.limit_per_class,
        },
        "splits": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "test_at_val_eer": test_at_val,
        "tar_at_far": tar_at_far,
        "elapsed_sec": round(elapsed, 2),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    if args.scores_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.scores_csv)), exist_ok=True)
        with open(args.scores_csv, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(
                ["split", "pair_id", "label", "score", "no_face", "ms",
                 "image1", "image2", "face1", "face2"]
            )
            for r in val_rows + test_rows:
                wr.writerow(
                    [
                        r["split"],
                        r["pair_id"],
                        r["label"],
                        round(float(r["score"]), 6),
                        int(r["no_face"]),
                        r["ms"],
                        f"{r['name1']}/{r['name1']}_{r['num1']:04d}.jpg",
                        f"{r['name2']}/{r['name2']}_{r['num2']:04d}.jpg",
                        "ok" if r["face1"] else "no_face",
                        "ok" if r["face2"] else "no_face",
                    ]
                )

    print(
        f"\nLFW: {len(val_rows)} val + {len(test_rows)} test cặp | "
        f"{len(cache)} ảnh unique ({sum(1 for v in cache.values() if v is None)} no-face) | "
        f"{elapsed:.1f}s"
    )
    for split_name, m in (("val", val_metrics), ("test", test_metrics)):
        print(
            f"[{split_name}] EER={m['eer']['eer']:.4f}@{m['eer']['threshold']:.3f} "
            f"AUC={m['auc']:.4f} | @thr {threshold:.3f}: "
            f"FAR={m['at_threshold']['far']:.4f} FRR={m['at_threshold']['frr']:.4f} "
            f"acc={m['at_threshold']['accuracy']:.4f} | no-face {m['n_no_face_pairs']}"
        )
    print(
        f"[test @val-EER {test_at_val['at_threshold']['threshold']:.3f}] "
        f"FAR={test_at_val['at_threshold']['far']:.4f} "
        f"FRR={test_at_val['at_threshold']['frr']:.4f} "
        f"acc={test_at_val['at_threshold']['accuracy']:.4f}"
    )
    print(f"Đã ghi {args.out}")
    if args.scores_csv:
        print(f"Đã ghi {args.scores_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
