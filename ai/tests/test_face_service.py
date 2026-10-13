import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.face_service import FaceService  # noqa: E402


def test_detector_uses_model_input_size():
    # version-RFB-640 needs a fixed 480x640 (HxW) input; a wrong size makes
    # onnxruntime raise and the detector silently fall back to Haar cascades.
    assert FaceService()._det_size == (640, 480)


def test_preprocess_shape():
    img = np.zeros((120, 90, 3), dtype=np.uint8)
    x = FaceService._preprocess_ultraface(img, (640, 480))
    assert x.shape == (1, 3, 480, 640)
    assert x.dtype == np.float32


def test_postprocess_decodes_and_suppresses():
    # Normalised (x1, y1, x2, y2) boxes + per-anchor scores.
    scores = np.array([[[0.1, 0.9], [0.2, 0.8], [0.9, 0.05]]], dtype=np.float32)
    boxes = np.array(
        [[[0.1, 0.1, 0.5, 0.5], [0.12, 0.12, 0.52, 0.52], [0.6, 0.6, 0.9, 0.9]]],
        dtype=np.float32,
    )
    out = FaceService._postprocess_ultraface({"scores": scores, "boxes": boxes}, (100, 100))
    # Box 1 is suppressed by NMS against box 0; box 2 is below the score floor.
    assert out == [(10, 10, 40, 40)]


def test_nms_keeps_distinct_boxes():
    dets = [(0, 0, 10, 10, 0.9), (1, 1, 10, 10, 0.8), (50, 50, 10, 10, 0.7)]
    kept = FaceService._nms(dets, 0.4)
    assert len(kept) == 2
    assert kept[0][4] == 0.9 and kept[1][4] == 0.7


def test_detect_face_boxes_runs():
    svc = FaceService()
    try:
        svc._init_detector()
    except Exception:
        pytest.skip("detector model not available")
    result = svc.detect_face_boxes(np.zeros((240, 320, 3), dtype=np.uint8))
    assert isinstance(result, list)
