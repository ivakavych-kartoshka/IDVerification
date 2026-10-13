import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.active_liveness_service import ActiveLivenessService  # noqa: E402


def test_analyze_no_frames():
    svc = ActiveLivenessService()
    result = svc.analyze([])
    assert result["status"] == "FAIL"
    assert result["frames"] == 0


def test_is_frozen_true_for_identical_frames():
    frame = np.full((32, 32, 3), 100, dtype=np.uint8)
    assert ActiveLivenessService._is_frozen([frame, frame.copy(), frame.copy()], 1.5) is True


def test_is_frozen_false_for_changing_frames():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = np.full((32, 32, 3), 255, dtype=np.uint8)
    assert ActiveLivenessService._is_frozen([a, b], 1.5) is False


def test_is_frozen_false_for_single_frame():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    assert ActiveLivenessService._is_frozen([frame], 1.5) is False
