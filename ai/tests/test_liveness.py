import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.fasnet import (  # noqa: E402
    FASNetAntiSpoof,
    _get_new_box,
    crop_face,
)

MODEL_ENV = "AI_FAS_MODEL"


def _model_available() -> bool:
    return FASNetAntiSpoof().available()


@pytest.fixture(scope="module")
def fas():
    svc = FASNetAntiSpoof()
    if not svc.available():
        pytest.skip("MiniFASNetV2 checkpoint or torch not available")
    return svc


def test_get_new_box_stays_within_image():
    left, top, right, bottom = _get_new_box(200, 100, (10, 20, 40, 40), 2.7)
    assert all(isinstance(v, int) for v in (left, top, right, bottom))
    assert 0 <= left < right <= 199
    assert 0 <= top < bottom <= 99
    # Expanded box must be larger than the original 40x40 face box.
    assert (right - left) >= 40
    assert (bottom - top) >= 40


def test_crop_face_returns_target_size():
    img = np.zeros((320, 240, 3), dtype=np.uint8)
    crop = crop_face(img, (60, 40, 120, 120), 2.7, 80, 80)
    assert crop.shape == (80, 80, 3)
    assert crop.dtype == np.uint8


def test_predict_uses_0_255_input_range(fas):
    """Regression: the reference pipeline feeds [0, 255] (it disables /255).

    A /255 normalisation makes every genuine face look like a spoof, so assert
    the tensor handed to the network is never scaled down into [0, 1].
    """
    fas._load()
    captured = {}

    def _capture(module, inputs):
        captured["x"] = inputs[0].detach()

    handle = fas._model.register_forward_pre_hook(_capture)
    try:
        img = np.full((160, 160, 3), 200, dtype=np.uint8)
        fas.predict(img, (30, 30, 100, 100))
    finally:
        handle.remove()

    x = captured["x"]
    assert float(x.max()) > 1.0
    assert float(x.max()) <= 255.0


def test_predict_returns_three_class_probabilities(fas):
    img = np.full((160, 160, 3), 128, dtype=np.uint8)
    result = fas.predict(img, (30, 30, 100, 100))
    assert result["engine"] == "MiniFASNetV2"
    assert len(result["scores"]) == 3
    assert result["label"] in {"real", "print", "replay"}
    assert abs(sum(result["scores"]) - 1.0) < 1e-2
    assert result["is_live"] == (result["label"] == "real")
