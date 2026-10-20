"""Tests cho document tamper detection (hướng A).

Không cần model đã train: test kiến trúc (shape/params) và các tín hiệu
heuristic. Phần chạy model sẽ tự skip nếu thiếu torch hoặc checkpoint.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.tamper_service import (  # noqa: E402
    OK,
    SUSPICIOUS,
    TAMPERED,
    TamperService,
    _blockiness,
    _letterbox,
    _low_hf_fraction,
    _unletterbox_box,
    mask_to_regions,
)


# --------------------------------------------------------------------- model
def test_tamper_net_shapes():
    torch = pytest.importorskip("torch")
    from app.services.tamper_net import TamperNet, count_params

    model = TamperNet(input_size=64)
    out = model(torch.rand(2, 3, 64, 64))
    assert out["logit"].shape == (2,)
    assert out["mask_logits"].shape == (2, 1, 64, 64)
    # Model phải đủ nhẹ để chạy CPU cạnh OCR/face trong cùng service.
    assert count_params(model) < 3_000_000


def test_tamper_net_requires_size_divisible_by_block():
    pytest.importorskip("torch")
    from app.services.tamper_net import TamperNet

    with pytest.raises(AssertionError):
        TamperNet(input_size=100)  # 100 % 8 != 0


def test_tamper_loss_is_finite():
    torch = pytest.importorskip("torch")
    from app.services.tamper_net import TamperLoss, TamperNet

    model = TamperNet(input_size=64)
    out = model(torch.rand(2, 3, 64, 64))
    loss, parts = TamperLoss()(out, torch.tensor([0.0, 1.0]))
    assert torch.isfinite(loss)
    assert "loss_cls" in parts


# ---------------------------------------------------------------- geometry
def test_letterbox_roundtrip_box():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    canvas, meta = _letterbox(img, 256)
    assert canvas.shape == (256, 256, 3)
    # Box ở toạ độ canvas -> ảnh gốc phải nằm trong ảnh và không âm.
    box = _unletterbox_box([meta[0], meta[1], 50, 50], meta)
    assert box[0] >= 0 and box[1] >= 0 and box[2] > 0 and box[3] > 0
    assert box[0] + box[2] <= 200 and box[1] + box[3] <= 100


def test_mask_to_regions_finds_component():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:32, 16:32] = 1.0
    _, meta = _letterbox(np.zeros((128, 128, 3), dtype=np.uint8), 64)
    regions = mask_to_regions(mask, meta)
    assert len(regions) == 1
    assert regions[0]["area_frac"] == pytest.approx(1 / 16, abs=0.02)


# --------------------------------------------------------------- heuristic
def test_blockiness_detects_jpeg_grid():
    flat = np.full((64, 64), 128, dtype=np.float32)
    assert _blockiness(flat) == pytest.approx(0.0, abs=1e-6)

    grid = flat.copy()
    grid[:, 7::8] += 20.0  # tạo lưới giả JPEG 8px
    assert _blockiness(grid) > 0.1


def test_low_hf_fraction_high_on_flat_image():
    flat = np.full((64, 64), 128, dtype=np.float32)
    noisy = np.random.default_rng(0).integers(0, 255, (64, 64)).astype(np.float32)
    assert _low_hf_fraction(flat) > _low_hf_fraction(noisy)


def test_forensic_signals_with_and_without_photo_box():
    img = np.random.default_rng(1).integers(0, 255, (240, 320, 3)).astype(np.uint8)
    with_box = TamperService.forensic_signals(img, [80, 60, 120, 120])
    without = TamperService.forensic_signals(img, None)
    assert with_box["raw"]["boundary_energy"] is not None
    assert with_box["raw"]["noise_step"] is not None
    # Không có vùng chân dung (ảnh mặt sau) -> tín hiệu mép = None, không crash.
    assert without["raw"]["boundary_energy"] is None
    assert without["raw"]["noise_step"] is None
    assert without["raw"]["blockiness"] is not None


def test_signal_score_renormalises_weights():
    all_present = TamperService._signal_score(
        {"boundary_energy": 1.0, "noise_step": 1.0, "blockiness": 1.0, "low_hf_fraction": 1.0}
    )
    missing_boundary = TamperService._signal_score(
        {"boundary_energy": None, "noise_step": 1.0, "blockiness": 1.0, "low_hf_fraction": 1.0}
    )
    assert all_present == pytest.approx(1.0)
    assert missing_boundary == pytest.approx(1.0)


def test_heuristic_engine_never_rejects_alone():
    # Không có model -> dù score vượt threshold cũng chỉ SUSPICIOUS.
    assert TamperService._classify(0.99, model_loaded=False, threshold=0.55, suspect=0.35) == SUSPICIOUS
    assert TamperService._classify(0.99, model_loaded=True, threshold=0.55, suspect=0.35) == TAMPERED
    assert TamperService._classify(0.01, model_loaded=True, threshold=0.55, suspect=0.35) == OK


def test_analyze_without_model_returns_contract():
    img = np.random.default_rng(2).integers(0, 255, (300, 400, 3)).astype(np.uint8)
    res = TamperService().analyze(img)
    assert set(["enabled", "status", "tamper_score", "engine", "model_loaded", "regions"]) <= set(res)
    assert res["status"] in (OK, SUSPICIOUS, TAMPERED, "DISABLED")
    if not res["model_loaded"]:
        assert res["engine"] in ("heuristic", None)
        assert res["status"] != TAMPERED


def test_analyze_disabled():
    os.environ["AI_TAMPER_ENABLED"] = "false"
    try:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = TamperService().analyze(img)
        assert res["status"] == "DISABLED"
        assert res["enabled"] is False
    finally:
        os.environ.pop("AI_TAMPER_ENABLED", None)
