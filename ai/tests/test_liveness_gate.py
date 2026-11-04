import io
import os
import sys
from unittest.mock import Mock

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.liveness_service import LivenessService  # noqa: E402
from app.services.verify_service import VerifyService  # noqa: E402


def test_low_blur_returns_retake_without_calling_fas(monkeypatch):
    monkeypatch.setenv("AI_FAS_BLUR_GATE", "25")
    svc = LivenessService()
    monkeypatch.setattr(svc, "_blur_var", lambda _img: 5.0)
    monkeypatch.setattr(svc.fas, "available", lambda: True)
    monkeypatch.setattr(
        svc.fas,
        "predict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("MiniFASNet must not run when the quality gate fails")
        ),
    )

    res = svc.passive_liveness(
        np.zeros((64, 64, 3), dtype=np.uint8), face_bbox=(0, 0, 64, 64)
    )

    assert res["status"] == "LOW_QUALITY"
    assert res["label"] == "low_quality"
    assert res["is_live"] is False
    assert res["retake_required"] is True
    assert res["blur_var"] == 5.0


def test_blurry_frame_without_detectable_face_returns_retake(monkeypatch):
    monkeypatch.setenv("AI_FAS_BLUR_GATE", "25")
    svc = LivenessService()
    monkeypatch.setattr(svc, "_blur_var", lambda _img: 5.0)
    monkeypatch.setattr(svc.fas, "available", lambda: True)
    monkeypatch.setattr(
        svc,
        "_get_face_service",
        lambda: Mock(detect_face_boxes=lambda _img: []),
    )

    res = svc.passive_liveness(np.zeros((64, 64, 3), dtype=np.uint8))

    assert res["status"] == "LOW_QUALITY"
    assert res["retake_required"] is True


def test_quality_gate_can_be_disabled(monkeypatch):
    monkeypatch.setenv("AI_FAS_BLUR_GATE", "0")
    svc = LivenessService()
    monkeypatch.setattr(svc, "_blur_var", lambda _img: 5.0)
    monkeypatch.setattr(svc.fas, "available", lambda: True)
    monkeypatch.setattr(
        svc.fas,
        "predict",
        lambda *_args, **_kwargs: {
            "is_live": True,
            "label": "real",
            "real_prob": 0.9,
            "engine": "MiniFASNetV2",
        },
    )

    res = svc.passive_liveness(
        np.zeros((64, 64, 3), dtype=np.uint8), face_bbox=(0, 0, 64, 64)
    )

    assert res["status"] == "PASS"
    assert res["is_live"] is True
    assert res["retake_required"] is False
    assert res["engine"] == "MiniFASNetV2"


def test_sharp_image_reaches_fas(monkeypatch):
    monkeypatch.setenv("AI_FAS_BLUR_GATE", "25")
    svc = LivenessService()
    monkeypatch.setattr(svc, "_blur_var", lambda _img: 120.0)
    monkeypatch.setattr(svc.fas, "available", lambda: True)
    monkeypatch.setattr(
        svc.fas,
        "predict",
        lambda *_args, **_kwargs: {
            "is_live": True,
            "label": "real",
            "real_prob": 0.95,
            "engine": "MiniFASNetV2",
        },
    )

    res = svc.passive_liveness(
        np.zeros((64, 64, 3), dtype=np.uint8), face_bbox=(0, 0, 64, 64)
    )

    assert res["status"] == "PASS"
    assert res["retake_required"] is False
    assert res["blur_gate"] == 25.0


def test_decide_status_retry_versus_reject():
    low = {"is_live": False, "status": "LOW_QUALITY", "reason": "blurry"}
    ok = {"is_live": True, "status": "PASS"}

    # Blur makes face/active results unreliable: retry while the document is OK.
    assert VerifyService._decide_status("MATCH", low, True, False, False, 1, 1, False) == "RETRY"
    assert VerifyService._decide_status("NO_MATCH", low, False, False, False, 1, 1, False) == "RETRY"
    assert VerifyService._decide_status("MATCH", low, False, False, False, 1, 0, False) == "RETRY"

    # Independent document/security failures still reject.
    assert VerifyService._decide_status("MATCH", low, True, True, False, 1, 1, False) == "REJECTED"
    assert VerifyService._decide_status("MATCH", low, True, False, True, 1, 1, False) == "REJECTED"
    assert VerifyService._decide_status("MATCH", low, True, False, False, 0, 1, False) == "REJECTED"
    assert VerifyService._decide_status("MATCH", low, True, False, False, 1, 1, True) == "REJECTED"

    # The normal paths are unchanged.
    assert VerifyService._decide_status("MATCH", ok, True, False, False, 1, 1, False) == "VERIFIED"
    assert VerifyService._decide_status("NO_MATCH", ok, True, False, False, 1, 1, False) == "REJECTED"


def test_verify_returns_retry_for_blurry_selfie():
    def _png() -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (120, 120, 120)).save(buf, format="PNG")
        return buf.getvalue()

    svc = VerifyService.__new__(VerifyService)
    svc.ocr = Mock()
    svc.ocr.extract.return_value = {"lines": [], "full_text": "", "document_number": None}
    svc.qr = Mock()
    svc.qr.compare.return_value = {"status": "MATCH", "details": {}}
    svc.face = Mock()
    svc.face.face_match.return_value = {
        "match_score": 0.9,
        "status": "MATCH",
        "face_count_selfie": 1,
        "face_count_id": 1,
    }
    svc.face.detect_face_boxes.return_value = [(0, 0, 64, 64)]
    svc.liveness = Mock()
    svc.liveness.passive_liveness.return_value = {
        "is_live": False,
        "status": "LOW_QUALITY",
        "label": "low_quality",
        "retake_required": True,
        "reason": "blurry",
    }
    svc.active = Mock()
    svc.tamper = Mock()
    svc.tamper.analyze.return_value = {"status": "OK", "tamper_score": 0.0, "regions": []}
    svc.tamper_enabled = True
    svc.qr_enabled = False
    svc.qr_require = False
    svc.qr_strict = True
    svc.qr_strict_name = False

    res = svc.verify(_png(), None, _png(), None)

    assert res["status"] == "RETRY"
    assert res["retake_required"] is True
    assert res["retake_reason"] == "blurry"
    assert res["rejection_reason"] is None
