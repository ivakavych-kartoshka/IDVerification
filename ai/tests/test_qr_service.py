import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.qr_service import QRService  # noqa: E402

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "cccd_back_qr.png"
)
PAYLOAD = (
    "068096013455|251043755|Trần Chí Hào|08101996|Nam|"
    "Tổ 2, Lộc Thắng, Bảo Lâm, Lâm Đồng|30082021"
)


def test_parse_with_legacy_id():
    f = QRService.parse_cccd(PAYLOAD)
    assert f["document_number"] == "068096013455"
    assert f["legacy_id"] == "251043755"
    assert f["full_name"] == "Trần Chí Hào"
    assert f["dob"] == "08101996"
    assert f["gender"] == "Nam"
    assert f["issue_date"] == "30082021"
    assert f["native_place"].startswith("Tổ 2")


def test_parse_without_legacy_id():
    f = QRService.parse_cccd("068096013455|Trần Chí Hào|08101996|Nữ|Hà Nội|30082021")
    assert f["document_number"] == "068096013455"
    assert "legacy_id" not in f
    assert f["gender"] == "Nữ"
    assert f["dob"] == "08101996"


def test_compare_match_ignores_accents_and_separators():
    svc = QRService()
    f = svc.parse_cccd(PAYLOAD)
    ocr = "068096013455 TRAN CHI HAO 08/10/1996 Nam"
    r = svc.compare(f, ocr)
    assert r["status"] == "MATCH"
    assert r["score"] == 1.0
    assert set(r["matched_fields"]) == {"document_number", "full_name", "dob", "gender"}


def test_compare_mismatch_document_number():
    svc = QRService()
    f = svc.parse_cccd(PAYLOAD)
    r = svc.compare(f, "012345678901 TRAN VAN B 01/01/1990 Nam")
    assert r["status"] == "MISMATCH"
    assert "document_number" in r["mismatched_fields"]


def test_compare_partial_when_only_name_differs():
    svc = QRService()
    f = svc.parse_cccd(PAYLOAD)
    r = svc.compare(f, "068096013455 08/10/1996 Nam")
    assert r["status"] == "PARTIAL"
    assert r["mismatched_fields"] == ["full_name"]


def test_compare_not_found_when_no_qr():
    svc = QRService()
    assert svc.compare({}, "anything")["status"] == "NOT_FOUND"


def test_compare_not_found_when_no_ocr():
    svc = QRService()
    f = svc.parse_cccd(PAYLOAD)
    assert svc.compare(f, "")["status"] == "NOT_FOUND"


@pytest.mark.skipif(not os.path.exists(FIXTURE), reason="fixture image missing")
def test_decode_fixture_image():
    svc = QRService()
    with open(FIXTURE, "rb") as fh:
        content = fh.read()
    res = svc.decode(content)
    assert res["found"] is True
    assert res["fields"]["document_number"] == "068096013455"
    assert res["fields"]["full_name"] == "Trần Chí Hào"
