import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ocr_service import OCRService  # noqa: E402


def test_prefers_twelve_digit_cccd():
    text = "So: 012345678901 cap ngay 01/01/2020"
    assert OCRService._find_document_number(text) == "012345678901"


def test_prefers_twelve_over_nine():
    text = "CMND 123456789 / CCCD 012345678901"
    assert OCRService._find_document_number(text) == "012345678901"


def test_falls_back_to_nine_digit_cmnd():
    assert OCRService._find_document_number("CMND so 123456789") == "123456789"


def test_returns_none_when_absent():
    assert OCRService._find_document_number("no number here") is None
    assert OCRService._find_document_number("") is None
