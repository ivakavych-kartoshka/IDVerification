from typing import Any, Dict, List, Optional, Tuple
import io
import re
import unicodedata

import cv2
import numpy as np
from PIL import Image

# Vietnamese CCCD/CC chip card QR payload (pipe-separated), documented as:
#   <document_number>|<legacy_id?>|<full_name>|<dob>|<gender>|<native_place>|<issue_date>
# The legacy 9-digit CMND field is optional, so parsing below is order-tolerant
# and classifies fields by shape instead of relying on fixed positions.
GENDER_TOKENS = {"nam", "nu", "male", "female", "m", "f"}


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def _normalize(text: str) -> str:
    """Lowercase, accent-free, alphanumeric-only representation for matching."""
    return re.sub(r"[^a-z0-9]", "", _strip_accents(text).lower())


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


class QRService:
    """Decode the QR code embedded on the back of a CCCD/CC card.

    The QR is *not* a cryptographic proof (the printed code can be copied), so
    it is used as a second, structured data source that is cross-checked against
    the OCR text. A QR/OCR mismatch is a useful tampering signal.
    """

    def __init__(self):
        self._det = cv2.QRCodeDetector()
        try:
            self._det_aruco = cv2.QRCodeDetectorAruco()
        except Exception:
            self._det_aruco = None
        try:
            from pyzbar.pyzbar import decode as _zbar_decode
            self._zbar = _zbar_decode
        except Exception:
            self._zbar = None

    # ------------------------------------------------------------ decoding
    @staticmethod
    def _variants(bgr: np.ndarray):
        """Yield progressively preprocessed images to improve QR detection."""
        h, w = bgr.shape[:2]
        longest = max(h, w)
        if longest > 2400:
            factor = 2400.0 / longest
            bgr = cv2.resize(bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

        yield bgr
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        yield gray

        # Only upscale small images; upscaling large ones is slow and pointless.
        if max(bgr.shape[:2]) < 1600:
            for scale in (2, 3):
                up = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                yield up
                yield cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

        try:
            th = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
            )
            yield th
        except Exception:
            pass

    def _decode_variant(self, img: np.ndarray) -> List[Tuple[str, str]]:
        found: List[Tuple[str, str]] = []
        for name, det in (("opencv", self._det), ("opencv_aruco", self._det_aruco)):
            if det is None:
                continue
            try:
                ok, texts, _pts, _ = det.detectAndDecodeMulti(img)
                if ok and texts:
                    for t in texts:
                        if t:
                            found.append((str(t), name))
            except Exception:
                pass
            try:
                t, _pts, _ = det.detectAndDecode(img)
                if t:
                    found.append((str(t), name))
            except Exception:
                pass
            if found:
                return found
        if self._zbar is not None:
            try:
                if img.ndim == 3:
                    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    pil = Image.fromarray(img)
                for r in self._zbar(pil):
                    data = r.data.decode("utf-8", errors="ignore")
                    if data:
                        found.append((data, "pyzbar"))
            except Exception:
                pass
        return found

    def decode(self, content: bytes, filename: str = "image.jpg") -> Dict[str, Any]:
        bgr = None
        try:
            bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            bgr = None
        if bgr is None:
            try:
                rgb = np.array(Image.open(io.BytesIO(content)).convert("RGB"))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                return {"found": False, "payload": None, "fields": None, "reader": None}

        for variant in self._variants(bgr):
            hits = self._decode_variant(variant)
            if hits:
                payload, reader = hits[0]
                return {
                    "found": True,
                    "payload": payload,
                    "fields": self.parse_cccd(payload),
                    "reader": reader,
                }
        return {"found": False, "payload": None, "fields": None, "reader": None}

    # ------------------------------------------------------------- parsing
    @staticmethod
    def parse_cccd(payload: Optional[str]) -> Dict[str, Any]:
        if not payload:
            return {}
        parts = [p.strip() for p in payload.split("|")]
        fields: Dict[str, Any] = {"raw_parts": parts}
        dates: List[str] = []
        texts: List[str] = []

        for part in parts:
            if not part:
                continue
            if re.fullmatch(r"\d{12}", part) and "document_number" not in fields:
                fields["document_number"] = part
            elif re.fullmatch(r"\d{9}", part) and "legacy_id" not in fields:
                fields["legacy_id"] = part
            elif re.fullmatch(r"\d{8}", part):
                dates.append(part)
            elif _normalize(part) in GENDER_TOKENS and "gender" not in fields:
                fields["gender"] = part
            else:
                texts.append(part)

        if dates:
            fields["dob"] = dates[0]
        if len(dates) > 1:
            fields["issue_date"] = dates[1]
        if texts:
            fields["full_name"] = texts[0]
        if len(texts) > 1:
            fields["native_place"] = texts[1]
        return fields

    # ---------------------------------------------------------- comparison
    def compare(self, qr_fields: Optional[Dict[str, Any]], ocr_text: str) -> Dict[str, Any]:
        """Cross-check decoded QR fields against the OCR text.

        Returns a consistency verdict:
          NOT_FOUND  - no QR data (or nothing to verify against)
          MATCH      - every verifiable field is present in the OCR text
          PARTIAL    - some fields match, some do not
          MISMATCH   - document number present but absent from OCR (strong signal)
        """
        qr_fields = qr_fields or {}
        ocr_text = ocr_text or ""
        norm_text = _normalize(ocr_text)
        ocr_digits = _digits(ocr_text)
        ocr_is_empty = not norm_text and not ocr_digits

        details: Dict[str, Dict[str, Any]] = {}
        matched: List[str] = []
        mismatched: List[str] = []
        not_checked: List[str] = []

        def record(field: str, qr_val: Optional[str], ok: Optional[bool]) -> None:
            if qr_val is None or ok is None:
                not_checked.append(field)
                if qr_val is not None:
                    details[field] = {"qr": qr_val, "ocr": None, "match": None}
                return
            details[field] = {"qr": qr_val, "ocr": ocr_text if ok else None, "match": ok}
            (matched if ok else mismatched).append(field)

        if ocr_is_empty:
            return {
                "status": "NOT_FOUND",
                "score": 0.0,
                "checked_fields": [],
                "matched_fields": [],
                "mismatched_fields": [],
                "not_checked_fields": list(qr_fields.keys()),
                "details": {},
                "message": "No OCR text to compare with QR",
            }

        doc = qr_fields.get("document_number")
        if doc:
            record("document_number", doc, doc in ocr_digits if ocr_digits else None)

        name = qr_fields.get("full_name")
        if name:
            name_norm = _normalize(name)
            record("full_name", name, name_norm in norm_text if name_norm else None)

        dob = qr_fields.get("dob")
        if dob:
            record("dob", dob, _digits(dob) in ocr_digits if _digits(dob) else None)

        gender = qr_fields.get("gender")
        if gender:
            record("gender", gender, _normalize(gender) in norm_text)

        checked = matched + mismatched
        score = round(len(matched) / len(checked), 4) if checked else 0.0

        if not qr_fields:
            status = "NOT_FOUND"
        elif not checked:
            status = "NOT_FOUND"
        elif "document_number" in mismatched:
            status = "MISMATCH"
        elif not mismatched:
            status = "MATCH"
        else:
            status = "PARTIAL"

        return {
            "status": status,
            "score": score,
            "checked_fields": checked,
            "matched_fields": matched,
            "mismatched_fields": mismatched,
            "not_checked_fields": not_checked,
            "details": details,
        }
