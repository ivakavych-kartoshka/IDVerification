from typing import Any, Dict, List, Optional, Tuple
import os
import re
from datetime import date, datetime
import numpy as np
from PIL import Image, ImageOps
from app.services.ocr_service import OCRService
from app.services.face_service import FaceService
from app.services.liveness_service import LivenessService
from app.services.active_liveness_service import ActiveLivenessService
from app.services.qr_service import QRService
from app.services.tamper_service import TamperService


class VerifyService:
    def __init__(self):
        self.ocr = OCRService()
        self.face = FaceService()
        self.liveness = LivenessService()
        self.active = ActiveLivenessService()
        self.qr = QRService()
        self.tamper = TamperService()
        self.tamper_enabled = self._env_flag("AI_TAMPER_ENABLED", True)
        self.tamper_advisory = self._env_flag("AI_TAMPER_ADVISORY", True)
        self.qr_enabled = self._env_flag("AI_QR_ENABLED", True)
        self.qr_require = self._env_flag("AI_QR_REQUIRE", False)
        self.qr_strict = self._env_flag("AI_QR_STRICT", True)
        self.qr_strict_name = self._env_flag("AI_QR_STRICT_NAME", False)
        try:
            self.min_age = int(os.environ.get("AI_MIN_AGE", "18"))
        except (TypeError, ValueError):
            self.min_age = 18
        if self.min_age < 0:
            self.min_age = 18
    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() not in ("false", "0", "no", "off")

    def _evaluate_qr(
        self, qr_fields: Dict[str, Any], consistency: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Decide whether the QR/OCR cross-check should reject the request."""
        if not self.qr_enabled:
            return False, None
        if not qr_fields:
            if self.qr_require:
                return True, "QR code not found on document"
            return False, None
        if self.qr_strict and consistency.get("status") == "MISMATCH":
            return True, "QR/OCR mismatch on document number"
        if self.qr_strict_name:
            detail = (consistency.get("details") or {}).get("full_name")
            if detail and detail.get("match") is False:
                return True, "QR/OCR mismatch on full name"
        return False, None

    def _blur_score(self, img_arr: np.ndarray) -> float:
        try:
            import cv2
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            var = lap.var()
            return float(var)
        except Exception:
            return 0.0

    @staticmethod
    def _parse_dob(value: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        text = str(value).strip()
        if not text:
            return None
        # Normalise common separators.
        candidates = [
            text,
            text.replace("/", "").replace("-", "").replace(".", "").replace(" ", ""),
        ]
        formats = (
            "%d%m%Y",
            "%Y%m%d",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d.%m.%Y",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%m/%d/%Y",
        )
        for fmt in formats:
            for cand in candidates:
                try:
                    return datetime.strptime(cand, fmt).date()
                except ValueError:
                    continue
        # Last resort: pull the first 8 consecutive digits.
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 8:
            for fmt in ("%d%m%Y", "%Y%m%d"):
                try:
                    return datetime.strptime(digits[:8], fmt).date()
                except ValueError:
                    continue
        return None

    def _age_from_dob(self, dob: Any, ref_date: Optional[date] = None) -> Optional[int]:
        birth = self._parse_dob(dob)
        if birth is None:
            return None
        ref = ref_date or date.today()
        if birth > ref:
            return None
        years = ref.year - birth.year
        if (ref.month, ref.day) < (birth.month, birth.day):
            years -= 1
        return years

    def _age_details(self, dob: Any, ref_date: Optional[date] = None) -> Dict[str, Any]:
        birth = self._parse_dob(dob)
        age = self._age_from_dob(dob, ref_date)
        min_age = getattr(self, "min_age", 18)
        underage = bool(age is not None and age < min_age)
        return {
            "dob": str(dob).strip() if dob is not None else None,
            "birth_date": birth.isoformat() if birth else None,
            "age": age,
            "min_age": min_age,
            "is_underage": underage,
            "is_under_18": underage,
        }

    def _check_age(self, dob_or_fields: Any, ref_date: Optional[date] = None) -> Tuple[bool, Optional[str]]:
        """Return ``(reject, reason)`` for an age check.

        Accepts either a raw DOB string/digits or a mapping containing a
        ``dob``/``date_of_birth`` field, so callers can pass QR fields directly.
        """
        dob = dob_or_fields
        if isinstance(dob_or_fields, dict):
            dob = (
                dob_or_fields.get("dob")
                or dob_or_fields.get("date_of_birth")
                or dob_or_fields.get("dateOfBirth")
                or dob_or_fields.get("birth_date")
            )
        details = self._age_details(dob, ref_date)
        min_age = getattr(self, "min_age", 18)
        if details["is_underage"]:
            age_txt = details["age"] if details["age"] is not None else "?"
            return True, f"Under {min_age} years old (age {age_txt}; chưa đủ {min_age} tuổi)"
        return False, None

    # Public aliases used by callers/tests that prefer different names.
    def check_age(self, dob_or_fields: Any, ref_date: Optional[date] = None) -> Tuple[bool, Optional[str]]:
        return self._check_age(dob_or_fields, ref_date)

    def is_under_18(self, dob_or_fields: Any, ref_date: Optional[date] = None) -> bool:
        reject, _ = self._check_age(dob_or_fields, ref_date)
        return reject

    def is_underage(self, dob_or_fields: Any, ref_date: Optional[date] = None) -> bool:
        return self.is_under_18(dob_or_fields, ref_date)

    @staticmethod
    def _extract_dob_from_text(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        # Prefer an explicitly labelled date of birth.
        m = re.search(
            r"(?:ng[aà]y sinh|date of birth|dob|birth date|sinh)\s*[:\-]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4}|\d{8})",
            text,
            re.IGNORECASE,
        )
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _decide_status(
        face_match_status: str,
        passive_liveness: Dict[str, Any],
        active_ok: bool,
        qr_reject: bool = False,
        tamper_reject: bool = False,
        face_count_id: int = 0,
        face_count_selfie: int = 0,
        multiple_faces: bool = False,
        age_reject: bool = False,
        underage_reject: bool = False,
        **kwargs: Any,
    ) -> str:
        age_reject = bool(
            age_reject
            or underage_reject
            or kwargs.get("age_reject")
            or kwargs.get("underage")
            or kwargs.get("underage_reject")
            or kwargs.get("is_underage")
            or kwargs.get("under_18")
        )
        """Resolve the top-level verdict.

        A low-quality selfie (``LOW_QUALITY``) returns ``RETRY`` rather than a
        spoof rejection, provided there is no independent document/security
        failure. Face-match and active-liveness failures are not hard blockers
        in that case because blur makes both unreliable. Document problems
        (missing ID portrait, QR mismatch, tamper, multiple faces) still reject.
        """
        if (
            face_match_status == "MATCH"
            and passive_liveness.get("is_live", True)
            and active_ok
            and not qr_reject
            and not tamper_reject
            and not age_reject
            and not multiple_faces
            and face_count_id > 0
            and face_count_selfie > 0
        ):
            return "VERIFIED"

        document_or_security_fail = (
            qr_reject or tamper_reject or age_reject or multiple_faces or face_count_id == 0
        )
        if (
            passive_liveness.get("status") == "LOW_QUALITY"
            and not document_or_security_fail
        ):
            return "RETRY"
        return "REJECTED"

    def verify(
        self,
        front_b: bytes,
        back_b: Optional[bytes],
        selfie_b: Optional[bytes],
        liveness_frames: Optional[List[bytes]] = None,
    ) -> Dict[str, Any]:
        import io

        def _load_rgb(content: Optional[bytes]):
            if not content:
                return None
            img = Image.open(io.BytesIO(content))
            return np.array(ImageOps.exif_transpose(img).convert("RGB"))

        front_arr = _load_rgb(front_b)
        back_arr = _load_rgb(back_b)
        selfie_arr = _load_rgb(selfie_b)

        ocr_front = self.ocr.extract(front_b)
        ocr_back = self.ocr.extract(back_b) if back_b else {"lines": [], "full_text": "", "document_number": None}

        qr_front = self.qr.decode(front_b) if self.qr_enabled else {"found": False, "payload": None, "fields": None, "reader": None}
        qr_back = (
            self.qr.decode(back_b)
            if (self.qr_enabled and back_b)
            else {"found": False, "payload": None, "fields": None, "reader": None}
        )
        qr_fields: Dict[str, Any] = (
            (qr_back.get("fields") if qr_back.get("found") else None)
            or (qr_front.get("fields") if qr_front.get("found") else None)
            or {}
        )

        combined_ocr = " ".join([ocr_front.get("full_text") or "", ocr_back.get("full_text") or ""])
        # Age check uses the structured DOB from the QR code, falling back to
        # an explicitly labelled date of birth in the OCR text when needed.
        if not qr_fields.get("dob"):
            fallback_dob = self._extract_dob_from_text(combined_ocr)
            if fallback_dob:
                qr_fields = dict(qr_fields)
                qr_fields["dob"] = fallback_dob
        qr_consistency = self.qr.compare(qr_fields, combined_ocr)

        # The QR carries structured data, so fall back to it when OCR fails to
        # read the document number.
        if not ocr_front.get("document_number") and qr_fields.get("document_number"):
            ocr_front["document_number"] = qr_fields["document_number"]

        doc_quality = {"front_blur_var": self._blur_score(front_arr), "back_blur_var": self._blur_score(back_arr) if back_arr is not None else 0.0}
        face_res = self.face.face_match(selfie_b, front_b) if (selfie_b and front_b) else {"match_score": 0.0, "status": "MISSING", "face_count_selfie": 0, "face_count_id": 0}
        face = {"face_count_selfie": face_res.get("face_count_selfie", 0), "face_count_id": face_res.get("face_count_id", 0)}
        face_match = {"match_score": face_res.get("match_score", 0.0), "status": face_res.get("status", "MISSING")}
        passive_liveness = self.liveness.passive_liveness(selfie_arr) if selfie_arr is not None else {"is_live": True, "status": "SIMPLE_OK", "blur_var": 0.0}

        selfie_face_count = len(self.face.detect_face_boxes(selfie_arr)) if selfie_arr is not None else 0
        multiple_faces = selfie_face_count > 1

        if liveness_frames:
            active_liveness = self.active.analyze(liveness_frames)
        else:
            active_liveness = {"status": "NOT_PROVIDED", "message": "No liveness frames provided"}

        active_ok = active_liveness.get("status") == "PASS"
        qr_reject, qr_reason = self._evaluate_qr(qr_fields, qr_consistency)
        age_reject, age_reason = self._check_age(qr_fields)
        age_details = self._age_details(qr_fields.get("dob") if isinstance(qr_fields, dict) else qr_fields)
        tamper = (
            self.tamper.analyze(front_arr)
            if (self.tamper_enabled and front_arr is not None)
            else {"enabled": False, "status": "DISABLED", "tamper_score": 0.0, "regions": []}
        )
        tamper_reject = (
            False if getattr(self, "tamper_advisory", True) else tamper.get("status") == "TAMPERED"
        )

        status = self._decide_status(
            face_match.get("status", ""),
            passive_liveness,
            active_ok,
            qr_reject,
            tamper_reject,
            face.get("face_count_id", 0),
            face.get("face_count_selfie", 0),
            multiple_faces,
            age_reject=age_reject,
        )

        retake_reason = None
        rejection = None
        if status == "RETRY":
            retake_reason = passive_liveness.get("reason") or (
                "Selfie image too blurry; please retake in better focus/lighting"
            )
        elif status == "REJECTED":
            reasons = []
            if age_reject and age_reason:
                reasons.append(age_reason)
            else:
                if face_match.get("status") != "MATCH":
                    reasons.append("Face mismatch" if face_match.get("status") == "NO_MATCH" else "Face match too weak")
                if not passive_liveness.get("is_live", True):
                    if passive_liveness.get("status") == "LOW_QUALITY":
                        reasons.append("Selfie image too blurry (retake recommended)")
                    else:
                        label = passive_liveness.get("label")
                        suffix = f" ({label})" if label and label not in ("no_face",) else ""
                        reasons.append("Passive liveness failed" + suffix)
                if not active_ok:
                    reasons.append("Active liveness failed: " + str(active_liveness.get("message", "")))
                if face.get("face_count_id", 0) == 0 or face.get("face_count_selfie", 0) == 0:
                    reasons.append("No face detected")
                if multiple_faces:
                    reasons.append(f"Multiple faces in selfie ({selfie_face_count})")
                if qr_reject and qr_reason:
                    reasons.append(qr_reason)
                if tamper_reject:
                    reasons.append("Document tamper detected (suspected photo/content substitution)")
            rejection = "; ".join(reasons) or "Verification failed"

        return {
            "status": status,
            "retake_required": status == "RETRY",
            "retake_reason": retake_reason,
            "age_check": age_details,
            "age": age_details.get("age"),
            "is_underage": age_details.get("is_underage", False),
            "underage": age_details.get("is_underage", False),
            "age_rejected": age_reject,
            "ocr": {"front": ocr_front, "back": ocr_back},
            "qr": {"front": qr_front, "back": qr_back, "fields": qr_fields, "enabled": self.qr_enabled},
            "qr_consistency": qr_consistency,
            "tamper": tamper,
            "document_quality": doc_quality,
            "face": face,
            "face_match": face_match,
            "selfie_face_count": selfie_face_count,
            "multiple_faces": multiple_faces,
            "passive_liveness": passive_liveness,
            "active_liveness": active_liveness,
            "rejection_reason": rejection,
        }
