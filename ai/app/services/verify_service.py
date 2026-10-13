from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
from PIL import Image, ImageOps
from app.services.ocr_service import OCRService
from app.services.face_service import FaceService
from app.services.liveness_service import LivenessService
from app.services.active_liveness_service import ActiveLivenessService
from app.services.qr_service import QRService


class VerifyService:
    def __init__(self):
        self.ocr = OCRService()
        self.face = FaceService()
        self.liveness = LivenessService()
        self.active = ActiveLivenessService()
        self.qr = QRService()
        self.qr_enabled = self._env_flag("AI_QR_ENABLED", True)
        self.qr_require = self._env_flag("AI_QR_REQUIRE", False)
        self.qr_strict = self._env_flag("AI_QR_STRICT", True)
        self.qr_strict_name = self._env_flag("AI_QR_STRICT_NAME", False)

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

        status = "PROCESSING"
        if face_match.get("status") == "MATCH" and passive_liveness.get("is_live", True) and active_ok and not qr_reject and not multiple_faces:
            status = "VERIFIED"
        elif (
            face_match.get("status") != "MATCH"
            or not passive_liveness.get("is_live", True)
            or not active_ok
            or face.get("face_count_id", 0) == 0
            or face.get("face_count_selfie", 0) == 0
            or qr_reject
            or multiple_faces
        ):
            status = "REJECTED"

        rejection = None
        if status == "REJECTED":
            reasons = []
            if face_match.get("status") != "MATCH":
                reasons.append("Face mismatch" if face_match.get("status") == "NO_MATCH" else "Face match too weak")
            if not passive_liveness.get("is_live", True):
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
            rejection = "; ".join(reasons) or "Verification failed"
        elif status == "VERIFIED":
            rejection = None

        return {
            "status": status,
            "ocr": {"front": ocr_front, "back": ocr_back},
            "qr": {"front": qr_front, "back": qr_back, "fields": qr_fields, "enabled": self.qr_enabled},
            "qr_consistency": qr_consistency,
            "document_quality": doc_quality,
            "face": face,
            "face_match": face_match,
            "selfie_face_count": selfie_face_count,
            "multiple_faces": multiple_faces,
            "passive_liveness": passive_liveness,
            "active_liveness": active_liveness,
            "rejection_reason": rejection,
        }
