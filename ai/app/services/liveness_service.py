import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from app.services.fasnet import FASNetAntiSpoof


class LivenessService:
    """Passive (silent) liveness.

    Primary engine is the bundled MiniFASNetV2 anti-spoofing model, which looks
    at texture cues to tell a live face apart from a printed photo or a screen
    replay. When the model or torch is unavailable it falls back to a simple
    Laplacian-variance blur check so the service keeps functioning.

    A low-quality gate runs *before* the classifier: very blurry frames are
    returned as ``LOW_QUALITY`` (retake) instead of being classified as a
    spoof, because MiniFASNet confuses blur with print/replay texture. Disable
    the gate with ``AI_FAS_BLUR_GATE=0``.
    """

    # Fallback threshold when the model is unavailable, and the default gate
    # applied before MiniFASNet (override with AI_FAS_BLUR_GATE).
    BLUR_THRESHOLD = 25.0

    def __init__(self):
        self.fas = FASNetAntiSpoof()
        self._face_service = None
        try:
            self.blur_gate = float(
                os.environ.get("AI_FAS_BLUR_GATE", str(self.BLUR_THRESHOLD))
            )
        except ValueError:
            self.blur_gate = self.BLUR_THRESHOLD

    def _get_face_service(self):
        if self._face_service is None:
            from app.services.face_service import FaceService

            self._face_service = FaceService()
        return self._face_service

    @staticmethod
    def _blur_var(img_rgb: np.ndarray) -> float:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _face_crop(
        img_rgb: np.ndarray, face_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in face_bbox)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return img_rgb[y1:y2, x1:x2]

    def _low_quality_result(self, blur: float) -> Dict[str, Any]:
        return {
            "is_live": False,
            "blur_var": blur,
            "blur_gate": self.blur_gate,
            "status": "LOW_QUALITY",
            "label": "low_quality",
            "retake_required": True,
            "reason": "Selfie image too blurry; please retake in better focus/lighting",
            "engine": "quality-gate",
        }

    def passive_liveness(
        self,
        img_rgb: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        try:
            blur = self._blur_var(img_rgb)
        except Exception:
            blur = 0.0

        # Preserve the old fallback path when the model is unavailable; face
        # detection is only needed by the model path and the quality gate.
        if not self.fas.available():
            is_live = blur > self.BLUR_THRESHOLD
            return {
                "is_live": bool(is_live),
                "blur_var": blur,
                "blur_gate": self.blur_gate,
                "retake_required": False,
                "status": "PASS" if is_live else "SUSPICIOUS",
                "engine": "blur",
            }

        if face_bbox is None:
            try:
                boxes = self._get_face_service().detect_face_boxes(img_rgb)
            except Exception:
                boxes = []
            if not boxes:
                # A detector failure on a very blurry frame is itself a retake
                # signal; otherwise keep the existing NO_FACE behaviour.
                if self.blur_gate > 0 and blur < self.blur_gate:
                    return self._low_quality_result(blur)
                return {
                    "is_live": True,
                    "blur_var": blur,
                    "blur_gate": self.blur_gate,
                    "retake_required": False,
                    "status": "NO_FACE",
                    "label": "no_face",
                    "engine": "MiniFASNetV2",
                }
            face_bbox = max(boxes, key=lambda b: b[2] * b[3])

        # Measure sharpness on the face, not the background: a smooth wall can
        # otherwise make a sharp selfie look blurry.
        try:
            face_blur = self._blur_var(self._face_crop(img_rgb, face_bbox))
        except Exception:
            face_blur = blur

        if self.blur_gate > 0 and face_blur < self.blur_gate:
            return self._low_quality_result(face_blur)

        try:
            result = self.fas.predict(img_rgb, face_bbox)
        except Exception as exc:
            is_live = face_blur > self.BLUR_THRESHOLD
            return {
                "is_live": bool(is_live),
                "blur_var": face_blur,
                "blur_gate": self.blur_gate,
                "retake_required": False,
                "status": "PASS" if is_live else "SUSPICIOUS",
                "engine": "blur",
                "error": str(exc),
            }

        result["blur_var"] = face_blur
        result["blur_gate"] = self.blur_gate
        result["retake_required"] = False
        result["status"] = "PASS" if result["is_live"] else "SUSPICIOUS"
        return result
