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
    """

    BLUR_THRESHOLD = 25.0

    def __init__(self):
        self.fas = FASNetAntiSpoof()
        self._face_service = None

    def _get_face_service(self):
        if self._face_service is None:
            from app.services.face_service import FaceService

            self._face_service = FaceService()
        return self._face_service

    @staticmethod
    def _blur_var(img_rgb: np.ndarray) -> float:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def passive_liveness(
        self,
        img_rgb: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        try:
            blur = self._blur_var(img_rgb)
        except Exception:
            blur = 0.0

        if not self.fas.available():
            is_live = blur > self.BLUR_THRESHOLD
            return {
                "is_live": bool(is_live),
                "blur_var": blur,
                "status": "PASS" if is_live else "SUSPICIOUS",
                "engine": "blur",
            }

        if face_bbox is None:
            try:
                boxes = self._get_face_service().detect_face_boxes(img_rgb)
            except Exception:
                boxes = []
            if not boxes:
                # Face detection is handled separately; do not double-reject here.
                return {
                    "is_live": True,
                    "blur_var": blur,
                    "status": "NO_FACE",
                    "label": "no_face",
                    "engine": "MiniFASNetV2",
                }
            face_bbox = max(boxes, key=lambda b: b[2] * b[3])

        try:
            result = self.fas.predict(img_rgb, face_bbox)
        except Exception as exc:
            is_live = blur > self.BLUR_THRESHOLD
            return {
                "is_live": bool(is_live),
                "blur_var": blur,
                "status": "PASS" if is_live else "SUSPICIOUS",
                "engine": "blur",
                "error": str(exc),
            }

        result["blur_var"] = blur
        result["status"] = "PASS" if result["is_live"] else "SUSPICIOUS"
        return result
