from typing import Any, Dict, List, Sequence
import io
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

# Head-pose thresholds (degrees).
STRAIGHT_MAX = 12.0  # |yaw| <= this => looking straight
TURN_MIN = 18.0      # |yaw| >= this => turned to one side
MIN_FACE_RATIO = 0.8


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("false", "0", "no", "off")


class ActiveLivenessService:
    """Active (challenge-response) liveness using head pose.

    The user is asked to look straight, then turn the head to one side and the
    other. We estimate the yaw angle for every captured frame and verify that
    the required poses actually happened (a static photo / replayed image cannot
    produce two opposite head rotations).
    """

    def __init__(self):
        self._landmarker = None
        self._mp = None
        self.static_check = _env_flag("AI_ACTIVE_STATIC_CHECK", True)
        self.require_blink = _env_flag("AI_ACTIVE_REQUIRE_BLINK", False)
        self.static_diff = float(os.environ.get("AI_ACTIVE_STATIC_DIFF", "1.5"))

    # ------------------------------------------------------------------ setup
    def _get_landmarker(self):
        if self._landmarker is not None:
            return self._landmarker
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models",
            "face_landmarker.task",
        )
        if not os.path.exists(model_path):
            raise RuntimeError(f"Face landmarker model not found: {model_path}")

        options = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        return self._landmarker

    @staticmethod
    def _to_rgb(image) -> np.ndarray:
        if isinstance(image, (bytes, bytearray, memoryview)):
            img = Image.open(io.BytesIO(bytes(image)))
            img = ImageOps.exif_transpose(img).convert("RGB")
            return np.array(img)
        return np.ascontiguousarray(image)

    # ------------------------------------------------------------------- pose
    def pose(self, image) -> Dict[str, Any]:
        """Return head pose for a single image."""
        arr = self._to_rgb(image)
        landmarker = self._get_landmarker()
        mp = self._mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            return {"face_found": False, "yaw": None, "pitch": None, "roll": None, "blink": None}

        blink = None
        if result.face_blendshapes:
            bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
            left, right = bs.get("eyeBlinkLeft"), bs.get("eyeBlinkRight")
            if left is not None and right is not None:
                blink = round(float(min(left, right)), 3)

        yaw = pitch = roll = None
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
            r = matrix[:3, :3]
            sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
            # yaw = rotation about the vertical axis (turn left/right). This is
            # the signal the challenge relies on. pitch/roll are informational.
            yaw = math.degrees(math.atan2(-r[2, 0], sy))
            pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
            roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))

        if yaw is None:
            lms = result.face_landmarks[0]
            h, w = arr.shape[:2]

            def pt(i: int):
                return lms[i].x * w, lms[i].y * h

            nose = pt(1)
            left_eye = pt(33)
            right_eye = pt(263)
            eye_dist = max(abs(right_eye[0] - left_eye[0]), 1e-6)
            yaw = math.degrees(
                math.atan2(nose[0] - (left_eye[0] + right_eye[0]) / 2.0, eye_dist)
            )

        return {
            "face_found": True,
            "yaw": round(float(yaw), 2),
            "pitch": round(float(pitch), 2) if pitch is not None else None,
            "roll": round(float(roll), 2) if roll is not None else None,
            "blink": blink,
        }

    @staticmethod
    def _is_frozen(arrs: Sequence[np.ndarray], diff_thresh: float) -> bool:
        """Detect a photo/screen replay as a set of near-identical frames."""
        if len(arrs) < 2:
            return False
        thumbs = []
        for a in arrs:
            try:
                g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
                thumbs.append(cv2.resize(g, (64, 64)).astype(np.float32))
            except Exception:
                return False
        diffs = [float(np.mean(np.abs(thumbs[i] - thumbs[i - 1]))) for i in range(1, len(thumbs))]
        return max(diffs) < diff_thresh

    # ---------------------------------------------------------------- verdict
    def analyze(self, frames: Sequence[Any]) -> Dict[str, Any]:
        """Verify a challenge sequence made of look-straight / turn-left / turn-right."""
        total = len(frames)
        if total == 0:
            return {
                "status": "FAIL",
                "message": "No liveness frames provided",
                "face_ratio": 0.0,
                "frames": 0,
                "yaw_sequence": [],
            }

        arrs: List[Any] = []
        for frame in frames:
            try:
                arrs.append(self._to_rgb(frame))
            except Exception:
                arrs.append(None)

        yaws: List[float] = []
        blinks: List[float] = []
        faces = 0
        for arr in arrs:
            if arr is None:
                continue
            r = self.pose(arr)
            if r.get("face_found"):
                faces += 1
                yaws.append(float(r["yaw"]))
                if r.get("blink") is not None:
                    blinks.append(float(r["blink"]))

        ratio = faces / total
        straight_ok = any(abs(y) <= STRAIGHT_MAX for y in yaws)
        left_ok = any(y <= -TURN_MIN for y in yaws)
        right_ok = any(y >= TURN_MIN for y in yaws)
        yaw_range = (max(yaws) - min(yaws)) if yaws else 0.0
        movement_ok = yaw_range >= TURN_MIN

        valid = [a for a in arrs if a is not None]
        frozen = self.static_check and self._is_frozen(valid, self.static_diff)
        blink_detected = any(b >= 0.5 for b in blinks)
        eyes_open_seen = any(b <= 0.2 for b in blinks)

        reasons: List[str] = []
        if ratio < MIN_FACE_RATIO:
            reasons.append("Face not detected in enough frames")
        if not straight_ok:
            reasons.append("No straight (frontal) pose detected")
        if not (left_ok and right_ok):
            reasons.append("Did not turn head to both sides")
        if not movement_ok:
            reasons.append("Insufficient head movement")
        if frozen:
            reasons.append("Frames appear static (possible photo/screen replay)")
        if self.require_blink and not blink_detected:
            reasons.append("No blink detected")

        passed = (
            ratio >= MIN_FACE_RATIO
            and straight_ok
            and left_ok
            and right_ok
            and movement_ok
            and not frozen
            and (blink_detected or not self.require_blink)
        )

        return {
            "status": "PASS" if passed else "FAIL",
            "face_ratio": round(ratio, 3),
            "frames": total,
            "faces_detected": faces,
            "straight_ok": straight_ok,
            "left_ok": left_ok,
            "right_ok": right_ok,
            "yaw_range": round(float(yaw_range), 2),
            "yaw_sequence": [round(y, 2) for y in yaws],
            "frozen": frozen,
            "blink_detected": blink_detected,
            "eyes_open_seen": eyes_open_seen,
            "thresholds": {"straight_max": STRAIGHT_MAX, "turn_min": TURN_MIN},
            "message": "OK" if passed else "; ".join(reasons),
        }
