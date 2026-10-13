"""MiniFASNetV2 silent face anti-spoofing (passive liveness).

Architecture and preprocessing follow the reference project
`minivision-ai/Silent-Face-Anti-Spoofing` so the shipped checkpoint
(`models/2.7_80x80_MiniFASNetV2.pth`) loads unchanged:

* input: 80x80 face crop, expanded around the detected face box by a scale
  factor (2.7 for this checkpoint). The reference pipeline feeds values in
  [0, 255] (its ``to_tensor`` deliberately skips the usual ``/255`` step) in
  BGR channel order, so we match that exactly;
* output: 3 logits; index 1 is the real/live class (0 and 2 are spoof).

The model classifies the *presentation* (print / screen replay) from texture in
the RGB image. It cannot detect a 3D mask or a mannequin reliably.
"""
import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is an optional heavy dependency
    TORCH_AVAILABLE = False


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("false", "0", "no", "off")


# --------------------------------------------------------------------------- #
# MiniFASNet architecture (trimmed copy of the reference implementation)
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:
    from torch.nn import (
        AdaptiveAvgPool2d,
        BatchNorm1d,
        BatchNorm2d,
        Conv2d,
        Flatten,
        Linear,
        Module,
        PReLU,
        ReLU,
        Sequential,
    )

    class Conv_block(Module):
        def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
            super().__init__()
            self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups,
                               stride=stride, padding=padding, bias=False)
            self.bn = BatchNorm2d(out_c)
            self.prelu = PReLU(out_c)

        def forward(self, x):
            return self.prelu(self.bn(self.conv(x)))

    class Linear_block(Module):
        def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
            super().__init__()
            self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel,
                               groups=groups, stride=stride, padding=padding, bias=False)
            self.bn = BatchNorm2d(out_c)

        def forward(self, x):
            return self.bn(self.conv(x))

    class Depth_Wise(Module):
        def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
            super().__init__()
            c1_in, c1_out = c1
            c2_in, c2_out = c2
            c3_in, c3_out = c3
            self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
            self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
            self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
            self.residual = residual

        def forward(self, x):
            if self.residual:
                short_cut = x
            x = self.project(self.conv_dw(self.conv(x)))
            return short_cut + x if self.residual else x

    class Residual(Module):
        def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
            super().__init__()
            modules = []
            for i in range(num_block):
                modules.append(Depth_Wise(c1[i], c2[i], c3[i], residual=True,
                                          kernel=kernel, padding=padding, stride=stride, groups=groups))
            self.model = Sequential(*modules)

        def forward(self, x):
            return self.model(x)

    class MiniFASNet(Module):
        def __init__(self, keep, embedding_size, conv6_kernel=(7, 7), drop_p=0.0,
                     num_classes=3, img_channel=3):
            super().__init__()
            self.embedding_size = embedding_size
            self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
            self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])

            c1 = [(keep[1], keep[2])]
            c2 = [(keep[2], keep[3])]
            c3 = [(keep[3], keep[4])]
            self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])

            c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
            c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
            c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
            self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

            c1 = [(keep[16], keep[17])]
            c2 = [(keep[17], keep[18])]
            c3 = [(keep[18], keep[19])]
            self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])

            c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]),
                  (keep[31], keep[32]), (keep[34], keep[35])]
            c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]),
                  (keep[32], keep[33]), (keep[35], keep[36])]
            c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]),
                  (keep[33], keep[34]), (keep[36], keep[37])]
            self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

            c1 = [(keep[37], keep[38])]
            c2 = [(keep[38], keep[39])]
            c3 = [(keep[39], keep[40])]
            self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])

            c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
            c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
            c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
            self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
            self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
            self.conv_6_flatten = Flatten()
            self.linear = Linear(512, embedding_size, bias=False)
            self.bn = BatchNorm1d(embedding_size)
            self.drop = torch.nn.Dropout(p=drop_p)
            self.prob = Linear(embedding_size, num_classes, bias=False)

        def forward(self, x):
            out = self.conv1(x)
            out = self.conv2_dw(out)
            out = self.conv_23(out)
            out = self.conv_3(out)
            out = self.conv_34(out)
            out = self.conv_4(out)
            out = self.conv_45(out)
            out = self.conv_5(out)
            out = self.conv_6_sep(out)
            out = self.conv_6_dw(out)
            out = self.conv_6_flatten(out)
            if self.embedding_size != 512:
                out = self.linear(out)
            out = self.bn(out)
            out = self.drop(out)
            return self.prob(out)

    _KEEP_1_8M_ = [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
                   13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
                   52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
                   128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512]

    def MiniFASNetV2(embedding_size=128, conv6_kernel=(7, 7), drop_p=0.2, num_classes=3, img_channel=3):
        return MiniFASNet(_KEEP_1_8M_, embedding_size, conv6_kernel, drop_p, num_classes, img_channel)


# --------------------------------------------------------------------------- #
# Cropping (reference CropImage._get_new_box + crop)
# --------------------------------------------------------------------------- #
def _get_new_box(src_w: int, src_h: int, bbox, scale: float) -> Tuple[int, int, int, int]:
    x, y, box_w, box_h = bbox[0], bbox[1], bbox[2], bbox[3]
    scale = min((src_h - 1) / max(box_h, 1), min((src_w - 1) / max(box_w, 1), scale))

    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y

    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2

    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1

    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def crop_face(img: np.ndarray, bbox, scale: float, out_w: int = 80, out_h: int = 80) -> np.ndarray:
    src_h, src_w = img.shape[:2]
    left, top, right, bottom = _get_new_box(src_w, src_h, bbox, scale)
    crop = img[top:bottom + 1, left:right + 1]
    if crop.size == 0:
        crop = img
    return cv2.resize(crop, (out_w, out_h))


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class FASNetAntiSpoof:
    """Passive (silent) anti-spoofing using the bundled MiniFASNetV2 checkpoint."""

    REAL_INDEX = 1  # 0 / 2 are spoof classes

    def __init__(self):
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_model = os.path.join(here, "models", "2.7_80x80_MiniFASNetV2.pth")
        self.model_path = os.environ.get("AI_FAS_MODEL") or default_model
        self.scale = float(os.environ.get("AI_FAS_SCALE", "2.7"))
        self.real_threshold = float(os.environ.get("AI_FAS_REAL_THRESHOLD", "0.5"))
        self.enabled = _env_flag("AI_FAS_ENABLED", True)
        self.tta = _env_flag("AI_FAS_TTA", True)
        self._model = None
        self._device = None

    def available(self) -> bool:
        return self.enabled and TORCH_AVAILABLE and os.path.exists(self.model_path)

    def _load(self):
        if self._model is not None:
            return
        requested = (os.environ.get("AI_FAS_DEVICE") or "auto").lower()
        if requested == "cpu":
            device = "cpu"
        elif requested == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        kernel = ((80 + 15) // 16, (80 + 15) // 16)  # get_kernel(80, 80) -> (5, 5)
        model = MiniFASNetV2(conv6_kernel=kernel)

        state_dict = torch.load(self.model_path, map_location="cpu")
        if next(iter(state_dict)).startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval().to(self._device)
        self._model = model

    def _softmax(self, tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self._model(tensor)
            return torch.softmax(logits, dim=1)[0].cpu().numpy().astype(float)

    def predict(self, img_rgb: np.ndarray, bbox) -> Dict[str, Any]:
        self._load()
        crop = crop_face(img_rgb, bbox, self.scale, 80, 80)
        # The checkpoint was trained with cv2 (BGR) images in [0, 255]; the
        # reference `to_tensor` keeps that range (its /255 step is disabled).
        bgr = crop[:, :, ::-1] if crop.ndim == 3 and crop.shape[2] == 3 else crop
        x = np.ascontiguousarray(bgr.astype(np.float32).transpose(2, 0, 1)[None, ...])
        tensor = torch.from_numpy(x).to(self._device)

        probs = self._softmax(tensor)
        if self.tta:
            # Horizontal-flip test-time augmentation stabilises the verdict.
            probs = (probs + self._softmax(torch.flip(tensor, dims=[3]))) / 2.0

        index = int(np.argmax(probs))
        real_prob = float(probs[self.REAL_INDEX])
        is_real = index == self.REAL_INDEX and real_prob >= self.real_threshold
        label = "real" if index == self.REAL_INDEX else ("print" if index == 0 else "replay")

        return {
            "is_live": bool(is_real),
            "label": label,
            "real_prob": round(real_prob, 4),
            "spoof_prob": round(1.0 - real_prob, 4),
            "scores": [round(float(p), 4) for p in probs],
            "threshold": self.real_threshold,
            "engine": "MiniFASNetV2",
        }
