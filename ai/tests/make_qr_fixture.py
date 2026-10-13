"""Regenerate tests/fixtures/cccd_back_qr.png (a fake CCCD back with a QR).

Requires an extra dev dependency:
    pip install qrcode[pil]

Run from the ai/ directory:
    python tests/make_qr_fixture.py
    python tests/make_qr_fixture.py "<your payload>" tests/fixtures/custom.png
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_PAYLOAD = (
    "068096013455|251043755|Trần Chí Hào|08101996|Nam|"
    "Tổ 2, Lộc Thắng, Bảo Lâm, Lâm Đồng|30082021"
)


def build(payload: str = DEFAULT_PAYLOAD, out_path: str = None) -> str:
    import qrcode

    out_path = out_path or os.path.join(HERE, "fixtures", "cccd_back_qr.png")

    base_path = os.path.normpath(os.path.join(ROOT, "..", "back.png"))
    base = cv2.imread(base_path)
    if base is None:
        base = np.full((400, 640, 3), 235, np.uint8)

    qr = np.array(qrcode.make(payload).convert("L"))

    side = 160
    qr = cv2.resize(qr, (side, side), interpolation=cv2.INTER_AREA)

    pad = 14
    canvas = np.full((side + 2 * pad, side + 2 * pad), 255, np.uint8)
    canvas[pad : pad + side, pad : pad + side] = qr

    h, w = canvas.shape
    x0, y0 = 20, 20
    base[y0 : y0 + h, x0 : x0 + w] = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, base)
    return out_path


if __name__ == "__main__":
    payload = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PAYLOAD
    out = sys.argv[2] if len(sys.argv) > 2 else None
    print("Wrote", build(payload, out))
