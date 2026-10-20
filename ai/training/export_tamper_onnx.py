"""Export TamperNet (.pth) -> models/tamper.onnx cho service dùng runtime ONNX.

Ví dụ::

    cd ai
    python training/export_tamper_onnx.py --weights models/tamper_net.pth --out models/tamper.onnx

Sau đó trỏ service sang file ONNX::

    set AI_TAMPER_MODEL=models/tamper.onnx

Output ONNX gồm 2 tensor: ``logit`` (B,) và ``mask_logits`` (B,1,H,W) - khớp
đúng thứ tự ``_predict()`` trong ``tamper_service.py`` đọc theo ``get_outputs()``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.tamper_net import INPUT_SIZE, TamperNet  # noqa: E402


class _Wrapper(torch.nn.Module):
    """ONNX chỉ nhận tensor vào/ra, nên gộp dict output thành tuple."""

    def __init__(self, model: TamperNet):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["logit"], out["mask_logits"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Export TamperNet sang ONNX")
    ap.add_argument("--weights", default="models/tamper_net.pth")
    ap.add_argument("--out", default="models/tamper.onnx")
    ap.add_argument("--size", type=int, default=INPUT_SIZE)
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print(f"Không thấy checkpoint: {args.weights}")
        return 1

    model = TamperNet(input_size=args.size)
    state = torch.load(args.weights, map_location="cpu")
    if next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dummy = torch.from_numpy(np.zeros((1, 3, args.size, args.size), dtype=np.float32))
    torch.onnx.export(
        _Wrapper(model).eval(),
        dummy,
        args.out,
        input_names=["image"],
        output_names=["logit", "mask_logits"],
        opset_version=18,
    )
    print(f"Đã export -> {args.out}")

    # Smoke-test: chạy ONNX và so với torch để bắt lệch output order.
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out = sess.run([o.name for o in sess.get_outputs()], {sess.get_inputs()[0].name: dummy.numpy()})
    print("ONNX outputs:", [o.name for o in sess.get_outputs()], [np.asarray(o).shape for o in out])
    with torch.no_grad():
        ref = _Wrapper(model)(dummy)
    print("max diff vs torch:", float(np.abs(np.asarray(out[0]) - ref[0].numpy()).max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
