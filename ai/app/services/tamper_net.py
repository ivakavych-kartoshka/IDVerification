"""TamperNet: lightweight multi-stream document tamper detector (skeleton).

Model cho hướng nghiên cứu A - phát hiện giả mạo CCCD/khác (thay thế ảnh chân
dung + inpainting vùng chữ), lấy cảm hứng từ:

* TruFor (CVPR'23)      - tín hiệu forensic (noise/high-pass) + transformer
                          + 2 head: phân loại ảnh & localization mask.
* EdgeDoc (ICCVW'25,    - CNN-transformer nhẹ + NoisePrint, hạng 3 DeepID
  DeepID Challenge)       Challenge @ ICCV 2025 (định benchmark FantasyID).
* JEI'25 multistream    - tách 3 luồng feature: texture / tần số / noise.

Kiến trúc (input SxS, mặc định S=256, chia hết cho 8)::

    RGB stream      3 --conv/s2--> [DSConv x3] ---------> 96ch @ S/8
    Forensic stream 3 --SRM(4 hp-filter)--> [DSConv x2] -> 64ch @ S/8
    Frequency stream3 --block-DCT AC ratio--> conv -------> 16ch @ S/8
                                     concat + 1x1 fuse --> 128ch @ S/8
                                             /                    \
                     cls: pool 4x4 -> 16 tokens -> 2 x Transformer
                          Encoder -> mean -> MLP -> logit (bona fide?)
                     mask: 3x upsample (x8) -> 1x1 conv -> pixel logit

Model cố ý rất nhỏ (~1-2M tham số) vì:
  * bộ train công khai FantasyID chỉ có 786 bona fide + 1572 forged;
  * inference chạy CPU cạnh OCR/face trong cùng service FastAPI.

Train: ``python training/train_tamper.py --data <fantasyid_root>`` -> ra
``models/tamper_net.pth`` (hoặc export ONNX ``models/tamper.onnx``), service
`tamper_service.py` tự load file đó; không có model -> fallback heuristic.

Chuẩn hoá input: float32 RGB trong **[0, 1]** (KHÔNG standardize 0-mean vì
tỉ lệ AC/DC của DCT và tín hiệu SRM đều giả định nền giá trị thực).
"""
from typing import Dict

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Phải chia hết cho 8 (block DCT 8x8) - service letterbox về size này.
INPUT_SIZE = 256
BLOCK = 8


# --------------------------------------------------------------------------- #
# Fixed (non-learned) forensic front-ends
# --------------------------------------------------------------------------- #
def _srm_bank() -> torch.Tensor:
    """4 bộ lọc SRM 5x5 kinh điển (steganalysis/forensics), dạng (4,5,5)."""
    k_lap = torch.zeros(5, 5)
    k_lap[1:4, 1:4] = torch.tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
    )

    k_hx = torch.tensor(
        [
            [-1.0, 2.0, -2.0, 2.0, -1.0],
            [2.0, -4.0, 4.0, -4.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, 4.0, -4.0, 4.0, -2.0],
            [1.0, -2.0, 2.0, -2.0, 1.0],
        ]
    )
    k_hy = k_hx.t().contiguous()

    k_lap3 = torch.zeros(5, 5)
    k_lap3[1:4, 1:4] = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])

    return torch.stack([k_lap, k_hx, k_hy, k_lap3], dim=0)


class SRMFilterBank(nn.Module):
    """RGB -> gray -> 4 bộ lọc high-pass cố định (requires_grad=False)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("gray_w", torch.tensor([0.299, 0.587, 0.114]).view(1, 1, 1, 3))
        self.register_buffer("srm_w", _srm_bank().view(4, 1, 5, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,S,S) -> (B,4,S,S)
        gray = (x * self.gray_w.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        return F.conv2d(gray, self.srm_w, padding=2)


class BlockDCTHF(nn.Module):
    """Tỉ lệ năng lượng AC/DC trung bình của mỗi block 8x8 (DCT-II khác biệt).

    Ảnh inpainting/smoothing bất thường sẽ có map HF phẳng (nhiều block AC~0);
    ảnh in lại/hiện lưới JPEG thìHF map có cấu trúc block rõ rệt.
    """

    def __init__(self, block: int = BLOCK):
        super().__init__()
        n = block
        idx = torch.arange(n, dtype=torch.float32)
        alpha = torch.ones(n)
        alpha[0] = 1.0 / math.sqrt(2.0)
        basis = alpha.unsqueeze(1) * torch.cos(math.pi * (idx.unsqueeze(0) + 0.5) * idx.unsqueeze(1) / n)
        self.register_buffer("basis", basis * math.sqrt(2.0 / n))  # (n, n) orte-đỉnh

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,S,S) -> (B,1,S/8,S/8)
        gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        blocks = gray.unfold(2, BLOCK, BLOCK).unfold(3, BLOCK, BLOCK)  # (B,1,N,N,8,8)
        coef = torch.matmul(torch.matmul(self.basis, blocks), self.basis.t())
        total = (coef ** 2).sum(dim=(-1, -2))
        dc = coef[..., 0, 0] ** 2
        return (total - dc) / (total + 1e-6)


# --------------------------------------------------------------------------- #
# Backbone pieces
# --------------------------------------------------------------------------- #
def _gn(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class DSConv(nn.Module):
    """Depthwise-separable conv block (dw 3x3 -> pw 1x1, GN + GELU)."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
        self.gn1 = nn.GroupNorm(_gn(in_c), in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.gn2 = nn.GroupNorm(_gn(out_c), out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gn1(self.dw(x)))
        return self.act(self.gn2(self.pw(x)))


class TamperNet(nn.Module):
    """Cls (bona fide vs forged) + mask (vùng giả mạo) trên ảnh thẻ."""

    def __init__(self, input_size: int = INPUT_SIZE, d_model: int = 128, depth: int = 2, num_heads: int = 4):
        super().__init__()
        assert input_size % BLOCK == 0, "input_size phải chia hết cho 8"
        self.input_size = input_size
        self.srm = SRMFilterBank()
        self.dct = BlockDCTHF()

        # Luồng 1: RGB (appearance/layout)
        self.rgb_in = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.GroupNorm(_gn(32), 32), nn.GELU()
        )
        self.rgb_stages = nn.Sequential(DSConv(32, 64, 2), DSConv(64, 64, 1), DSConv(64, 96, 2))

        # Luồng 2: forensic (SRM high-pass)
        self.fr_in = nn.Sequential(
            nn.Conv2d(4, 24, 3, 2, 1, bias=False), nn.GroupNorm(_gn(24), 24), nn.GELU()
        )
        self.fr_stages = nn.Sequential(DSConv(24, 48, 2), DSConv(48, 64, 2))

        # Luồng 3: tần số (block DCT AC ratio @ S/8, chỉ cần conv nhẹ)
        self.frq_in = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False), nn.GroupNorm(_gn(16), 16), nn.GELU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(96 + 64 + 16, d_model, 1, bias=False),
            nn.GroupNorm(_gn(d_model), d_model),
            nn.GELU(),
        )

        # Head cls: pool về 4x4 = 16 token (attention rẻ) thay vì 32x32 = 1024 token.
        enc_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model * 2, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, depth, enable_nested_tensor=False)
        self.pos = nn.Parameter(torch.zeros(1, 16, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))

        # Head mask: /8 -> /4 -> /2 -> /1
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(d_model, 64, 4, 2, 1), nn.GroupNorm(_gn(64), 64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(_gn(32), 32), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.GroupNorm(_gn(16), 16), nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = x.shape[-2:]
        ph, pw = (BLOCK - h % BLOCK) % BLOCK, (BLOCK - w % BLOCK) % BLOCK
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")

        f_rgb = self.rgb_stages(self.rgb_in(x))            # (B, 96, S/8, S/8)
        f_fr = self.fr_stages(self.fr_in(self.srm(x)))     # (B, 64, S/8, S/8)
        f_frq = self.frq_in(self.dct(x))                   # (B, 16, S/8, S/8)
        feats = self.fuse(torch.cat([f_rgb, f_fr, f_frq], dim=1))  # (B, 128, S/8, S/8)

        tokens = F.adaptive_avg_pool2d(feats, 4).flatten(2).transpose(1, 2)  # (B,16,C)
        enc = self.encoder(tokens + self.pos)
        logit = self.cls_head(enc.mean(dim=1)).squeeze(-1)  # (B,)

        mask = self.dec(feats)[..., :h, :w]                 # (B,1,H,W)
        return {"logit": logit, "mask_logits": mask}


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
class TamperLoss(nn.Module):
    """BCE ảnh-level + BCE pixel-level (mask yếu nếu không có GT mask).

    ``masks=None`` -> chỉ loss cls (weak image-level supervision, đủ để
    fine-tune bước đầu trên FantasyID); có GT mask -> thêm mask_weight*mask.
    """

    def __init__(self, mask_weight: float = 0.5, pos_weight: float = 3.0):
        super().__init__()
        self.mask_weight = mask_weight
        # pos_weight được đăng ký là buffer -> tự .to(device) cùng model/loss,
        # tránh lỗi "found at least two devices" khi train trên GPU.
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.mask_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels: torch.Tensor, masks=None):
        cls = self.cls_loss(outputs["logit"], labels.float().view(-1))
        total = cls
        parts = {"loss_cls": float(cls.detach())}
        if masks is not None and "mask_logits" in outputs:
            gt = F.interpolate(masks.float().unsqueeze(1) if masks.dim() == 3 else masks.float(),
                               size=outputs["mask_logits"].shape[-2:], mode="nearest")
            m = self.mask_loss(outputs["mask_logits"], gt)
            total = total + self.mask_weight * m
            parts["loss_mask"] = float(m.detach())
        parts["loss_total"] = float(total.detach())
        return total, parts


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":  # smoke test: python -m app.services.tamper_net
    m = TamperNet(input_size=64)
    out = m(torch.rand(2, 3, 64, 64))
    print(f"params={count_params(m):,}")
    print("logit:", tuple(out["logit"].shape), "mask:", tuple(out["mask_logits"].shape))
    loss, parts = TamperLoss()(out, torch.tensor([0, 1]))
    print("loss ok:", parts, float(loss.detach()))
