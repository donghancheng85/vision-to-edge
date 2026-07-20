"""YOLOv11 architecture: building blocks and full model.

Architecture diagram (small variant, 640×640 input)
────────────────────────────────────────────────────
 Backbone
   Conv(3→32,s2)  Conv(32→64,s2)  C3k2(64→128)
   Conv(128→128,s2)  C3k2(128→256) ─────────────── P3  (s8,  256ch)
   Conv(256→256,s2)  C3k2(256→256,c3k) ─────────── P4  (s16, 256ch)
   Conv(256→512,s2)  C3k2(512→512,c3k)  SPPF
   C2PSA(512→512) ─────────────────────────────── P5  (s32, 512ch)

 FPN (top-down)
   up(P5) ┐                                         256ch  s16
   cat P4 ┘ → C3k2                               ─ h13
   up(h13) ┐                                        128ch  s8
   cat P3  ┘ → C3k2                             ─ out_s8   ← P3 detect

 PAN (bottom-up)
   down(out_s8) ┐                                   256ch  s16
   cat h13      ┘ → C3k2                        ─ out_s16  ← P4 detect
   down(out_s16) ┐                                  512ch  s32
   cat P5        ┘ → C3k2                       ─ out_s32  ← P5 detect

 Detect([out_s8, out_s16, out_s32])
   reg branch → 4·reg_max outputs per point (DFL bins)
   cls branch → nc outputs per point

Key innovations vs YOLOv8
  • C3k2 replaces C2f: same CSP structure but optionally uses 3×3+3×3
    bottlenecks (c3k=True) in deeper layers for larger receptive fields.
  • C2PSA added in the neck: injects global context via multi-head
    self-attention over half the channels without a large compute cost.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Scale configurations  (depth_mult, width_mult, max_channels)
# ─────────────────────────────────────────────────────────────────────────────
_SCALES: dict[str, tuple[float, float, int]] = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00,  512),
    "l": (1.00, 1.00,  512),
    "x": (1.00, 1.50,  512),
}


def _autopad(k: int) -> int:
    """Padding so stride-1 conv keeps H/W unchanged."""
    return k // 2


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class Conv(nn.Module):
    """Conv2d + BatchNorm2d + SiLU — the atomic unit in YOLOv11.

    SiLU(x) = x · σ(x) is empirically better than ReLU for detection
    because its smooth gradient helps with small object training.
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k), groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Residual bottleneck: two Conv3×3 with an optional skip connection.

    The skip connection only applies when c1 == c2 and shortcut=True.
    Without it the block still acts as a non-linear feature transform.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        ch = int(c2 * e)
        self.cv1 = Conv(c1, ch, 3)
        self.cv2 = Conv(ch, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k2(nn.Module):
    """Cross-Stage-Partial block with 2-path split (YOLOv11's main stage).

    Why CSP?  Splitting the feature map into two branches and only running
    the bottleneck stack on one of them halves the computation in the heavy
    part of the block, while the skip branch preserves gradient flow and
    feature reuse.

    When c3k=True the bottlenecks use a slightly larger kernel pair,
    giving deeper layers more spatial context without a big cost increase.

    Forward path:
        x → cv1 → [branch-A | branch-B → BN_1 → ... → BN_n]
        cat(A, B_outputs) → cv2 → output
    """

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False,
                 e: float = 0.5, shortcut: bool = True):
        super().__init__()
        self.c = int(c2 * e)                    # hidden channels in each branch
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # The bottleneck repeat count n is scaled by depth_mult before passing in
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))   # [branch-A, branch-B_init]
        y.extend(m(y[-1]) for m in self.m)       # accumulate bottleneck outputs
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast.

    Applies the same MaxPool kernel 3 times sequentially then concatenates
    all intermediate results.  This is equivalent to three parallel pools
    with kernel sizes k, 2k−1, 3k−2, but faster because no padding change.
    Captures multi-scale spatial context before the FPN neck.
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        ch = c1 // 2
        self.cv1 = Conv(c1, ch, 1)
        self.cv2 = Conv(ch * 4, c2, 1)
        self.m   = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x: Tensor) -> Tensor:
        y = self.cv1(x)
        y1 = self.m(y)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([y, y1, y2, y3], dim=1))


class PSABlock(nn.Module):
    """Pre-norm transformer block: self-attention + FFN with residuals.

    Used inside C2PSA to inject global context.  The spatial dimensions
    are flattened to a sequence so every position attends to every other.
    """

    def __init__(self, c: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        self.attn  = nn.MultiheadAttention(c, num_heads, dropout=0.0, batch_first=True)
        self.ff    = nn.Sequential(nn.Linear(c, c * 2), nn.GELU(), nn.Linear(c * 2, c))

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        s = x.flatten(2).transpose(1, 2)            # [B, H·W, C]
        n = self.norm1(s)
        s = s + self.attn(n, n, n, need_weights=False)[0]
        s = s + self.ff(self.norm2(s))
        return s.transpose(1, 2).view(B, C, H, W)


class C2PSA(nn.Module):
    """Cross-Stage-Partial with Positional Self-Attention (new in YOLOv11).

    Splits the channels in half; applies PSA blocks to one half only,
    then merges.  This gives global context at a fraction of the cost
    of full-channel attention.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c   = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.m   = nn.Sequential(*[PSABlock(self.c, max(1, self.c // 64)) for _ in range(n)])

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.cv1(x).chunk(2, dim=1)
        return self.cv2(torch.cat([a, self.m(b)], dim=1))


class DFL(nn.Module):
    """Distribution Focal Loss integral layer.

    Converts a predicted probability distribution over ``c1`` discrete bins
    into a continuous offset value by computing the expectation:

        offset = Σ_i  i · softmax(logits)_i

    The weight matrix [0, 1, ..., c1-1] is fixed (not trained).
    Only the upstream Conv that produces the distribution logits is trained.

    Input  shape: [B, 4·c1, N]   (4 coords, c1 bins each, N anchor points)
    Output shape: [B, 4,    N]   (one predicted offset per coord per point)
    """

    def __init__(self, c1: int = 16) -> None:
        super().__init__()
        self.c1   = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        b, _, a = x.shape                          # [B, 4·c1, N]
        return (
            self.conv(
                x.view(b, 4, self.c1, a)
                 .transpose(2, 1)                  # [B, c1, 4, N]
                 .softmax(1)                        # distribution over bins
            ).view(b, 4, a)
        )


class Detect(nn.Module):
    """Anchor-free, decoupled detection head.

    For each of ``nl`` feature scales, two separate branches predict:
        • Regression: 4·reg_max values (left/top/right/bottom DFL distributions)
        • Classification: nc logits

    'Decoupled' means cls and reg use different conv paths, avoiding the
    gradient conflict that harms single-branch heads.

    Training output : list of [B, 4·reg_max+nc, Hi, Wi] (one per scale)
    Inference output: [B, N, 4+nc]  decoded xyxy boxes + sigmoid scores
    """

    reg_max: int = 16

    def __init__(self, nc: int = 80, ch: tuple[int, ...] = ()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.no = nc + self.reg_max * 4

        c2 = max(self.reg_max * 4, ch[0] // 4, 16)   # reg branch channels
        c3 = max(ch[0], min(nc, 100))                  # cls branch channels

        self.cv2 = nn.ModuleList(                      # regression branches
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(                      # classification branches
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3),
                          nn.Conv2d(c3, nc, 1))
            for x in ch
        )
        self.dfl    = DFL(self.reg_max)
        self.stride = torch.tensor([8.0, 16.0, 32.0]) # set in YOLOv11.__init__

    def forward(self, x: list[Tensor]) -> list[Tensor] | Tensor:
        for i, xi in enumerate(x):
            x[i] = torch.cat([self.cv2[i](xi), self.cv3[i](xi)], dim=1)
        if self.training:
            return x                                   # raw feature maps
        return self._decode(x)

    def _decode(self, x: list[Tensor]) -> Tensor:
        """Decode raw outputs → [B, N, 4+nc] with xyxy pixel boxes."""
        anchors_all: list[Tensor] = []
        strides_all: list[Tensor] = []
        for xi, s in zip(x, self.stride):
            h, w = xi.shape[2:]
            ys = torch.arange(h, device=xi.device, dtype=torch.float32) + 0.5
            xs = torch.arange(w, device=xi.device, dtype=torch.float32) + 0.5
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            pts = torch.stack([gx, gy], dim=-1).reshape(-1, 2) * s
            anchors_all.append(pts)
            strides_all.append(pts.new_full((pts.shape[0], 1), s))

        anc     = torch.cat(anchors_all)              # [N, 2] pixel coords
        strides = torch.cat(strides_all)              # [N, 1]

        # Flatten and split
        flat = torch.cat([xi.flatten(2) for xi in x], dim=2).transpose(1, 2)  # [B,N,no]
        box_dist, cls_logits = flat.split([4 * self.reg_max, self.nc], dim=-1)

        # DFL integral → offsets in grid-cell units → pixel units
        dist4 = box_dist.view(*flat.shape[:2], 4, self.reg_max).softmax(-1)
        bins  = torch.arange(self.reg_max, device=flat.device, dtype=torch.float32)
        offsets_px = (dist4 * bins).sum(-1) * strides.unsqueeze(0)  # [B,N,4] pixels

        lt, rb = offsets_px.chunk(2, dim=-1)          # [B,N,2] each
        boxes  = torch.cat([anc - lt, anc + rb], dim=-1)   # xyxy pixels

        return torch.cat([boxes, cls_logits.sigmoid()], dim=-1)  # [B,N,4+nc]


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class YOLOv11(nn.Module):
    """YOLOv11 object detector.

    Args:
        model_size: ``n`` / ``s`` / ``m`` / ``l`` / ``x``
        nc:         Number of detection classes (80 for COCO).

    Quick start::

        model = YOLOv11("s", nc=80).cuda()
        model.train()
        # dummy forward — training returns list of raw feature maps
        out = model(torch.zeros(2, 3, 640, 640).cuda())
        # out[0].shape → [2, 4*16+80, 80, 80]  (P3, stride 8)
        # out[1].shape → [2, 4*16+80, 40, 40]  (P4, stride 16)
        # out[2].shape → [2, 4*16+80, 20, 20]  (P5, stride 32)
    """

    def __init__(self, model_size: str = "s", nc: int = 80) -> None:
        super().__init__()
        assert model_size in _SCALES, f"model_size ∈ {list(_SCALES)}"
        depth, width, max_ch = _SCALES[model_size]

        def c(base: int) -> int:
            return min(int(base * width), max_ch)

        def d(base: int) -> int:
            return max(round(base * depth), 1)

        # ── Backbone ─────────────────────────────────────────────────────────
        self.b0  = Conv(3,      c(64),   3, 2)           # ↓2
        self.b1  = Conv(c(64),  c(128),  3, 2)           # ↓4
        self.b2  = C3k2(c(128), c(256),  d(2))
        self.b3  = Conv(c(256), c(256),  3, 2)           # ↓8
        self.b4  = C3k2(c(256), c(512),  d(2))           # P3
        self.b5  = Conv(c(512), c(512),  3, 2)           # ↓16
        self.b6  = C3k2(c(512), c(512),  d(2), c3k=True) # P4
        self.b7  = Conv(c(512), c(1024), 3, 2)           # ↓32
        self.b8  = C3k2(c(1024), c(1024), d(2), c3k=True)
        self.b9  = SPPF(c(1024), c(1024))
        self.b10 = C2PSA(c(1024), c(1024), d(2))         # P5

        ch_p3 = c(512)                                    # P3 channels
        ch_p4 = c(512)                                    # P4 channels
        ch_p5 = c(1024)                                   # P5 channels

        # ── FPN: top-down path ────────────────────────────────────────────────
        self.up1  = nn.Upsample(scale_factor=2, mode="nearest")
        self.h13  = C3k2(ch_p5 + ch_p4, c(512),  d(2))  # after cat(up(P5), P4)

        self.up2  = nn.Upsample(scale_factor=2, mode="nearest")
        self.h16  = C3k2(c(512) + ch_p3, c(256), d(2))  # after cat(up(h13), P3)  ← s8

        # ── PAN: bottom-up path ───────────────────────────────────────────────
        self.h17  = Conv(c(256), c(256), 3, 2)
        self.h19  = C3k2(c(256) + c(512),  c(512),  d(2))               # ← s16

        self.h20  = Conv(c(512), c(512), 3, 2)
        self.h22  = C3k2(c(512) + ch_p5, c(1024), d(2), c3k=True)      # ← s32

        # ── Detection head ────────────────────────────────────────────────────
        det_ch = (c(256), c(512), c(1024))
        self.detect         = Detect(nc=nc, ch=det_ch)
        self.detect.stride  = torch.tensor([8.0, 16.0, 32.0])

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Stabilise detection head: bias cls logits so initial confidence ≈ 1%
        for cv3, s in zip(self.detect.cv3, self.detect.stride):
            b = cv3[-1].bias.data
            b += math.log(8 / (640 / float(s)) ** 2)
            b[: self.detect.nc] += math.log(0.6 / (self.detect.nc - 0.99))

    def forward(self, x: Tensor) -> list[Tensor] | Tensor:
        # ── Backbone ──
        x   = self.b0(x)
        x   = self.b1(x)
        x   = self.b2(x)
        x   = self.b3(x)
        p3  = self.b4(x)             # P3: stride 8
        x   = self.b5(p3)
        p4  = self.b6(x)             # P4: stride 16
        x   = self.b7(p4)
        x   = self.b8(x)
        x   = self.b9(x)
        p5  = self.b10(x)            # P5: stride 32

        # ── FPN ──
        x   = self.up1(p5)
        h13 = self.h13(torch.cat([x, p4], dim=1))

        x   = self.up2(h13)
        s8  = self.h16(torch.cat([x, p3], dim=1))   # stride-8 feature map

        # ── PAN ──
        x   = self.h17(s8)
        s16 = self.h19(torch.cat([x, h13], dim=1))  # stride-16 feature map

        x   = self.h20(s16)
        s32 = self.h22(torch.cat([x, p5], dim=1))   # stride-32 feature map

        return self.detect([s8, s16, s32])
