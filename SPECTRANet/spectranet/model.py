"""RGB-guided depth upsampling network.

Architecture:
  - RGB encoder: pretrained MobileNetV2 with feature taps at strides 2/4/8/16
  - Depth encoder: lightweight CNN over the bicubic-upsampled lowres depth
    (+ confidence channel) at strides 1/2/4/8/16
  - Decoder: progressive bilinear upsampling that fuses RGB and depth features
    at every scale (concat + 3x3 conv blocks)
  - Output: scalar residual added to the bicubic baseline (in normalized space)

The choice of MobileNetV2 (over ResNet) follows the spec — it converts cleanly
to CoreML for iPhone deployment while still providing strong edge guidance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def _conv_bn_relu(in_ch: int, out_ch: int, k: int = 3) -> nn.Sequential:
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class RGBEncoder(nn.Module):
    """MobileNetV2 wrapper that exposes 4 multi-scale feature maps."""

    # taps after these MobileNetV2 stages (output strides):
    #   features[1]  ->  16 ch, stride 2
    #   features[3]  ->  24 ch, stride 4
    #   features[6]  ->  32 ch, stride 8
    #   features[13] ->  96 ch, stride 16
    TAP_INDICES = (1, 3, 6, 13)
    TAP_CHANNELS = (16, 24, 32, 96)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        mnv2 = mobilenet_v2(weights=weights)
        # we only need features[0..13]
        self.stem = mnv2.features[0]                                       # stride 2, 32 ch (intermediate)
        self.stage1 = mnv2.features[1]                                     # stride 2, 16 ch  (tap 0)
        self.stage2 = nn.Sequential(mnv2.features[2], mnv2.features[3])    # stride 4, 24 ch  (tap 1)
        self.stage3 = nn.Sequential(*[mnv2.features[i] for i in range(4, 7)])    # stride 8, 32 ch (tap 2)
        self.stage4 = nn.Sequential(*[mnv2.features[i] for i in range(7, 14)])   # stride 16, 96 ch (tap 3)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)   # stride 2,  16
        f2 = self.stage2(f1)  # stride 4,  24
        f3 = self.stage3(f2)  # stride 8,  32
        f4 = self.stage4(f3)  # stride 16, 96
        return [f1, f2, f3, f4]

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)
        self.train()


class DepthEncoder(nn.Module):
    """Lightweight CNN over depth + confidence channels at the target res.

    Produces features at strides 1, 2, 4, 8, 16 to mirror the RGB encoder.
    """

    def __init__(self, in_ch: int = 2, widths: Tuple[int, int, int, int, int] = (16, 16, 24, 32, 64)):
        super().__init__()
        self.stem = _conv_bn_relu(in_ch, widths[0], k=3)        # stride 1
        self.down1 = nn.Sequential(_conv_bn_relu(widths[0], widths[1], k=3), nn.MaxPool2d(2))   # stride 2
        self.down2 = nn.Sequential(_conv_bn_relu(widths[1], widths[2], k=3), nn.MaxPool2d(2))   # stride 4
        self.down3 = nn.Sequential(_conv_bn_relu(widths[2], widths[3], k=3), nn.MaxPool2d(2))   # stride 8
        self.down4 = nn.Sequential(_conv_bn_relu(widths[3], widths[4], k=3), nn.MaxPool2d(2))   # stride 16

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        d0 = self.stem(x)     # stride 1
        d1 = self.down1(d0)   # stride 2
        d2 = self.down2(d1)   # stride 4
        d3 = self.down3(d2)   # stride 8
        d4 = self.down4(d3)   # stride 16
        return [d0, d1, d2, d3, d4]


class FuseUpBlock(nn.Module):
    """Bilinear-upsample + concat (RGB tap, depth tap, prev decoder feat) + 2x conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            _conv_bn_relu(in_ch, out_ch, k=3),
            _conv_bn_relu(out_ch, out_ch, k=3),
        )

    def forward(self, x: torch.Tensor, *skips: torch.Tensor) -> torch.Tensor:
        # upsample x to skip resolution, concat all
        target_size = skips[0].shape[-2:]
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = torch.cat([x] + list(skips), dim=1)
        return self.fuse(x)


class RGBGuidedDepthUpsampler(nn.Module):
    """End-to-end model: RGB+depth -> residual -> denormalized depth output."""

    def __init__(self, pretrained_rgb: bool = True, residual_scale: float = 0.2):
        super().__init__()
        self.rgb_enc = RGBEncoder(pretrained=pretrained_rgb)
        self.depth_enc = DepthEncoder(in_ch=2)  # depth + confidence
        self.residual_scale = residual_scale

        rgb_chs = RGBEncoder.TAP_CHANNELS                # (16, 24, 32, 96)
        d_widths = (16, 16, 24, 32, 64)                  # depth feature widths at strides 1..16

        # decoder progressively fuses from stride 16 up to stride 1
        # bottom: depth_16 (64) + rgb_16 (96)
        self.bot = _conv_bn_relu(d_widths[4] + rgb_chs[3], 96, k=3)

        # up to stride 8: prev (96) + rgb_8 (32) + depth_8 (32)
        self.up3 = FuseUpBlock(96 + rgb_chs[2] + d_widths[3], 64)
        # up to stride 4: prev (64) + rgb_4 (24) + depth_4 (24)
        self.up2 = FuseUpBlock(64 + rgb_chs[1] + d_widths[2], 48)
        # up to stride 2: prev (48) + rgb_2 (16) + depth_2 (16)
        self.up1 = FuseUpBlock(48 + rgb_chs[0] + d_widths[1], 32)
        # up to stride 1: prev (32) + depth_1 (16)
        self.up0 = FuseUpBlock(32 + d_widths[0], 16)

        self.head = nn.Sequential(
            _conv_bn_relu(16, 16, k=3),
            nn.Conv2d(16, 1, 3, padding=1),
        )

        # init head's last conv to zero so the model starts as the identity
        # (output == bicubic baseline). Stabilizes early training.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        rgb: torch.Tensor,
        bicubic_norm: torch.Tensor,
        conf_hi: torch.Tensor,
    ) -> torch.Tensor:
        """Returns predicted depth in *normalized* units (i.e. depth / DEPTH_MAX)."""
        depth_in = torch.cat([bicubic_norm, conf_hi], dim=1)  # (B, 2, H, W)
        rgb_feats = self.rgb_enc(rgb)            # 4 maps, strides 2, 4, 8, 16
        depth_feats = self.depth_enc(depth_in)   # 5 maps, strides 1, 2, 4, 8, 16

        # bottom fusion at stride 16
        b = self.bot(torch.cat([depth_feats[4], rgb_feats[3]], dim=1))

        # decoder up
        u3 = self.up3(b, rgb_feats[2], depth_feats[3])      # stride 8
        u2 = self.up2(u3, rgb_feats[1], depth_feats[2])     # stride 4
        u1 = self.up1(u2, rgb_feats[0], depth_feats[1])     # stride 2
        u0 = self.up0(u1, depth_feats[0])                   # stride 1

        residual = self.head(u0)                             # (B, 1, H, W)
        out = bicubic_norm + self.residual_scale * residual
        return out  # still in normalized units (~[0,1])

    def freeze_rgb(self):
        self.rgb_enc.freeze()

    def unfreeze_rgb(self):
        self.rgb_enc.unfreeze()


def count_parameters(m: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable
