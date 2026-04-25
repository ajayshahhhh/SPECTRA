"""Standalone laptop inference: RGB + iPhone LiDAR -> metric depth.

Loads `best.pt` and runs a single frame through the network, writing both
a colormap PNG (for viewing) and a float32 .npy (for downstream use).

Run:
    python infer.py \
        --ckpt best.pt \
        --rgb sample_inputs/rgb.png \
        --lowres sample_inputs/lowres_depth.png \
        --conf sample_inputs/confidence.png \
        --out output_depth.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from mvg.dataset import (
    DEPTH_MAX, DEPTH_MIN, IMAGENET_MEAN, IMAGENET_STD,
    NATIVE_LOW, MM_TO_M, _rotate_np,
)
from mvg.model import RGBGuidedDepthUpsampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--rgb", required=True)
    p.add_argument("--lowres", required=True, help="lowres iPhone depth PNG (uint16, mm)")
    p.add_argument("--conf", required=True, help="confidence PNG (uint8, 0/1/2)")
    p.add_argument("--out", default="output_depth.png", help="colormap PNG output")
    p.add_argument("--out_npy", default="", help="optional float32 .npy output (default: alongside --out)")
    p.add_argument("--target_h", type=int, default=768)
    p.add_argument("--target_w", type=int, default=1024)
    p.add_argument("--sky_direction", default="Up",
                   choices=["Up", "Down", "Left", "Right"],
                   help="ARKit sky direction; rotates inputs so 'up' is up")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"device: {device}")

    # --- Load inputs (mirror dataset.py preprocessing) ---
    rgb_np = np.array(Image.open(args.rgb).convert("RGB"))
    lo_mm  = np.array(Image.open(args.lowres))
    conf   = np.array(Image.open(args.conf))

    # Apply ARKit sky-direction rotation if needed
    rgb_np = _rotate_np(rgb_np, args.sky_direction)
    lo_mm  = _rotate_np(lo_mm,  args.sky_direction)
    conf   = _rotate_np(conf,   args.sky_direction)

    H, W = args.target_h, args.target_w
    upsample_factor = W // NATIVE_LOW[1]
    lh, lw = H // upsample_factor, W // upsample_factor

    # Resize RGB to target res; depth/conf already at low res (192x256 ≈ lh×lw)
    rgb_pil = Image.fromarray(rgb_np).resize((W, H), Image.BILINEAR)
    rgb_np = np.array(rgb_pil)
    if lo_mm.shape != (lh, lw):
        lo_mm = np.array(Image.fromarray(lo_mm).resize((lw, lh), Image.NEAREST))
        conf  = np.array(Image.fromarray(conf).resize((lw, lh), Image.NEAREST))

    # Convert to meters; zero out low-confidence iPhone pixels (per dataset spec)
    lo_m = lo_mm.astype(np.float32) / MM_TO_M
    lo_filtered = np.where(conf >= 2, lo_m, 0.0).astype(np.float32)

    # To torch tensors with batch dim
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    rgb_t = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(rgb_t.squeeze(0)).unsqueeze(0)
    lo_t = torch.from_numpy(lo_filtered).unsqueeze(0).unsqueeze(0)
    conf_t = torch.from_numpy(conf.astype(np.uint8)).unsqueeze(0).unsqueeze(0)

    # Bicubic baseline + high-res confidence-2 mask (model inputs)
    bicubic = F.interpolate(lo_t, size=(H, W), mode="bicubic", align_corners=False).clamp(0, DEPTH_MAX)
    bicubic_norm = (bicubic / DEPTH_MAX).clamp(0, 1)
    conf_hi = F.interpolate(
        (conf_t == 2).float(), size=(H, W), mode="bilinear", align_corners=False,
    )

    rgb_t = rgb_t.to(device)
    bicubic_norm = bicubic_norm.to(device)
    conf_hi = conf_hi.to(device)

    # --- Load model + run ---
    model = RGBGuidedDepthUpsampler(pretrained_rgb=False).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)

    with torch.no_grad():
        pred_norm = model(rgb_t, bicubic_norm, conf_hi)
        pred_m = (pred_norm * DEPTH_MAX).clamp(DEPTH_MIN, DEPTH_MAX)

    pred_m_np = pred_m.squeeze().cpu().numpy().astype(np.float32)

    # --- Save outputs ---
    out_png = Path(args.out)
    out_npy = Path(args.out_npy) if args.out_npy else out_png.with_suffix(".npy")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Colormap visualization (matplotlib if available, else simple normalize-to-uint8)
    try:
        from matplotlib import pyplot as plt
        plt.figure(figsize=(10, 7.5))
        plt.imshow(pred_m_np, cmap="turbo", vmin=DEPTH_MIN, vmax=DEPTH_MAX)
        plt.colorbar(label="depth (m)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=110, bbox_inches="tight")
        plt.close()
    except ImportError:
        norm = (pred_m_np - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
        Image.fromarray((norm.clip(0, 1) * 255).astype(np.uint8)).save(out_png)

    np.save(out_npy, pred_m_np)
    print(f"wrote {out_png}  ({pred_m_np.shape[0]}x{pred_m_np.shape[1]} metric depth)")
    print(f"wrote {out_npy}")
    print(f"depth range in this frame: {pred_m_np.min():.2f} – {pred_m_np.max():.2f} m")


if __name__ == "__main__":
    main()
