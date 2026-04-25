"""ARKitScenes dataset for RGB-guided depth upsampling.

Loads the four pre-aligned modalities (wide RGB, lowres iPhone depth,
confidence, highres Faro GT) and produces tensors at a chosen target
resolution. Includes confidence-aware filtering of the iPhone input,
ImageNet RGB normalization, and a residual-style baseline (bicubic
upsample of the lowres depth) so the model learns a delta on top.
"""
from __future__ import annotations

import os
import random
from glob import glob
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# canonical native resolutions (after sky-direction rotation to "Up")
NATIVE_LOW = (192, 256)  # H, W
NATIVE_HIGH = (1440, 1920)
MM_TO_M = 1000.0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# depth clip range, in meters (indoor scenes)
DEPTH_MIN = 0.5
DEPTH_MAX = 10.0


def _rotate_np(img: np.ndarray, direction: str) -> np.ndarray:
    """Rotate HxW or HxWxC array by 90/180 degrees per ARKit sky direction."""
    if direction == "Up":
        return img
    if direction == "Left":
        # 90° clockwise == np.rot90(k=-1)
        return np.rot90(img, k=-1).copy()
    if direction == "Right":
        return np.rot90(img, k=1).copy()
    if direction == "Down":
        return np.rot90(img, k=2).copy()
    raise ValueError(f"unknown sky_direction {direction!r}")


def _resize_image(arr: np.ndarray, size: Tuple[int, int], mode: str) -> np.ndarray:
    """Resize HxW or HxWxC numpy array to (H, W). mode in {"bilinear","nearest"}."""
    h, w = size
    if arr.shape[:2] == (h, w):
        return arr
    if arr.ndim == 2:
        pil = Image.fromarray(arr)
        resample = Image.BILINEAR if mode == "bilinear" else Image.NEAREST
        return np.array(pil.resize((w, h), resample=resample))
    # HxWxC uint8 RGB
    pil = Image.fromarray(arr)
    return np.array(pil.resize((w, h), resample=Image.BILINEAR))


class ARKitUpsampleDataset(Dataset):
    """ARKitScenes RGB-guided depth-upsampling dataset.

    Returns a dict per sample with:
      - rgb           : (3, H, W)  float32, ImageNet-normalized
      - lowres        : (1, h, w)  float32  meters (zeros = invalid input)
      - lowres_conf   : (1, h, w)  uint8 in {0,1,2}
      - bicubic       : (1, H, W)  float32  meters, bicubic upsample of conf-filtered lowres
      - bicubic_norm  : (1, H, W)  float32  bicubic / DEPTH_MAX, clipped to [0,1]
      - conf_hi       : (1, H, W)  float32  confidence==2 mask, bilinear-upsampled to high-res
      - gt            : (1, H, W)  float32  meters (Faro GT, zeros = invalid)
      - valid         : (1, H, W)  bool     gt valid mask (clipped to DEPTH_MIN..DEPTH_MAX)
      - identifier    : str
    """

    def __init__(
        self,
        root: str,
        split: str,
        target_hw: Tuple[int, int] = (768, 1024),
        crop_hw: Tuple[int, int] | None = None,
        augment: bool = False,
        confidence_threshold: int = 2,
    ) -> None:
        super().__init__()
        assert split in ("train", "val")
        self.root = os.path.expanduser(root)
        self.split = split
        self.target_hw = target_hw
        self.crop_hw = crop_hw
        self.augment = augment
        self.confidence_threshold = confidence_threshold

        # cleanly map native lowres aspect 4:3 onto target (4:3)
        H, W = target_hw
        assert H % 4 == 0 and W % 4 == 0, "target must be divisible by 4"
        self.upsample_factor = W // NATIVE_LOW[1]  # iPhone lowres → target
        self.lowres_hw = (H // self.upsample_factor, W // self.upsample_factor)

        split_folder = "Training" if split == "train" else "Validation"
        self.split_dir = os.path.join(self.root, split_folder)
        meta = pd.read_csv(os.path.join(self.root, "metadata.csv"))
        meta = meta[meta["fold"] == split_folder]

        # build sample list, skip non-directory artifacts (e.g. partial .tmp downloads)
        self.samples = []  # (video_id, frame_filename, sky_direction)
        for video_id, sky in zip(meta["video_id"], meta["sky_direction"]):
            video_folder = os.path.join(self.split_dir, str(video_id))
            if not os.path.isdir(video_folder):
                continue
            wide_dir = os.path.join(video_folder, "wide")
            if not os.path.isdir(wide_dir):
                continue
            for fpath in sorted(glob(os.path.join(wide_dir, "*.png"))):
                fname = os.path.basename(fpath)
                # confirm all four modalities exist for this frame
                ok = all(
                    os.path.isfile(os.path.join(video_folder, sub, fname))
                    for sub in ("highres_depth", "lowres_depth", "confidence")
                )
                if ok:
                    self.samples.append((str(video_id), fname, sky))

        self._rgb_norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

        if augment:
            self._color_jitter = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
        else:
            self._color_jitter = None

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_png(path: str) -> np.ndarray:
        return np.array(Image.open(path))

    def _load_one(self, video_id: str, fname: str, sky: str):
        base = os.path.join(self.split_dir, video_id)
        rgb = self._load_png(os.path.join(base, "wide", fname))                 # (1440, 1920, 3) uint8
        gt_mm = self._load_png(os.path.join(base, "highres_depth", fname))      # (1440, 1920) uint16
        lo_mm = self._load_png(os.path.join(base, "lowres_depth", fname))       # (192, 256) uint16
        conf = self._load_png(os.path.join(base, "confidence", fname))          # (192, 256) uint8

        rgb = _rotate_np(rgb, sky)
        gt_mm = _rotate_np(gt_mm, sky)
        lo_mm = _rotate_np(lo_mm, sky)
        conf = _rotate_np(conf, sky)
        return rgb, gt_mm, lo_mm, conf

    def __getitem__(self, idx: int):
        video_id, fname, sky = self.samples[idx]
        rgb, gt_mm, lo_mm, conf = self._load_one(video_id, fname, sky)

        H, W = self.target_hw
        lh, lw = self.lowres_hw

        # Resize RGB and Faro GT down to target res; lowres iPhone depth and
        # confidence to the matching low-res. Native lowres is exactly lh×lw
        # when target == 4× native, so no resize needed in the standard config.
        rgb = _resize_image(rgb, (H, W), mode="bilinear")
        gt_mm = _resize_image(gt_mm, (H, W), mode="nearest")
        if lo_mm.shape != (lh, lw):
            lo_mm = _resize_image(lo_mm, (lh, lw), mode="nearest")
            conf = _resize_image(conf, (lh, lw), mode="nearest")

        # mm -> meters; preserve zero (invalid)
        gt = gt_mm.astype(np.float32) / MM_TO_M  # (H, W)
        lo = lo_mm.astype(np.float32) / MM_TO_M  # (lh, lw)

        # iPhone depth: zero out low-confidence pixels (input cleanup per spec)
        lo_filtered = np.where(conf >= self.confidence_threshold, lo, 0.0).astype(np.float32)

        # to torch
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        gt_t = torch.from_numpy(gt).unsqueeze(0)                        # (1, H, W)
        lo_t = torch.from_numpy(lo_filtered).unsqueeze(0)               # (1, lh, lw)
        conf_t = torch.from_numpy(conf.astype(np.uint8)).unsqueeze(0)   # (1, lh, lw)

        # GT valid mask: must be within sensible range (clipped) and non-zero
        valid = (gt_t > DEPTH_MIN) & (gt_t < DEPTH_MAX)

        # --- augmentation (training only) ---
        if self.augment:
            # color jitter on RGB only
            if self._color_jitter is not None:
                rgb_t = self._color_jitter(rgb_t)

            # horizontal flip (RGB, lowres, GT, valid, conf)
            if random.random() < 0.5:
                rgb_t = torch.flip(rgb_t, dims=[2])
                gt_t = torch.flip(gt_t, dims=[2])
                valid = torch.flip(valid, dims=[2])
                lo_t = torch.flip(lo_t, dims=[2])
                conf_t = torch.flip(conf_t, dims=[2])

            # random crop: keep low-res and high-res aligned by upsample_factor
            if self.crop_hw is not None:
                ch, cw = self.crop_hw
                lch, lcw = ch // self.upsample_factor, cw // self.upsample_factor
                # pick low-res crop origin first to keep things aligned
                ly = random.randint(0, lh - lch)
                lx = random.randint(0, lw - lcw)
                hy, hx = ly * self.upsample_factor, lx * self.upsample_factor
                lo_t = lo_t[:, ly:ly + lch, lx:lx + lcw]
                conf_t = conf_t[:, ly:ly + lch, lx:lx + lcw]
                rgb_t = rgb_t[:, hy:hy + ch, hx:hx + cw]
                gt_t = gt_t[:, hy:hy + ch, hx:hx + cw]
                valid = valid[:, hy:hy + ch, hx:hx + cw]
                H, W = ch, cw

        # build bicubic baseline at full target res from filtered lowres
        # (a NaN-safe simple bicubic — just uses zeros where missing)
        bicubic = F.interpolate(
            lo_t.unsqueeze(0).float(), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze(0)
        bicubic.clamp_(min=0.0, max=DEPTH_MAX)

        # high-res confidence-2 mask, useful as model input + analysis
        conf_hi = F.interpolate(
            (conf_t == self.confidence_threshold).float().unsqueeze(0),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(0)

        # normalize for model: depth / DEPTH_MAX in [0,1]
        bicubic_norm = (bicubic / DEPTH_MAX).clamp(0.0, 1.0)

        rgb_t = self._rgb_norm(rgb_t)

        return {
            "rgb": rgb_t,
            "lowres": lo_t,
            "lowres_conf": conf_t,
            "bicubic": bicubic,
            "bicubic_norm": bicubic_norm,
            "conf_hi": conf_hi,
            "gt": gt_t,
            "valid": valid,
            "identifier": f"{video_id}/{fname}",
        }


def make_loaders(
    root: str,
    target_hw: Tuple[int, int],
    train_crop_hw: Tuple[int, int] | None,
    batch_size: int,
    num_workers: int,
):
    from torch.utils.data import DataLoader

    train_set = ARKitUpsampleDataset(
        root=root, split="train", target_hw=target_hw, crop_hw=train_crop_hw, augment=True,
    )
    val_set = ARKitUpsampleDataset(
        root=root, split="val", target_hw=target_hw, crop_hw=None, augment=False,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=True, persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_set, val_set
