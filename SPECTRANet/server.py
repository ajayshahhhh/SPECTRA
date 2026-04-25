"""SPECTRANet HTTP inference server — runs on GX10, called by the iPhone app.

Start it from the SPECTRANet directory:
    pip install fastapi uvicorn python-multipart
    uvicorn server:app --host 0.0.0.0 --port 8000

The iPhone sends a multipart/form-data POST to /infer with:
  - rgb   (file): JPEG bytes of the camera frame, any resolution
  - depth (file): raw float32 LE bytes, meters, lH×lW (ARKit sceneDepth)
  - conf  (file): raw uint8 bytes, lH×lW, values {0,1,2}
  - lH    (int field): low-res depth height  (usually 192)
  - lW    (int field): low-res depth width   (usually 256)

Response: 200 OK, image/jpeg — colorized depth map (landscape, ~50-100 KB).
  Headers: X-Center-Distance, X-Min-Depth, X-Max-Depth (float strings, meters).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import Response
from PIL import Image
from torchvision import transforms as T

from spectranet.dataset import DEPTH_MAX, DEPTH_MIN, IMAGENET_MEAN, IMAGENET_STD
from spectranet.model import RGBGuidedDepthUpsampler

CKPT = Path(__file__).parent / "best.pt"
TARGET_H, TARGET_W = 768, 1024

# Turbo-inspired HSV LUT matching DepthProcessor.swift exactly:
# red (near) → yellow → green → cyan → blue (far), hue 0→240°
_LUT: np.ndarray = np.zeros((256, 3), dtype=np.uint8)
for _i in range(256):
    _t = _i / 255.0
    _hue = _t * (240.0 / 360.0)
    _s = _hue * 6
    _si = int(_s) % 6
    _f = _s - int(_s)
    _q = 1 - _f
    _rgb = [(1, _f, 0), (_q, 1, 0), (0, 1, _f), (0, _q, 1), (_f, 0, 1), (1, 0, _q)][_si]
    _LUT[_i] = [min(255, int(v * 255)) for v in _rgb]


def _colorize(pred_m: np.ndarray):
    """Return (jpeg_bytes, center_dist, min_d, max_d). pred_m is (H,W) float32 meters."""
    valid = (pred_m > 0) & np.isfinite(pred_m)
    if not valid.any():
        return None, 0.0, 0.0, 0.0

    min_d = float(pred_m[valid].min())
    max_d = float(pred_m[valid].max())
    if max_d <= min_d:
        return None, 0.0, min_d, max_d

    norm = np.zeros(pred_m.shape, dtype=np.float32)
    norm[valid] = (pred_m[valid] - min_d) / (max_d - min_d) * 255.0
    idx = np.clip(norm, 0, 255).astype(np.uint8)

    H, W = pred_m.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[valid, :3] = _LUT[idx[valid]]
    rgba[valid, 3] = 255

    # Center distance (5×5 patch)
    cy, cx = H // 2, W // 2
    patch = pred_m[cy - 2:cy + 3, cx - 2:cx + 3]
    patch_vals = patch[(patch > 0) & np.isfinite(patch)]
    center_dist = float(patch_vals.mean()) if len(patch_vals) > 0 else 0.0

    jpeg_buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").convert("RGB").save(jpeg_buf, format="JPEG", quality=85)
    return jpeg_buf.getvalue(), center_dist, min_d, max_d


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


app = FastAPI()

_device = _pick_device()
print(f"[SPECTRANet] loading model on {_device} …")
_model = RGBGuidedDepthUpsampler(pretrained_rgb=False).to(_device).eval()
_ckpt = torch.load(CKPT, map_location=_device, weights_only=False)
_state = _ckpt["model"] if "model" in _ckpt else _ckpt
_model.load_state_dict(_state)

if _device.type == "cuda":
    # Warm-up: ensure CUDA kernels are loaded before first real request
    print("[SPECTRANet] warming up …")
    with torch.no_grad(), torch.amp.autocast("cuda"):
        _dummy_rgb   = torch.zeros(1, 3, TARGET_H, TARGET_W, device=_device)
        _dummy_depth = torch.zeros(1, 1, TARGET_H, TARGET_W, device=_device)
        _model(_dummy_rgb, _dummy_depth, _dummy_depth)
print("[SPECTRANet] ready — listening for frames.")


@app.post("/infer")
async def infer(
    rgb: UploadFile,
    depth: UploadFile,
    conf: UploadFile,
    lH: int = Form(...),
    lW: int = Form(...),
) -> Response:
    H, W = TARGET_H, TARGET_W

    # RGB: JPEG → resize → ImageNet-normalise → (1,3,H,W)
    rgb_bytes = await rgb.read()
    rgb_pil = Image.open(io.BytesIO(rgb_bytes)).convert("RGB").resize((W, H), Image.BILINEAR)
    rgb_np = np.array(rgb_pil)
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
    rgb_t = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(rgb_t).unsqueeze(0)

    # Depth: raw float32 LE, meters, lH×lW
    lo_m = np.frombuffer(await depth.read(), dtype="<f4").reshape(lH, lW).copy()

    # Confidence: raw uint8, lH×lW, values {0,1,2}
    conf_np = np.frombuffer(await conf.read(), dtype=np.uint8).reshape(lH, lW).copy()

    lo_filtered = np.where(conf_np >= 2, lo_m, 0.0).astype(np.float32)

    lo_t   = torch.from_numpy(lo_filtered).unsqueeze(0).unsqueeze(0)
    conf_t = torch.from_numpy(conf_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    bicubic      = F.interpolate(lo_t, (H, W), mode="bicubic", align_corners=False).clamp(0, DEPTH_MAX)
    bicubic_norm = (bicubic / DEPTH_MAX).clamp(0, 1)
    conf_hi      = F.interpolate((conf_t == 2).float(), (H, W), mode="bilinear", align_corners=False)

    import time
    t0 = time.perf_counter()
    autocast_ctx = torch.amp.autocast("cuda") if _device.type == "cuda" else torch.amp.autocast("cpu", enabled=False)
    with torch.no_grad(), autocast_ctx:
        pred_norm = _model(rgb_t.to(_device), bicubic_norm.to(_device), conf_hi.to(_device))
        pred_m    = (pred_norm * DEPTH_MAX).clamp(DEPTH_MIN, DEPTH_MAX)
    print(f"[infer] {(time.perf_counter()-t0)*1000:.0f} ms  device={_device}", flush=True)

    pred_np = pred_m.squeeze().cpu().numpy().astype(np.float32)
    jpeg_bytes, center_dist, min_d, max_d = _colorize(pred_np)

    if jpeg_bytes is None:
        return Response(status_code=204)

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "X-Center-Distance": f"{center_dist:.4f}",
            "X-Min-Depth":       f"{min_d:.4f}",
            "X-Max-Depth":       f"{max_d:.4f}",
        },
    )
