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

Response: 200 OK, image/jpeg — colorized depth map (landscape).
  Headers: X-Center-Distance, X-Min-Depth, X-Max-Depth (float strings, meters).
"""
from __future__ import annotations

import io
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from PIL import Image
from torchvision import transforms as T

try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    _jpeg = TurboJPEG()
except Exception:
    _jpeg = None

from spectranet.dataset import DEPTH_MAX, DEPTH_MIN, IMAGENET_MEAN, IMAGENET_STD
from spectranet.model import RGBGuidedDepthUpsampler

CKPT = Path(__file__).parent / "best.pt"
TARGET_H, TARGET_W = 768, 1024

# HSV LUT — conventional depth mapping: near=red, far=blue
_LUT_NP: np.ndarray = np.zeros((256, 3), dtype=np.uint8)
for _i in range(256):
    _t = _i / 255.0
    _hue = _t * (210.0 / 360.0)  # index 0 (near) → red (0°), index 255 (far) → blue (210°), avoiding magenta wrap
    _s = _hue * 6
    _si = int(_s) % 6
    _f = _s - int(_s)
    _q = 1 - _f
    _rgb = [(1, _f, 0), (_q, 1, 0), (0, 1, _f), (0, _q, 1), (_f, 0, 1), (1, 0, _q)][_si]
    _LUT_NP[_i] = [min(255, int(v * 255)) for v in _rgb]


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

# GPU LUT tensor for fast colorization on device
_LUT_GPU: torch.Tensor | None = None

# Static tensors and CUDA graph (set up below if on CUDA)
_graph: torch.cuda.CUDAGraph | None = None
_static_rgb:      torch.Tensor | None = None
_static_bicubic:  torch.Tensor | None = None
_static_conf_hi:  torch.Tensor | None = None
_static_out:      torch.Tensor | None = None

# Temporal smoothing for center distance (EMA filter)
_smoothed_distance: float | None = None
_SMOOTHING_ALPHA = 0.3  # Lower = more smoothing, higher = more responsive

if _device.type == "cuda":
    _LUT_GPU = torch.from_numpy(_LUT_NP).to(_device)

    # torch.jit.trace — faster than eager, no Python.h needed unlike torch.compile
    print("[SPECTRANet] tracing model …")
    with torch.no_grad():
        _dummy     = torch.zeros(1, 1, TARGET_H, TARGET_W, device=_device)
        _dummy_rgb = torch.zeros(1, 3, TARGET_H, TARGET_W, device=_device)
        _model = torch.jit.trace(_model, (_dummy_rgb, _dummy, _dummy))

    # CUDA graph: capture the forward pass once, replay with zero CPU overhead
    print("[SPECTRANet] capturing CUDA graph …")
    _static_rgb     = torch.zeros(1, 3, TARGET_H, TARGET_W, device=_device)
    _static_bicubic = torch.zeros(1, 1, TARGET_H, TARGET_W, device=_device)
    _static_conf_hi = torch.zeros(1, 1, TARGET_H, TARGET_W, device=_device)

    # Warm-up outside the graph (required before capture)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for _ in range(3):
            _model(_static_rgb, _static_bicubic, _static_conf_hi)

    _graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.amp.autocast("cuda"), torch.cuda.graph(_graph):
        _static_out = _model(_static_rgb, _static_bicubic, _static_conf_hi)

print("[SPECTRANet] ready — listening for frames.")


def _colorize_gpu(pred_m: torch.Tensor):
    """Colorize on GPU, return (jpeg_bytes, center_dist, min_d, max_d).
    pred_m: (H, W) float32 on _device, already clamped."""
    global _smoothed_distance

    valid = (pred_m > 0) & torch.isfinite(pred_m)
    if not valid.any():
        return None, 0.0, 0.0, 0.0

    vals = pred_m[valid]
    actual_min = vals.min().item()
    actual_max = vals.max().item()

    # Use FIXED depth range for consistent absolute color mapping
    # 0.5m → red (index 0), 5m+ → blue (index 255)
    idx = ((pred_m - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255).clamp(0, 255).long()
    idx[~valid] = 0

    H, W = pred_m.shape
    rgb = _LUT_GPU[idx.view(-1)].view(H, W, 3)         # (H,W,3) uint8
    alpha = (valid.to(torch.uint8) * 255).unsqueeze(-1) # (H,W,1)
    rgba = torch.cat([rgb, alpha], dim=-1)              # (H,W,4)

    # Center distance (11×11 patch, matching SPECTRALive)
    cy, cx = H // 2, W // 2
    patch = pred_m[cy - 5:cy + 6, cx - 5:cx + 6]
    patch_vals = patch[(patch > 0) & torch.isfinite(patch)]
    raw_dist = patch_vals.mean().item() if patch_vals.numel() > 0 else 0.0

    # Apply temporal smoothing (EMA filter)
    if _smoothed_distance is None:
        _smoothed_distance = raw_dist
    else:
        _smoothed_distance = _SMOOTHING_ALPHA * raw_dist + (1 - _SMOOTHING_ALPHA) * _smoothed_distance
    center_dist = _smoothed_distance

    rgba_np = rgba.cpu().numpy()
    rgb_out = rgba_np[:, :, :3]
    if _jpeg is not None:
        return _jpeg.encode(rgb_out, quality=70, pixel_format=TJPF_RGB), center_dist, actual_min, actual_max
    buf = io.BytesIO()
    Image.fromarray(rgb_out).save(buf, format="JPEG", quality=70)
    return buf.getvalue(), center_dist, actual_min, actual_max


def _colorize_cpu(pred_np: np.ndarray):
    """CPU fallback colorize."""
    global _smoothed_distance

    valid = (pred_np > 0) & np.isfinite(pred_np)
    if not valid.any():
        return None, 0.0, 0.0, 0.0
    actual_min = float(pred_np[valid].min())
    actual_max = float(pred_np[valid].max())

    # Use FIXED depth range for consistent absolute color mapping
    # 0.5m → red (index 0), 5m+ → blue (index 255)
    norm = np.zeros_like(pred_np)
    norm[valid] = (pred_np[valid] - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255.0
    idx = np.clip(norm, 0, 255).astype(np.uint8)
    H, W = pred_np.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[valid, :3] = _LUT_NP[idx[valid]]
    rgba[valid, 3] = 255
    cy, cx = H // 2, W // 2
    patch = pred_np[cy - 5:cy + 6, cx - 5:cx + 6]
    patch_vals = patch[(patch > 0) & np.isfinite(patch)]
    raw_dist = float(patch_vals.mean()) if len(patch_vals) > 0 else 0.0

    # Apply temporal smoothing (EMA filter)
    if _smoothed_distance is None:
        _smoothed_distance = raw_dist
    else:
        _smoothed_distance = _SMOOTHING_ALPHA * raw_dist + (1 - _SMOOTHING_ALPHA) * _smoothed_distance
    center_dist = _smoothed_distance

    rgb_out = rgba[:, :, :3]
    if _jpeg is not None:
        return _jpeg.encode(rgb_out, quality=70, pixel_format=TJPF_RGB), center_dist, actual_min, actual_max
    buf = io.BytesIO()
    Image.fromarray(rgb_out).save(buf, format="JPEG", quality=70)
    return buf.getvalue(), center_dist, actual_min, actual_max


@app.post("/infer")
async def infer(
    rgb: UploadFile,
    depth: UploadFile,
    conf: UploadFile,
    lH: int = Form(...),
    lW: int = Form(...),
) -> Response:
    H, W = TARGET_H, TARGET_W

    # RGB: JPEG → resize → ImageNet-normalise → GPU tensor (1,3,H,W)
    rgb_bytes = await rgb.read()
    if _jpeg is not None:
        rgb_np = _jpeg.decode(rgb_bytes, pixel_format=TJPF_RGB)
        rgb_np = np.array(Image.fromarray(rgb_np).resize((W, H), Image.BILINEAR))
    else:
        rgb_np = np.array(Image.open(io.BytesIO(rgb_bytes)).convert("RGB").resize((W, H), Image.BILINEAR))
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
    rgb_t = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(rgb_t).unsqueeze(0).to(_device)

    # Depth + confidence: to GPU immediately, all upsampling on GPU
    lo_m    = np.frombuffer(await depth.read(), dtype="<f4").reshape(lH, lW).copy()
    conf_np = np.frombuffer(await conf.read(), dtype=np.uint8).reshape(lH, lW).copy()

    lo_filtered = np.where(conf_np >= 2, lo_m, 0.0).astype(np.float32)

    lo_t   = torch.from_numpy(lo_filtered).unsqueeze(0).unsqueeze(0).to(_device)
    conf_t = torch.from_numpy(conf_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(_device)

    # All interpolation on GPU
    bicubic      = F.interpolate(lo_t, (H, W), mode="bicubic", align_corners=False).clamp(0, DEPTH_MAX)
    bicubic_norm = (bicubic / DEPTH_MAX).clamp(0, 1)
    conf_hi      = F.interpolate((conf_t == 2).float(), (H, W), mode="bilinear", align_corners=False)

    t0 = time.perf_counter()
    if _graph is not None:
        # Copy inputs into the static tensors the graph was captured with, then replay
        _static_rgb.copy_(rgb_t)
        _static_bicubic.copy_(bicubic_norm)
        _static_conf_hi.copy_(conf_hi)
        _graph.replay()
        pred_m = (_static_out * DEPTH_MAX).clamp(DEPTH_MIN, DEPTH_MAX)
    else:
        autocast_ctx = torch.amp.autocast("cuda") if _device.type == "cuda" else torch.amp.autocast("cpu", enabled=False)
        with torch.no_grad(), autocast_ctx:
            pred_norm = _model(rgb_t, bicubic_norm, conf_hi)
            pred_m    = (pred_norm * DEPTH_MAX).clamp(DEPTH_MIN, DEPTH_MAX)
    print(f"[infer] {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)

    if _LUT_GPU is not None:
        jpeg_bytes, center_dist, min_d, max_d = _colorize_gpu(pred_m.squeeze())
    else:
        jpeg_bytes, center_dist, min_d, max_d = _colorize_cpu(
            pred_m.squeeze().cpu().numpy().astype(np.float32)
        )

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


def _run_infer(rgb_np, lo_m, conf_np):
    """Shared inference logic for both HTTP and WebSocket endpoints."""
    H, W = TARGET_H, TARGET_W

    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
    rgb_t = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(rgb_t).unsqueeze(0).to(_device)

    lo_filtered = np.where(conf_np >= 2, lo_m, 0.0).astype(np.float32)
    lo_t   = torch.from_numpy(lo_filtered).unsqueeze(0).unsqueeze(0).to(_device)
    conf_t = torch.from_numpy(conf_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(_device)

    bicubic      = F.interpolate(lo_t, (H, W), mode="bicubic", align_corners=False).clamp(0, DEPTH_MAX)
    bicubic_norm = (bicubic / DEPTH_MAX).clamp(0, 1)
    conf_hi      = F.interpolate((conf_t == 2).float(), (H, W), mode="bilinear", align_corners=False)

    if _graph is not None:
        _static_rgb.copy_(rgb_t)
        _static_bicubic.copy_(bicubic_norm)
        _static_conf_hi.copy_(conf_hi)
        _graph.replay()
        pred_m = (_static_out * DEPTH_MAX).clamp(DEPTH_MIN, DEPTH_MAX)
    else:
        with torch.no_grad():
            pred_norm = _model(rgb_t, bicubic_norm, conf_hi)
            pred_m    = (pred_norm * DEPTH_MAX).clamp(DEPTH_MIN, DEPTH_MAX)

    if _LUT_GPU is not None:
        return _colorize_gpu(pred_m.squeeze())
    return _colorize_cpu(pred_m.squeeze().cpu().numpy().astype(np.float32))


@app.websocket("/ws")
async def ws_infer(websocket: WebSocket):
    """WebSocket endpoint — persistent connection, one frame per message.

    Client sends binary:
      [4B uint32 lH][4B uint32 lW][4B uint32 jpeg_len]
      [jpeg_bytes][zlib_depth_bytes][zlib_conf_bytes]

    Server replies binary:
      [4B uint32 jpeg_len][4B float32 center][4B float32 min_d][4B float32 max_d]
      [jpeg_bytes]
    """
    await websocket.accept()
    print(f"[WS] client connected: {websocket.client}")
    try:
        while True:
            data = await websocket.receive_bytes()
            try:
                lH, lW, jpeg_len = struct.unpack_from(">III", data, 0)
                offset = 12
                jpeg_bytes_in = data[offset: offset + jpeg_len];  offset += jpeg_len
                # NSData.compressed(using: .zlib) produces raw DEFLATE (no header)
                dec = zlib.decompressobj(wbits=-15)
                depth_raw = dec.decompress(bytes(data[offset:]))
                conf_raw  = zlib.decompress(dec.unused_data, wbits=-15)

                if _jpeg is not None:
                    rgb_np = _jpeg.decode(bytes(jpeg_bytes_in), pixel_format=TJPF_RGB)
                    rgb_np = np.array(Image.fromarray(rgb_np).resize(
                        (TARGET_W, TARGET_H), Image.BILINEAR))
                else:
                    rgb_np = np.array(Image.open(io.BytesIO(bytes(jpeg_bytes_in))).convert("RGB")
                                      .resize((TARGET_W, TARGET_H), Image.BILINEAR))

                lo_m    = np.frombuffer(depth_raw, dtype="<f4").reshape(lH, lW).copy()
                conf_np = np.frombuffer(conf_raw,  dtype=np.uint8).reshape(lH, lW).copy()

                jpeg_out, center, min_d, max_d = _run_infer(rgb_np, lo_m, conf_np)
                if jpeg_out is None:
                    continue

                header = struct.pack(">Ifff", len(jpeg_out), center, min_d, max_d)
                await websocket.send_bytes(header + jpeg_out)
            except Exception as e:
                print(f"[WARN] frame error (skipping): {e}")

    except WebSocketDisconnect:
        pass
