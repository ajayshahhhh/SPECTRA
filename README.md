# SPECTRA
### Sparse-to-dense dEpth CompleTion with Rgb-guided Arkit upsampling

SPECTRA turns the iPhone Pro's sparse LiDAR signal into a dense, metric-scale depth map. It's two pieces:

- **[SPECTRANet](SPECTRANet/)** — an RGB-guided depth upsampler trained on ARKitScenes. Predicts a residual on top of a bicubic baseline; ships as both a PyTorch checkpoint and a ~2 MB fp16 CoreML model.
- **[SPECTRALive](SPECTRALive/)** — a SwiftUI / ARKit iOS app that streams the iPhone's LiDAR depth, overlays a colormap on the camera feed, and lets you capture RGB+depth composites for sharing.

---

## Motivation

The iPhone 15 Pro's scene depth is metric but sparse — 192×256, with a per-pixel confidence mask. Plain bicubic upsampling gets you to camera resolution but smears edges and ignores the RGB image entirely. SPECTRANet uses MobileNetV2 features from the RGB frame to sharpen the upsample along real surface boundaries, while keeping the bicubic baseline as an anchor so it stays metrically faithful.

The model is small and CoreML-friendly on purpose: the goal is something that runs on-device, not a server round-trip.

---

## Pipeline

```
iPhone wide RGB (1440x1920)  +  LiDAR depth (192x256, mm)  +  confidence (192x256, {0,1,2})
                          │
            confidence-2 mask, mm → meters
                          │
                          ▼
                  Bicubic baseline (target res)
                          │
                          ▼
       ┌──────────────────────────────────────────┐
       │  SPECTRANet (RGBGuidedDepthUpsampler)    │
       │  ─ MobileNetV2 RGB encoder (4 taps)      │
       │  ─ Lightweight depth encoder (5 taps)    │
       │  ─ Decoder fuses RGB+depth at each scale │
       │  ─ Outputs scalar residual               │
       └──────────────────────────────────────────┘
                          │
                          ▼
                  bicubic + 0.2 * residual
                          │
                          ▼
              Dense metric depth (768x1024, m)
```

The decoder's final conv is zero-initialized, so the model starts as the bicubic identity and learns deltas from there.

---

## SPECTRANet

Self-contained inference bundle in [SPECTRANet/mvg_laptop_bundle/](SPECTRANet/mvg_laptop_bundle/).

**Model.** `RGBGuidedDepthUpsampler` ([mvg/model.py](SPECTRANet/mvg_laptop_bundle/mvg/model.py)) — MobileNetV2 (ImageNet-pretrained) feature taps at strides 2/4/8/16, parallel depth+confidence encoder at strides 1/2/4/8/16, decoder that bilinearly upsamples and concatenates RGB and depth features at each scale.

**Inputs (all aligned to 768×1024).**

| Field | Shape | Notes |
|---|---|---|
| `rgb` | `(1, 3, 768, 1024)` float32 | ImageNet-normalized |
| `bicubic_norm` | `(1, 1, 768, 1024)` float32 | LiDAR bicubic-upsampled, divided by 10.0, zero where confidence < 2 |
| `conf_hi` | `(1, 1, 768, 1024)` float32 | `(confidence == 2)` bilinear-upsampled |

**Output.** `pred_norm` ∈ `[0, 1]`; multiply by `DEPTH_MAX = 10.0` for meters.

**Run on a sample frame.**
```bash
cd SPECTRANet/mvg_laptop_bundle
pip install -r requirements.txt

python infer.py \
  --ckpt best.pt \
  --rgb sample_inputs/rgb.png \
  --lowres sample_inputs/lowres_depth.png \
  --conf sample_inputs/confidence.png \
  --out output_depth.png
```

This writes `output_depth.png` (turbo colormap visualization) and `output_depth.npy` (float32 metric depth, HxW).

**iOS deployment.** [`mvg_depth.mlpackage`](SPECTRANet/mvg_laptop_bundle/mvg_depth.mlpackage) is the same network in fp16 (~2 MB, iOS 17+). Drag into Xcode to inspect the I/O spec.

---

## SPECTRALive

iOS app (Xcode project in [SPECTRALive/](SPECTRALive/)). Requires a LiDAR-equipped iPhone Pro / iPad Pro.

**Two modes** (entry point: [HomeView.swift](SPECTRALive/SPECTRALive/HomeView.swift)):
- **Live Depth** — runs `ARWorldTrackingConfiguration` with `.sceneDepth` semantics, processes each `ARFrame` at 15 Hz, and overlays an HSV colormap of the depth map on top of the camera feed. Displays the center-pixel distance and a min/max depth scale. Tap the shutter to save a composite RGB+depth PNG to Documents and share it.
- **ML Depth** — placeholder for the on-device SPECTRANet inference path; the CoreML model exists in the repo but isn't wired into the app yet.

Key files:
- [DepthProcessor.swift](SPECTRALive/SPECTRALive/DepthProcessor.swift) — pulls `sceneDepth.depthMap` and `confidenceMap`, masks out confidence-0 / non-finite pixels, normalizes via vDSP, applies an HSV LUT.
- [CaptureManager.swift](SPECTRALive/SPECTRALive/CaptureManager.swift) — composites the RGB frame, depth overlay, and color scale into a shareable PNG.
- [ARViewContainer.swift](SPECTRALive/SPECTRALive/ARViewContainer.swift) — the AR session and frame delegate.

---

## Dataset

Trained on the [ARKitScenes](https://github.com/apple/ARKitScenes) upsampling split. Each sample provides four pre-aligned modalities — wide RGB (1440×1920), iPhone LiDAR depth (192×256, mm), confidence (192×256, {0,1,2}), and Faro laser-scan high-res depth (1440×1920, mm) as ground truth.

```bash
git clone https://github.com/apple/ARKitScenes.git
cd ARKitScenes
pip install -r requirements.txt
python3 download_data.py upsampling \
  --video_id_csv depth_upsampling/upsampling_train_val_splits.csv \
  --download_dir /data/arkitscenes/
```

Dataset preprocessing details (sky-direction rotation, confidence-aware filtering, residual baseline construction) live in [mvg/dataset.py](SPECTRANet/mvg_laptop_bundle/mvg/dataset.py).

---

## References

```bibtex
@inproceedings{dehghan2021arkitscenes,
  title={ARKitScenes: A Diverse Real-World Dataset for 3D Indoor Scene Understanding},
  author={Dehghan, Gilad and others},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2021}
}

@inproceedings{sandler2018mobilenetv2,
  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={CVPR},
  year={2018}
}
```

---

*Built at LA Hacks 2026.*
