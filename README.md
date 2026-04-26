# SPECTRA
### Sparse-to-dense dEpth CompleTion via RGB-guided upsampling

SPECTRA turns the sparse, low-resolution depth map from a cheap consumer LiDAR into a dense, full-resolution, metric-scale depth map — entirely on-device. No cloud, no external compute, no special hardware beyond an iPhone 15 Pro.

---

## The Problem

Industrial LiDAR sensors (Velodyne, Ouster) deliver dense, reliable depth — but cost $8,000–$75,000 per unit. Apple's iPhone 15 Pro ships a solid-state LiDAR scanner at roughly $3–$25 in component cost, but the raw output is only 192×256 pixels: sparse, noisy, and too low-resolution for reliable scene understanding or robotics navigation.

SPECTRA closes this gap. The same $10 solid-state LiDAR module that would otherwise produce unusable depth becomes the foundation for a full dense depth map when paired with SPECTRANet.

---

## What We Built

**SPECTRANet** — a lightweight neural network that fuses the iPhone's sparse LiDAR with its RGB camera to produce dense, edge-accurate, metric depth maps at full camera resolution.

**SPECTRALive** — an iOS app (Swift + ARKit) that runs SPECTRANet entirely on-device via ZETIC Melange, streaming real-time dense depth from iPhone LiDAR + camera with zero cloud dependency.

---

## Architecture

SPECTRANet is a two-encoder, one-decoder network that predicts a residual correction on top of a bicubic baseline.

```
Inputs:
  rgb           (3, 768, 1024)   iPhone camera, ImageNet-normalized
  bicubic_norm  (1, 768, 1024)   LiDAR depth upsampled to full res, ÷10.0
  conf_hi       (1, 768, 1024)   ARKit confidence==2 mask, bilinear-upsampled

RGB Encoder (MobileNetV2, ImageNet pretrained)
  f1  stride 2   16ch
  f2  stride 4   24ch
  f3  stride 8   32ch
  f4  stride 16  96ch

Depth Encoder (lightweight CNN, 2-channel input: depth + conf)
  d0  stride 1   16ch
  d1  stride 2   16ch
  d2  stride 4   24ch
  d3  stride 8   32ch
  d4  stride 16  64ch

Fusion Decoder (FuseUpBlocks — bilinear upsample + concat + 2× conv)
  bot   [d4 + f4]         → 96ch   stride 16
  up3   [bot + f3 + d3]   → 64ch   stride 8
  up2   [up3 + f2 + d2]   → 48ch   stride 4
  up1   [up2 + f1 + d1]   → 32ch   stride 2
  up0   [up1 + d0]        → 16ch   stride 1

Head
  3×3 conv + BN + ReLU → 1×1 conv (zero-initialized) → residual Δ

Output
  pred_norm = bicubic_norm + 0.2 × Δ
  pred_m    = pred_norm × 10.0        (meters, range 0.5–10m)
```

**Key design choices:**
- MobileNetV2 chosen over ResNet for clean CoreML/ZETIC Melange export to Apple Neural Engine
- Residual learning (predicting Δ not absolute depth) makes training stable from epoch 0
- ARKit confidence filtering: pixels below confidence 2 are zeroed before training — the model never learns from unreliable LiDAR readings (glass, dark surfaces, mirrors)
- The confidence mask is passed as a second depth encoder channel, explicitly signaling where LiDAR is trustworthy vs. where the model should rely on RGB

**Training loss:**
```
total = 1.0 × L1(pred_m, gt, valid)
      + 0.05 × edge_aware_smoothness(pred_m, rgb)
      + 0.10 × (1 - SSIM)(pred_m, gt, valid)
```

**Optimizer:** AdamW — decoder + depth encoder at lr=1e-4, RGB encoder at lr=2e-5. RGB encoder frozen for first 5 epochs. CosineAnnealingLR. bf16 AMP.

---

## Dataset

**ARKitScenes** (upsampling split) — publicly available, no institutional access required.

Each scene provides four aligned modalities:

| Modality | Resolution | Source |
|---|---|---|
| Wide RGB | 1440×1920×3 | iPhone wide camera |
| High-res depth (GT) | 1440×1920 | Faro laser scanner |
| Low-res depth | 192×256 | iPhone LiDAR |
| Confidence | 192×256 | ARKit |

Download:
```bash
git clone https://github.com/apple/ARKitScenes.git
cd ARKitScenes
pip install -r requirements.txt
python3 download_data.py upsampling \
  --video_id_csv depth_upsampling/upsampling_train_val_splits.csv \
  --download_dir /data/arkitscenes/
```

**Scope:** Indoor scenes, 0.5–10m. Not designed for outdoor use or ranges beyond 10m.

---

## Results

Evaluated on the ARKitScenes upsampling validation set. All metrics over valid pixels (GT depth in 0.5–10m).

| Method | RMSE↓ | MAE↓ | δ1↑ | δ2↑ | δ3↑ |
|---|---|---|---|---|---|
| Bicubic baseline | 0.2171 | 0.0730 | 0.9476 | 0.9550 | 0.9630 |
| Marigold-DC (zero-shot) | 0.0989 | 0.0724 | 0.9730 | 0.9973 | 0.9995 |
| Marigold-DC (LoRA) | 0.1007 | 0.0735 | 0.9714 | 0.9970 | 0.9995 |
| SPECTRANet (L1 only) | 0.0709 | 0.0257 | 0.9851 | 0.9953 | 0.9984 |
| **SPECTRANet (ours)** | **0.0552** | **0.0216** | **0.9909** | **0.9973** | **0.9991** |

**δ1 = 0.9909** — 99.1% of pixels are within ±25% of ground truth depth. SPECTRANet achieves 75% lower RMSE than bicubic, 1.8× better RMSE than Marigold-DC, and 3.4× better MAE than Marigold-DC.

*δi = fraction of valid pixels where max(pred/gt, gt/pred) < 1.25^i. Higher is better. RMSE and MAE in meters, lower is better.*

---

## Hardware

### Training — ASUS Ascent GX10

SPECTRANet was trained on the full ARKitScenes dataset on the GX10's NVIDIA GB10 Grace Blackwell Superchip. The 128GB unified LPDDR5x memory pool eliminated gradient checkpointing — full batch sizes, full resolution, full speed. 1 petaFLOP AI performance, Ubuntu, PyTorch + bf16 AMP.

### Deployment — ZETIC Melange + iPhone 15 Pro

After training, `best.pt` was uploaded to [ZETIC Melange](https://mlange.zetic.ai), which automatically converted it to an NPU-optimized binary for Apple's Neural Engine and generated the Swift SDK (`Linjfeng/SPECTRA`, version 1). SPECTRALive loads the model with:

```swift
let model = try ZeticMLangeModel(
    personalKey: "dev_780921dd4eb84275a5d431dd8dcb90b5",
    name: "Linjfeng/SPECTRA",
    version: 1,
    modelMode: ModelMode.RUN_AUTO
)
```

Runs entirely on the A17 Pro Neural Engine. No GX10, no internet, no cloud at inference time. ~2MB on-device.

---

## Setup — Python Inference

```bash
git clone https://github.com/ajayshahhhh/SPECTRA.git
cd SPECTRA
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Run on a single frame:
```bash
python infer.py \
  --ckpt best.pt \
  --rgb sample_inputs/rgb.png \
  --lowres sample_inputs/lowres_depth.png \
  --conf sample_inputs/confidence.png \
  --out output_depth.png
```

Outputs `output_depth.png` (turbo colormap) and `output_depth.npy` (float32 meters).

**Input format:**
- `--rgb` — any resolution PNG, resized internally to 768×1024
- `--lowres` — 192×256 uint16 PNG, millimeters
- `--conf` — 192×256 uint8 PNG, values in {0, 1, 2}

---

## Setup — SPECTRALive (iOS)

Requirements: Xcode 15+, iPhone 15 Pro (LiDAR required), iOS 17+

1. Open `SPECTRALive/SPECTRALive.xcodeproj`
2. File → Add Package Dependencies → `https://github.com/zetic-ai/ZeticMLangeiOS.git` → exact version **1.6.0**
3. Select your iPhone as build target, sign with your Apple ID under Signing & Capabilities
4. Build in **Release** mode for best inference FPS: Edit Scheme → Run → Build Configuration → Release
5. ⌘R to build and run

On first launch the model downloads to the device. After that it runs fully offline.

---

## Applications

| Use Case | Description |
|---|---|
| Home Robotics | Dense 3D obstacle map from a $10 sensor — furniture legs, pets, dropped objects |
| AR Occlusion | Per-pixel metric depth for physically correct AR object placement |
| Accessibility | Real-time metric distance warnings ("obstacle 0.8m ahead") on any iPhone Pro |
| Robot Grasping | Edge-accurate depth at object boundaries for reliable grasp pose estimation |
| Smart Home Security | 3D shape-based presence detection — distinguishing a pet from a person |
| Prosthetic Vision | Real-time depth feedback for prosthetic arm grasp control without a dedicated rig |

---

## References

1. Velodyne "HDL-64E"; Ouster "OS1". Industrial LiDAR retail $8k–$75k (2020–2023).
2. Frumusanu, A. AnandTech, 2020; iFixit teardown. Apple solid-state LiDAR BOM est. $3–$25.
3. Song et al. "Depth Completion with Twin Surface Extrapolation at Occlusion Boundaries." CVPR 2021.
4. Dehghan et al. "ARKitScenes: A Diverse Real-World Dataset for 3D Indoor Scene Understanding." NeurIPS Datasets & Benchmarks, 2021.
5. iRobot. "Roomba Product Specifications." irobot.com.

---

## Team

Built in 36 hours at LA Hacks 2026 — Stanford University.

| Name | Role |
|---|---|
| Ajay Shah | iOS app (SPECTRALive), website, system integration |
| William Wang | Model architecture, training pipeline |
| Benjamin Jiang | Dataset preprocessing, model training on GX10 |
| Junfeng Lin | ZETIC Melange deployment, CoreML conversion |
