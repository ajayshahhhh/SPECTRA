# SPECTRA
### Sparse-to-dense dEpth CompleTion with diffusion and ARkit

SPECTRA fine-tunes [Marigold](https://github.com/prs-eth/Marigold) — a diffusion-based monocular depth estimator — on ARKitScenes RGB+depth pairs, then runs [Marigold-DC](https://github.com/prs-eth/marigold-dc) at inference to produce dense, metric-scale depth maps guided by sparse iPhone LiDAR. The entire pipeline runs locally on the ASUS Ascent GX10.

---

## Motivation

Monocular depth models like Marigold produce stunning relative depth maps but lack metric scale — they can't tell you that the table is 1.2 meters away, only that it's closer than the wall. Meanwhile, iPhone 15 Pro's LiDAR sensor gives you metric scale but is sparse and low-resolution. SPECTRA bridges these two: it uses the rich generative prior of a diffusion model, fine-tuned on real indoor iPhone data, and anchors it to metric scale using the sparse LiDAR signal.

---

## Pipeline

```
iPhone RGB + Sparse LiDAR
         │
         ▼
  ┌─────────────┐
  │  Marigold   │  ← Fine-tuned on ARKitScenes
  │  (UNet)     │
  └─────────────┘
         │
         ▼
  Marigold-DC Inference
  (sparse LiDAR guidance)
         │
         ▼
  Dense Metric Depth Map (meters)
```

**Step 1 — Data Preprocessing**
Load the ARKitScenes upsampling split. Each scene provides high-res RGB, low-res sparse LiDAR depth (from iPhone ARKit), and high-res laser scan depth as ground truth. Pairs are aligned and normalized for training.

**Step 2 — Fine-tuning Marigold**
We fine-tune the Marigold UNet denoiser on ARKitScenes RGB+depth pairs, starting from the pretrained `prs-eth/marigold-depth-lcm-v1-0` checkpoint. The VAE encoder/decoder is frozen — only the denoiser is updated. We use a scale-and-shift invariant loss to handle the affine ambiguity in Marigold's training objective.

**Step 3 — Marigold-DC Inference**
At inference, Marigold-DC takes an RGB image and sparse LiDAR depth as input. The sparse depth acts as a guidance signal during the diffusion reverse process, anchoring the prediction to metric scale. Output is a dense float32 depth map in meters.

**Step 4 — Evaluation**
We evaluate against ARKitScenes high-res depth ground truth using standard metrics: AbsRel, RMSE, and δ1 accuracy.

---

## Dataset

[ARKitScenes](https://github.com/apple/ARKitScenes) — upsampling split. Publicly available, no institutional access required.

Download:
```bash
git clone https://github.com/apple/ARKitScenes.git
cd ARKitScenes
pip install -r requirements.txt

python3 download_data.py upsampling \
  --video_id_csv depth_upsampling/upsampling_train_val_splits.csv \
  --download_dir /data/arkitscenes/
```

---

## Hardware

Built and trained on the **ASUS Ascent GX10** — NVIDIA GB10 Grace Blackwell Superchip, 128GB unified LPDDR5x memory, 1 petaFLOP AI performance. The large unified memory pool makes fine-tuning a Stable Diffusion-based architecture practical without gradient checkpointing hacks.

---

## Interface Contract

| Field | Type | Description |
|---|---|---|
| `rgb` | `H x W x 3, uint8` | RGB image from iPhone camera |
| `sparse_depth` | `H x W, float32` | LiDAR depth in meters, 0 where missing |
| `dense_depth` (output) | `H x W, float32` | Dense depth map in meters |

---

## Setup

```bash
git clone https://github.com/your-org/spectra.git
cd spectra

conda create -n spectra python=3.10
conda activate spectra
pip install -r requirements.txt
```

**Fine-tune:**
```bash
python train.py \
  --data_dir /data/arkitscenes/ \
  --checkpoint prs-eth/marigold-depth-lcm-v1-0 \
  --output_dir ./checkpoints/
```

**Inference:**
```bash
python infer.py \
  --checkpoint ./checkpoints/best.ckpt \
  --rgb path/to/image.png \
  --sparse_depth path/to/lidar.npy \
  --output depth_out.npy
```

---

## References

```bibtex
@inproceedings{ke2024repurposing,
  title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author={Ke, Bingxin and Obukhov, Anton and Huang, Shengyu and Metzger, Nando and Daudt, Rodrigo Caye and Schindler, Konrad},
  booktitle={CVPR},
  year={2024}
}

@article{garcia2024marigolddc,
  title={Marigold-DC: Zero-Shot Monocular Depth Completion with Guided Diffusion},
  author={Garcia, Massimiliano and Guizilini, Vitor and Ambrus, Rares and Dai, Angela},
  year={2024}
}

@inproceedings{dehghan2021arkitscenes,
  title={ARKitScenes: A Diverse Real-World Dataset for 3D Indoor Scene Understanding},
  author={Dehghan, Gilad and others},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2021}
}
```

---

*Built at LA Hacks 2026*
