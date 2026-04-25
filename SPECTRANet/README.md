# mvg laptop demo bundle

Self-contained inference setup: take a (RGB, lowres iPhone-LiDAR depth,
confidence) triplet and produce a refined high-resolution metric depth map.

## Setup

    python -m venv venv
    source venv/bin/activate          # or venv\Scripts\activate on Windows
    pip install -r requirements.txt

## Run on a sample frame

    python infer.py \
      --ckpt best.pt \
      --rgb sample_inputs/rgb.png \
      --lowres sample_inputs/lowres_depth.png \
      --conf sample_inputs/confidence.png \
      --out output_depth.png

This writes `output_depth.png` (a colormap visualization) and
`output_depth.npy` (the metric depth in meters, float32, shape HxW).

## Run on your own iPhone capture

The model expects the four ARKit modalities, all aligned:
- RGB: any resolution; will be resized internally to 768x1024
- lowres depth: 192x256, uint16 PNG, values in millimeters
- confidence: 192x256, uint8 PNG, values in {0, 1, 2}

If you have an ARKit capture from an iPhone Pro / iPad Pro, you already
have all three (ARFrame.capturedImage, sceneDepth.depthMap,
sceneDepth.confidenceMap).

## iPhone (CoreML) deployment

`mvg_depth.mlpackage` is the same model converted for iOS (fp16, ~2 MB,
minimum_deployment_target=iOS17). Drag it into Xcode to inspect the
input/output spec.

Inputs (all Float32 MultiArrays):
- `rgb`           (1, 3, 768, 1024)  ImageNet-normalized: (x/255 - mean) / std,
                                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- `bicubic_norm`  (1, 1, 768, 1024)  LiDAR depth in meters, bicubic-upsampled
                                     from 256x192 to 1024x768, divided by 10.0,
                                     zero where confidence < 2
- `conf_hi`       (1, 1, 768, 1024)  (confidence == 2).float() bilinear-upsampled

Output:
- `pred_norm`     (1, 1, 768, 1024)  depth in [0,1]; multiply by 10.0 for meters
