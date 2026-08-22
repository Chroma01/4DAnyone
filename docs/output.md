# Output

With the default `--data_dir data`, one inference run writes:

```bash
data/
├── gvhmr/
│   └── results/<clip>/
│       ├── motion.json
│       └── motion.safetensors
└── fdanyone/
    └── <clip>/
        ├── metadata.json
        ├── cameras.json
        ├── skeletons/00.mp4 ... <N-1>.mp4
        └── videos/
            ├── sparse/{00,04,09,12,14,19}.mp4  # default 24-view RCP proposals
            └── dense/00.mp4 ... <N-1>.mp4
```

`<clip>` is the input filename without its extension. Results are written atomically; a run never overwrites an existing result, and temporary intermediates are removed after a successful run.

## GVHMR motion

`motion.safetensors` contains the global and input-camera SMPL parameters, per-frame camera intrinsics, and observed ViTPose keypoints recovered from the input. `motion.json` records the GVHMR revision, timeline, frame identity, raster, and coordinate convention. When generation is retried with the same input clip, the validated motion result is reused instead of recomputed.

## 4DAnyone videos

`videos/dense/` contains `views_per_layer × len(layer_pitches)` final views. Camera IDs are layer-major: every pitch layer contains the requested evenly spaced yaw views before the next layer begins. Yaw `0` faces the person's front, positive pitch places the camera above the subject, and the default camera `00` is therefore frontal. `views_per_layer` must be divisible by 4 or 6. Runs with at most six targets generate them directly; larger runs use RCP by default, with four or six proposal videos following `views_per_group` and stored in `videos/sparse/`. The first four proposals are used as target-stage references. `skeletons/` contains one synchronized Goliath40 conditioning video per final camera, with IDs matching `videos/dense/`. All videos contain 121 frames at 704×1280 and use the selected input clip FPS. In `auto` mode, the source FPS is preserved unless it can be evenly reduced to 24, 25, or 30 FPS. Audio is not copied.

Before generation, the canonical source frames are segmented by the pinned BiRefNet revision at its standard 1024×1024 inference resolution, using FP16 batches of four. Their union mask is cropped with a 4% margin and a center-contain policy. Projected input-camera anatomy, foreground support, and ViTPose agreement classify the source as full-body, half-body, or close-up. The generated views then share one sequence-level camera solve:

- the base radius protects the 95th-percentile body height and 80th-percentile body width around the full camera ring;
- confident half-body and close-up inputs additionally adapt target height and focal length;
- target and RCP conditioning use a center-aspect crop, while skeleton line width follows projected 3D body scale.

`metadata.json` records this preprocessing solve alongside the resolved layer pitches, yaw range, grouping, input timeline, motion and checkpoint identity, timings, and resource measurements. `cameras.json` stores each camera's layer, pitch, yaw, intrinsics, and pose in OpenCV camera coordinates and the canonical 4DAnyone Y-up human world.

## Nerfstudio export

Nerfstudio consumes images rather than MP4 files. Export one synchronized timestamp after inference:

```bash
python scripts/export_nerfstudio.py \
  --result_dir data/fdanyone/<clip> \
  --frame_index 60
```

This creates:

```bash
data/nerfstudio/<clip>/frame_060/
├── transforms.json
├── images/00.jpg ... <N-1>.jpg
└── masks/00.png ... <N-1>.png
```

The exporter runs the same pinned standard BiRefNet model used by inference, at 1024×1024 in FP16 batches of four. Missing BiRefNet files are downloaded automatically into `--model_dir` on first use. Its soft predictions are thresholded at 0.5 and written as single-channel binary PNGs: white pixels are foreground and black pixels are excluded from training.

`transforms.json` follows the [Nerfstudio `OPENCV` dataset schema](https://docs.nerf.studio/quickstart/data_conventions.html). Every frame contains both `file_path` and `mask_path`; Nerfstudio loads those masks automatically. Intrinsics are stored per image so each exported frame remains self-contained. Camera poses are converted from 4DAnyone's OpenCV/Y-up convention to Nerfstudio's OpenGL/Z-up camera-to-world convention. The directory can be passed directly to Splatfacto:

```bash
ns-train splatfacto \
  --data data/nerfstudio/<clip>/frame_060 \
  --pipeline.model.background-color random
```

The masks remove generated background pixels from the image loss. The explicit random rendering background also discourages transparent Gaussians from relying on one fixed composite color.

Each export is one static multi-view timestamp. Exporting all 121 frames into one vanilla Nerfstudio dataset would mix a moving person across time; dynamic methods should instead use a method-specific temporal data parser.
