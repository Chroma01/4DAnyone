<p align="center"><a href="https://4danyone.github.io/"><img src="docs/assets/logo_title.png" width="300" alt="4DAnyone"></a></p>

<h2 align="center">4DAnyone: Create Anyone in 4D from a Casual Monocular Video</h2>

<p align="center"><a href="https://4danyone.github.io/"><strong>Project Page</strong></a> &nbsp;|&nbsp; <a href="https://arxiv.org/abs/2608.20335"><strong>Paper</strong></a></p>

<p align="center"><img src="docs/assets/teaser.gif" width="100%" alt="4DAnyone teaser"></p>

<p align="center">4DAnyone turns a casual monocular video into multi-view videos, enabling downstream 4DGS reconstruction.</p>

## Installation

```bash
git clone https://github.com/ant-research/4DAnyone.git
cd 4DAnyone
git submodule update --init third_party/GVHMR

conda create -n 4danyone python=3.11 -y
conda activate 4danyone
pip install -r requirements.txt
```

For faster inference, optionally install [FlashAttention-3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) or [SageAttention](https://github.com/thu-ml/SageAttention).

Missing models and examples are downloaded automatically on first use. You can also download them manually:

```bash
python scripts/download_smplx.py
python scripts/download_model.py
python scripts/download_example.py
```

## Inference

### Generate target views

```bash
# Generate 6 evenly spaced views around the subject
python inference.py \
    --video_path "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4" \
    --views_per_layer 6

# Generate 24 evenly spaced views around the subject
python inference.py \
    --video_path "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4" \
    --views_per_layer 24

# Generate 48 views with 3 pitch layers and 16 views per layer
python inference.py \
    --video_path "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4" \
    --views_per_layer 16 --layer_pitches '[-10,15,35]'

# Generate 8 views over the frontal 180-degree arc
python inference.py \
    --video_path "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4" \
    --views_per_layer 8 --start_yaw -90 --yaw_span 180

# Generate one view at pitch 15° and yaw 60°
python inference.py \
    --video_path "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4" \
    --views_per_layer 1 --layer_pitches '[15]' --start_yaw 60
```

Camera layout options:

- `layer_pitches`: pitch angle of each camera layer, in degrees; positive values place the cameras above the person.
- `views_per_layer`: number of evenly spaced views in each camera layer. Total views are `views_per_layer × len(layer_pitches)`.
- `start_yaw`: horizontal angle of the first view, in degrees; yaw `0` is the front view.
- `yaw_span`: horizontal range covered by each camera layer, in degrees.

Run `python inference.py --help` to see all inference options.

### Output

With the default `--data_dir data`, results follow this layout. See the [output documentation](docs/output.md) for the complete format.

```text
data/
├── gvhmr/results/<clip>/          # reusable motion-recovery result
└── fdanyone/<clip>/
    ├── metadata.json              # run settings, timings, resources
    ├── cameras.json               # the final N-camera rig
    ├── skeletons/00.mp4 ... <N-1>.mp4
    └── videos/
        ├── sparse/{00,04,09,12,14,19}.mp4  # default 24-view RCP proposals
        └── dense/00.mp4 ... <N-1>.mp4       # generated target views
```

### Custom data

Use a handheld-like video that:

- has a portrait 9:16 aspect ratio;
- shows one person, either full body or upper body;
- contains at least 121 frames;
- has only mild camera movement.

## 3DGS Reconstruction

Install [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) before running `ns-train`.

```bash
# Export frame 0 across all generated views
python scripts/export_nerfstudio.py \
    --result_dir data/fdanyone/<clip> \
    --frame_index 0

# Train foreground-only 3DGS
ns-train splatfacto \
    --data data/nerfstudio/<clip>/frame_000 \
    --pipeline.model.background-color random
```

## Todos

- [ ] Low-memory inference (<32 GB)
- [ ] Faster inference with TensorRT and sparse attention
- [ ] Support 4DGS reconstruction with an open-source method

## Citation

If you find 4DAnyone useful or interesting, please cite our work and consider giving the repository a star ⭐:

```bibtex
@article{jin2026fdanyone,
  title={4DAnyone: Create Anyone in 4D from a Casual Monocular Video},
  author={Jin, Yudong and Xie, Tao and Zhang, Qihang and Shen, Zehong and Xu, Zhen and Shen, Yujun and Bao, Hujun and Zhou, Xiaowei and Xu, Yinghao},
  journal={arXiv preprint arXiv:2608.20335},
  year={2026},
  url={https://arxiv.org/abs/2608.20335}
}
```
