# OxFlow: Longitudinal CT Tumor Synthesis with OT Masks + Flow Matching

Official implementation of *Spatiotemporal Modeling of Tumor Dynamics via Optimal Transport for
Enhanced Longitudinal Analysis*, MICCAI 2026.

A generative framework for synthesizing intermediate tumor CT slices between two longitudinal
timepoints. A tumor mask at an intermediate time is obtained by optimal-transport (OT)
interpolation of the endpoint masks, and a mask-conditioned ControlNet trained with flow matching
generates the corresponding CT image. The result is a densely sampled, geometrically consistent
tumor trajectory from sparsely sampled follow-up scans.

```
Longitudinal CT + masks ──► preprocess ──► longitudinal_data.pt
                                                 │
                                                 ▼
                                 train ControlNet (flow matching)
                                                 │
                                                 ▼
                  OT mask interpolation + flow-matching ODE sampling
                                                 │
                                                 ▼
                                      synthetic_data.pt
```

This repository contains code only. No patient data, labels, or trained weights are included.

---

## Installation

```bash
git clone <this-repo> oxflow && cd oxflow
conda create -n oxflow python=3.12 -y && conda activate oxflow
pip install -r requirement.txt
pip install -e .
```

Tested with Python 3.12 and PyTorch 2.9. All commands below are run from the repository root.
`src/` holds the library (`data`, `models`, `ot`, `train`, `inference`) and `script/` the
command-line entry points used below.

---

## Data Preparation

### Expected input layout

One directory per patient, with co-registered CT volumes and binary tumor masks per timepoint:

```
data/
└── <patient_id>/                # integer ID
    ├── img/
    │   ├── <patient_id>_1_<scan>_img.nii.gz
    │   ├── <patient_id>_2_<scan>_img.nii.gz
    │   └── ...
    └── msk/
        ├── <patient_id>_1_<scan>_gt.nii.gz
        └── ...
```

The timepoint index is parsed from the filename by `extract_timepoint_from_filename` in
[preprocess_slices.py](src/data/preprocess_slices.py). Adapt that function if your naming differs.
All timepoints of a patient must have the same number of axial slices (register volumes first).

### Preprocess

```bash
python -m src.data.preprocess_slices \
    --data_root data \
    --output data_processed/longitudinal_data.pt
```

CT intensities are clipped to [-1000, 400] HU, rescaled to [-1, 1] and resized to 256×256.

### Tensor format

`longitudinal_data.pt` is a list of dicts, one per (patient, slice position):

| Key | Type | Description |
|-----|------|-------------|
| `patient_id` | int | patient ID |
| `slice_idx` | int | axial slice index |
| `timepoints` | list[int] | sorted timepoint indices |
| `ct_slices` | tensor (T, 1, H, W) | CT at every timepoint |
| `masks` | tensor (T, 1, H, W) | tumor mask at every timepoint |
| `has_tumor` | bool | tumor present at any timepoint |
| `has_tumor_per_timepoint` | list[bool] | per-timepoint tumor flag |

`LongitudinalCTDataset` in [dataset_slices.py](src/data/dataset_slices.py) reads this file:

```python
from src.data.dataset_slices import LongitudinalCTDataset

ds = LongitudinalCTDataset('data_processed/longitudinal_data.pt', mode='eval', filter_no_tumor=True)
pid = ds.get_available_patients()[0]
s = ds.get_patient_slices(pid)[0]
s['ct_series'], s['mask_series'], s['timepoints']   # (T,1,256,256), (T,1,256,256), [1, 2, ...]
```

With `mode='train'` each item is instead a single `(ct, mask)` pair at a random timepoint, which
is what the generative model is trained on.

---

## 1. Train the ControlNet (flow matching)

```bash
python -m src.train.train_controlnet_fm \
    --data_path data_processed/ \
    --ckpt_path ckpts/ \
    --epochs 300 --batch_size 16 --lr 1e-4
```

Conditioning is `concat(mask, masked_background)` where the tumor region is filled with the mean
value. The loss is velocity MSE with 2× weight inside the tumor region. An EMA copy of the
weights is kept. Checkpoints are written to `ckpts/controlnet_fm_epoch{N}.pth` every
`--save_interval` epochs plus `ckpts/controlnet_fm_latest.pth`. Add `--wandb` for logging.
Time sampling, EMA decay and the other hyper-parameters are exposed as flags; see `--help`.

---

## 2. Generate synthetic intermediate timepoints

For every (patient, slice) with consecutive tumor-bearing timepoints, interpolate the mask with
OT and inpaint a CT at each requested fraction of the interval:

```bash
python script/generate/generate_synthetic_data.py \
    --data_path data_processed/longitudinal_data.pt \
    --ckpt_path ckpts/controlnet_fm_latest.pth \
    --output_dir data_processed/synthetic \
    --interp_steps 0.25 0.5 0.75
```

Output `synthetic_data.pt` is a list of dicts with `ct`, `mask`, `patient_id`, `slice_idx`,
`interp_t` and `pair` (the two source timepoints). Generation is resumable through
`manifest.json` in the output directory.

Solver settings (`--steps`, `--method`, `--controlnet_scale`) and batching are flags; see `--help`.

### Ablations and baselines

Same model with a different mask interpolation, and a pixel-space baseline with no generative
model. The evaluation scripts below look for these output directories by default.

```bash
python script/generate/generate_synthetic_data.py ... --interp_method linear --output_dir data_processed/synthetic_linear_mask
python script/generate/generate_synthetic_data.py ... --interp_method random --output_dir data_processed/synthetic_random_mask
python script/generate/generate_pixel_interp.py \
    --data_path data_processed/longitudinal_data.pt --output_dir data_processed/synthetic_pixel_interp
```

### OT mask interpolation as a library

```python
from src.ot.interpolation import compute_ot_interpolated_mask

mask_t = compute_ot_interpolated_mask(mask_t0, mask_t1, t=0.5)   # (1, H, W) tensors in, binary mask out
```

---

## 3. Evaluation

Three complementary checks: does a segmentation model find the tumor where the conditioning mask
put it, how close is the image to the real follow-up scan, and is the tumor volume trajectory
smooth.

### Geometric fidelity via segmentation (nnU-Net Dice)

Requires a 2D nnU-Net trained on your real slices first. `script/train/prepare_nnunet_data.py`
converts the tensor file into an nnU-Net dataset, and its docstring lists the training commands.

```bash
python script/eval/eval_segmentation_dice.py \
    --synthetic_path data_processed/synthetic/synthetic_data.pt \
    --dataset_id 1 --config 2d --fold 0 --output_dir outputs/eval_dice_synthetic
```

`script/eval/compare_dice.py` then tabulates several methods with a paired test.

### Image similarity to the real follow-up

```bash
python script/eval/compare_ssim.py \
    --data_path data_processed/longitudinal_data.pt --region tumor
```

### Mask trajectory smoothness

```bash
python script/eval/compute_trajectory_metrics.py \
    --data_path data_processed/longitudinal_data.pt --output_dir outputs/trajectory_metrics --plot
```

### Qualitative figure

OT masks, OT + FM and random-matching + FM for one patient:

```bash
python script/visualize/visualize_method_comparison.py \
    --patient_id <PATIENT_ID> --fm_ckpt ckpts/controlnet_fm_latest.pth
```

## Method notes

**Flow matching.** With `z = t·x + (1-t)·ε`, the network predicts the clean image `x̂` and
the velocity `v = (x̂ - z) / (1-t)` drives the ODE from noise (t=0) to image (t=1), integrated
with 50 Heun steps. The whole image is generated from the conditioning `concat(mask, masked_bg)`;
the background is not re-imposed during sampling.

**OT mask interpolation.** Each mask is converted to a point cloud, points are matched between
the two endpoints by optimal transport, and every matched pair is linearly displaced to time `t`
before rasterising back to a mask. `linear` (pixel-wise blend) and `random` (random matching)
are available as ablations.

---

## License

This code is made publicly available under the PolyForm Noncommercial License 1.0.0 for
non-commercial research, academic, and educational use. Commercial use is not permitted under
this license. See [LICENSE](LICENSE) for the full terms and [NOTICE](NOTICE) for the required
copyright notice.

This software is for research use only. It is not a medical device, has not been validated for
clinical use, and must not be used for diagnosis or treatment decisions. Synthetic images produced
by this code are model outputs, not observations of a patient, and must be labelled as such in any
downstream use.
