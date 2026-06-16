# BoneSeg — nnUNet Bone Segmentation

Train a bone segmentation model using [nnUNet v2](https://github.com/MIC-DKFZ/nnUNet)
on [Total Segmentator](https://github.com/wasserth/TotalSegmentator) annotated CT data.

---

## Project Structure

```
BoneSeg/
├── data/                       # Raw Total Segmentator data
│   ├── s0000.nii.gz            #   CT image
│   ├── s0000-seg.nii.gz        #   Segmentation mask (104-class Total Segmentator)
│   └── ...
├── configs/
│   ├── default.yaml            # ← Main experiment config (3-D CT)
│   ├── 2d.yaml                 # ← 2-D config (ultrasound / endoscopy)
│   └── labels.yaml             # ← Total Segmentator label definitions
├── src/
│   ├── __init__.py
│   ├── utils.py                # Path / logging / env helpers + data_type switch
│   ├── label_config.py         # Label remapping logic (3-D)
│   ├── data_prep.py           # nnUNet dataset preparation (3-D)
│   ├── data_prep_2d.py         # nnUNet dataset preparation (2-D images)
│   ├── image_io.py             # 2-D image/mask IO (jpg/png ↔ nnUNet PNG)
│   └── slices_builder.py       # 2.5-D slice extraction and multi-channel IO
├── scripts/
│   ├── inspect_labels.py       # Utility: inspect actual label values in data
│   ├── 01_prepare_data.py      # Step 1: prepare nnUNet dataset (3-D)
│   ├── 01b_prepare_data_25d.py # Step 1b: prepare 2.5-D multi-channel dataset
│   ├── 02_train.py             # Step 2: train 3-D model
│   ├── 02_train_25d.py         # Step 2b: train 2.5-D model
│   ├── 03_predict.py           # Step 3: inference (3-D)
│   ├── 03_predict_25d.py       # Step 3b: inference (2.5-D)
│   └── 04_evaluate.py          # Step 4: evaluate predictions
└── nnunet_workspace/           # Created automatically
    ├── nnUNet_raw/             #   nnUNet input data
    ├── nnUNet_preprocessed/    #   nnUNet preprocessed data
    └── nnUNet_results/         #   Training results / checkpoints
```

---

## Prerequisites

### 1. Install nnUNet v2

```bash
conda create -n boneseg python=3.10 -y
conda activate boneseg
pip install nnunetv2
pip install nibabel scipy pyyaml
```

> **Note**: nnUNet v2 requires PyTorch with CUDA. Install the appropriate
> PyTorch version for your GPU from https://pytorch.org before running pip.

### 2. Verify your data

```
data/
  s0000.nii.gz        — CT image
  s0000-seg.nii.gz    — Total Segmentator segmentation (104 classes)
  s0001.nii.gz
  s0001-seg.nii.gz
  ...
```

---

## Quick Start

All steps are run from the project root (`D:\codes\work-projects\BoneSeg\`).

### Step 0 — Inspect label values (recommended first step)

Verify that the label IDs in `configs/labels.yaml` match the actual values
in your segmentation files:

```bash
python scripts/inspect_labels.py --config configs/default.yaml
```

The output shows which Total Segmentator label IDs are present in each case
and cross-references them with the bone groups defined in `labels.yaml`.
Adjust `bone_groups` / `active_groups` in `configs/labels.yaml` if needed.

---

### Step 1 — Prepare nnUNet dataset

```bash
python scripts/01_prepare_data.py --config configs/default.yaml
```

This script:
1. Reads raw CT images and segmentation masks
2. Remaps Total Segmentator labels → target classes (binary or multi-class)
3. Writes nnUNet-formatted `imagesTr/` and `labelsTr/` directories
4. Generates `dataset.json`
5. Optionally runs `nnUNetv2_plan_and_preprocess`

Use `--dry_run` to preview without writing files:

```bash
python scripts/01_prepare_data.py --config configs/default.yaml --dry_run
```

Use `--auto_preprocess` to skip the interactive prompt and run preprocessing automatically (useful for CI/CD):

```bash
python scripts/01_prepare_data.py --config configs/default.yaml --auto_preprocess
```

---

### Step 2 — Train

```bash
python scripts/02_train.py --config configs/default.yaml
```

Options:

| Flag | Description |
|------|-------------|
| `--fold 0` | Train a specific fold (0–4, default: from config) |
| `--all_folds` | Train all 5 folds sequentially |
| `--preprocess_only` | Only run preprocessing, skip training |
| `--skip_preprocess` | Skip preprocessing (already done) |
| `--continue_training` | Resume from latest checkpoint |

Example — train all folds for ensemble inference:

```bash
python scripts/02_train.py --config configs/default.yaml --all_folds
```

---

### Step 3 — Predict

```bash
# From a folder of raw CT images (auto-renamed to *_0000.nii.gz):
python scripts/03_predict.py \
    --config configs/default.yaml \
    --input_raw data/ \
    --output predictions/

# From already-formatted nnUNet images (*_0000.nii.gz):
python scripts/03_predict.py \
    --config configs/default.yaml \
    --input /path/to/images_formatted \
    --output predictions/

# Ensemble all folds:
python scripts/03_predict.py \
    --config configs/default.yaml \
    --input_raw data/ \
    --output predictions/ \
    --fold all
```

---

### Step 4 — Evaluate

```bash
python scripts/04_evaluate.py \
    --config configs/default.yaml \
    --pred_dir predictions/ \
    --gt_dir   data/
```

Outputs:
- Per-case Dice coefficient and HD95 (Hausdorff distance 95th percentile)
- Summary table printed to console
- `predictions/evaluation_metrics.csv`

Use `--no_hd95` to skip Hausdorff distance computation (faster).

---

## Configuration Reference

### `configs/default.yaml` — key parameters

| Section | Key | Description |
|---------|-----|-------------|
| `paths` | `raw_data_dir` | Raw Total Segmentator data folder |
| `paths` | `nnunet_workspace` | nnUNet workspace root |
| `dataset` | `id` | nnUNet dataset ID (integer) |
| `dataset` | `name` | Dataset name (used in directory name) |
| `segmentation` | `mode` | `"binary"` (all bones→1) or `"multiclass"` (per-group) |
| `preprocessing` | `num_processes` | Parallel processes for preprocessing |
| `training` | `configuration` | `"3d_fullres"`, `"3d_lowres"`, `"2d"` |
| `training` | `fold` | CV fold to train (0–4) |
| `training` | `trainer` | nnUNet trainer class name |
| `inference` | `fold` | Fold(s) for prediction: `0` or `"all"` |
| `inference` | `checkpoint` | `"checkpoint_best.pth"` or `"checkpoint_final.pth"` |
| `inference` | `disable_tta` | Disable test-time augmentation |

### `configs/labels.yaml` — key sections

| Key | Description |
|-----|-------------|
| `all_labels` | Complete Total Segmentator v1 label map (reference) |
| `bone_groups` | Named bone groups with source label IDs |
| `active_groups` | Which groups to include in segmentation |
| `multiclass_mapping` | Group → class ID mapping for multi-class mode |

---

## Segmentation Modes

### Binary (default)
All active bone groups → label 1. Network predicts bone vs. background.

```yaml
segmentation:
  mode: "binary"
```

Output labels: `{background: 0, bone: 1}`

### Multi-class
Each bone group gets its own output class. More detailed but harder to learn.

```yaml
segmentation:
  mode: "multiclass"
```

Output labels (default):
```
0: background
1: vertebrae
2: ribs
3: sternum
4: pelvis
5: shoulder_girdle
6: femur
```

---

## nnUNet Training Configurations

| Config | Description | GPU RAM |
|--------|-------------|---------|
| `2d` | 2D U-Net per slice | ~4 GB |
| `3d_lowres` | 3D U-Net, down-sampled patches | ~8 GB |
| `3d_fullres` | 3D U-Net, full-resolution patches | ~16 GB |
| `3d_cascade_fullres` | 3d_lowres → 3d_fullres cascade | ~16 GB + time |

For bone segmentation with CT images, **`3d_fullres` is recommended** as bones
are three-dimensional structures. Use `3d_lowres` if you run out of GPU memory.

---

## 2-D Image Segmentation (ultrasound / endoscopy)

The same pipeline can segment **native 2-D images** (e.g. ultrasound or
endoscopy `.jpg`/`.png`) instead of 3-D CT volumes. This is selected purely by
the config: setting `data_type: "2d"` switches every step
(`01`/`02`/`03`/`04`) onto the 2-D code paths. The 3-D behaviour is unchanged
when `data_type` is absent (it defaults to `"3d"`).

### Input layout

Images and masks live in two parallel directory trees with **identical
relative paths** — the tree can be arbitrarily nested:

```
/data0/train/images/C847675/CS20250714439/1/<uid>.jpg     ← image
/data0/train/mask/C847675/CS20250714439/1/<uid>.jpg       ← annotation mask
```

Every leaf image becomes one nnUNet training case. The case name is derived
from its path relative to the image root (e.g.
`C847675-CS20250714439-1-<uid>`), and a `manifest.json` is written next to the
dataset mapping each case back to its source image/mask.

### How the adaptation works

- **No Total Segmentator remapping** — 2-D masks are already foreground/
  background, so `labels` are declared directly in `configs/2d.yaml`.
- **nnUNet native 2-D format** — images are written as `imagesTr/{case}_0000.png`
  and labels as `labelsTr/{case}.png` with `file_ending: ".png"`. Grayscale
  images → 1 input channel; RGB → 3 channels (`channel_names` set accordingly).
- **Lossy-JPEG masks are binarised** — mask pixels `> mask_threshold` become
  foreground `1`, recovering a clean `{0,1}` label map from noisy JPEGs.
- **Training is unchanged** — `training.configuration: "2d"` makes nnUNet train
  a 2-D U-Net; `scripts/02_train.py` needs no modality-specific logic.

### 2-D Workflow

```bash
# 1. Prepare the nnUNet 2-D dataset from the nested image/mask trees
python scripts/01_prepare_data.py --config configs/2d.yaml

# 2. Train the 2-D U-Net (uses the "2d" nnUNet configuration)
python scripts/02_train.py --config configs/2d.yaml

# 3. Predict on a folder of raw 2-D images (nested or flat, jpg/png)
python scripts/03_predict.py --config configs/2d.yaml \
    --input_raw /data0/test/images --output predictions_2d/

# 4. Evaluate predictions against ground-truth masks (raw nested tree or flat)
python scripts/04_evaluate.py --config configs/2d.yaml \
    --pred_dir predictions_2d/ --gt_dir /data0/test/mask
```

### Mirror the input directory tree (`--mirror_tree`)

By default 2-D prediction writes a **flat** folder of `{case}.png` masks
(convenient for evaluation). Add `--mirror_tree` to instead reproduce the input
directory layout: each prediction is written back under the **same relative
path** as its source image, so the output tree mirrors the input tree
one-for-one. Only image files are predicted (`.jpg`/`.png`/...); other files
(`.dcm`/`.pdf`/`.xlsx`/...) are ignored.

```bash
python scripts/03_predict.py --config configs/2d.yaml \
    --input_raw 甲状腺图像报告 --output 甲状腺图像报告pred --mirror_tree
```

```
甲状腺图像报告/                         甲状腺图像报告pred/
└── 甲状腺乳头状癌800/        →        └── 甲状腺乳头状癌800/
    └── 2026-03-30/49664962_唐月芳/         └── 2026-03-30/49664962_唐月芳/
        └── CS.../<uid>.jpg                     └── CS.../<uid>.png   ← predicted mask
```

`--mirror_tree` requires `data_type: "2d"` and `--input_raw`. It can also be
enabled by default via `inference.mirror_tree: true` in `configs/2d.yaml`.
The flat-output behaviour (without the flag) is unchanged, and the 3-D CT path
is unaffected.

### 2-D Configuration (`configs/2d.yaml`)

| Section | Key | Description |
|---------|-----|-------------|
| (top-level) | `data_type` | `"2d"` to enable the 2-D pipeline |
| `paths` | `images_dir` / `masks_dir` | Roots of the nested image / mask trees |
| `segmentation` | `labels` | nnUNet label block (e.g. `{background: 0, target: 1}`) |
| `data` | `color_mode` | `"grayscale"` (1ch, ultrasound) or `"rgb"` (3ch, endoscopy) |
| `data` | `mask_threshold` | Grayscale cut-off for binarising JPEG masks |
| `data` | `binary_mask` | `true` to binarise; `false` to keep raw class-ID pixels |
| `training` | `configuration` | must be `"2d"` |

---

## 2.5-D Mode (Bonus)

nnUNet v2 natively supports **multi-channel inputs**: each input channel is a
separate `_XXXX.nii.gz` file belonging to the same case.  The 2.5-D mode
exploits this by stacking adjacent CT slices as separate channels, giving a
2-D U-Net access to 3-D anatomical context.

### How it works

Each training sample is a 2-D slice (index `k`) paired with `N` adjacent
slices stacked as `N` input channels:

```
Channel 0: CT slice k-2  (or zero-padded if out of bounds)
Channel 1: CT slice k-1
Channel 2: CT slice k     ← centre / primary
Channel 3: CT slice k+1
Channel 4: CT slice k+2  (or zero-padded if out of bounds)
```

With `num_channels=5` and `channel_depth=1`, each channel is a single 2-D slice.
With `channel_depth=3`, each channel is a thin 3-D sub-volume (e.g. 3 slices
centred on `k-2`, `k-1`, etc.) — this gives even richer depth context.

Key advantages of 2.5-D:
- **3-D context for 2-D U-Net**: each forward pass sees neighbouring slices
- **Compatible with 2-D augmentations**: nnUNet's built-in 2-D augmentation
  applies seamlessly to each multi-channel slice sample
- **No custom trainer needed**: works with the standard `nnUNetTrainer`
- **Memory-efficient**: much cheaper than full 3-D U-Net, especially on high-resolution CT

### 2.5-D Workflow

```bash
# 1. Inspect labels (same as standard workflow)
python scripts/inspect_labels.py --config configs/default.yaml

# 2. Prepare 2.5-D dataset (creates one nnUNet case per (case, slice) pair)
python scripts/01b_prepare_data_25d.py --config configs/default.yaml

# 3. Train the 2.5-D model (uses the 2d configuration, but with 5 input channels)
python scripts/02_train_25d.py --config configs/default.yaml

# 4. Inference
python scripts/03_predict_25d.py --config configs/default.yaml \
    --input_raw /path/to/images --output predictions_25d/

# 5. Evaluate
python scripts/04_evaluate.py --config configs/default.yaml \
    --pred_dir predictions_25d/ --gt_dir data/
```

### 2.5-D Configuration

In `configs/default.yaml`:

```yaml
segmentation_25d:
  mode: "binary"
  num_channels: 5     # adjacent slices per sample (3 = k-1,k,k+1, 5 = k-2..k+2)
  channel_depth: 1    # 1 = single-slice channels, 3 = thin 3-D sub-volumes

training_25d:
  configuration: "2d"  # must be "2d" — 2-D U-Net on multi-channel inputs
  fold: 0
  num_gpus: 1
  num_proc_da: 4
  use_amp: true
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_channels` | 5 | Number of adjacent-slice channels per sample (must be odd) |
| `channel_depth` | 1 | Slices per channel sub-volume (1=single 2-D slice, 3=3-D sub-volume) |
| `configuration` | `"2d"` | nnUNet config — must be `"2d"` for 2.5-D |

---

## Tips for Small Datasets

This project includes 12 cases, which is a small dataset for medical image
segmentation. nnUNet's automatic configuration handles this well, but consider:

1. **Use 5-fold cross-validation** (`--all_folds`) to maximize training data
2. **Binary mode** (fewer output classes) is easier to learn on small data
3. **3d_fullres** is generally preferable even on small datasets
4. nnUNet automatically applies heavy data augmentation — trust the process
5. Default 1000 epochs may be sufficient; monitor validation metrics

---

## Troubleshooting

### "No bone voxels found after remapping"
Run `python scripts/inspect_labels.py` to check actual label IDs in your data.
The label IDs may differ between Total Segmentator v1 and v2.

### "CUDA out of memory"
Switch to `3d_lowres` configuration in `configs/default.yaml`:
```yaml
training:
  configuration: "3d_lowres"
```

### nnUNet environment variables not found
The scripts set these automatically. If running nnUNet commands directly,
export them manually:
```bash
export nnUNet_raw="D:/codes/work-projects/BoneSeg/nnunet_workspace/nnUNet_raw"
export nnUNet_preprocessed="D:/codes/work-projects/BoneSeg/nnunet_workspace/nnUNet_preprocessed"
export nnUNet_results="D:/codes/work-projects/BoneSeg/nnunet_workspace/nnUNet_results"
```

### Windows PowerShell environment variables
```powershell
$env:nnUNet_raw          = "D:\codes\work-projects\BoneSeg\nnunet_workspace\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\codes\work-projects\BoneSeg\nnunet_workspace\nnUNet_preprocessed"
$env:nnUNet_results      = "D:\codes\work-projects\BoneSeg\nnunet_workspace\nnUNet_results"
```
