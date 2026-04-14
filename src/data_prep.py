"""
nnUNet v2 dataset preparation.

Responsibilities:
  - Discover image / segmentation pairs in the raw data directory
  - Remap Total Segmentator labels → target segmentation classes
  - Write nnUNet-formatted imagesTr / labelsTr directories
  - Generate dataset.json
"""

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import nibabel.processing as nib_proc
import numpy as np

from .label_config import LabelConfig
from .utils import ensure_dir, get_dataset_dir_name, get_nnunet_dirs

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_data_pairs(raw_data_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Discover (image, segmentation) pairs in *raw_data_dir*.

    Naming convention expected:
        {case_id}.nii.gz          — CT image
        {case_id}-seg.nii.gz      — Total Segmentator segmentation mask

    Returns
    -------
    List of (image_path, seg_path) tuples, sorted by case ID.
    """
    pairs: List[Tuple[Path, Path]] = []
    for seg_file in sorted(raw_data_dir.glob("*-seg.nii.gz")):
        case_id = seg_file.name.replace("-seg.nii.gz", "")
        img_file = raw_data_dir / f"{case_id}.nii.gz"
        if img_file.exists():
            pairs.append((img_file, seg_file))
        else:
            logger.warning("Segmentation found but image missing — skipped: %s", seg_file.name)

    if not pairs:
        raise FileNotFoundError(
            f"No (image, seg) pairs found in '{raw_data_dir}'. "
            "Expected files named '{id}.nii.gz' and '{id}-seg.nii.gz'."
        )

    logger.info("Found %d data pairs in '%s'.", len(pairs), raw_data_dir)
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Label remapping
# ─────────────────────────────────────────────────────────────────────────────

def remap_labels(
    seg_data: np.ndarray,
    remap: Dict[int, int],
) -> np.ndarray:
    """
    Remap a segmentation volume using the provided lookup table.

    Any source label NOT in *remap* maps to 0 (background).
    Output dtype is uint8 (supports up to 255 classes).
    """
    if not remap:
        return np.zeros(seg_data.shape, dtype=np.uint8)

    out = np.zeros(seg_data.shape, dtype=np.uint8)

    if seg_data.size == 0:
        return out

    # Vectorised: build a lookup array covering the max label ID seen in data.
    # Labels not in remap stay 0 (background), which matches the initial state.
    max_label = max(max(int(k) for k in remap), int(seg_data.max()))
    lookup = np.zeros(max_label + 1, dtype=np.uint8)
    for src, dst in remap.items():
        lookup[int(src)] = int(dst)
    out = lookup[seg_data]
    return out


def load_and_remap_seg(seg_path: Path, remap: Dict[int, int]) -> Tuple[np.ndarray, Any]:
    """
    Load a segmentation NIfTI file, apply label remapping, and return
    the remapped array together with the original NIfTI header/affine.

    NOTE: This function is kept for backwards compatibility and for cases
    where direct label inspection is needed outside the nnUNet dataset
    preparation flow. In the normal data-preparation pipeline, use
    ``_write_label()`` instead, which handles resampling and file writing
    in one step.
    """
    img = nib.load(str(seg_path))
    data = np.asarray(img.dataobj, dtype=np.int32)

    if data.size == 0:
        logger.warning("  %s: segmentation file is empty (shape=%s).", seg_path.name, data.shape)
        return np.zeros(data.shape, dtype=np.uint8), img

    remapped = remap_labels(data, remap)

    # Diagnostics
    unique_src = np.unique(data)
    unique_dst = np.unique(remapped)
    bone_voxels = int((remapped > 0).sum())
    total_voxels = int(remapped.size)
    pct = 100.0 * bone_voxels / total_voxels
    logger.debug(
        "  %s: src labels=%s -> dst labels=%s  (bone=%.1f%%)",
        seg_path.name, unique_src[unique_src > 0].tolist(),
        unique_dst[unique_dst > 0].tolist(), pct,
    )
    if bone_voxels == 0:
        logger.warning(
            "  %s: NO bone voxels found after remapping! "
            "Check that labels.yaml label IDs match the actual data.",
            seg_path.name,
        )

    return remapped, img


# ─────────────────────────────────────────────────────────────────────────────
# Dataset splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_cases(
    pairs: List[Tuple[Path, Path]],
    train_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Split pairs into train / (optional) test sets.

    When train_fraction == 1.0 all pairs go to train (nnUNet CV handles validation).
    Returns (train_pairs, test_pairs).
    """
    if train_fraction >= 1.0:
        return pairs, []

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_train = max(1, round(len(shuffled) * train_fraction))
    return shuffled[:n_train], shuffled[n_train:]


# ─────────────────────────────────────────────────────────────────────────────
# Affine / voxel-grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _seg_shares_voxel_grid(
    seg_nib: nib.Nifti1Image,
    ref_nib: nib.Nifti1Image,
    *,
    origin_atol: float = 1e-2,
    direction_atol: float = 1e-3,
) -> bool:
    """
    Return True when *seg_nib* occupies the same voxel grid as *ref_nib*.

    TotalSegmentator often stores segmentations with the same array shape and
    origin as the CT, but with wrong voxel-size / zoom metadata (e.g. 1.5 mm
    instead of 1.0 mm).  There are also cases where zooms nominally match but
    the affine differs by tiny float-precision rounding.

    We detect "same grid" by checking:
      1. Shapes are identical.
      2. Origins (translation column of the affine) are within *origin_atol*.
      3. Direction cosines (normalised rotation columns) are within
         *direction_atol* — meaning the axes point the same way even if the
         zoom scale differs.

    If all three hold, the voxel arrays are 1-to-1 aligned and no resampling
    is needed; we can just stamp the CT affine onto the segmentation.
    """
    if seg_nib.shape[:3] != ref_nib.shape[:3]:
        return False

    seg_origin = seg_nib.affine[:3, 3]
    ref_origin = ref_nib.affine[:3, 3]
    if not np.allclose(seg_origin, ref_origin, atol=origin_atol):
        return False

    # Normalise rotation columns to get direction cosines
    seg_rot = seg_nib.affine[:3, :3]
    ref_rot = ref_nib.affine[:3, :3]
    seg_norms = np.linalg.norm(seg_rot, axis=0)
    ref_norms = np.linalg.norm(ref_rot, axis=0)

    # Guard against zero-norm columns (should never happen for valid NIfTI)
    if np.any(seg_norms < 1e-8) or np.any(ref_norms < 1e-8):
        return False

    seg_dirs = seg_rot / seg_norms
    ref_dirs = ref_rot / ref_norms
    if not np.allclose(seg_dirs, ref_dirs, atol=direction_atol):
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# nnUNet file writing
# ─────────────────────────────────────────────────────────────────────────────

def _write_image(src_img_path: Path, dst_path: Path) -> None:
    """Copy a CT image to nnUNet imagesTr directory with modality suffix _0000."""
    shutil.copy2(str(src_img_path), str(dst_path))


def _write_label(
    seg_path: Path,
    img_path: Path,
    dst_path: Path,
    remap: Dict[int, int],
) -> None:
    """
    Remap Total Segmentator labels and write the result aligned to the CT voxel grid.

    Uses nibabel for reading (applies scl_slope in float64 → correct int32 labels)
    and nibabel.processing.resample_from_to with order=0 (nearest-neighbor) for
    resampling when truly needed.

    When the segmentation shares the same voxel grid as the CT (same shape,
    origin, direction cosines — only zooms differ), resampling is skipped
    entirely: the remapped array is saved directly with the CT affine.
    """
    # Load CT as the resampling reference (nibabel preserves the affine exactly)
    ref_nib = nib.load(str(img_path))

    # Load seg with nibabel: np.asarray with dtype=int32 applies scl_slope in
    # float64 then truncates to int, giving correct integer label values.
    seg_nib  = nib.load(str(seg_path))
    seg_data = np.asarray(seg_nib.dataobj, dtype=np.int32)

    # Remap labels
    remapped_array = remap_labels(seg_data, remap)

    if remapped_array.size == 0:
        logger.warning(
            "  %s: segmentation file is empty (shape=%s). "
            "Writing empty label mask.",
            seg_path.name, remapped_array.shape,
        )

    # Diagnostics
    bone_voxels = int((remapped_array > 0).sum())
    if bone_voxels == 0:
        logger.warning(
            "  %s: NO bone voxels found after remapping! "
            "Check that labels.yaml label IDs match the actual data.",
            seg_path.name,
        )
    else:
        pct = 100.0 * bone_voxels / remapped_array.size
        logger.debug("  %s: bone voxels=%.1f%%", seg_path.name, pct)

    # --- Decide whether to resample or shortcut -------------------------
    same_grid = _seg_shares_voxel_grid(seg_nib, ref_nib)

    if same_grid:
        # Seg is already on the CT voxel grid — skip resampling entirely.
        # Just save the remapped array with the CT affine.
        seg_zooms = tuple(round(float(z), 3) for z in seg_nib.header.get_zooms()[:3])
        ref_zooms = tuple(round(float(z), 3) for z in ref_nib.header.get_zooms()[:3])
        if seg_zooms != ref_zooms:
            logger.info(
                "    Same voxel grid (zooms %s ≠ %s but shape+origin+directions match)"
                " — saving with CT affine, no resampling.",
                seg_zooms, ref_zooms,
            )
        else:
            logger.info(
                "    Same voxel grid — saving with CT affine, no resampling.",
            )
        resampled_nib = nib.Nifti1Image(
            remapped_array.astype(np.uint8), ref_nib.affine,
        )
    else:
        # Genuinely different grids — resample via nearest-neighbor.
        remapped_nib = nib.Nifti1Image(
            remapped_array.astype(np.uint8), seg_nib.affine, seg_nib.header,
        )
        remapped_nib.set_data_dtype(np.uint8)
        logger.info(
            "    Resampling '%s' to CT space  (seg: shape=%s zooms=%s → CT: shape=%s zooms=%s)",
            seg_path.name,
            seg_nib.shape[:3],
            tuple(round(float(z), 3) for z in seg_nib.header.get_zooms()[:3]),
            ref_nib.shape[:3],
            tuple(round(float(z), 3) for z in ref_nib.header.get_zooms()[:3]),
        )
        resampled_nib = nib_proc.resample_from_to(remapped_nib, ref_nib, order=0, cval=0)

    nib.save(resampled_nib, str(dst_path))


# ─────────────────────────────────────────────────────────────────────────────
# dataset.json generation
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_json(
    labels: Dict[str, int],
    num_training: int,
    num_test: int = 0,
) -> Dict[str, Any]:
    """
    Build the dataset.json dict required by nnUNet v2.

    Parameters
    ----------
    labels : {"background": 0, "bone": 1, ...}
    num_training : number of training cases
    num_test : number of test cases (0 = no separate test set)
    """
    dataset = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    if num_test > 0:
        dataset["numTest"] = num_test
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def prepare_nnunet_dataset(
    cfg: Dict[str, Any],
    label_cfg: LabelConfig,
    dry_run: bool = False,
) -> Path:
    """
    Convert raw Total Segmentator data into nnUNet v2 dataset format.

    Parameters
    ----------
    cfg : loaded default.yaml config dict
    label_cfg : LabelConfig instance
    dry_run : if True, print what would be done but don't write files

    Returns
    -------
    Path to the created dataset directory (inside nnUNet_raw/).
    """
    # ── Config extraction ──────────────────────────────────────────────────
    raw_data_dir   = Path(cfg["paths"]["raw_data_dir"])
    seg_mode       = cfg["segmentation"]["mode"]
    class_override = cfg["segmentation"].get("class_override") or {}
    train_fraction = float(cfg["dataset"].get("train_fraction", 1.0))
    split_seed     = int(cfg["dataset"].get("split_seed", 42))

    # ── Label config ───────────────────────────────────────────────────────
    remap          = label_cfg.get_remap(seg_mode, class_override)
    nnunet_labels  = label_cfg.get_nnunet_labels(seg_mode, class_override)
    label_cfg.describe(seg_mode)

    if not remap:
        raise ValueError(
            "Label remap is empty — no active bone groups produce valid labels. "
            "Check configs/labels.yaml active_groups."
        )

    # ── nnUNet directory layout ────────────────────────────────────────────
    nnunet_dirs  = get_nnunet_dirs(cfg)
    dataset_name = get_dataset_dir_name(cfg)
    dataset_dir  = nnunet_dirs["raw"] / dataset_name

    images_tr_dir = dataset_dir / "imagesTr"
    labels_tr_dir = dataset_dir / "labelsTr"
    images_ts_dir = dataset_dir / "imagesTs"

    if not dry_run:
        ensure_dir(images_tr_dir)
        ensure_dir(labels_tr_dir)

    # ── Discover data pairs ────────────────────────────────────────────────
    all_pairs = find_data_pairs(raw_data_dir)
    train_pairs, test_pairs = split_cases(all_pairs, train_fraction, split_seed)

    logger.info(
        "Split: %d training, %d test cases.", len(train_pairs), len(test_pairs)
    )

    if dry_run:
        logger.info("[DRY RUN] Would create dataset at: %s", dataset_dir)
        logger.info("[DRY RUN] Would write %d training cases.", len(train_pairs))
        return dataset_dir

    # ── Write training cases ───────────────────────────────────────────────
    case_names: List[str] = []
    for img_path, seg_path in train_pairs:
        case_id   = img_path.stem.replace(".nii", "")  # e.g. 's0000'
        case_name = case_id                             # nnUNet case name

        img_dst   = images_tr_dir / f"{case_name}_0000.nii.gz"
        label_dst = labels_tr_dir / f"{case_name}.nii.gz"

        logger.info("  [train] %s → %s", img_path.name, img_dst.name)
        _write_image(img_path, img_dst)

        logger.info("  [label] %s → %s (remapping labels)", seg_path.name, label_dst.name)
        _write_label(seg_path, img_path, label_dst, remap)

        case_names.append(case_name)

    # ── Write test cases (optional) ────────────────────────────────────────
    if test_pairs:
        ensure_dir(images_ts_dir)
        for img_path, _ in test_pairs:
            case_id = img_path.stem.replace(".nii", "")
            img_dst = images_ts_dir / f"{case_id}_0000.nii.gz"
            logger.info("  [test] %s → %s", img_path.name, img_dst.name)
            _write_image(img_path, img_dst)

    # ── Write dataset.json ─────────────────────────────────────────────────
    dataset_json = build_dataset_json(
        labels=nnunet_labels,
        num_training=len(train_pairs),
        num_test=len(test_pairs),
    )
    json_path = dataset_dir / "dataset.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(dataset_json, fh, indent=2)

    logger.info("Wrote dataset.json → %s", json_path)
    logger.info(
        "Dataset '%s' prepared successfully (%d classes, %d train cases).",
        dataset_name, len(nnunet_labels) - 1, len(train_pairs),
    )

    return dataset_dir
