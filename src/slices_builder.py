"""
2.5D slice builder — extract adjacent slices and stack them as multi-channel input.

nnUNet v2 supports multi-channel inputs natively: each channel is a separate
_XXXX.nii.gz file belonging to the same case.  The network's first convolutional
layer automatically adapts its ``in_channels`` parameter to the number of channels
declared in ``channel_names``.

For 2.5D we treat each axial slice as the centre of a 5-slice window
(channels 0..4 = slices k-2, k-1, k, k+1, k+2).  The 2D U-Net sees a 5-channel
2D map per forward pass, giving it 3D context without the full 3D decoder cost.

Design decisions
----------------
- Each (case, slice_position) pair becomes its own nnUNet case so that nnUNet's
  built-in training / validation splitting works without any trainer customisation.
- Context slices are extracted as thin 3-D sub-volumes (3 slices each: one
  source slice + 1-slice padding on each side) to keep file sizes small.
- Edge slices are zero-padded rather than dropped, so every axial position
  in the original volume has a training sample.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core slice-extraction logic
# ─────────────────────────────────────────────────────────────────────────────

def _extract_adjacent_slices_3d(
    volume: np.ndarray,
    center_idx: int,
    depth: int = 3,
) -> np.ndarray:
    """
    Extract a ``depth``-slice sub-volume centred on *center_idx* from *volume*.

    The returned array has shape ``(depth, H, W)`` even at volume boundaries
    where slices are zero-padded.

    Parameters
    ----------
    volume : np.ndarray
        3-D array with shape (D, H, W).
    center_idx : int
        Axial slice index of the centre slice.
    depth : int
        Total number of slices in the sub-volume (must be odd).  Default 3
        gives 1 source slice + 1 padding on each side.  Set to 5 for a
        wider context (1 source + 2 padding each side).

    Returns
    -------
    np.ndarray
        3-D array of shape ``(depth, H, W)``.
    """
    D, H, W = volume.shape
    half = depth // 2
    out = np.zeros((depth, H, W), dtype=volume.dtype)

    for d in range(depth):
        src_idx = center_idx - half + d
        if 0 <= src_idx < D:
            out[d] = volume[src_idx]
    return out


def _extract_center_slice_2d(
    volume: np.ndarray,
    center_idx: int,
) -> np.ndarray:
    """
    Extract a single 2-D slice (shape H×W) from *volume* at *center_idx*.
    Returns zeros of shape (H, W) if *center_idx* is out of bounds.
    """
    D, H, W = volume.shape
    if 0 <= center_idx < D:
        return volume[center_idx].copy()
    return np.zeros((H, W), dtype=volume.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample 3-D sub-volume (all channels together)
# ─────────────────────────────────────────────────────────────────────────────

def build_25d_subvolume(
    ct_volume: np.ndarray,
    seg_volume: np.ndarray,
    center_idx: int,
    num_channels: int = 5,
    channel_depth: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build a 2.5-D training sample from a CT and its segmentation.

    The sample consists of:
      - ``num_channels`` 3-D sub-volumes, each of shape ``(channel_depth, H, W)``,
        stacked along a new channel axis → shape ``(num_channels, channel_depth, H, W)``.
      - A single 2-D label slice of shape ``(H, W)`` centred on ``center_idx``.

    Channel layout (for the default 5-channel, 3-depth setting):

    ============  ============================================================
    Channel      Content
    ============  ============================================================
    0            Sub-volume centred on ``center_idx - 2`` (3 slices)
    1            Sub-volume centred on ``center_idx - 1`` (3 slices)
    2            Centre slice ``center_idx`` as a 2-D slice padded to (3,H,W)
    3            Sub-volume centred on ``center_idx + 1`` (3 slices)
    4            Sub-volume centred on ``center_idx + 2`` (3 slices)
    ============  ============================================================

    Parameters
    ----------
    ct_volume, seg_volume : np.ndarray
        3-D CT image and segmentation mask.
    center_idx : int
        Axial slice index that receives the label.
    num_channels : int
        Number of adjacent-slice channels (must be odd; default 5).
    channel_depth : int
        Number of slices per sub-volume (must be odd; default 3 → 1 source + 2 padding).

    Returns
    -------
    input_4d : np.ndarray   shape ``(num_channels, channel_depth, H, W)``
    label_2d : np.ndarray   shape ``(H, W)``
    context_indices : List[int]
        The source slice indices used for each channel (for logging/debugging).
    """
    if num_channels % 2 != 1:
        raise ValueError(f"num_channels must be odd, got {num_channels}")
    if channel_depth % 2 != 1:
        raise ValueError(f"channel_depth must be odd, got {channel_depth}")

    D, H, W = ct_volume.shape
    half_ch = num_channels // 2   # e.g. 2 for 5 channels → slices k-2 … k+2
    half_d  = channel_depth // 2  # e.g. 1 for 3 depth  → 1 source ± 1

    # Label slice
    label_2d = _extract_center_slice_2d(seg_volume, center_idx)

    # Build each channel
    input_channels: List[np.ndarray] = []
    context_indices: List[int] = []

    for ch in range(num_channels):
        src_idx = center_idx - half_ch + ch          # source centre slice for this channel
        context_indices.append(src_idx)

        if ch == half_ch:
            # Centre channel: single 2-D slice, padded to channel_depth
            ch_vol = _extract_adjacent_slices_3d(ct_volume, center_idx, channel_depth)
        else:
            ch_vol = _extract_adjacent_slices_3d(ct_volume, src_idx, channel_depth)

        input_channels.append(ch_vol)

    # Stack: (C, depth, H, W)
    input_4d = np.stack(input_channels, axis=0).astype(np.float32)
    return input_4d, label_2d, context_indices


# ─────────────────────────────────────────────────────────────────────────────
# Per-slice label remapping (used by 2.5D pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def remap_labels_single(
    label_slice: np.ndarray,
    remap: Dict[int, int],
) -> np.ndarray:
    """
    Remap a 2-D label slice using the provided lookup table.

    Any source label NOT in *remap* maps to 0 (background).
    Output dtype is uint8.
    """
    if not remap:
        return np.zeros(label_slice.shape, dtype=np.uint8)

    label_work = label_slice.astype(np.int32)
    out = np.zeros(label_slice.shape, dtype=np.uint8)

    if label_work.size == 0:
        return out

    max_label = max(max(int(k) for k in remap), int(label_work.max()))
    lookup = np.zeros(max_label + 1, dtype=np.uint8)
    for src, dst in remap.items():
        lookup[int(src)] = int(dst)
    out = lookup[label_work]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4-D file helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_channels(
    channels_4d: np.ndarray,
    case_name: str,
    output_dir: Path,
) -> None:
    """
    Write each channel of a multi-channel case as a separate _000X.nii.gz file.

    Parameters
    ----------
    channels_4d : np.ndarray  shape (num_channels, channel_depth, H, W)
    case_name   : str         e.g. "s0000_00042"
    output_dir  : Path        imagesTr directory
    """
    num_channels = channels_4d.shape[0]
    for ch in range(num_channels):
        ch_vol   = channels_4d[ch]
        out_path = output_dir / f"{case_name}_000{chr(ord('0') + ch)}.nii.gz"
        nib.save(nib.Nifti1Image(ch_vol.astype(np.int16), np.eye(4)), str(out_path))


def write_label(
    label_2d: np.ndarray,
    case_name: str,
    output_dir: Path,
) -> None:
    """
    Write a single-case label file.

    The label is stored as shape (1, H, W) to satisfy nnUNet's 3-D requirement.
    The identity affine (np.eye(4)) is a placeholder; nnUNet uses the shape only
    for the label and does not use the affine for label data.
    """
    label_3d = np.expand_dims(label_2d.astype(np.uint8), axis=0)
    label_path = output_dir / f"{case_name}.nii.gz"
    nib.save(nib.Nifti1Image(label_3d, np.eye(4)), str(label_path))


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def prepare_25d_dataset(
    cfg: Dict[str, Any],
    label_cfg: "LabelConfig",  # forward ref to avoid circular import
    dry_run: bool = False,
) -> Tuple[Path, Path, Path, int]:
    """
    Convert raw Total Segmentator data into 2.5-D nnUNet dataset format.

    For each (case, axial_position) pair this creates one nnUNet case with
    ``num_channels`` input channels (adjacent slices) and a single 2-D label.

    Parameters
    ----------
    cfg       : loaded default.yaml dict
    label_cfg : LabelConfig instance
    dry_run   : if True, print what would be done without writing files

    Returns
    -------
    (dataset_dir, images_tr_dir, labels_tr_dir, num_total_samples)
    """
    from .label_config import LabelConfig
    from .utils import ensure_dir, get_dataset_dir_name, get_nnunet_dirs

    # ── Config extraction ──────────────────────────────────────────────────
    raw_data_dir   = Path(cfg["paths"]["raw_data_dir"])
    seg_mode       = cfg["segmentation_25d"].get("mode", "binary")
    class_override = cfg["segmentation"].get("class_override") or {}

    # 2.5D parameters
    num_channels   = int(cfg["segmentation_25d"].get("num_channels", 5))
    channel_depth  = int(cfg["segmentation_25d"].get("channel_depth", 3))

    logger.info(
        "2.5D config: num_channels=%d, channel_depth=%d, mode=%s",
        num_channels, channel_depth, seg_mode,
    )

    # ── Label config ───────────────────────────────────────────────────────
    remap         = label_cfg.get_remap(seg_mode, class_override)
    nnunet_labels = label_cfg.get_nnunet_labels(seg_mode, class_override)
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

    if not dry_run:
        ensure_dir(images_tr_dir)
        ensure_dir(labels_tr_dir)

    # ── Discover data pairs ────────────────────────────────────────────────
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

    if dry_run:
        logger.info("[DRY RUN] Would create dataset at: %s", dataset_dir)
        return dataset_dir, images_tr_dir, labels_tr_dir, 0

    # ── Per-case, per-slice processing ─────────────────────────────────────
    all_case_names: List[str] = []
    processed_slices: Dict[str, int] = {}

    for img_path, seg_path in pairs:
        case_id = img_path.stem.replace(".nii", "")

        # Load volumes
        ct_nib  = nib.load(str(img_path))
        seg_nib = nib.load(str(seg_path))

        ct_vol  = np.asarray(ct_nib.dataobj,  dtype=np.int16)
        seg_vol = np.asarray(seg_nib.dataobj, dtype=np.int32)

        D, H, W = ct_vol.shape

        logger.info("  Processing %s — shape %s, %d slices", case_id, (D, H, W), D)

        for z in range(D):
            # Build 2.5D sample
            input_4d, label_2d, ctx_idx = build_25d_subvolume(
                ct_vol, seg_vol, z,
                num_channels=num_channels,
                channel_depth=channel_depth,
            )

            # Remap labels
            remapped_label = remap_labels_single(label_2d, remap)

            logger.debug(
                "    Slice %d (ctx=%s): bone=%.1f%%",
                z, ctx_idx,
                100.0 * remapped_label.sum() / remapped_label.size
                if remapped_label.size > 0 else 0.0,
            )

            # Case name includes original case ID and slice index
            slice_case_name = f"{case_id}_s{z:05d}"

            # Write channels to imagesTr
            write_channels(input_4d, slice_case_name, images_tr_dir)

            # Write label to labelsTr
            write_label(remapped_label, slice_case_name, labels_tr_dir)

            all_case_names.append(slice_case_name)

        processed_slices[case_id] = D

    num_total = len(all_case_names)

    # ── Write dataset.json ─────────────────────────────────────────────────
    # nnUNet v2: channel_names keys are "0", "1", ... and values are string labels
    channel_names: Dict[str, str] = {
        f"{ch}": f"CT_adjacent_{ch}" for ch in range(num_channels)
    }

    dataset_json: Dict[str, Any] = {
        "channel_names": channel_names,
        "labels": nnunet_labels,
        "numTraining": num_total,
        "file_ending": ".nii.gz",
    }
    json_path = dataset_dir / "dataset.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(dataset_json, fh, indent=2)

    logger.info(
        "2.5D dataset '%s' prepared: %d samples (%s) from %d cases.",
        dataset_name,
        num_total,
        " / ".join(f"{k}={v}" for k, v in processed_slices.items()),
        len(pairs),
    )

    return dataset_dir, images_tr_dir, labels_tr_dir, num_total
