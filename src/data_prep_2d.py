"""
nnUNet v2 dataset preparation for 2-D images (ultrasound / endoscopy).

This is the 2-D counterpart of :mod:`src.data_prep` (which handles 3-D CT
NIfTI volumes). It is selected automatically when the experiment config sets
``data_type: "2d"`` (see :func:`src.utils.get_data_type`).

Input layout
------------
Images and masks live in two parallel directory trees with **identical
relative paths** (only the root differs)::

    {images_dir}/C847675/CS20250714439/1/<uid>.jpg      ← image
    {masks_dir}/C847675/CS20250714439/1/<uid>.jpg       ← annotation mask

The tree can be arbitrarily nested; every leaf image is one training sample.

Output
------
A flat nnUNet v2 2-D dataset::

    Dataset{ID}_{name}/
        imagesTr/{case}_0000.png     ← image (grayscale → 1ch, rgb → 3ch)
        labelsTr/{case}.png          ← binarised label map ({0,1})
        dataset.json                 ← file_ending=".png"
        manifest.json                ← {case: {image, mask}} for traceability

Responsibilities:
  - Discover (image, mask) pairs across the nested directory trees
  - Convert source JPEGs to nnUNet's lossless PNG format
  - Binarise lossy JPEG masks into clean integer label maps
  - Generate dataset.json + a manifest mapping cases back to source files
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import image_io
from .utils import ensure_dir, get_dataset_dir_name, get_nnunet_dirs

logger = logging.getLogger(__name__)


# A (case_name, image_path, mask_path) triple describing one 2-D sample.
Pair = Tuple[str, Path, Path]


# ─────────────────────────────────────────────────────────────────────────────
# Case naming
# ─────────────────────────────────────────────────────────────────────────────

def make_case_name(rel_path: Path) -> str:
    """
    Build a filesystem-safe, unique nnUNet case identifier from a relative path.

    The nested directory components and the file stem are joined with ``-`` so
    the case name stays human-traceable, e.g.::

        C847675/CS20250714439/1/1.2.410...567.jpg
            → C847675-CS20250714439-1-1.2.410...567

    Any character outside ``[A-Za-z0-9.-]`` is replaced with ``-`` so the name
    is safe on every filesystem. The ``_0000`` channel suffix that nnUNet
    appends later is unaffected because it is added after this name.
    """
    parts = rel_path.with_suffix("").parts
    raw = "-".join(parts)
    # Keep dots (DICOM-style UIDs rely on them); sanitise everything else.
    safe = re.sub(r"[^A-Za-z0-9.\-]", "-", raw)
    # Collapse runs of separators introduced by sanitisation.
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    return safe


# ─────────────────────────────────────────────────────────────────────────────
# Data discovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_matching_mask(masks_dir: Path, rel_path: Path) -> Optional[Path]:
    """
    Locate the mask that corresponds to an image at *rel_path*.

    The mask is expected at the same relative path under *masks_dir*. The
    image and mask may use different extensions, so the exact name is tried
    first and then any supported extension with the same stem.
    """
    exact = masks_dir / rel_path
    if exact.exists():
        return exact

    candidate_dir = masks_dir / rel_path.parent
    if not candidate_dir.is_dir():
        return None
    stem = rel_path.stem
    for ext in image_io.IMAGE_EXTENSIONS:
        cand = candidate_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def find_image_mask_pairs(images_dir: Path, masks_dir: Path) -> List[Pair]:
    """
    Discover (case_name, image_path, mask_path) triples.

    Walks *images_dir* recursively; for every image it looks up the matching
    mask under *masks_dir* at the same relative path. Images without a mask are
    skipped with a warning. Raises FileNotFoundError if nothing usable is found.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: '{images_dir}'.")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found: '{masks_dir}'.")

    pairs: List[Pair] = []
    seen_names: Dict[str, Path] = {}
    n_missing = 0

    for img_path in image_io.find_images_recursive(images_dir):
        rel = img_path.relative_to(images_dir)
        mask_path = _find_matching_mask(masks_dir, rel)
        if mask_path is None:
            logger.warning("No mask found for image — skipped: %s", rel)
            n_missing += 1
            continue

        case_name = make_case_name(rel)
        if case_name in seen_names:
            logger.error(
                "Case name collision: '%s' from both '%s' and '%s'. "
                "Skipping the latter.",
                case_name, seen_names[case_name], rel,
            )
            continue
        seen_names[case_name] = rel
        pairs.append((case_name, img_path, mask_path))

    if n_missing:
        logger.warning("%d image(s) had no matching mask and were skipped.", n_missing)

    if not pairs:
        raise FileNotFoundError(
            f"No (image, mask) pairs found.\n"
            f"  images_dir = '{images_dir}'\n"
            f"  masks_dir  = '{masks_dir}'\n"
            "Expected the mask tree to mirror the image tree with identical "
            "relative paths."
        )

    logger.info("Found %d (image, mask) pairs.", len(pairs))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_pairs(
    pairs: List[Pair],
    train_fraction: float,
    seed: int,
) -> Tuple[List[Pair], List[Pair]]:
    """
    Split pairs into (train, test). When ``train_fraction >= 1.0`` all pairs go
    to train and nnUNet's built-in cross-validation handles the validation set.
    """
    if train_fraction >= 1.0:
        return pairs, []
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_train = max(1, round(len(shuffled) * train_fraction))
    return shuffled[:n_train], shuffled[n_train:]


# ─────────────────────────────────────────────────────────────────────────────
# Label config (2-D is self-contained — no Total Segmentator remapping)
# ─────────────────────────────────────────────────────────────────────────────

def get_2d_labels(cfg: Dict[str, Any]) -> Dict[str, int]:
    """
    Return the nnUNet ``labels`` block for a 2-D dataset.

    2-D masks are already foreground/background (or per-class) annotations, so
    there is no Total Segmentator remapping. Labels are read directly from
    ``segmentation.labels`` in the config, defaulting to a binary task.
    """
    seg = cfg.get("segmentation", {})
    labels = seg.get("labels")
    if labels:
        return {str(k): int(v) for k, v in labels.items()}
    return {"background": 0, "target": 1}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def prepare_2d_dataset(cfg: Dict[str, Any], dry_run: bool = False) -> Path:
    """
    Convert nested 2-D image/mask directories into nnUNet v2 dataset format.

    Parameters
    ----------
    cfg : loaded 2-D experiment config (``data_type: "2d"``)
    dry_run : if True, only report what would be done; write nothing.

    Returns the path to the created dataset directory (inside nnUNet_raw/).
    """
    # ── Config extraction ──────────────────────────────────────────────────
    images_dir     = Path(cfg["paths"]["images_dir"])
    masks_dir      = Path(cfg["paths"]["masks_dir"])
    train_fraction = float(cfg["dataset"].get("train_fraction", 1.0))
    split_seed     = int(cfg["dataset"].get("split_seed", 42))

    data_cfg       = cfg.get("data", {})
    color_mode     = str(data_cfg.get("color_mode", image_io.COLOR_MODE_GRAYSCALE))
    mask_threshold = int(data_cfg.get("mask_threshold", 127))
    binary_mask    = bool(data_cfg.get("binary_mask", True))

    nnunet_labels  = get_2d_labels(cfg)
    num_channels   = image_io.num_channels_for_mode(color_mode)
    channel_names  = image_io.channel_names_for_mode(color_mode)

    logger.info("=" * 60)
    logger.info("2-D dataset preparation")
    logger.info("  images_dir   : %s", images_dir)
    logger.info("  masks_dir    : %s", masks_dir)
    logger.info("  color_mode   : %s (%d channel(s))", color_mode, num_channels)
    logger.info("  mask binarise: %s (threshold=%d)", binary_mask, mask_threshold)
    logger.info("  labels       : %s", nnunet_labels)
    logger.info("=" * 60)

    # ── nnUNet directory layout ────────────────────────────────────────────
    nnunet_dirs   = get_nnunet_dirs(cfg)
    dataset_name  = get_dataset_dir_name(cfg)
    dataset_dir   = nnunet_dirs["raw"] / dataset_name
    images_tr_dir = dataset_dir / "imagesTr"
    labels_tr_dir = dataset_dir / "labelsTr"

    # ── Discover + split data pairs ────────────────────────────────────────
    all_pairs = find_image_mask_pairs(images_dir, masks_dir)
    train_pairs, test_pairs = split_pairs(all_pairs, train_fraction, split_seed)
    logger.info("Split: %d training, %d test cases.", len(train_pairs), len(test_pairs))

    if dry_run:
        logger.info("[DRY RUN] Would create dataset at: %s", dataset_dir)
        logger.info("[DRY RUN] Would write %d training cases.", len(train_pairs))
        for case_name, img_path, mask_path in train_pairs[:5]:
            logger.info("[DRY RUN]   %s  ←  %s | %s", case_name, img_path, mask_path)
        if len(train_pairs) > 5:
            logger.info("[DRY RUN]   ... (%d more)", len(train_pairs) - 5)
        return dataset_dir

    ensure_dir(images_tr_dir)
    ensure_dir(labels_tr_dir)

    # ── Write training cases ───────────────────────────────────────────────
    manifest: Dict[str, Dict[str, str]] = {}
    empty_masks = 0

    for case_name, img_path, mask_path in train_pairs:
        image = image_io.load_image(img_path, color_mode)
        label = image_io.load_mask_as_label(
            mask_path, threshold=mask_threshold, binary=binary_mask,
        )

        if image.shape[:2] != label.shape[:2]:
            logger.warning(
                "  %s: image %s and mask %s differ in size — skipped.",
                case_name, image.shape[:2], label.shape[:2],
            )
            continue

        fg = int((label > 0).sum())
        if fg == 0:
            empty_masks += 1
            logger.debug("  %s: mask has no foreground pixels.", case_name)

        image_io.save_nnunet_image(image, case_name, images_tr_dir)
        image_io.save_label(label, case_name, labels_tr_dir)

        manifest[case_name] = {
            "image": str(img_path),
            "mask": str(mask_path),
        }
        logger.info("  [train] %s (fg=%.2f%%)", case_name, 100.0 * fg / label.size)

    if empty_masks:
        logger.warning(
            "%d training mask(s) had no foreground after binarisation — "
            "verify the mask_threshold / mask format.", empty_masks,
        )

    # ── Write test cases (images only; optional) ───────────────────────────
    if test_pairs:
        images_ts_dir = dataset_dir / "imagesTs"
        ensure_dir(images_ts_dir)
        for case_name, img_path, _ in test_pairs:
            image = image_io.load_image(img_path, color_mode)
            image_io.save_nnunet_image(image, case_name, images_ts_dir)
            logger.info("  [test] %s", case_name)

    # ── Write dataset.json ─────────────────────────────────────────────────
    dataset_json: Dict[str, Any] = {
        "channel_names": channel_names,
        "labels": nnunet_labels,
        "numTraining": len(manifest),
        "file_ending": image_io.NNUNET_2D_FILE_ENDING,
    }
    if test_pairs:
        dataset_json["numTest"] = len(test_pairs)

    with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as fh:
        json.dump(dataset_json, fh, indent=2)

    # ── Write manifest (case → source files) ───────────────────────────────
    with open(dataset_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    logger.info(
        "2-D dataset '%s' prepared: %d train cases (%d classes) → %s",
        dataset_name, len(manifest), len(nnunet_labels) - 1, dataset_dir,
    )
    return dataset_dir
