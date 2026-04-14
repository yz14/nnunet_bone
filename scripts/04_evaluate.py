"""
Step 4 — Evaluate predicted bone segmentation against ground truth.

What this script does:
  1. Remaps ground truth Total Segmentator masks to the same target classes
     used during training (using the same label remapping logic)
  2. Computes Dice coefficient and Hausdorff distance (HD95) per case and class
  3. Saves a per-case CSV and prints a summary table

Usage:
    python scripts/04_evaluate.py \\
        --config configs/default.yaml \\
        --pred_dir /path/to/predictions \\
        --gt_dir   /path/to/ground_truth_segs

    # Ground truth can be raw Total Segmentator masks (auto-remapped) or
    # already-remapped masks (use --gt_already_remapped flag):
    python scripts/04_evaluate.py \\
        --config configs/default.yaml \\
        --pred_dir /path/to/predictions \\
        --gt_dir   /path/to/remapped_masks \\
        --gt_already_remapped
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep import remap_labels
from src.label_config import LabelConfig
from src.utils import ensure_dir, load_config, set_nnunet_env, setup_logging


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute the Dice similarity coefficient between two binary masks.

    Both masks must be boolean arrays of the same shape.

    Returns
    -------
    float
        Dice score in [0.0, 1.0]. Returns NaN when both masks are empty,
        since the Dice value of two empty sets is mathematically 1.0 but
        is not meaningful in a segmentation evaluation context.
    """
    pred_bool = pred.astype(bool)
    gt_bool   = gt.astype(bool)

    pred_sum = int(pred_bool.sum())
    gt_sum   = int(gt_bool.sum())

    # Both empty — no foreground in either prediction or ground truth.
    # Returning 1.0 would be misleading in a segmentation evaluation context.
    if pred_sum == 0 and gt_sum == 0:
        return float("nan")

    intersection = pred_bool & gt_bool
    denom = pred_sum + gt_sum
    if denom == 0:
        # Prediction empty but ground truth non-empty (or vice versa).
        return 0.0

    return 2.0 * intersection.sum() / denom


def hausdorff_95(pred: np.ndarray, gt: np.ndarray, spacing: Optional[Tuple[float, ...]] = None) -> float:
    """
    Compute the 95th-percentile Hausdorff distance between binary masks.

    Parameters
    ----------
    pred, gt : binary bool arrays of equal shape
    spacing  : voxel spacing in mm (z, y, x order); if None, uses unit spacing

    Returns nan if either mask is empty.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("scipy not available — HD95 skipped (install scipy to enable).")
        return float("nan")

    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)

    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return float("nan")

    sampling = spacing if spacing is not None else (1.0,) * pred_b.ndim

    # Distances from gt boundary to pred (and vice versa)
    dt_pred = distance_transform_edt(~pred_b, sampling=sampling)
    dt_gt   = distance_transform_edt(~gt_b,   sampling=sampling)

    hd_gt2pred = dt_pred[gt_b]
    hd_pred2gt = dt_gt[pred_b]

    combined = np.concatenate([hd_gt2pred, hd_pred2gt])
    return float(np.percentile(combined, 95))


# ─────────────────────────────────────────────────────────────────────────────
# Per-class evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_case(
    pred_data: np.ndarray,
    gt_data:   np.ndarray,
    class_ids: List[int],
    spacing:   Optional[Tuple[float, ...]],
    compute_hd95: bool,
) -> Dict[int, Dict[str, float]]:
    """
    Compute Dice (and optionally HD95) for each class in a single case.

    Returns {class_id: {"dice": float, "hd95": float}}
    """
    results: Dict[int, Dict[str, float]] = {}
    for cls in class_ids:
        pred_mask = pred_data == cls
        gt_mask   = gt_data   == cls
        dice = dice_coefficient(pred_mask, gt_mask)
        hd95 = hausdorff_95(pred_mask, gt_mask, spacing) if compute_hd95 else float("nan")
        results[cls] = {"dice": dice, "hd95": hd95}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_nii(path: Path) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Load NIfTI and return (data_int32, spacing_mm)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.int32)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return data, spacing


def find_prediction_files(pred_dir: Path) -> Dict[str, Path]:
    """Return {case_id: pred_path} for all .nii.gz files in pred_dir."""
    result: Dict[str, Path] = {}
    for f in sorted(pred_dir.glob("*.nii.gz")):
        case_id = f.name.replace(".nii.gz", "")
        result[case_id] = f
    return result


def find_gt_file(gt_dir: Path, case_id: str, already_remapped: bool) -> Optional[Path]:
    """Locate the ground-truth file for *case_id*."""
    if already_remapped:
        candidates = [
            gt_dir / f"{case_id}.nii.gz",
        ]
    else:
        # Raw Total Segmentator naming: s0000-seg.nii.gz
        candidates = [
            gt_dir / f"{case_id}-seg.nii.gz",
            gt_dir / f"{case_id}.nii.gz",
        ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Results output
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(
    all_results: List[Dict],
    class_names: Dict[int, str],
    compute_hd95: bool,
) -> None:
    """Print a nicely formatted summary table."""
    logger = logging.getLogger(__name__)
    class_ids = sorted(class_names.keys())

    # Header
    header = f"{'Case':<20}"
    for cls in class_ids:
        name = class_names[cls][:10]
        header += f"  {name:>10}_Dice"
        if compute_hd95:
            header += f"  {name:>10}_HD95"
    logger.info(header)
    logger.info("-" * len(header))

    # Per-case
    dice_per_class: Dict[int, List[float]] = {cls: [] for cls in class_ids}

    for row in all_results:
        line = f"{row['case']:<20}"
        for cls in class_ids:
            dice = row["metrics"].get(cls, {}).get("dice", float("nan"))
            hd95 = row["metrics"].get(cls, {}).get("hd95", float("nan"))
            line += f"  {dice:>14.4f}"
            if compute_hd95:
                hd95_str = f"{hd95:.2f}" if not np.isnan(hd95) else "  nan"
                line += f"  {hd95_str:>14}"
            if not np.isnan(dice):
                dice_per_class[cls].append(dice)
        logger.info(line)

    # Mean
    logger.info("-" * len(header))
    mean_line = f"{'MEAN':<20}"
    for cls in class_ids:
        vals = dice_per_class[cls]
        mean_dice = float(np.mean(vals)) if vals else float("nan")
        mean_line += f"  {mean_dice:>14.4f}"
        if compute_hd95:
            mean_line += f"  {'---':>14}"
    logger.info(mean_line)


def save_csv(
    all_results: List[Dict],
    class_names: Dict[int, str],
    output_path: Path,
    compute_hd95: bool,
) -> None:
    class_ids = sorted(class_names.keys())
    fieldnames = ["case"]
    for cls in class_ids:
        fieldnames.append(f"class{cls}_{class_names[cls]}_dice")
        if compute_hd95:
            fieldnames.append(f"class{cls}_{class_names[cls]}_hd95")

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            csv_row: Dict[str, object] = {"case": row["case"]}
            for cls in class_ids:
                m = row["metrics"].get(cls, {})
                dice = m.get("dice", float("nan"))
                hd95 = m.get("hd95", float("nan"))
                csv_row[f"class{cls}_{class_names[cls]}_dice"] = f"{dice:.4f}"
                if compute_hd95:
                    csv_row[f"class{cls}_{class_names[cls]}_hd95"] = (
                        f"{hd95:.2f}" if not np.isnan(hd95) else ""
                    )
            writer.writerow(csv_row)

    logging.getLogger(__name__).info("Saved per-case metrics to: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate nnUNet bone segmentation predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pred_dir", required=True,
                        help="Directory with predicted *.nii.gz masks")
    parser.add_argument("--gt_dir", required=True,
                        help="Directory with ground-truth segmentation files")
    parser.add_argument("--gt_already_remapped", action="store_true",
                        help="GT masks are already remapped (not raw Total Segmentator)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save CSV results (default: pred_dir)")
    parser.add_argument("--no_hd95", action="store_true",
                        help="Skip Hausdorff distance (faster)")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    cfg       = load_config(args.config)
    set_nnunet_env(cfg)
    label_cfg = LabelConfig(cfg["paths"]["labels_config"])

    seg_mode   = cfg["segmentation"]["mode"]
    override   = cfg["segmentation"].get("class_override") or {}
    remap      = label_cfg.get_remap(seg_mode, override)
    class_map  = label_cfg.get_nnunet_labels(seg_mode, override)
    # Foreground classes only (exclude background=0)
    class_names = {v: k for k, v in class_map.items() if v != 0}
    class_ids   = sorted(class_names.keys())

    compute_hd95 = not args.no_hd95
    pred_dir     = Path(args.pred_dir)
    gt_dir       = Path(args.gt_dir)
    output_dir   = Path(args.output_dir) if args.output_dir else pred_dir
    ensure_dir(output_dir)

    # ── Find predictions ──────────────────────────────────────────────────
    pred_files = find_prediction_files(pred_dir)
    if not pred_files:
        logger.error("No predictions found in '%s'.", pred_dir)
        sys.exit(1)

    logger.info("Evaluating %d prediction(s) ...", len(pred_files))

    # ── Per-case evaluation ───────────────────────────────────────────────
    all_results: List[Dict] = []
    missing_gt   = 0

    for case_id, pred_path in sorted(pred_files.items()):
        gt_path = find_gt_file(gt_dir, case_id, args.gt_already_remapped)
        if gt_path is None:
            logger.warning("No ground truth found for case '%s' — skipped.", case_id)
            missing_gt += 1
            continue

        pred_data, spacing = load_nii(pred_path)
        gt_raw, _          = load_nii(gt_path)

        if not args.gt_already_remapped:
            gt_data = remap_labels(gt_raw, remap)
        else:
            gt_data = gt_raw.astype(np.uint8)

        if pred_data.shape != gt_data.shape:
            logger.warning(
                "Shape mismatch for '%s': pred=%s, gt=%s — skipped.",
                case_id, pred_data.shape, gt_data.shape,
            )
            continue

        metrics = evaluate_case(pred_data, gt_data, class_ids, spacing, compute_hd95)
        all_results.append({"case": case_id, "metrics": metrics})

        # Brief per-case log
        dice_vals = [f"{class_names[c]}={metrics[c]['dice']:.3f}" for c in class_ids]
        logger.info("  %s — %s", case_id, ", ".join(dice_vals))

    if missing_gt:
        logger.warning("%d case(s) skipped due to missing ground truth.", missing_gt)

    if not all_results:
        logger.error("No cases evaluated successfully.")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary_table(all_results, class_names, compute_hd95)

    # ── Save CSV ──────────────────────────────────────────────────────────
    if cfg["evaluation"].get("save_csv", True):
        csv_path = output_dir / "evaluation_metrics.csv"
        save_csv(all_results, class_names, csv_path, compute_hd95)


if __name__ == "__main__":
    main()
