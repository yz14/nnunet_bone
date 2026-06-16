"""
Step 3 — Run inference with the trained nnUNet bone segmentation model.

What this script does:
  1. Sets nnUNet environment variables
  2. Prepares an input folder from the specified images
  3. Runs nnUNetv2_predict

Usage:
    # Predict on a folder of raw CT images (must be _0000.nii.gz format):
    python scripts/03_predict.py --input /path/to/images --output /path/to/preds

    # Using config defaults (reads input/output from config paths):
    python scripts/03_predict.py --config configs/default.yaml --input /path/to/input_images --output /path/to/output_preds

    # Use a specific fold:
    python scripts/03_predict.py --config configs/default.yaml \\
        --input /path/to/images --output /path/to/preds --fold 0

    # Ensemble all folds:
    python scripts/03_predict.py --config configs/default.yaml \\
        --input /path/to/images --output /path/to/preds --fold all

    # Predict on raw TotalSeg images (auto-rename with _0000 suffix):
    CUDA_VISIBLE_DEVICES=1 python scripts/03_predict.py --config configs/default.yaml --input_raw /path/to/totalseg_data --output /path/to/preds
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import image_io
from src.tree_io_2d import prepare_input_tree_2d, scatter_predictions_to_tree
from src.utils import (
    ensure_dir,
    get_data_type,
    load_config,
    set_nnunet_env,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nnUNet v2 bone segmentation inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input folder with images named *_0000.nii.gz (nnUNet format)",
    )
    parser.add_argument(
        "--input_raw", type=str, default=None,
        help="Input folder with raw images named *.nii.gz (auto-converted to *_0000.nii.gz)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output folder for predicted segmentation masks",
    )
    parser.add_argument(
        "--mirror_tree", action="store_true",
        help=(
            "2-D + --input_raw only: write each prediction back under the same "
            "relative path as its source image, mirroring the input directory "
            "tree (e.g. '<dir>' -> '<dir>pred'). Without this flag, predictions "
            "are written to a flat folder."
        ),
    )
    parser.add_argument(
        "--fold", default=None,
        help="Fold(s) to use: 0/1/2/3/4 or 'all' (default: from config)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint name: 'checkpoint_best.pth' or 'checkpoint_final.pth' (default: from config)",
    )
    parser.add_argument(
        "--save_probabilities", action="store_true",
        help="Save softmax probability maps in addition to hard predictions",
    )
    parser.add_argument(
        "--disable_tta", action="store_true",
        help="Disable test-time augmentation (faster but slightly lower quality)",
    )
    parser.add_argument(
        "--log_level", default="INFO",
        help="Logging level: DEBUG / INFO / WARNING",
    )
    return parser.parse_args()


def prepare_input_folder(raw_input_dir: Path) -> Path:
    """
    Copy raw *.nii.gz images into a temporary folder, renaming them to
    *_0000.nii.gz as required by nnUNet v2.

    Returns path to the temporary folder (caller should clean up).
    """
    logger = logging.getLogger(__name__)
    tmp_dir = Path(tempfile.mkdtemp(prefix="boneseg_predict_"))

    converted = 0
    for img_path in sorted(raw_input_dir.glob("*.nii.gz")):
        if img_path.name.endswith("-seg.nii.gz"):
            continue  # skip segmentation masks
        stem = img_path.name.replace(".nii.gz", "")
        dst  = tmp_dir / f"{stem}_0000.nii.gz"
        shutil.copy2(str(img_path), str(dst))
        converted += 1

    logger.info("Prepared %d images in temporary input folder: %s", converted, tmp_dir)
    if converted == 0:
        raise FileNotFoundError(
            f"No '*.nii.gz' image files found in '{raw_input_dir}'."
        )
    return tmp_dir


def prepare_input_folder_2d(raw_input_dir: Path, color_mode: str) -> Path:
    """
    Convert raw 2-D images (jpg / png, possibly nested) into a flat temporary
    folder of nnUNet-format ``{case}_0000.png`` files.

    Case names are derived from each image's path relative to *raw_input_dir*
    (same scheme as data preparation) so predictions can later be matched back
    to their source images. Returns the path to the temporary folder.

    This reuses :func:`src.tree_io_2d.prepare_input_tree_2d` and discards the
    case→relative-path mapping, since the flat path does not preserve the input
    directory layout.
    """
    tmp_dir, _ = prepare_input_tree_2d(raw_input_dir, color_mode)
    return tmp_dir


def run_predict(
    cfg: dict,
    input_dir: Path,
    output_dir: Path,
    fold: str,
    checkpoint: str,
    save_probabilities: bool,
    disable_tta: bool,
) -> None:
    """Run nnUNetv2_predict."""
    logger = logging.getLogger(__name__)

    dataset_id    = int(cfg["dataset"]["id"])
    configuration = cfg["training"]["configuration"]
    trainer       = cfg["training"].get("trainer", "nnUNetTrainer")
    step_size     = float(cfg["inference"].get("step_size", 0.5))

    ensure_dir(output_dir)

    # nnUNetv2_predict accepts -f 0 1 2 3 4 (multiple values) for ensemble,
    # or -f 0 for a single fold. "all" is expanded to all 5 folds here.
    fold_args: list[str] = []
    if str(fold).lower() == "all":
        fold_args = ["-f", "0", "1", "2", "3", "4"]
    else:
        fold_args = ["-f", str(fold)]

    cmd = [
        "nnUNetv2_predict",
        "-d", str(dataset_id),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-c", configuration,
        *fold_args,
        "-chk", checkpoint,
        "-step_size", str(step_size),
        "-tr", trainer,
    ]

    if save_probabilities:
        cmd.append("--save_probabilities")

    if disable_tta:
        cmd.append("--disable_tta")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info(line)
    if result.returncode != 0:
        logger.error(
            "nnUNetv2_predict exited with code %d.\nSTDERR:\n%s",
            result.returncode, result.stderr or "(no stderr)",
        )
        sys.exit(result.returncode)

    logger.info("Predictions written to: %s", output_dir)


def main() -> None:
    args   = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    set_nnunet_env(cfg)

    # ── Resolve fold ──────────────────────────────────────────────────────
    if args.fold is not None:
        fold = args.fold
    else:
        fold_cfg = cfg["inference"].get("fold", 0)
        fold = str(fold_cfg)

    # ── Resolve checkpoint ────────────────────────────────────────────────
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        checkpoint = cfg["inference"].get("checkpoint", "checkpoint_best.pth")

    save_probs  = args.save_probabilities or cfg["inference"].get("save_probabilities", False)
    disable_tta = args.disable_tta or cfg["inference"].get("disable_tta", False)

    # ── Prepare input folder ──────────────────────────────────────────────
    data_type = get_data_type(cfg)
    color_mode = cfg.get("data", {}).get("color_mode", image_io.COLOR_MODE_GRAYSCALE)
    mirror_tree = args.mirror_tree or cfg["inference"].get("mirror_tree", False)
    logger.info("Data type: %s", data_type)

    # The mirror-tree output mode reconstructs the input directory layout, which
    # only makes sense for raw 2-D image trees discovered via --input_raw.
    if mirror_tree and (data_type != "2d" or args.input_raw is None):
        logger.error(
            "--mirror_tree requires data_type '2d' and --input_raw "
            "(raw nested image tree). Got data_type='%s', input_raw=%s.",
            data_type, args.input_raw,
        )
        sys.exit(1)

    tmp_dir: "Path | None" = None      # temporary nnUNet input folder
    tmp_out_dir: "Path | None" = None  # temporary flat prediction folder (mirror mode)
    case_to_rel: dict = {}

    if args.input is not None:
        input_dir = Path(args.input)
        if not input_dir.exists():
            logger.error("Input directory not found: %s", input_dir)
            sys.exit(1)
    elif args.input_raw is not None:
        raw_dir = Path(args.input_raw)
        if not raw_dir.exists():
            logger.error("Raw input directory not found: %s", raw_dir)
            sys.exit(1)
        if data_type == "2d":
            if mirror_tree:
                tmp_dir, case_to_rel = prepare_input_tree_2d(raw_dir, color_mode)
            else:
                tmp_dir = prepare_input_folder_2d(raw_dir, color_mode)
        else:
            tmp_dir = prepare_input_folder(raw_dir)
        input_dir = tmp_dir
    else:
        logger.error("Provide --input (nnUNet format) or --input_raw (raw images).")
        sys.exit(1)

    final_output_dir = Path(args.output)
    # In mirror mode, nnUNet predicts into a flat temporary folder; predictions
    # are then scattered into the mirrored tree at *final_output_dir*.
    if mirror_tree:
        tmp_out_dir = Path(tempfile.mkdtemp(prefix="seg2d_predict_out_"))
        predict_output_dir = tmp_out_dir
    else:
        predict_output_dir = final_output_dir

    try:
        run_predict(
            cfg=cfg,
            input_dir=input_dir,
            output_dir=predict_output_dir,
            fold=fold,
            checkpoint=checkpoint,
            save_probabilities=save_probs,
            disable_tta=disable_tta,
        )
        if mirror_tree:
            scatter_predictions_to_tree(
                predict_output_dir, case_to_rel, final_output_dir,
            )
    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
        if tmp_out_dir is not None and tmp_out_dir.exists():
            shutil.rmtree(str(tmp_out_dir), ignore_errors=True)

    if mirror_tree:
        logger.info("Done. Predictions mirror the input tree at: %s", final_output_dir)
    else:
        logger.info("Done. To evaluate predictions, run:")
        logger.info("  python scripts/04_evaluate.py --config %s "
                    "--pred_dir %s --gt_dir <ground_truth_dir>",
                    args.config, final_output_dir)


if __name__ == "__main__":
    main()
