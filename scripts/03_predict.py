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

from src.utils import ensure_dir, get_dataset_dir_name, load_config, set_nnunet_env, setup_logging


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
    tmp_dir: "Path | None" = None

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
        tmp_dir = prepare_input_folder(raw_dir)
        input_dir = tmp_dir
    else:
        logger.error("Provide --input (nnUNet format) or --input_raw (raw images).")
        sys.exit(1)

    output_dir = Path(args.output)

    try:
        run_predict(
            cfg=cfg,
            input_dir=input_dir,
            output_dir=output_dir,
            fold=fold,
            checkpoint=checkpoint,
            save_probabilities=save_probs,
            disable_tta=disable_tta,
        )
    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

    logger.info("Done. To evaluate predictions, run:")
    logger.info("  python scripts/04_evaluate.py --config %s "
                "--pred_dir %s --gt_dir <ground_truth_dir>",
                args.config, output_dir)


if __name__ == "__main__":
    main()
