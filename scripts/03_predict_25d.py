"""
Step 3b — Run 2.5-D inference with the trained nnUNet model.

Usage:
    # Predict on raw CT images:
    python scripts/03_predict_25d.py --config configs/default.yaml \
        --input_raw /path/to/images --output /path/to/preds

    # Ensemble all folds:
    python scripts/03_predict_25d.py --config configs/default.yaml \
        --input_raw /path/to/images --output /path/to/preds --fold all
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_dataset_dir_name, load_config, set_nnunet_env, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 2.5-D nnUNet v2 bone segmentation inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input folder with pre-formatted nnUNet images (*_000X.nii.gz)",
    )
    parser.add_argument(
        "--input_raw", type=str, default=None,
        help="Input folder with raw *.nii.gz CT images (auto-converted to 2.5-D format)",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--fold", default=None,
        help="Fold(s): 0/1/2/3/4 or 'all' (default: from config)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint: 'checkpoint_best.pth' or 'checkpoint_final.pth'",
    )
    parser.add_argument(
        "--save_probabilities", action="store_true",
        help="Save softmax probability maps",
    )
    parser.add_argument(
        "--disable_tta", action="store_true",
        help="Disable test-time augmentation",
    )
    parser.add_argument(
        "--log_level", default="INFO",
    )
    return parser.parse_args()


def prepare_input_folder_25d(raw_input_dir: Path) -> Path:
    """
    Prepare 2.5-D multi-channel input from raw CT images.

    For each ``*.nii.gz`` file found in *raw_input_dir*, this function:
      1. Loads the full 3-D volume
      2. Extracts adjacent-slice sub-volumes for each axial slice
      3. Writes them as separate nnUNet case files (one per slice)

    Returns path to the temporary folder (caller should clean up).
    """
    logger = logging.getLogger(__name__)
    tmp_dir = Path(tempfile.mkdtemp(prefix="boneseg_25d_predict_"))

    num_channels  = 5
    channel_depth  = 3
    half_ch  = num_channels // 2
    half_d   = channel_depth // 2

    for nii_path in sorted(raw_input_dir.glob("*.nii.gz")):
        if nii_path.name.endswith("-seg.nii.gz"):
            continue

        vol = nib.load(str(nii_path))
        data = np.asarray(vol.dataobj, dtype=np.int16)
        D, H, W = data.shape
        case_id = nii_path.stem.replace(".nii", "")

        logger.info("  Preparing %d slices for %s", D, case_id)

        for z in range(D):
            slice_case_name = f"{case_id}_s{z:05d}"

            for ch in range(num_channels):
                src_idx = z - half_ch + ch
                subvol  = _extract_subvolume(data, src_idx, channel_depth)
                out_path = tmp_dir / f"{slice_case_name}_000{chr(ord('0') + ch)}.nii.gz"
                nib.save(nib.Nifti1Image(subvol.astype(np.int16), np.eye(4)), str(out_path))

    logger.info("Prepared 2.5-D input folder: %s", tmp_dir)
    return tmp_dir


def _extract_subvolume(volume: np.ndarray, center_idx: int, depth: int) -> np.ndarray:
    """Extract a depth-slice sub-volume centred on center_idx, zero-padded at boundaries."""
    D, H, W = volume.shape
    half = depth // 2
    out = np.zeros((depth, H, W), dtype=volume.dtype)
    for d in range(depth):
        src = center_idx - half + d
        if 0 <= src < D:
            out[d] = volume[src]
    return out


def run_predict(
    cfg: dict,
    input_dir: Path,
    output_dir: Path,
    fold: str,
    checkpoint: str,
    save_probabilities: bool,
    disable_tta: bool,
) -> None:
    """Run nnUNetv2_predict with 2.5-D configuration."""
    logger = logging.getLogger(__name__)

    dataset_id    = int(cfg["dataset"]["id"])
    configuration = cfg["training_25d"].get("configuration", "2d")
    trainer       = cfg["training_25d"].get("trainer", "nnUNetTrainer")
    step_size     = float(cfg["inference_25d"].get("step_size", 0.5))

    ensure_dir(output_dir)

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

    if "training_25d" not in cfg:
        logger.error("Config missing 'training_25d' section.")
        sys.exit(1)

    set_nnunet_env(cfg)

    # Resolve fold
    if args.fold is not None:
        fold = args.fold
    else:
        fold = str(cfg["inference_25d"].get("fold", 0))

    # Resolve checkpoint
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        checkpoint = cfg["inference_25d"].get("checkpoint", "checkpoint_best.pth")

    save_probs  = args.save_probabilities or cfg["inference_25d"].get("save_probabilities", False)
    disable_tta = args.disable_tta or cfg["inference_25d"].get("disable_tta", False)

    # Prepare input
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
        tmp_dir = prepare_input_folder_25d(raw_dir)
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

    logger.info(
        "2.5-D inference complete. Predictions in: %s", output_dir
    )
    logger.info(
        "To evaluate, run: python scripts/04_evaluate.py --config %s "
        "--pred_dir %s --gt_dir <ground_truth_dir>",
        args.config, output_dir,
    )


if __name__ == "__main__":
    main()
