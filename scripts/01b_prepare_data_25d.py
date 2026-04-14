"""
Step 1b — Prepare 2.5-D nnUNet v2 dataset from raw Total Segmentator data.

Unlike the standard 3-D preparation (01_prepare_data.py), this script creates
one nnUNet training case per (original_case, axial_slice) pair.  Each such
case contains multiple adjacent slices stacked as separate input channels,
giving a 2-D U-Net access to 3-D anatomical context.

Usage:
    python scripts/01b_prepare_data_25d.py
    python scripts/01b_prepare_data_25d.py --config configs/default.yaml
    python scripts/01b_prepare_data_25d.py --config configs/default.yaml --dry_run
    python scripts/01b_prepare_data_25d.py --config configs/default.yaml --auto_preprocess
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.label_config import LabelConfig
from src.slices_builder import prepare_25d_dataset
from src.utils import load_config, set_nnunet_env, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare 2.5-D nnUNet v2 dataset from Total Segmentator data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without writing any files",
    )
    parser.add_argument(
        "--auto_preprocess", action="store_true",
        help="Automatically run nnUNet preprocessing without interactive prompt",
    )
    parser.add_argument(
        "--log_level", default="INFO",
        help="Logging level: DEBUG / INFO / WARNING",
    )
    return parser.parse_args()


def run_nnunet_plan_and_preprocess(cfg: dict) -> None:
    """
    Run nnUNetv2_plan_and_preprocess for the 2.5-D dataset.
    The 2.5-D dataset uses the same dataset ID as the standard dataset (configured
    in default.yaml) but with a distinct channel configuration; nnUNet will create
    a separate preprocessing folder under nnUNet_preprocessed/ for it.
    """
    logger = logging.getLogger(__name__)

    dataset_id   = int(cfg["dataset"]["id"])
    configuration = cfg["training_25d"].get("configuration", "2d")
    num_procs   = int(cfg["preprocessing"].get("num_processes", 4))
    verify      = cfg["preprocessing"].get("verify_dataset_integrity", True)

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "-c", configuration,
        "-np", str(num_procs),
    ]
    if verify:
        cmd.append("--verify_dataset_integrity")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        logger.info("STDOUT:\n%s", result.stdout)
    if result.returncode != 0:
        logger.error(
            "nnUNetv2_plan_and_preprocess exited with code %d.\nSTDERR:\n%s",
            result.returncode,
            result.stderr or "(no stderr)",
        )
        logger.error(
            "Fix the errors above, then re-run this command manually:\n"
            "  nnUNetv2_plan_and_preprocess -d %s -c %s -np %s%s",
            dataset_id,
            configuration,
            num_procs,
            " --verify_dataset_integrity" if verify else "",
        )
        sys.exit(result.returncode)
    logger.info("nnUNetv2_plan_and_preprocess finished successfully.")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # ── Load configuration ────────────────────────────────────────────────
    logger.info("Loading config: %s", args.config)
    cfg = load_config(args.config)

    if "segmentation_25d" not in cfg:
        logger.error(
            "Config is missing the required 'segmentation_25d' section. "
            "Add it to configs/default.yaml before running this script."
        )
        sys.exit(1)

    # ── Set nnUNet environment variables ──────────────────────────────────
    if not args.dry_run:
        set_nnunet_env(cfg)

    # ── Load label configuration ──────────────────────────────────────────
    label_cfg = LabelConfig(cfg["paths"]["labels_config"])

    # ── Prepare 2.5-D dataset ────────────────────────────────────────────
    dataset_dir, images_tr_dir, labels_tr_dir, num_samples = prepare_25d_dataset(
        cfg, label_cfg, dry_run=args.dry_run,
    )

    if args.dry_run:
        logger.info("[DRY RUN] No files were written. Remove --dry_run to proceed.")
        return

    logger.info("2.5-D Dataset created at: %s", dataset_dir)
    logger.info("  Total samples (case x slices): %d", num_samples)
    logger.info("  Images dir : %s", images_tr_dir)
    logger.info("  Labels dir : %s", labels_tr_dir)

    # ── Optionally run preprocessing ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("2.5-D dataset preparation complete.")
    print(f"  Dataset dir  : {dataset_dir}")
    print(f"  Dataset ID   : {cfg['dataset']['id']}")
    print(f"  Total samples: {num_samples}")
    print()
    print("Next step — run nnUNet preprocessing:")
    print(f"  python scripts/02_train_25d.py --config {args.config}")
    print()

    auto_run = (
        args.auto_preprocess
        or args.dry_run
        or not sys.stdin.isatty()
    )

    if auto_run:
        answer = "y"
    else:
        answer = input("Run nnUNetv2_plan_and_preprocess now? [y/N] ").strip().lower()

    if answer == "y":
        run_nnunet_plan_and_preprocess(cfg)
    else:
        dataset_id    = int(cfg["dataset"]["id"])
        configuration = cfg["training_25d"].get("configuration", "2d")
        num_procs    = int(cfg["preprocessing"].get("num_processes", 4))
        print("\nTo preprocess manually, run:")
        print(
            f"  nnUNetv2_plan_and_preprocess -d {dataset_id} "
            f"-c {configuration} "
            f"-np {num_procs} "
            f"--verify_dataset_integrity"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
