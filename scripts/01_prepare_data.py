"""
Step 1 — Prepare nnUNet v2 dataset from raw Total Segmentator data.

What this script does:
  1. Reads raw CT images and Total Segmentator segmentation masks
  2. Remaps labels according to configs/labels.yaml + segmentation mode
  3. Writes nnUNet-formatted imagesTr / labelsTr directories
  4. Generates dataset.json

After running this script, proceed with:
    python scripts/02_train.py --config configs/default.yaml

Usage:
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --config configs/default.yaml
    python scripts/01_prepare_data.py --config configs/default.yaml --dry_run
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep import prepare_nnunet_dataset
from src.label_config import LabelConfig
from src.utils import load_config, set_nnunet_env, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare nnUNet v2 dataset from Total Segmentator data.",
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
    Run nnUNetv2_plan_and_preprocess after dataset creation.
    Optionally called at the end of this script.

    Parameters
    ----------
    cfg : dict
        Loaded configuration dictionary.
    auto_yes : bool
        If True, skip the interactive prompt and run immediately.
    """
    logger = logging.getLogger(__name__)

    dataset_id   = int(cfg["dataset"]["id"])
    num_procs    = int(cfg["preprocessing"].get("num_processes", 4))
    verify       = cfg["preprocessing"].get("verify_dataset_integrity", True)
    configuration = cfg["training"]["configuration"]

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "-c", configuration,
        "-np", str(num_procs),
    ]
    if verify:
        cmd.append("--verify_dataset_integrity")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
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

    # ── Set nnUNet environment variables ──────────────────────────────────
    if not args.dry_run:
        set_nnunet_env(cfg)

    # ── Load label configuration ──────────────────────────────────────────
    label_cfg = LabelConfig(cfg["paths"]["labels_config"])

    # ── Prepare dataset ───────────────────────────────────────────────────
    dataset_dir = prepare_nnunet_dataset(cfg, label_cfg, dry_run=args.dry_run)
    # dataset_dir = '/data0/yzhen/projects/BoneSeg/nnunet_workspace/nnUNet_raw/Dataset100_BoneSegmentation'

    if args.dry_run:
        logger.info("[DRY RUN] No files were written. Remove --dry_run to proceed.")
        return

    logger.info("Dataset created at: %s", dataset_dir)

    # ── Optionally run preprocessing ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("Dataset preparation complete.")
    print(f"  Dataset dir : {dataset_dir}")
    print(f"  Dataset ID  : {cfg['dataset']['id']}")
    print()
    print("Next step — run nnUNet preprocessing:")
    print(f"  python scripts/02_train.py --config {args.config}")
    print()

    # Support both --auto_preprocess flag and non-interactive (CI/CD) environments.
    # In non-interactive mode (stdin is not a TTY), skip the prompt automatically.
    auto_run = (
        args.auto_preprocess
        or args.dry_run  # dry_run implies no interactive input
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
        configuration = cfg["training"]["configuration"]
        num_procs     = int(cfg["preprocessing"].get("num_processes", 4))
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
