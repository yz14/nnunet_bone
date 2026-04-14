"""
Step 2b — Train the nnUNet bone segmentation model using 2.5-D input.

This script is identical to 02_train.py except that it:
  1. Uses the ``training_25d`` section of default.yaml for configuration
  2. Trains on the 2.5-D dataset prepared by scripts/01b_prepare_data_25d.py

Usage:
    python scripts/02_train_25d.py --config configs/default.yaml
    python scripts/02_train_25d.py --config configs/default.yaml --fold 1
    python scripts/02_train_25d.py --config configs/default.yaml --all_folds
    python scripts/02_train_25d.py --config configs/default.yaml --preprocess_only
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    get_dataset_dir_name,
    get_nnunet_dirs,
    load_config,
    set_nnunet_env,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train nnUNet v2 bone segmentation model (2.5-D).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Override fold number (0-4). Defaults to value in config.",
    )
    parser.add_argument(
        "--all_folds", action="store_true",
        help="Train all 5 folds sequentially (ignores --fold).",
    )
    parser.add_argument(
        "--preprocess_only", action="store_true",
        help="Only run plan_and_preprocess, do not start training.",
    )
    parser.add_argument(
        "--skip_preprocess", action="store_true",
        help="Skip plan_and_preprocess and go directly to training.",
    )
    parser.add_argument(
        "--continue_training", action="store_true",
        help="Continue training from latest checkpoint (adds --c flag).",
    )
    parser.add_argument(
        "--log_level", default="INFO",
        help="Logging level: DEBUG / INFO / WARNING",
    )
    return parser.parse_args()


def run_plan_and_preprocess(cfg: dict) -> None:
    """Run nnUNetv2_plan_and_preprocess for the 2.5-D dataset."""
    logger = logging.getLogger(__name__)

    dataset_id    = int(cfg["dataset"]["id"])
    configuration = cfg["training_25d"].get("configuration", "2d")
    num_procs     = int(cfg["preprocessing"].get("num_processes", 4))
    verify        = cfg["preprocessing"].get("verify_dataset_integrity", True)

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
            "plan_and_preprocess failed with code %d.\nSTDERR:\n%s",
            result.returncode,
            result.stderr or "(no stderr)",
        )
        sys.exit(result.returncode)
    logger.info("plan_and_preprocess finished successfully.")


def run_training(cfg: dict, fold: int, continue_training: bool = False) -> None:
    """Run nnUNetv2_train for a single fold (2.5-D configuration)."""
    import os as _os
    logger = logging.getLogger(__name__)

    dataset_id    = int(cfg["dataset"]["id"])
    configuration = cfg["training_25d"].get("configuration", "2d")
    trainer       = cfg["training_25d"].get("trainer", "nnUNetTrainer")
    num_gpus      = int(cfg["training_25d"].get("num_gpus", 1))
    use_amp       = cfg["training_25d"].get("use_amp", True)
    num_proc_da   = int(cfg["training_25d"].get("num_proc_da", 4))

    # Set number of data augmentation worker processes.
    _os.environ["nnUNet_n_proc_DA"] = str(num_proc_da)
    logger.info("nnUNet_n_proc_DA = %d  (data augmentation workers)", num_proc_da)

    # Disable torch.compile on Windows — Triton is Linux-only
    if _os.name == "nt":
        _os.environ["nnUNet_compile"] = "false"
        logger.info("nnUNet_compile = false  (Triton unavailable on Windows)")

    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        configuration,
        str(fold),
        "-tr", trainer,
        "-num_gpus", str(num_gpus),
    ]

    if continue_training:
        cmd.append("--c")

    logger.info(
        "2.5-D Training fold %d  |  config: %s  |  dataset: %d  |  trainer: %s  |  AMP: %s",
        fold, configuration, dataset_id, trainer, "ON" if use_amp else "OFF",
    )
    logger.info("Command: %s", " ".join(cmd))
    logger.info("=" * 60)

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info(line)
    if result.returncode != 0:
        logger.error(
            "Training (fold %d) exited with code %d.\nSTDERR:\n%s",
            fold, result.returncode, result.stderr or "(no stderr)",
        )
        sys.exit(result.returncode)

    logger.info("Training fold %d completed successfully.", fold)


def find_best_configuration(cfg: dict) -> None:
    """Run nnUNetv2_find_best_configuration after all folds."""
    logger = logging.getLogger(__name__)
    dataset_id    = int(cfg["dataset"]["id"])
    configuration = cfg["training_25d"].get("configuration", "2d")

    cmd = [
        "nnUNetv2_find_best_configuration",
        str(dataset_id),
        "-c", configuration,
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info(line)
    if result.returncode != 0:
        logger.error(
            "find_best_configuration exited with code %d.\nSTDERR:\n%s",
            result.returncode, result.stderr or "(no stderr)",
        )
    else:
        logger.info("find_best_configuration finished.")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)

    # Verify 2.5D config section exists
    if "training_25d" not in cfg:
        logger.error(
            "Config is missing the required 'training_25d' section. "
            "Add it to configs/default.yaml before running this script."
        )
        sys.exit(1)

    set_nnunet_env(cfg)

    # ── Determine folds to train ───────────────────────────────────────────
    if args.all_folds:
        folds = list(range(5))
    else:
        fold_cfg = cfg["training_25d"].get("fold", 0)
        folds = [args.fold if args.fold is not None else int(fold_cfg)]

    # ── Verify dataset exists ─────────────────────────────────────────────
    nnunet_dirs  = get_nnunet_dirs(cfg)
    dataset_name = get_dataset_dir_name(cfg)
    dataset_raw  = nnunet_dirs["raw"] / dataset_name

    if not dataset_raw.exists():
        logger.error(
            "Dataset directory not found: %s\n"
            "Run 'python scripts/01b_prepare_data_25d.py' first.",
            dataset_raw,
        )
        sys.exit(1)

    # ── Preprocessing ─────────────────────────────────────────────────────
    preprocessed_dir = nnunet_dirs["preprocessed"] / dataset_name
    needs_preprocess = not preprocessed_dir.exists() or not any(preprocessed_dir.iterdir())

    if not args.skip_preprocess and (needs_preprocess or args.preprocess_only):
        logger.info("Running plan_and_preprocess ...")
        run_plan_and_preprocess(cfg)
    elif args.skip_preprocess:
        logger.info("Skipping plan_and_preprocess as requested.")
    else:
        logger.info("Preprocessed data already exists — skipping plan_and_preprocess.")

    if args.preprocess_only:
        logger.info("--preprocess_only set. Done.")
        return

    # ── Training ──────────────────────────────────────────────────────────
    for fold in folds:
        run_training(cfg, fold, continue_training=args.continue_training)

    # ── Find best config after all-fold training ────────────────────────────
    if args.all_folds:
        logger.info("All folds done. Running find_best_configuration ...")
        find_best_configuration(cfg)

    logger.info("2.5-D Training pipeline complete.")
    logger.info(
        "Next step: python scripts/03_predict_25d.py --config %s", args.config
    )


if __name__ == "__main__":
    main()
