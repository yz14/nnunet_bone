"""
Step 2 — Train the nnUNet bone segmentation model.

What this script does:
  1. Sets nnUNet environment variables
  2. Optionally runs plan_and_preprocess (if not done yet)
  3. Launches nnUNetv2_train for the configured fold / configuration

Prerequisites:
  - scripts/01_prepare_data.py must have been run first

Usage:
    python scripts/02_train.py
    python scripts/02_train.py --config configs/default.yaml
    python scripts/02_train.py --config configs/default.yaml --fold 1
    python scripts/02_train.py --config configs/default.yaml --all_folds
    python scripts/02_train.py --config configs/default.yaml --preprocess_only
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    get_data_type,
    get_dataset_dir_name,
    get_nnunet_dirs,
    load_config,
    set_nnunet_env,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train nnUNet v2 bone segmentation model.",
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
    """Run nnUNetv2_plan_and_preprocess."""
    logger = logging.getLogger(__name__)

    dataset_id    = int(cfg["dataset"]["id"])
    configuration = cfg["training"]["configuration"]
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
    """Run nnUNetv2_train for a single fold."""
    import os as _os
    logger = logging.getLogger(__name__)

    dataset_id    = int(cfg["dataset"]["id"])
    configuration = cfg["training"]["configuration"]
    trainer       = cfg["training"].get("trainer", "nnUNetTrainer")
    num_gpus      = int(cfg["training"].get("num_gpus", 1))
    use_amp       = cfg["training"].get("use_amp", True)
    num_proc_da   = cfg["training"].get("num_proc_da", 12)
    # ↓ 新增
    pretrained_weights = cfg["training"].get("pretrained_weights", None)

    _os.environ["nnUNet_n_proc_DA"] = str(int(num_proc_da))
    logger.info("nnUNet_n_proc_DA = %d", int(num_proc_da))

    if _os.name == "nt":
        _os.environ["nnUNet_compile"] = "false"

    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        configuration,
        str(fold),
        "-tr", trainer,
        "-num_gpus", str(num_gpus),
    ]

    # ↓ 新增：只要 config 里填了路径就透传给 nnUNet
    if pretrained_weights:
        cmd += ["-pretrained_weights", pretrained_weights]
        logger.info("Pretrained weights: %s", pretrained_weights)

    if continue_training:
        cmd.append("--c")

    logger.info(
        "Training fold %d | config: %s | dataset: %d | trainer: %s | AMP: %s",
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
    """
    Run nnUNetv2_find_best_configuration after all folds to determine
    the optimal ensemble / single-fold setup for inference.
    """
    logger = logging.getLogger(__name__)
    dataset_id    = int(cfg["dataset"]["id"])
    configuration = cfg["training"]["configuration"]

    # dataset_name_or_id is a positional argument (no -d flag)
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
    set_nnunet_env(cfg)

    # Training itself is modality-agnostic: nnUNet decides 2-D vs 3-D from
    # `training.configuration` (e.g. "2d" for the 2-D image pipeline).
    logger.info(
        "Data type: %s | nnUNet configuration: %s",
        get_data_type(cfg), cfg["training"]["configuration"],
    )

    # ── Determine folds to train ───────────────────────────────────────────
    if args.all_folds:
        folds = list(range(5))
    else:
        fold_cfg = cfg["training"].get("fold", 0)
        folds = [args.fold if args.fold is not None else int(fold_cfg)]

    # ── Verify dataset exists ─────────────────────────────────────────────
    nnunet_dirs  = get_nnunet_dirs(cfg)
    dataset_name = get_dataset_dir_name(cfg)
    dataset_raw  = nnunet_dirs["raw"] / dataset_name

    if not dataset_raw.exists():
        logger.error(
            "Dataset directory not found: %s\n"
            "Run 'python scripts/01_prepare_data.py' first.",
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

    # ── Find best config after all-fold training ───────────────────────────
    if args.all_folds:
        logger.info("All folds done. Running find_best_configuration ...")
        find_best_configuration(cfg)

    logger.info("Training pipeline complete.")
    logger.info(
        "Next step: python scripts/03_predict.py --config %s", args.config
    )


if __name__ == "__main__":
    main()
