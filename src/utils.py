"""
Utility functions: logging, path management, config loading.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Union

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(
    log_level: str = "INFO",
    log_file: Union[str, Path, None] = None,
) -> None:
    """Configure root logger with console (and optional file) handler."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Return the project root (parent of `src/`)."""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file and return as dict. Raises FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load the main experiment config (default.yaml or similar).
    Resolves relative paths in the config relative to the project root.
    Raises KeyError if required top-level sections are missing.
    """
    cfg = load_yaml(config_path)
    root = get_project_root()

    # Validate required top-level sections
    required_sections = ["paths", "dataset", "segmentation"]
    for section in required_sections:
        if section not in cfg:
            raise KeyError(
                f"Config '{config_path}' is missing required section: '{section}'. "
                f"Available sections: {list(cfg.keys())}"
            )

    # Resolve relative path strings under `paths:` section
    if "paths" in cfg:
        for key, val in cfg["paths"].items():
            if isinstance(val, str) and not Path(val).is_absolute():
                cfg["paths"][key] = str(root / val)

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# nnUNet environment helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_nnunet_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """
    Derive the three required nnUNet environment directories from config.

    Returns a dict with keys: raw, preprocessed, results.
    Raises KeyError if 'paths.nnunet_workspace' is missing from config.
    """
    if "nnunet_workspace" not in cfg.get("paths", {}):
        raise KeyError(
            "Config is missing 'paths.nnunet_workspace'. "
            "Add it to your default.yaml or ensure load_config() was used."
        )
    workspace = Path(cfg["paths"]["nnunet_workspace"])
    return {
        "raw":          workspace / "nnUNet_raw",
        "preprocessed": workspace / "nnUNet_preprocessed",
        "results":      workspace / "nnUNet_results",
    }


def set_nnunet_env(cfg: Dict[str, Any]) -> None:
    """
    Set nnUNet v2 environment variables (nnUNet_raw, nnUNet_preprocessed,
    nnUNet_results) from config paths, and create the directories.
    """
    dirs = get_nnunet_dirs(cfg)
    for key, path in dirs.items():
        ensure_dir(path)

    os.environ["nnUNet_raw"]          = str(dirs["raw"])
    os.environ["nnUNet_preprocessed"] = str(dirs["preprocessed"])
    os.environ["nnUNet_results"]      = str(dirs["results"])

    logger = logging.getLogger(__name__)
    logger.info("nnUNet_raw          = %s", dirs["raw"])
    logger.info("nnUNet_preprocessed = %s", dirs["preprocessed"])
    logger.info("nnUNet_results      = %s", dirs["results"])


def get_dataset_dir_name(cfg: Dict[str, Any]) -> str:
    """Return nnUNet v2 dataset directory name, e.g. 'Dataset100_BoneSegmentation'."""
    if "dataset" not in cfg:
        raise KeyError(
            "Config is missing 'dataset' section. "
            "Add it to your default.yaml."
        )
    ds_id   = int(cfg["dataset"]["id"])
    ds_name = cfg["dataset"].get("name")
    if ds_name is None:
        raise KeyError("Config is missing 'dataset.name'.")
    return f"Dataset{ds_id:03d}_{ds_name}"
