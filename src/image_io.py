"""
2-D image IO helpers for ultrasound / endoscopy segmentation.

The 3-D CT pipeline works with NIfTI volumes (``nibabel``). 2-D natural images
(ultrasound, endoscopy) are stored as ``.jpg`` / ``.png`` instead, so this
module provides the equivalent read / write primitives used by the 2-D
data-preparation, prediction and evaluation code paths.

nnUNet v2 native 2-D format
---------------------------
nnUNet reads 2-D natural images with its ``NaturalImage2DIO`` reader, which
expects **one file per case channel group** named ``{CASE}_{XXXX}.png``:

* a grayscale ``.png`` → a single input channel
* an RGB ``.png``      → three input channels (R, G, B)

Labels are stored as ``labelsTr/{CASE}.png`` holding the integer class IDs
(``0`` = background, ``1`` = target, ...) as raw pixel values — they are **not**
scaled to the 0-255 display range.

Everything here intentionally avoids any dependency on ``nibabel`` so the 2-D
pipeline can run in environments that only have Pillow + numpy installed.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Image extensions recognised when discovering raw 2-D images / masks.
IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
)

# nnUNet v2 always stores 2-D natural images as PNG (lossless) regardless of
# the source format, so masks/labels survive without JPEG compression noise.
NNUNET_2D_FILE_ENDING = ".png"

# Supported colour modes for the input images.
COLOR_MODE_GRAYSCALE = "grayscale"
COLOR_MODE_RGB = "rgb"


# ─────────────────────────────────────────────────────────────────────────────
# Channel / mode helpers
# ─────────────────────────────────────────────────────────────────────────────

def num_channels_for_mode(color_mode: str) -> int:
    """Return the number of input channels produced by *color_mode*."""
    mode = color_mode.strip().lower()
    if mode == COLOR_MODE_GRAYSCALE:
        return 1
    if mode == COLOR_MODE_RGB:
        return 3
    raise ValueError(
        f"Unknown color_mode '{color_mode}'. Expected "
        f"'{COLOR_MODE_GRAYSCALE}' or '{COLOR_MODE_RGB}'."
    )


def channel_names_for_mode(color_mode: str) -> Dict[str, str]:
    """
    Build the nnUNet ``dataset.json`` ``channel_names`` block for *color_mode*.

    grayscale → {"0": "image"}
    rgb       → {"0": "R", "1": "G", "2": "B"}
    """
    mode = color_mode.strip().lower()
    if mode == COLOR_MODE_GRAYSCALE:
        return {"0": "image"}
    if mode == COLOR_MODE_RGB:
        return {"0": "R", "1": "G", "2": "B"}
    raise ValueError(
        f"Unknown color_mode '{color_mode}'. Expected "
        f"'{COLOR_MODE_GRAYSCALE}' or '{COLOR_MODE_RGB}'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reading
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path, color_mode: str = COLOR_MODE_GRAYSCALE) -> np.ndarray:
    """
    Load a 2-D image and return it as a uint8 numpy array.

    Parameters
    ----------
    path : Path
        Source image file (jpg / png / ...).
    color_mode : str
        ``"grayscale"`` → returns shape ``(H, W)``.
        ``"rgb"``       → returns shape ``(H, W, 3)``.

    The image is converted to the requested mode so downstream code never has
    to guess the source channel layout.
    """
    mode = color_mode.strip().lower()
    with Image.open(path) as img:
        if mode == COLOR_MODE_GRAYSCALE:
            arr = np.asarray(img.convert("L"))
        elif mode == COLOR_MODE_RGB:
            arr = np.asarray(img.convert("RGB"))
        else:
            raise ValueError(
                f"Unknown color_mode '{color_mode}'. Expected "
                f"'{COLOR_MODE_GRAYSCALE}' or '{COLOR_MODE_RGB}'."
            )
    return np.ascontiguousarray(arr)


def load_mask_as_label(
    path: Path,
    threshold: int = 127,
    binary: bool = True,
) -> np.ndarray:
    """
    Load an annotation mask and convert it to an integer label map (uint8).

    Masks shipped as ``.jpg`` are lossy: a nominally binary mask contains
    compression noise around the foreground edges, so a simple ``> threshold``
    binarisation is applied to recover a clean ``{0, 1}`` label map.

    Parameters
    ----------
    path : Path
        Mask file (jpg / png / ...).
    threshold : int
        Grayscale cut-off (0-255). Pixels strictly greater than this become
        foreground (1). Only used when ``binary`` is True.
    binary : bool
        When True (default) the mask is binarised to ``{0, 1}``. When False
        the raw grayscale values are returned unchanged (uint8) — useful when
        the mask already encodes integer multi-class IDs as pixel values.
    """
    gray = load_image(path, COLOR_MODE_GRAYSCALE)
    if binary:
        return (gray > int(threshold)).astype(np.uint8)
    return gray.astype(np.uint8)


def load_label_png(path: Path) -> np.ndarray:
    """
    Load a label / prediction PNG produced by nnUNet, preserving the exact
    integer class IDs stored as pixel values. Returns a uint8 array.
    """
    return load_image(path, COLOR_MODE_GRAYSCALE).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────────────

def save_nnunet_image(
    image: np.ndarray,
    case_name: str,
    output_dir: Path,
) -> Path:
    """
    Write a 2-D image as the single nnUNet input file ``{case}_0000.png``.

    A grayscale array (H, W) yields a 1-channel PNG; an RGB array (H, W, 3)
    yields a 3-channel PNG. nnUNet's ``NaturalImage2DIO`` reader expands these
    into 1 / 3 input channels respectively at training / inference time.

    Returns the path that was written.
    """
    out_path = output_dir / f"{case_name}_0000{NNUNET_2D_FILE_ENDING}"
    Image.fromarray(image.astype(np.uint8)).save(out_path)
    return out_path


def save_label(
    label: np.ndarray,
    case_name: str,
    output_dir: Path,
) -> Path:
    """
    Write an integer label map as ``{case}.png`` with raw class-ID pixel
    values (NOT scaled to 0-255). Returns the path that was written.
    """
    out_path = output_dir / f"{case_name}{NNUNET_2D_FILE_ENDING}"
    Image.fromarray(label.astype(np.uint8)).save(out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_images_recursive(root: Path) -> List[Path]:
    """
    Recursively find all image files under *root* (sorted, deterministic).

    Matches any extension in :data:`IMAGE_EXTENSIONS` (case-insensitive).
    """
    found: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            found.append(p)
    return found
