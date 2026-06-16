"""
Directory-tree-preserving IO for 2-D prediction.

The default 2-D prediction path (:mod:`scripts.03_predict`) flattens every raw
image into a single folder of nnUNet ``{case}_0000.png`` inputs and produces a
**flat** folder of ``{case}.png`` predictions. That is convenient for batch
evaluation but loses the original nested layout of the source data.

This module adds an alternative that mirrors the input directory tree: each
prediction is written back under the **same relative path** as its source
image, so an input tree like::

    甲状腺图像报告/甲状腺乳头状癌800/2026-03-30/49664962_唐月芳/CS.../<uid>.jpg

produces an output tree::

    甲状腺图像报告pred/甲状腺乳头状癌800/2026-03-30/49664962_唐月芳/CS.../<uid>.png

Only image files (:data:`src.image_io.IMAGE_EXTENSIONS`) are predicted; any
other files in the tree (``.dcm`` / ``.pdf`` / ``.xlsx`` / ``.rar`` / ...) are
ignored. Predictions are nnUNet 2-D label PNGs, so the mirrored files keep the
source file stem but use a ``.png`` extension.

The mapping from nnUNet case name back to the source relative path is captured
at input-preparation time (rather than reverse-engineering :func:`make_case_name`,
which is intentionally lossy), so it stays robust for arbitrary file names.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

from . import image_io
from .data_prep_2d import make_case_name

logger = logging.getLogger(__name__)


# Maps an nnUNet case name → its source path relative to the raw input root.
CaseToRel = Dict[str, Path]


def prepare_input_tree_2d(
    raw_input_dir: Path,
    color_mode: str,
) -> Tuple[Path, CaseToRel]:
    """
    Convert a nested tree of raw 2-D images into a flat nnUNet input folder,
    recording how each case maps back to its source relative path.

    This is the tree-aware counterpart of the flat preparation used by the
    default prediction path: it produces the exact same ``{case}_0000.png``
    files (so nnUNet inference is identical), but additionally returns a
    ``{case_name: rel_path}`` mapping so predictions can later be scattered
    back into a directory tree mirroring *raw_input_dir*.

    Parameters
    ----------
    raw_input_dir : Path
        Root of the nested raw-image tree (may contain non-image files, which
        are ignored).
    color_mode : str
        ``"grayscale"`` or ``"rgb"`` — see :func:`src.image_io.load_image`.

    Returns
    -------
    (tmp_dir, case_to_rel)
        *tmp_dir* is a fresh temporary folder of ``{case}_0000.png`` inputs
        (the caller is responsible for deleting it). *case_to_rel* maps each
        case name to its source path relative to *raw_input_dir*.

    Raises
    ------
    FileNotFoundError
        If no image files are found under *raw_input_dir*.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="seg2d_predict_"))

    case_to_rel: CaseToRel = {}
    converted = 0
    for img_path in image_io.find_images_recursive(raw_input_dir):
        rel = img_path.relative_to(raw_input_dir)
        case_name = make_case_name(rel)

        if case_name in case_to_rel:
            logger.warning(
                "Case name collision: '%s' from both '%s' and '%s'. "
                "Keeping the first; the latter is skipped.",
                case_name, case_to_rel[case_name], rel,
            )
            continue

        image = image_io.load_image(img_path, color_mode)
        image_io.save_nnunet_image(image, case_name, tmp_dir)
        case_to_rel[case_name] = rel
        converted += 1

    logger.info(
        "Prepared %d 2-D images (tree mode) in temporary input folder: %s",
        converted, tmp_dir,
    )
    if converted == 0:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise FileNotFoundError(
            f"No 2-D image files {image_io.IMAGE_EXTENSIONS} found in "
            f"'{raw_input_dir}'."
        )
    return tmp_dir, case_to_rel


def scatter_predictions_to_tree(
    flat_pred_dir: Path,
    case_to_rel: CaseToRel,
    output_root: Path,
) -> int:
    """
    Copy flat nnUNet predictions back into a tree mirroring the input layout.

    For every ``{case_name: rel_path}`` entry, the prediction
    ``flat_pred_dir/{case_name}.png`` is copied to
    ``output_root/{rel_path with a .png suffix}``, creating parent directories
    as needed. Cases without a prediction file are warned about and skipped.

    Parameters
    ----------
    flat_pred_dir : Path
        Folder of flat nnUNet 2-D predictions (``{case}.png``).
    case_to_rel : CaseToRel
        Mapping returned by :func:`prepare_input_tree_2d`.
    output_root : Path
        Root of the mirrored output tree (created if missing).

    Returns
    -------
    int
        Number of prediction files written.
    """
    ending = image_io.NNUNET_2D_FILE_ENDING
    output_root.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0
    for case_name, rel in sorted(case_to_rel.items()):
        pred_path = flat_pred_dir / f"{case_name}{ending}"
        if not pred_path.exists():
            logger.warning(
                "No prediction produced for case '%s' (source '%s') — skipped.",
                case_name, rel,
            )
            missing += 1
            continue

        dst = output_root / rel.with_suffix(ending)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(pred_path), str(dst))
        written += 1

    if missing:
        logger.warning(
            "%d case(s) had no prediction file and were skipped.", missing,
        )
    logger.info(
        "Scattered %d prediction(s) into mirrored tree: %s", written, output_root,
    )
    return written
