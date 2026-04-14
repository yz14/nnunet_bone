"""
Inspect label values in the raw Total Segmentator segmentation files.

Usage:
    python scripts/inspect_labels.py --config configs/default.yaml
    python scripts/inspect_labels.py --config configs/default.yaml --max_cases 3

This script:
  1. Scans the raw data directory for segmentation files
  2. Prints all unique label IDs found in each file
  3. Cross-references them against the All-labels map in labels.yaml
  4. Reports which bone labels from the active groups ARE and ARE NOT present

Use the output to verify / adjust bone_groups and active_groups in labels.yaml
before running 01_prepare_data.py.
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.label_config import LabelConfig
from src.utils import load_config, setup_logging


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_unique_labels(nii_path: Path) -> np.ndarray:
    """Return sorted unique non-zero label values in a NIfTI segmentation file."""
    img = nib.load(str(nii_path))
    data = np.asarray(img.dataobj, dtype=np.int32)
    return np.unique(data[data > 0])


def print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Total Segmentator label values in raw data."
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to experiment config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--max_cases", type=int, default=None,
        help="Maximum number of cases to inspect (default: all)",
    )
    parser.add_argument(
        "--log_level", default="INFO",
        help="Logging level: DEBUG / INFO / WARNING (default: INFO)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # ── Load configs ──────────────────────────────────────────────────────
    cfg       = load_config(args.config)
    label_cfg = LabelConfig(cfg["paths"]["labels_config"])

    raw_data_dir = Path(cfg["paths"]["raw_data_dir"])
    all_labels   = label_cfg.all_labels          # {id: name}
    bone_ids     = set(label_cfg.get_active_label_ids())

    # ── Discover segmentation files ───────────────────────────────────────
    seg_files = sorted(raw_data_dir.glob("*-seg.nii.gz"))
    if not seg_files:
        logger.error("No '*-seg.nii.gz' files found in '%s'.", raw_data_dir)
        sys.exit(1)

    if args.max_cases is not None:
        seg_files = seg_files[: args.max_cases]

    logger.info("Inspecting %d segmentation file(s) in '%s'", len(seg_files), raw_data_dir)

    # ── Per-case inspection ───────────────────────────────────────────────
    all_seen_ids: "set[int]" = set()
    cases_missing_bone: list[str] = []

    for seg_path in seg_files:
        case_id = seg_path.name.replace("-seg.nii.gz", "")
        print_separator()
        print(f"Case: {case_id}  ({seg_path.name})")

        unique_ids = get_unique_labels(seg_path)
        all_seen_ids.update(int(x) for x in unique_ids)

        # Show named labels
        named = [(int(lid), all_labels.get(int(lid), f"<unknown:{lid}>")) for lid in unique_ids]
        print(f"  Unique labels ({len(named)}):  ", end="")
        if len(named) <= 30:
            print(", ".join(f"{lid}:{name}" for lid, name in named))
        else:
            first = ", ".join(f"{lid}:{name}" for lid, name in named[:15])
            last  = ", ".join(f"{lid}:{name}" for lid, name in named[-5:])
            print(f"{first} ... {last}")

        # Check bone coverage
        found_bone    = set(int(x) for x in unique_ids) & bone_ids
        missing_bone  = bone_ids - set(int(x) for x in unique_ids)
        bone_pct      = 100.0 * len(found_bone) / len(bone_ids) if bone_ids else 0.0

        print(f"  Bone labels present : {len(found_bone)}/{len(bone_ids)} ({bone_pct:.0f}%)")
        if missing_bone:
            missing_named = [
                f"{lid}:{all_labels.get(lid, '?')}" for lid in sorted(missing_bone)[:10]
            ]
            suffix = f" ... (+{len(missing_bone)-10} more)" if len(missing_bone) > 10 else ""
            print(f"  Missing bone labels : {', '.join(missing_named)}{suffix}")

        if found_bone:
            # Quick volume count (proxy for sanity check)
            img = nib.load(str(seg_path))
            data = np.asarray(img.dataobj, dtype=np.int32)
            bone_vox = int(np.isin(data, list(found_bone)).sum())
            total_vox = int(data.size)
            print(f"  Bone voxels         : {bone_vox:,} / {total_vox:,} ({100.*bone_vox/total_vox:.2f}%)")
        else:
            cases_missing_bone.append(case_id)

    # ── Global summary ────────────────────────────────────────────────────
    print_separator("═")
    print("GLOBAL SUMMARY")
    print_separator()

    all_seen_sorted = sorted(all_seen_ids)
    print(f"Total unique label IDs across all cases: {len(all_seen_sorted)}")
    print(f"  Range: {all_seen_sorted[0]} … {all_seen_sorted[-1]}")

    unlabeled_ids = [lid for lid in all_seen_sorted if lid not in all_labels]
    if unlabeled_ids:
        print(f"\n  WARNING: Unknown IDs (not in labels.yaml): {unlabeled_ids}")
        print("     -> Update 'all_labels' in configs/labels.yaml if these are new v2 classes.")

    globally_present_bone = all_seen_ids & bone_ids
    globally_absent_bone  = bone_ids - all_seen_ids
    print(f"\nActive bone groups   : {', '.join(label_cfg.active_groups)}")
    print(f"Active bone label IDs: {len(bone_ids)} total")
    print(f"  Present in dataset : {len(globally_present_bone)}")

    if globally_absent_bone:
        absent_named = [
            f"{lid}:{all_labels.get(lid, '?')}" for lid in sorted(globally_absent_bone)
        ]
        print(f"  Never seen         : {', '.join(absent_named)}")
        print("  -> Consider removing these from active_groups in labels.yaml")

    if cases_missing_bone:
        print(f"\n  WARNING: Cases with NO bone voxels: {cases_missing_bone}")
        print("     -> Verify label IDs or consider excluding these cases.")

    print_separator("═")
    print("Done. Adjust configs/labels.yaml if needed, then run:")
    print("  python scripts/01_prepare_data.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
