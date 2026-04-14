"""
Label configuration manager.

Responsibilities:
  - Load and validate labels.yaml
  - Build label-remapping tables (source Total Segmentator ID → output class)
  - Generate nnUNet dataset.json label section
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .utils import load_yaml

logger = logging.getLogger(__name__)


class LabelConfig:
    """
    Manages Total Segmentator label definitions and bone-group remapping.

    Parameters
    ----------
    labels_config_path : str or Path
        Path to labels.yaml.
    """

    def __init__(self, labels_config_path: Union[str, Path]) -> None:
        self._raw = load_yaml(labels_config_path)
        self._validate()

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def all_labels(self) -> Dict[int, str]:
        """Full Total Segmentator label map {label_id: name}."""
        return {int(k): str(v) for k, v in self._raw["all_labels"].items()}

    @property
    def bone_groups(self) -> Dict[str, Dict]:
        """Raw bone group definitions from labels.yaml."""
        return self._raw["bone_groups"]

    @property
    def active_groups(self) -> List[str]:
        """Names of bone groups that participate in the segmentation task."""
        return list(self._raw["active_groups"])

    @property
    def multiclass_mapping_yaml(self) -> Dict[str, int]:
        """Group-name → class-id mapping defined in labels.yaml."""
        return {str(k): int(v) for k, v in self._raw["multiclass_mapping"].items()}

    # ─────────────────────────────────────────────────────────────────────────
    # Label-set queries
    # ─────────────────────────────────────────────────────────────────────────

    def get_active_label_ids(self) -> List[int]:
        """Return sorted list of all source label IDs that belong to active groups."""
        ids: List[int] = []
        for group_name in self.active_groups:
            group = self.bone_groups.get(group_name)
            if group is None:
                logger.warning("Active group '%s' not found in bone_groups — skipped.", group_name)
                continue
            ids.extend(int(x) for x in group["labels"])
        return sorted(set(ids))

    # ─────────────────────────────────────────────────────────────────────────
    # Remapping tables
    # ─────────────────────────────────────────────────────────────────────────

    def get_binary_remap(self) -> Dict[int, int]:
        """
        Build remap table for **binary** mode.

        Returns {source_label_id: 1} for every active bone label.
        Non-bone voxels stay 0 (background) by convention.
        """
        return {label_id: 1 for label_id in self.get_active_label_ids()}

    def get_multiclass_remap(
        self, override: Optional[Dict[str, int]] = None
    ) -> Dict[int, int]:
        """
        Build remap table for **multi-class** mode.

        Parameters
        ----------
        override : dict, optional
            {group_name: class_id} pairs that override labels.yaml defaults.

        Returns
        -------
        Dict[int, int]
            {source_label_id: output_class_id}
        """
        group_to_class = dict(self.multiclass_mapping_yaml)
        if override:
            group_to_class.update({str(k): int(v) for k, v in override.items()})

        remap: Dict[int, int] = {}
        for group_name in self.active_groups:
            class_id = group_to_class.get(group_name)
            if class_id is None:
                logger.warning(
                    "No class ID for active group '%s' in multiclass_mapping — skipped.",
                    group_name,
                )
                continue
            group = self.bone_groups.get(group_name, {})
            for src_id in group.get("labels", []):
                remap[int(src_id)] = int(class_id)

        return remap

    def get_remap(
        self,
        mode: str,
        multiclass_override: Optional[Dict[str, int]] = None,
    ) -> Dict[int, int]:
        """
        Convenience wrapper. mode must be 'binary' or 'multiclass'.
        """
        if mode == "binary":
            return self.get_binary_remap()
        elif mode == "multiclass":
            return self.get_multiclass_remap(override=multiclass_override)
        else:
            raise ValueError(f"Unknown segmentation mode: '{mode}'. Expected 'binary' or 'multiclass'.")

    # ─────────────────────────────────────────────────────────────────────────
    # nnUNet dataset.json helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_nnunet_labels(
        self,
        mode: str,
        multiclass_override: Optional[Dict[str, int]] = None,
    ) -> Dict[str, int]:
        """
        Return the ``labels`` block for nnUNet v2 dataset.json.

        Format: {"background": 0, "bone": 1}  (binary)
             or {"background": 0, "vertebrae": 1, "ribs": 2, ...}  (multiclass)
        """
        if mode == "binary":
            return {"background": 0, "bone": 1}

        # multiclass: invert group_to_class mapping
        group_to_class = dict(self.multiclass_mapping_yaml)
        if multiclass_override:
            group_to_class.update(multiclass_override)

        labels: Dict[str, int] = {"background": 0}
        for group_name in self.active_groups:
            class_id = group_to_class.get(group_name)
            if class_id is not None:
                labels[group_name] = int(class_id)

        return labels

    def get_num_output_classes(
        self,
        mode: str,
        multiclass_override: Optional[Dict[str, int]] = None,
    ) -> int:
        """Return number of *foreground* classes (excluding background)."""
        if mode == "binary":
            return 1
        labels = self.get_nnunet_labels(mode, multiclass_override)
        return len(labels) - 1  # subtract background

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        required_keys = ["all_labels", "bone_groups", "active_groups", "multiclass_mapping"]
        for key in required_keys:
            if key not in self._raw:
                raise KeyError(f"labels.yaml is missing required key: '{key}'")

        for group in self.active_groups:
            if group not in self.bone_groups:
                logger.warning(
                    "Active group '%s' is not defined in bone_groups.", group
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Debug / display
    # ─────────────────────────────────────────────────────────────────────────

    def describe(self, mode: str) -> None:
        """Print a human-readable summary of the label configuration."""
        logger.info("=" * 60)
        logger.info("Label configuration summary")
        logger.info("  Mode            : %s", mode)
        logger.info("  Active groups   : %s", ", ".join(self.active_groups))
        active_ids = self.get_active_label_ids()
        if len(active_ids) <= 10:
            logger.info("  Active label IDs: %d IDs — %s", len(active_ids), active_ids)
        else:
            logger.info(
                "  Active label IDs: %d IDs — %s ... %s",
                len(active_ids), active_ids[:5], active_ids[-5:],
            )
        remap = self.get_remap(mode)
        output_classes = sorted(set(remap.values()))
        logger.info("  Output classes  : %s", output_classes)
        logger.info("=" * 60)
