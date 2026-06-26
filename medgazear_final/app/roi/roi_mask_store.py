"""Compressed storage for extracted ROI masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_roi_masks(
    output_dir: Path,
    seg_sop_instance_uid: str,
    masks: list[np.ndarray],
    roi_ids: list[str],
    frame_indices: list[int],
    ct_sop_instance_uids: list[str],
) -> Path | None:
    """Save accepted masks for one SEG object as a compressed NPZ."""

    if not masks:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_uid = "".join(char if char.isalnum() or char in ".-_" else "_" for char in seg_sop_instance_uid)
    path = output_dir / f"{safe_uid}.npz"
    np.savez_compressed(
        path,
        masks=np.stack([mask.astype(np.uint8) for mask in masks], axis=0),
        roi_ids=np.asarray(roi_ids),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        ct_sop_instance_uids=np.asarray(ct_sop_instance_uids),
    )
    return path


def load_roi_masks(path: Path) -> np.lib.npyio.NpzFile:
    """Load compressed ROI mask arrays."""

    return np.load(path, allow_pickle=False)
