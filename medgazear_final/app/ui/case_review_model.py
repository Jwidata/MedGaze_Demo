"""Case-level CT review model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app.ui.dicom_ct_loader import CTSeries, load_ct_series_for_case


@dataclass
class CaseReviewModel:
    patient_id: str
    series_uid: str
    ct_series: CTSeries
    roi_geometry: pd.DataFrame
    features: pd.DataFrame
    attention: pd.DataFrame
    gaze: pd.DataFrame

    @property
    def total_slices(self) -> int:
        return self.ct_series.total_slices

    def image_for_slice(self, slice_index: int) -> Image.Image | None:
        return self.ct_series.image_for_index(slice_index)

    def rois_on_slice(self, slice_index: int) -> pd.DataFrame:
        return self.roi_geometry[pd.to_numeric(self.roi_geometry["ct_stack_index"], errors="coerce").fillna(-1).astype(int) == int(slice_index)]

    def gaze_on_slice(self, session_id: str, roi_id: str, slice_index: int) -> pd.DataFrame:
        roi_rows = self.roi_geometry[self.roi_geometry["roi_id"].astype(str) == str(roi_id)]
        if roi_rows.empty or int(roi_rows.iloc[0]["ct_stack_index"]) != int(slice_index):
            return pd.DataFrame()
        return self.gaze[(self.gaze["session_id"].astype(str) == str(session_id)) & (self.gaze["roi_id"].astype(str) == str(roi_id))].copy()

    def case_gaze_samples(self) -> pd.DataFrame:
        mapping = self.roi_geometry[["roi_id", "ct_stack_index"]].drop_duplicates()
        rows = self.gaze.merge(mapping, on="roi_id", how="inner")
        return rows.sort_values(["timestamp_ms", "session_id"]).reset_index(drop=True)

    def roi_slice_indices(self) -> list[int]:
        return sorted(pd.to_numeric(self.roi_geometry["ct_stack_index"], errors="coerce").dropna().astype(int).unique().tolist())

    def summary(self) -> dict[str, object]:
        statuses = self.attention["rule_attention_status"].value_counts().to_dict() if "rule_attention_status" in self.attention else {}
        return {
            "patient_id": self.patient_id,
            "series_uid": self.series_uid,
            "total_slices": self.total_slices,
            "roi_slices": int(self.roi_geometry["ct_stack_index"].nunique()),
            "roi_count": int(len(self.roi_geometry)),
            "evaluated_roi_episodes": int(len(self.features)),
            "uncovered_seg_rois": int(max(0, len(self.roi_geometry) - self.features["roi_id"].nunique())) if "roi_id" in self.features else int(len(self.roi_geometry)),
            "reviewed": int(statuses.get("reviewed", 0)),
            "weakly_reviewed": int(statuses.get("weakly_reviewed", 0)),
            "not_reviewed": int(statuses.get("not_reviewed", 0)),
            "not_evaluated": int(statuses.get("not_evaluated", 0)),
        }


def build_case_review_model(case_id: str, data) -> CaseReviewModel:
    geometry = data.roi_geometry[data.roi_geometry["patient_id"].astype(str) == str(case_id)].copy()
    if geometry.empty:
        raise ValueError(f"No ROI geometry found for case {case_id}")
    series_uid = str(geometry.iloc[0]["ct_series_instance_uid"])
    ct_series = load_ct_series_for_case(case_id, series_uid, data.dicom_inventory)
    sop_to_stack_index = {slice_.sop_instance_uid: slice_.slice_index for slice_ in ct_series.slices}
    geometry["ct_stack_index"] = geometry["ct_sop_instance_uid"].astype(str).map(sop_to_stack_index).fillna(-1).astype(int)
    geometry = geometry[geometry["ct_stack_index"] >= 0].copy()
    case_roi_ids = set(geometry["roi_id"].astype(str))
    features = data.features[data.features["roi_id"].astype(str).isin(case_roi_ids)].copy()
    if "case_id" in features.columns:
        features = features[features["case_id"].astype(str) == str(case_id)].copy()
    if not features.empty:
        features = features.merge(geometry[["roi_id", "ct_stack_index", "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max", "centroid_x", "centroid_y"]], on="roi_id", how="left")
    attention = data.attention[data.attention["roi_id"].astype(str).isin(case_roi_ids)].copy()
    if not attention.empty:
        status = attention.sort_values("roi_id").drop_duplicates("roi_id")[["roi_id", "rule_attention_status"]]
        geometry = geometry.merge(status, on="roi_id", how="left")
        geometry["rule_attention_status"] = geometry["rule_attention_status"].fillna("not_evaluated")
    else:
        geometry["rule_attention_status"] = "not_evaluated"
    return CaseReviewModel(case_id, series_uid, ct_series, geometry, features, attention, data.gaze)


def load_roi_mask(row: pd.Series) -> np.ndarray | None:
    path = Path(str(row.get("mask_npz_path", "")))
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        roi_ids = [str(item) for item in data["roi_ids"]]
        index = roi_ids.index(str(row["roi_id"]))
        return data["masks"][index].astype(bool)
    except Exception:
        return None
