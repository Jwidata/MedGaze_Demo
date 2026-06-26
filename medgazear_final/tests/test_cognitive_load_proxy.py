from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.cognitive_load.attention_load_relation import attention_vs_cognitive_proxy, behavior_vs_cognitive_proxy
from app.cognitive_load.cognitive_proxy_engine import assign_cognitive_proxy_labels, distribution
from app.cognitive_load.cognitive_proxy_features import build_cognitive_proxy_features
from app.cognitive_load.cognitive_proxy_report import LIMITATION_TEXT, write_cognitive_limitations, write_cognitive_proxy_report


def test_proxy_score_generation() -> None:
    features = _feature_rows()
    proxy = build_cognitive_proxy_features(features)

    assert "cognitive_load_proxy_score" in proxy.columns
    assert proxy["cognitive_load_proxy_score"].between(0, 1).all()
    assert proxy.loc[0, "cognitive_load_proxy_score"] < proxy.loc[2, "cognitive_load_proxy_score"]


def test_low_medium_high_label_assignment() -> None:
    proxy = build_cognitive_proxy_features(_feature_rows())
    labels = assign_cognitive_proxy_labels(proxy)

    assert set(labels["cognitive_load_proxy"]) == {"low_load_proxy", "medium_load_proxy", "high_load_proxy"}
    assert labels.loc[0, "cognitive_load_proxy"] == "low_load_proxy"
    assert labels.loc[2, "cognitive_load_proxy"] == "high_load_proxy"


def test_attention_and_behavior_relations() -> None:
    labels = assign_cognitive_proxy_labels(build_cognitive_proxy_features(_feature_rows()))
    attention = pd.DataFrame(
        [
            {"session_id": "S1", "roi_id": "R1", "rule_attention_status": "reviewed"},
            {"session_id": "S2", "roi_id": "R2", "rule_attention_status": "weakly_reviewed"},
            {"session_id": "S3", "roi_id": "R3", "rule_attention_status": "not_reviewed"},
        ]
    )

    attention_table = attention_vs_cognitive_proxy(labels, attention)
    behavior_table = behavior_vs_cognitive_proxy(labels, _feature_rows())

    assert set(attention_table.columns) >= {"rule_attention_status", "low_load_proxy", "medium_load_proxy", "high_load_proxy"}
    assert set(behavior_table.columns) >= {"hidden_behavior_label", "low_load_proxy", "medium_load_proxy", "high_load_proxy"}


def test_report_generation_and_limitation_wording(tmp_path: Path) -> None:
    labels = assign_cognitive_proxy_labels(build_cognitive_proxy_features(_feature_rows()))
    dist = distribution(labels)
    relation = pd.DataFrame([{"rule_attention_status": "reviewed", "low_load_proxy": 1, "medium_load_proxy": 1, "high_load_proxy": 1, "total": 3}])
    behavior = pd.DataFrame([{"hidden_behavior_label": "expert_like_systematic_review", "low_load_proxy": 1, "medium_load_proxy": 1, "high_load_proxy": 1, "total": 3}])
    report = tmp_path / "report.md"
    limitations = tmp_path / "limitations.md"

    write_cognitive_proxy_report(report, dist, relation, behavior, 14)
    write_cognitive_limitations(limitations)

    assert LIMITATION_TEXT in report.read_text(encoding="utf-8")
    assert "Proxy Formula" in report.read_text(encoding="utf-8")
    assert LIMITATION_TEXT in limitations.read_text(encoding="utf-8")


def _feature_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("S1", "R1", "expert_like_systematic_review", 10),
            _row("S2", "R2", "partial_near_miss_review", 50),
            _row("S3", "R3", "high_load_fragmented_review", 100),
        ]
    )


def _row(session_id: str, roi_id: str, label: str, value: float) -> dict[str, object]:
    return {
        "session_id": session_id,
        "roi_id": roi_id,
        "case_id": "C1",
        "reader_id": "Reader1",
        "hidden_behavior_label": label,
        "gaze_dispersion_px": value,
        "scanpath_length_px": value,
        "roi_revisit_count": value / 10,
        "near_roi_revisit_count": value / 10,
        "slice_transition_count": value / 10,
        "adjacent_slice_toggle_count": value / 10,
        "delayed_attention_score": value / 100,
        "fixation_duration_variance": value,
        "saccade_like_ratio": value / 100,
        "dropout_ratio": value / 1000,
        "blink_ratio": value / 1000,
        "invalid_burst_ratio": value / 1000,
        "outside_ct_ratio": value / 1000,
        "background_gaze_ratio": value / 100,
    }
