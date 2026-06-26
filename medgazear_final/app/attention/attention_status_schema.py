"""Attention status schema and ordering."""

from __future__ import annotations

ATTENTION_STATUSES = ("reviewed", "weakly_reviewed", "not_reviewed", "not_evaluated")
REVIEW_QUEUE_SEVERITY = {"not_evaluated": 0, "not_reviewed": 1, "weakly_reviewed": 2, "reviewed": 3}


def validate_attention_status(status: str) -> None:
    if status not in ATTENTION_STATUSES:
        raise ValueError(f"Invalid attention status: {status}")
