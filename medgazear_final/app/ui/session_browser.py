"""Session and ROI browser helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SessionSelectionModel:
    sessions: pd.DataFrame
    index: int = 0

    def filtered(self, behavior_label: str | None = None) -> pd.DataFrame:
        if not behavior_label or behavior_label == "All":
            return self.sessions
        return self.sessions[self.sessions["hidden_behavior_label"] == behavior_label]

    def current(self, behavior_label: str | None = None) -> pd.Series | None:
        rows = self.filtered(behavior_label)
        if rows.empty:
            return None
        safe_index = max(0, min(self.index, len(rows) - 1))
        return rows.iloc[safe_index]

    def next(self, behavior_label: str | None = None) -> pd.Series | None:
        rows = self.filtered(behavior_label)
        if rows.empty:
            return None
        self.index = (self.index + 1) % len(rows)
        return self.current(behavior_label)

    def previous(self, behavior_label: str | None = None) -> pd.Series | None:
        rows = self.filtered(behavior_label)
        if rows.empty:
            return None
        self.index = (self.index - 1) % len(rows)
        return self.current(behavior_label)

    def set_by_session(self, session_id: str, behavior_label: str | None = None) -> pd.Series | None:
        rows = self.filtered(behavior_label).reset_index(drop=True)
        matches = rows.index[rows["session_id"].astype(str) == str(session_id)].tolist()
        if not matches:
            return self.current(behavior_label)
        self.index = int(matches[0])
        return self.current(behavior_label)
