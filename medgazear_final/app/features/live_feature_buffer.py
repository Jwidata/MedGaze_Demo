"""Small buffer for future live Tobii gaze samples."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class LiveFeatureBuffer:
    max_samples: int = 3600
    samples: deque[dict[str, object]] = field(default_factory=deque)

    def append(self, sample: dict[str, object]) -> None:
        self.samples.append(sample)
        while len(self.samples) > self.max_samples:
            self.samples.popleft()

    def snapshot(self) -> list[dict[str, object]]:
        return list(self.samples)
