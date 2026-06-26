"""Small helper for CT stack navigation state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SliceStackController:
    total_slices: int = 0
    current_index: int = 0

    def clamp(self, index: int) -> int:
        if self.total_slices <= 0:
            self.current_index = 0
        else:
            self.current_index = max(0, min(int(index), self.total_slices - 1))
        return self.current_index

    def step(self, delta: int) -> int:
        if self.total_slices <= 0:
            return 0
        self.current_index = (self.current_index + int(delta)) % self.total_slices
        return self.current_index
