"""Canonical gaze source interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


CanonicalGazeSample = dict[str, object]
GazeCallback = Callable[[CanonicalGazeSample], None]


@dataclass
class GazeSourceStatus:
    source_type: str
    connected: bool = False
    streaming: bool = False
    message: str = "not connected"


class GazeSource(Protocol):
    def start_stream(self, callback: GazeCallback) -> None:
        ...

    def stop_stream(self) -> None:
        ...

    def get_status(self) -> GazeSourceStatus:
        ...
