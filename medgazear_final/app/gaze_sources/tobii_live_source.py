"""Optional Tobii Research SDK live gaze source."""

from __future__ import annotations

import time
from typing import Any

from app.gaze_sources.gaze_source_interface import GazeCallback, GazeSourceStatus
from app.gaze_sources.tobii_status import CALIBRATION_REMINDER, NO_TRACKER_MESSAGE, SDK_MISSING_MESSAGE

try:
    import tobii_research as tr
except ImportError:  # pragma: no cover - depends on local machine SDK
    tr = None


class TobiiLiveSource:
    def __init__(self, coordinate_mapper=None) -> None:
        self.coordinate_mapper = coordinate_mapper
        self.tracker = None
        self.callback: GazeCallback | None = None
        self.streaming = False
        self.message = "not connected"

    def is_sdk_available(self) -> bool:
        return tr is not None

    def find_devices(self) -> list[Any]:
        if tr is None:
            self.message = SDK_MISSING_MESSAGE
            return []
        devices = list(tr.find_all_eyetrackers())
        self.message = f"{len(devices)} Tobii eye tracker(s) detected." if devices else NO_TRACKER_MESSAGE
        return devices

    def connect_first_device(self) -> bool:
        devices = self.find_devices()
        if not devices:
            self.tracker = None
            return False
        self.tracker = devices[0]
        self.message = f"connected: {getattr(self.tracker, 'model', 'Tobii eye tracker')} | {CALIBRATION_REMINDER}"
        return True

    def start_stream(self, callback: GazeCallback) -> None:
        self.callback = callback
        if tr is None:
            self.message = SDK_MISSING_MESSAGE
            return
        if self.tracker is None and not self.connect_first_device():
            return
        if self.tracker is None:
            self.message = NO_TRACKER_MESSAGE
            return
        self.tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self._on_gaze_data, as_dictionary=True)
        self.streaming = True
        self.message = f"streaming live gaze | {CALIBRATION_REMINDER}"

    def stop_stream(self) -> None:
        if tr is not None and self.tracker is not None and self.streaming:
            try:
                self.tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._on_gaze_data)
            except Exception:
                pass
        self.streaming = False
        if self.tracker is not None:
            self.message = f"connected, not streaming | {CALIBRATION_REMINDER}"

    def get_status(self) -> GazeSourceStatus:
        if tr is None:
            return GazeSourceStatus("tobii_live", connected=False, streaming=False, message=SDK_MISSING_MESSAGE)
        return GazeSourceStatus("tobii_live", connected=self.tracker is not None, streaming=self.streaming, message=self.message)

    def _on_gaze_data(self, gaze_data: dict[str, Any]) -> None:
        if self.callback is None:
            return
        sample = self._canonical_sample(gaze_data)
        self.callback(sample)

    def _canonical_sample(self, gaze_data: dict[str, Any]) -> dict[str, object]:
        left = gaze_data.get("left_gaze_point_on_display_area") or (float("nan"), float("nan"))
        right = gaze_data.get("right_gaze_point_on_display_area") or (float("nan"), float("nan"))
        left_valid = int(gaze_data.get("left_gaze_point_validity", 0) or 0) == 1
        right_valid = int(gaze_data.get("right_gaze_point_validity", 0) or 0) == 1
        valid_points = [point for point, valid in ((left, left_valid), (right, right_valid)) if valid]
        if valid_points:
            x_norm = sum(float(point[0]) for point in valid_points) / len(valid_points)
            y_norm = sum(float(point[1]) for point in valid_points) / len(valid_points)
            is_valid = True
        else:
            x_norm = 0.0
            y_norm = 0.0
            is_valid = False
        mapped = self.coordinate_mapper(x_norm, y_norm) if self.coordinate_mapper is not None else None
        image_x, image_y, is_outside_ct = mapped if mapped is not None else (0.0, 0.0, True)
        return {
            "source_type": "tobii_live",
            "timestamp_ms": float(gaze_data.get("device_time_stamp", time.time_ns() / 1_000_000)) / 1000.0,
            "screen_x": x_norm,
            "screen_y": y_norm,
            "gaze_x_norm": x_norm,
            "gaze_y_norm": y_norm,
            "image_x": float(image_x),
            "image_y": float(image_y),
            "is_valid": is_valid and not bool(is_outside_ct),
            "is_dropout": not is_valid,
            "is_blink": not is_valid,
            "is_outside_ct": bool(is_outside_ct),
            "is_ui_glance": bool(is_outside_ct),
        }
