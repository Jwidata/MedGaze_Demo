"""Optional real Tobii live gaze source."""

from __future__ import annotations

import time
from typing import Any

from app.gaze_sources.gaze_source_interface import GazeCallback, GazeSourceStatus
from app.gaze_sources import tobii_status
from app.gaze_sources.tobii_normalization import canonicalize_tobii_sample


class TobiiLiveSource:
    def __init__(self, coordinate_mapper=None, screen_geometry_provider=None) -> None:
        self.coordinate_mapper = coordinate_mapper
        self.screen_geometry_provider = screen_geometry_provider
        self.tracker: Any | None = None
        self.callback: GazeCallback | None = None
        self.streaming = False
        self.capture_state = "stopped"
        self.error: str | None = None
        self.last_payload = tobii_status.get_status_payload()

    def is_sdk_available(self) -> bool:
        return tobii_status.is_sdk_available()

    def find_devices(self) -> list[Any]:
        self.last_payload = tobii_status.get_status_payload()
        if self.last_payload.get("status_label") == tobii_status.SDK_ERROR_MESSAGE:
            self.error = str(self.last_payload.get("error") or tobii_status.SDK_ERROR_MESSAGE)
            return []
        return tobii_status.find_devices()

    def connect_first_device(self) -> bool:
        devices = self.find_devices()
        if not devices:
            self.tracker = None
            return False
        self.tracker = devices[0]
        self.last_payload = tobii_status.get_status_payload()
        self.error = None
        return True

    def start_stream(self, callback: GazeCallback) -> None:
        self.callback = callback
        self.capture_state = "error"
        self.error = None
        if not self.is_sdk_available():
            self.error = tobii_status.SDK_MISSING_MESSAGE
            return
        if self.tracker is None and not self.connect_first_device():
            if self.error is None:
                self.error = tobii_status.NO_TRACKER_MESSAGE
            return
        if self.tracker is None or tobii_status.tr is None:
            self.error = tobii_status.NO_TRACKER_MESSAGE
            return
        try:
            self.tracker.subscribe_to(tobii_status.tr.EYETRACKER_GAZE_DATA, self._on_gaze_data, as_dictionary=True)
            self.streaming = True
            self.capture_state = "recording"
            self.error = None
            self.last_payload = tobii_status.get_status_payload()
        except Exception as exc:  # pragma: no cover - SDK/device dependent
            self.streaming = False
            self.capture_state = "error"
            self.error = str(exc)

    def stop_stream(self) -> None:
        if tobii_status.tr is not None and self.tracker is not None and self.streaming:
            try:
                self.tracker.unsubscribe_from(tobii_status.tr.EYETRACKER_GAZE_DATA, self._on_gaze_data)
            except Exception:
                pass
        self.streaming = False
        self.capture_state = "stopped"

    def get_status(self) -> GazeSourceStatus:
        payload = self.get_status_payload()
        message = payload["status_label"]
        if payload.get("device_model"):
            message = f"{message}: {payload['device_model']}"
        if payload.get("error"):
            message = f"{message} | {payload['error']}"
        return GazeSourceStatus("live_tobii", connected=bool(payload["device_connected"]), streaming=self.streaming, message=str(message))

    def get_status_payload(self) -> dict[str, object]:
        payload = tobii_status.get_status_payload()
        payload = {**payload}
        payload["capture_state"] = self.capture_state
        payload["error"] = self.error or payload.get("error")
        if self.error:
            payload["device_state"] = tobii_status.DEVICE_STATE_ERROR
        elif self.streaming:
            payload["device_state"] = tobii_status.DEVICE_STATE_STREAMING
        elif self.tracker is not None and payload.get("device_connected"):
            payload["device_state"] = tobii_status.DEVICE_STATE_CONNECTED
        return payload

    def _on_gaze_data(self, gaze_data: dict[str, Any]) -> None:
        if self.callback is None:
            return
        self.callback(self._canonical_sample(gaze_data))

    def _canonical_sample(self, gaze_data: dict[str, Any]) -> dict[str, object]:
        screen_geometry = self.screen_geometry_provider() if self.screen_geometry_provider is not None else None
        return canonicalize_tobii_sample(gaze_data, coordinate_mapper=self.coordinate_mapper, screen_geometry=screen_geometry)
