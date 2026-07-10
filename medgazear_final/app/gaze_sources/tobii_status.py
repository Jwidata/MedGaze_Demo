"""Optional Tobii SDK status helpers."""

from __future__ import annotations

import sys
from typing import Any

try:
    import tobii_research as tr
except ImportError:  # pragma: no cover - optional dependency
    tr = None


SDK_MISSING_MESSAGE = "SDK missing"
NO_TRACKER_MESSAGE = "No tracker detected"
TRACKER_DETECTED_MESSAGE = "Tracker detected"
SDK_ERROR_MESSAGE = "SDK error"
CALIBRATION_REMINDER = "Calibrate with Tobii Manager before live capture."
DEVICE_STATE_NO_DEVICE = "NO_DEVICE"
DEVICE_STATE_DEVICE_FOUND = "DEVICE_FOUND"
DEVICE_STATE_CONNECTED = "CONNECTED"
DEVICE_STATE_STREAMING = "STREAMING"
DEVICE_STATE_ERROR = "ERROR"


def is_sdk_available() -> bool:
    return tr is not None


def get_sdk_status() -> str:
    return "installed" if is_sdk_available() else "missing"


def find_devices() -> list[Any]:
    if tr is None:
        return []
    return list(tr.find_all_eyetrackers())


def get_first_device_summary() -> dict[str, str | None]:
    devices = find_devices()
    if not devices:
        return {
            "device_model": None,
            "device_address": None,
            "device_serial": None,
            "device_name": None,
        }
    device = devices[0]
    return _device_summary(device)


def get_status_payload() -> dict[str, object]:
    if tr is None:
        return {
            "sdk_available": False,
            "device_connected": False,
            "device_count": 0,
            "device_model": None,
            "device_address": None,
            "device_serial": None,
            "status_label": SDK_MISSING_MESSAGE,
            "device_state": DEVICE_STATE_ERROR,
            "calibration_reminder": CALIBRATION_REMINDER,
            "error": _missing_sdk_detail(),
        }
    try:
        devices = find_devices()
        summary = _device_summary(devices[0]) if devices else {"device_model": None, "device_address": None, "device_serial": None}
        return {
            "sdk_available": True,
            "device_connected": bool(devices),
            "device_count": len(devices),
            "device_model": summary.get("device_model"),
            "device_address": summary.get("device_address"),
            "device_serial": summary.get("device_serial"),
            "status_label": TRACKER_DETECTED_MESSAGE if devices else NO_TRACKER_MESSAGE,
            "device_state": DEVICE_STATE_DEVICE_FOUND if devices else DEVICE_STATE_NO_DEVICE,
            "calibration_reminder": CALIBRATION_REMINDER,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - hardware/SDK dependent
        return {
            "sdk_available": True,
            "device_connected": False,
            "device_count": 0,
            "device_model": None,
            "device_address": None,
            "device_serial": None,
            "status_label": SDK_ERROR_MESSAGE,
            "device_state": DEVICE_STATE_ERROR,
            "calibration_reminder": CALIBRATION_REMINDER,
            "error": str(exc),
        }


def _device_summary(device: Any) -> dict[str, str | None]:
    return {
        "device_model": _safe_str(getattr(device, "model", None)),
        "device_address": _safe_str(getattr(device, "address", None)),
        "device_serial": _safe_str(getattr(device, "serial_number", None)),
        "device_name": _safe_str(getattr(device, "device_name", None)),
    }


def _safe_str(value: object) -> str | None:
    return None if value in (None, "") else str(value)


def _missing_sdk_detail() -> str:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if (sys.version_info.major, sys.version_info.minor) != (3, 10):
        return f"tobii-research wheels are primarily published for Python 3.10. Current interpreter is Python {version}."
    return "Install the optional Tobii SDK binding with: python -m pip install -r requirements-tobii.txt"
