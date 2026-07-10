"""Check optional Tobii SDK installation and connected devices."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    print(f"Python version: {platform.python_version()}")
    try:
        import tobii_research as tr
    except ImportError as exc:
        print("SDK import status: failed")
        print(f"Import error: {exc}")
        return 1

    print("SDK import status: success")
    try:
        devices = list(tr.find_all_eyetrackers())
    except Exception as exc:
        print(f"Unexpected SDK error: {exc}")
        return 1

    print(f"Devices found: {len(devices)}")
    for index, device in enumerate(devices, start=1):
        print(f"Device {index}:")
        print(f"  address: {getattr(device, 'address', None)}")
        print(f"  model: {getattr(device, 'model', None)}")
        print(f"  serial number: {getattr(device, 'serial_number', None)}")
        print(f"  device name: {getattr(device, 'device_name', None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
