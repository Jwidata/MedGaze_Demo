# Tobii Setup

Use Python 3.10. Current official `tobii_research` wheels on PyPI are published primarily for Python 3.10.

If you run MedGazeAR with Python 3.14 or another newer interpreter, pip may report:

```bash
ERROR: No matching distribution found for tobii-research>=2.1.0
```

That means the SDK is not installable in the current interpreter, even if the Tobii device is physically connected and calibrated.

Recommended Windows setup:

```bash
py -3.10 -m venv .venv310
.venv310\Scripts\activate
python -m pip install -r requirements-tobii.txt
python scripts/21_check_tobii_sdk.py
python scripts/20_launch_review_workstation.py --output-root outputs
```

Install the optional Tobii SDK binding:

```bash
python -m pip install -r requirements-tobii.txt
```

Run the SDK/device diagnostic:

```bash
python scripts/21_check_tobii_sdk.py
```

Before live capture:

- Connect the Tobii tracker.
- Calibrate with Tobii Manager before live capture.

Notes:

- If the Tobii SDK is missing, the MedGazeAR app should still launch.
- If no tracker is connected, the app should report that state without crashing.
- Live capture never falls back to synthetic replay.
