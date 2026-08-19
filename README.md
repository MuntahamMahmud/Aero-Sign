# Aerosign Gesture

Hand gesture detection for Aerosign-style drone commands using MediaPipe and OpenCV.

## Environment

- Python 3.11.9
- MediaPipe 0.10.14
- NumPy 2.4.6
- OpenCV (opencv-python) 5.0.0.93

## Files included

- `main.py` - camera capture, live gesture detection, and display.
- `gesture_control.py` - gesture classification using MediaPipe hand landmarks.
- `requirements` - required Python packages.
- `.gitignore` - excludes virtual environment files and temporary artifacts.

## Why `gesture_env` is excluded

The `gesture_env` folder is a local Python virtual environment. It is not tracked in Git because:

- it is machine-specific
- it contains installed packages and large binary files
- it can be recreated from `requirements`

## Setup

1. Create a virtual environment:

```powershell
python -m venv gesture_env
```

2. Activate it:

```powershell
.\gesture_env\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements
```

## Run

```powershell
python main.py
```

This script now defaults to MAVLink UDP on `udp:0.0.0.0:14550` at `57600` baud.

### Override the default MAVLink transport

```powershell
python main.py --mavlink COM3 --baud 57600
```

Or use a different UDP endpoint:

```powershell
python main.py --mavlink udp:127.0.0.1:14550
```

## Notes

If you want to reuse this repository on another machine, recreate the virtual environment and install from `requirements` rather than copying `gesture_env`. If you add new dependencies, update `requirements` accordingly.
