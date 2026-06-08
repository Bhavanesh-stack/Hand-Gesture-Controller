# AI Hand Gesture Controller

A real-time virtual hand mouse that uses your webcam, MediaPipe hand tracking,
and PyAutoGUI to let you control your computer cursor with hand gestures.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the controller
python hand_gesture_controller.py
```

## How It Works

| Gesture | Action |
|---|---|
| **Move index finger** | Cursor follows your fingertip |
| **Pinch** (thumb + index touch) | Left mouse click |
| **Press `q` / ESC** | Quit the application |

## Architecture

```
Webcam Frame
    │
    ▼
 cv2.flip (mirror)
    │
    ▼
 MediaPipe Hands  ──►  21 3D landmarks
    │
    ├──► Landmark 8 (Index Tip)  ──►  Map to screen coords  ──►  pyautogui.moveTo()
    │                                       │
    │                                  EMA Smoothing
    │
    └──► Landmark 4 (Thumb Tip)  ──►  Euclidean distance  ──►  if < threshold → click
```

## Key Math

### Euclidean Distance (pinch detection)

```
d = √[ (x₈ − x₄)² + (y₈ − y₄)² ]
```

MediaPipe returns normalised [0, 1] coordinates; when `d < 0.04` the
fingers are "pinched" and a click fires.

### Exponential Moving Average (cursor smoothing)

```
smoothed = α · raw  +  (1 − α) · prev_smoothed
```

`α = 0.35` by default.  Lower → smoother but laggier; higher → snappier
but jitterier.

## Tuning

All constants live at the top of `hand_gesture_controller.py`:

| Constant | Default | Purpose |
|---|---|---|
| `SMOOTHING_FACTOR` | 0.35 | EMA alpha (0 = max smooth, 1 = raw) |
| `PINCH_THRESHOLD` | 0.04 | Normalised distance to trigger click |
| `CLICK_COOLDOWN_SEC` | 0.4 | Seconds between consecutive clicks |
| `FRAME_MARGIN` | 0.1 | Dead-zone at camera edges (fraction) |
| `CAMERA_INDEX` | 0 | Which webcam to use |

## Troubleshooting

- **Cursor jumps to corners** → Increase `FRAME_MARGIN` to 0.15–0.2.
- **Accidental clicks** → Lower `PINCH_THRESHOLD` (e.g. 0.03) or raise `CLICK_COOLDOWN_SEC`.
- **Cursor too laggy** → Raise `SMOOTHING_FACTOR` towards 0.5–0.6.
- **Camera not detected** → Change `CAMERA_INDEX` to 1 or 2.
- **PyAutoGUI fail-safe** → Moving the mouse to the top-left corner of the
  screen will abort the script (built-in safety).

## Dependencies

- Python 3.9+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- PyAutoGUI (`pyautogui`)
- NumPy (`numpy`)
