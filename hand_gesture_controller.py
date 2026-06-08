"""
╔══════════════════════════════════════════════════════════════════════╗
║               AI Hand Gesture Controller – Virtual Hand Mouse       ║
║                                                                      ║
║  Tech Stack: Python · OpenCV · MediaPipe · PyAutoGUI · NumPy         ║
║                                                                      ║
║  Controls:                                                           ║
║    • Move index finger  → cursor follows                             ║
║    • Pinch (thumb + index finger together) → left click              ║
║    • Press 'q' or ESC   → quit                                       ║
╚══════════════════════════════════════════════════════════════════════╝

Distance-Calculation Math
─────────────────────────
We compute the **Euclidean distance** between the Thumb Tip (landmark 4)
and the Index Finger Tip (landmark 8) in *normalised* [0, 1] coordinate
space that MediaPipe returns.

    d = √[ (x₈ − x₄)² + (y₈ − y₄)² ]

Because MediaPipe landmarks are normalised to the frame dimensions,
`d` is a unitless ratio.  A value < PINCH_THRESHOLD (default 0.04)
indicates the two fingertips are touching → we fire a click.

Coordinate Smoothing
────────────────────
Raw landmark positions jitter frame-to-frame.  We apply an
**Exponential Moving Average (EMA)** – a single-pole low-pass filter:

    smoothed_x = α · raw_x  +  (1 − α) · prev_smoothed_x

α (SMOOTHING_FACTOR, default 0.35) controls responsiveness:
  • α → 1  ⟹  no smoothing (instant, jittery)
  • α → 0  ⟹  maximum smoothing (laggy, buttery)

0.35 is a balanced default; tune it to taste.
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ──────────────────────────────────────────────────────────────────────
# Configuration constants – tweak these to taste
# ──────────────────────────────────────────────────────────────────────

# Camera settings
CAMERA_INDEX: int = 0          # 0 = default webcam
CAMERA_WIDTH: int = 640        # capture resolution (width)
CAMERA_HEIGHT: int = 480       # capture resolution (height)

# Smoothing
SMOOTHING_FACTOR: float = 0.35  # EMA α  (0 = max smooth, 1 = no smooth)

# Pinch-to-click
PINCH_THRESHOLD: float = 0.04   # normalised distance to trigger a click
CLICK_COOLDOWN_SEC: float = 0.4  # seconds between consecutive clicks

# Screen-edge dead-zone padding (fraction of frame to ignore at edges,
# so you don't have to reach the very corner of the camera view).
FRAME_MARGIN: float = 0.1       # 10 % margin on each side

# PyAutoGUI safety
pyautogui.FAILSAFE = True       # move mouse to top-left corner to abort
pyautogui.PAUSE = 0             # no built-in pause between pyautogui calls

# ──────────────────────────────────────────────────────────────────────
# MediaPipe Hands initialisation
# ──────────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark indices we care about
INDEX_FINGER_TIP = 8
THUMB_TIP = 4


def euclidean_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """Return the Euclidean distance between two 2-D points.

    Math:  d = √[(x₂ − x₁)² + (y₂ − y₁)²]

    Parameters
    ----------
    p1, p2 : tuple[float, float]
        Each point is (x, y) in any consistent coordinate system.

    Returns
    -------
    float
        Non-negative scalar distance.
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def map_coordinates(
    norm_x: float,
    norm_y: float,
    screen_w: int,
    screen_h: int,
    margin: float = FRAME_MARGIN,
) -> Tuple[int, int]:
    """Map normalised hand coordinates to absolute screen pixel coordinates.

    We clamp the normalised values into [margin, 1 − margin] first so the
    user doesn't need to reach the extreme edges of the camera frame to
    hit the screen edges, then linearly interpolate to [0, screen_w/h).

    Parameters
    ----------
    norm_x, norm_y : float
        MediaPipe normalised coordinates in [0, 1].
    screen_w, screen_h : int
        Display resolution in pixels.
    margin : float
        Fraction of the frame to treat as dead-zone on each side.

    Returns
    -------
    (screen_x, screen_y) : tuple[int, int]
        Clamped pixel coordinates on the display.
    """
    # Clamp into the usable region
    clamped_x = max(margin, min(norm_x, 1.0 - margin))
    clamped_y = max(margin, min(norm_y, 1.0 - margin))

    # Re-scale [margin .. 1-margin] → [0 .. 1]
    usable = 1.0 - 2.0 * margin
    scaled_x = (clamped_x - margin) / usable
    scaled_y = (clamped_y - margin) / usable

    # Map to screen pixels
    screen_x = int(scaled_x * screen_w)
    screen_y = int(scaled_y * screen_h)

    # Final safety clamp
    screen_x = max(0, min(screen_x, screen_w - 1))
    screen_y = max(0, min(screen_y, screen_h - 1))

    return screen_x, screen_y


class GestureController:
    """Encapsulates the main loop: capture → detect → act → display."""

    def __init__(self) -> None:
        # Screen resolution
        self.screen_w, self.screen_h = pyautogui.size()

        # Smoothed cursor position (initialise to centre of screen)
        self.smooth_x: float = self.screen_w / 2
        self.smooth_y: float = self.screen_h / 2

        # Click state
        self.last_click_time: float = 0.0
        self.is_pinching: bool = False

        # FPS calculation
        self.fps_buffer: deque[float] = deque(maxlen=30)
        self.prev_frame_time: float = time.time()

        # Distance history for HUD sparkline (last N values)
        self.dist_history: deque[float] = deque(maxlen=60)

    # ─── main loop ────────────────────────────────────────────────

    def run(self) -> None:
        """Open the camera, process frames, and control the cursor."""

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam. Check CAMERA_INDEX.")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        print("─" * 60)
        print("  AI Hand Gesture Controller is RUNNING")
        print(f"  Screen resolution : {self.screen_w} × {self.screen_h}")
        print(f"  Pinch threshold   : {PINCH_THRESHOLD}")
        print(f"  Smoothing factor  : {SMOOTHING_FACTOR}")
        print("  Press 'q' or ESC to quit.")
        print("─" * 60)

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        ) as hands:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("[WARN] Empty frame – skipping.")
                    continue

                # 1) Flip horizontally so it mirrors the user
                frame = cv2.flip(frame, 1)

                # 2) Convert BGR → RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False  # perf hint

                # 3) Run hand detection
                results = hands.process(rgb_frame)

                # 4) Process landmarks if a hand is detected
                hand_detected = False
                pinch_dist = 0.0

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        hand_detected = True

                        # Draw the skeleton on the frame
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )

                        # Extract the landmarks we need
                        idx_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
                        thumb_tip = hand_landmarks.landmark[THUMB_TIP]

                        # ── Cursor movement ──────────────────────────
                        raw_x, raw_y = map_coordinates(
                            idx_tip.x,
                            idx_tip.y,
                            self.screen_w,
                            self.screen_h,
                        )

                        # Apply EMA smoothing
                        self.smooth_x = (
                            SMOOTHING_FACTOR * raw_x
                            + (1 - SMOOTHING_FACTOR) * self.smooth_x
                        )
                        self.smooth_y = (
                            SMOOTHING_FACTOR * raw_y
                            + (1 - SMOOTHING_FACTOR) * self.smooth_y
                        )

                        # Move the OS cursor
                        pyautogui.moveTo(
                            int(self.smooth_x),
                            int(self.smooth_y),
                            _pause=False,
                        )

                        # ── Click detection ──────────────────────────
                        pinch_dist = euclidean_distance(
                            (idx_tip.x, idx_tip.y),
                            (thumb_tip.x, thumb_tip.y),
                        )
                        self.dist_history.append(pinch_dist)

                        now = time.time()

                        if pinch_dist < PINCH_THRESHOLD:
                            if (
                                not self.is_pinching
                                and (now - self.last_click_time) > CLICK_COOLDOWN_SEC
                            ):
                                pyautogui.click(_pause=False)
                                self.last_click_time = now
                            self.is_pinching = True
                        else:
                            self.is_pinching = False

                        # Draw a line between thumb and index for visual feedback
                        h, w, _ = frame.shape
                        pt_idx = (int(idx_tip.x * w), int(idx_tip.y * h))
                        pt_thumb = (int(thumb_tip.x * w), int(thumb_tip.y * h))

                        line_color = (0, 0, 255) if self.is_pinching else (0, 255, 0)
                        cv2.line(frame, pt_idx, pt_thumb, line_color, 3)
                        cv2.circle(frame, pt_idx, 10, (255, 0, 255), cv2.FILLED)
                        cv2.circle(frame, pt_thumb, 10, (255, 200, 0), cv2.FILLED)

                # 5) HUD overlay
                self._draw_hud(frame, hand_detected, pinch_dist)

                # 6) Display the annotated frame
                cv2.imshow("AI Hand Gesture Controller", frame)

                # 7) Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Hand Gesture Controller stopped.")

    # ─── HUD rendering ───────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        hand_detected: bool,
        pinch_dist: float,
    ) -> None:
        """Draw an informational heads-up display on the frame."""

        # FPS calculation
        now = time.time()
        dt = now - self.prev_frame_time
        self.prev_frame_time = now
        if dt > 0:
            self.fps_buffer.append(1.0 / dt)
        avg_fps = sum(self.fps_buffer) / max(len(self.fps_buffer), 1)

        h, w, _ = frame.shape

        # Semi-transparent dark banner at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # FPS
        cv2.putText(
            frame,
            f"FPS: {avg_fps:.0f}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        # Hand status
        status_text = "Hand: TRACKING" if hand_detected else "Hand: NOT FOUND"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(
            frame,
            status_text,
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            1,
            cv2.LINE_AA,
        )

        if hand_detected:
            # Pinch distance indicator
            dist_text = f"Pinch Dist: {pinch_dist:.4f}"
            dist_color = (0, 0, 255) if pinch_dist < PINCH_THRESHOLD else (200, 200, 200)
            cv2.putText(
                frame,
                dist_text,
                (200, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                dist_color,
                1,
                cv2.LINE_AA,
            )

            # Click indicator
            if self.is_pinching:
                cv2.putText(
                    frame,
                    "CLICK!",
                    (200, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Cursor position
            cv2.putText(
                frame,
                f"Cursor: ({int(self.smooth_x)}, {int(self.smooth_y)})",
                (420, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        # Bottom bar: instructions
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 30), (w, h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
        cv2.putText(
            frame,
            "Pinch to click  |  Move index finger to move cursor  |  'q' / ESC to quit",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )


# ──────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    controller = GestureController()
    controller.run()
