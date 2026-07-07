"""gesture_vision.py — Main entry point for GestureVision.

Run:  python gesture_vision.py
Quit: press Q
"""

import time
import cv2
from hand_detector import HandDetector
from gesture_classifier import classify_gesture
from filters import get_filter, FILTER_MAP

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX    = 0          # Change to 1/2 if default webcam isn't used
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
DEBOUNCE_SEC    = 0.6        # Seconds before gesture switch is accepted
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# Gesture → display label
GESTURE_LABELS = {
    "OPEN_PALM":  "✋ Normal",
    "ONE":        "☝️  Grayscale",
    "TWO":        "✌️  Sepia",
    "THREE":      "🤟 Edge Detection",
    "FOUR":       "🖖 Cartoon",
    "FIST":       "✊ Invert",
    "THUMBS_UP":  "👍 Blur",
    "NONE":       "— Detecting...",
}
# ─────────────────────────────────────────────────────────────────────────────


def draw_ui(frame, gesture: str, fps: float):
    """Overlay gesture label, FPS, and gesture guide onto frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "GestureVision", (14, 38), FONT, 1.1, (255, 255, 255), 2)

    # Current filter label
    label = GESTURE_LABELS.get(gesture, gesture)
    cv2.putText(frame, f"Filter: {label}", (w // 2 - 160, 40), FONT, 0.9, (0, 230, 120), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 130, 40), FONT, 0.8, (200, 200, 200), 2)

    # Gesture guide — bottom panel
    guide_lines = [
        "0 fingers=Invert  1=Gray  2=Sepia",
        "3=Edges  4=Cartoon  5=Normal  Thumb=Blur",
    ]
    panel_y = h - 55
    cv2.rectangle(frame, (0, panel_y - 8), (w, h), (20, 20, 20), -1)
    for i, line in enumerate(guide_lines):
        cv2.putText(frame, line, (14, panel_y + 18 + i * 22), FONT, 0.55, (180, 180, 180), 1)

    return frame


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check CAMERA_INDEX in gesture_vision.py.")
        return

    detector         = HandDetector(max_hands=1)
    current_gesture  = "NONE"
    last_switch_time = 0.0
    prev_frame_time  = 0.0

    print("[GestureVision] Running — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural feel

        # ── Hand detection ────────────────────────────────────────────────────
        frame, all_landmarks = detector.find_hands(frame, draw=True)

        # ── Gesture classification with debounce ──────────────────────────────
        if all_landmarks:
            detected = classify_gesture(all_landmarks[0])
            now = time.time()
            if detected != current_gesture and (now - last_switch_time) > DEBOUNCE_SEC:
                current_gesture  = detected
                last_switch_time = now
        else:
            current_gesture = "NONE"

        # ── Apply filter ──────────────────────────────────────────────────────
        filter_fn = get_filter(current_gesture)
        frame     = filter_fn(frame)

        # ── FPS calculation ───────────────────────────────────────────────────
        now = time.time()
        fps = 1 / (now - prev_frame_time + 1e-9)
        prev_frame_time = now

        # ── Draw UI overlay ───────────────────────────────────────────────────
        frame = draw_ui(frame, current_gesture, fps)

        cv2.imshow("GestureVision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[GestureVision] Exited.")


if __name__ == "__main__":
    main()
