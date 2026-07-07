"""gesture_classifier.py — Finger counting from bounding-box landmarks."""

# Landmark layout (from hand_detector.py):
# 0 = wrist
# 1-4   = thumb  (MCP, PIP, DIP, TIP)
# 5-8   = index  (MCP, PIP, DIP, TIP)
# 9-12  = middle (MCP, PIP, DIP, TIP)
# 13-16 = ring   (MCP, PIP, DIP, TIP)
# 17-20 = pinky  (MCP, PIP, DIP, TIP)

FINGER_TIPS  = [4, 8, 12, 16, 20]   # TIP of each finger
FINGER_MIDS  = [2, 6, 10, 14, 18]   # PIP of each finger

FINGER_GESTURE_MAP = {
    0: "FIST",
    1: "ONE",
    2: "TWO",
    3: "THREE",
    4: "FOUR",
    5: "OPEN_PALM",
}


def count_fingers(landmarks: list) -> int:
    if not landmarks or len(landmarks) < 21:
        return -1
    count = 0
    for tip, mid in zip(FINGER_TIPS, FINGER_MIDS):
        # Finger is raised if tip is above (lower y value) than its mid joint
        if landmarks[tip][1] < landmarks[mid][1]:
            count += 1
    return count


def is_thumbs_up(landmarks: list) -> bool:
    if not landmarks or len(landmarks) < 21:
        return False
    thumb_up      = landmarks[4][1] < landmarks[0][1] - 40
    fingers_curled = all(landmarks[FINGER_TIPS[i]][1] > landmarks[FINGER_MIDS[i]][1] for i in range(1, 5))
    return thumb_up and fingers_curled


def classify_gesture(landmarks: list) -> str:
    if not landmarks:
        return "NONE"
    if is_thumbs_up(landmarks):
        return "THUMBS_UP"
    count = count_fingers(landmarks)
    return FINGER_GESTURE_MAP.get(count, "NONE")
