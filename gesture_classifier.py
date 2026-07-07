"""gesture_classifier.py — Converts hand landmarks into a gesture label."""

# MediaPipe landmark indices for fingertips and their PIP joints (knuckles)
TIP_IDS  = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
PIP_IDS  = [3, 6, 10, 14, 18]   # one joint below each tip

# Gesture name mapped by number of raised fingers
FINGER_GESTURE_MAP = {
    0: "FIST",
    1: "ONE",
    2: "TWO",
    3: "THREE",
    4: "FOUR",
    5: "OPEN_PALM",
}


def count_fingers(landmarks: list) -> int:
    """
    Count how many fingers are raised given a list of 21 (x, y) landmarks.
    Returns an integer 0–5.
    """
    if not landmarks or len(landmarks) < 21:
        return -1

    fingers_up = 0

    # Thumb: compare tip x with IP joint x (horizontal check for side of hand)
    if landmarks[TIP_IDS[0]][0] < landmarks[TIP_IDS[0] - 1][0]:
        fingers_up += 1

    # Other four fingers: tip y < pip y means finger is raised (y increases downward)
    for i in range(1, 5):
        if landmarks[TIP_IDS[i]][1] < landmarks[PIP_IDS[i]][1]:
            fingers_up += 1

    return fingers_up


def is_thumbs_up(landmarks: list) -> bool:
    """
    Detect a thumbs-up: thumb tip significantly above wrist,
    all other fingers curled (tips below their PIPs).
    """
    if not landmarks or len(landmarks) < 21:
        return False

    thumb_up = landmarks[4][1] < landmarks[0][1] - 40  # thumb tip well above wrist
    fingers_curled = all(
        landmarks[TIP_IDS[i]][1] > landmarks[PIP_IDS[i]][1]
        for i in range(1, 5)
    )
    return thumb_up and fingers_curled


def classify_gesture(landmarks: list) -> str:
    """
    Return a gesture string from landmarks.
    Priority: thumbs-up check first, then finger count.
    """
    if not landmarks:
        return "NONE"

    if is_thumbs_up(landmarks):
        return "THUMBS_UP"

    count = count_fingers(landmarks)
    return FINGER_GESTURE_MAP.get(count, "NONE")
