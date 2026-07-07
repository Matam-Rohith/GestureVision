"""hand_detector.py — MediaPipe Hands, compatible with mediapipe 0.10.30+"""

import cv2
import numpy as np
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_draw_module


class HandDetector:
    def __init__(self, max_hands: int = 1, detection_conf: float = 0.75, tracking_conf: float = 0.6):
        self.mp_hands = mp_hands_module
        self.mp_draw  = mp_draw_module
        self.hands    = mp_hands_module.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def find_hands(self, frame: np.ndarray, draw: bool = True):
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        all_landmarks = []

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_lms in results.multi_hand_landmarks:
                lm_list = [
                    (int(lm.x * w), int(lm.y * h))
                    for lm in hand_lms.landmark
                ]
                all_landmarks.append(lm_list)
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame, hand_lms,
                        self.mp_hands.HAND_CONNECTIONS
                    )

        return frame, all_landmarks
