"""hand_detector.py — MediaPipe-based hand landmark detector."""

import cv2
import mediapipe as mp


class HandDetector:
    """
    Wraps MediaPipe Hands to detect hand landmarks from a BGR frame.
    """

    def __init__(self, max_hands: int = 1, detection_conf: float = 0.75, tracking_conf: float = 0.6):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def find_hands(self, frame: "np.ndarray", draw: bool = True):
        """
        Process a BGR frame and return (annotated_frame, list_of_landmark_lists).
        Each landmark list contains 21 (x_pixel, y_pixel) tuples.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        all_landmarks = []

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_lms in results.multi_hand_landmarks:
                lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
                all_landmarks.append(lm_list)
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )

        return frame, all_landmarks
