"""hand_detector.py — MediaPipe Hands detector, compatible with mediapipe 0.10+"""

import cv2
import numpy as np

try:
    # mediapipe 0.10+ new Tasks API
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
    NEW_API = True
except (ImportError, AttributeError):
    NEW_API = False

if not NEW_API:
    import mediapipe as mp


class HandDetector:
    """
    Works with both mediapipe 0.9.x (legacy) and 0.10+ (Tasks API).
    Automatically picks the right backend.
    """

    def __init__(self, max_hands: int = 1, detection_conf: float = 0.75, tracking_conf: float = 0.6):
        self._use_legacy = False
        try:
            # Try legacy solutions API first (mediapipe <= 0.9.x)
            self.mp_hands = mp.solutions.hands
            self.mp_draw  = mp.solutions.drawing_utils
            self.hands    = self.mp_hands.Hands(
                max_num_hands=max_hands,
                min_detection_confidence=detection_conf,
                min_tracking_confidence=tracking_conf,
            )
            self._use_legacy = True
        except AttributeError:
            # mediapipe 0.10+ — use the python wrapper around solutions
            # that ships as mediapipe.python.solutions
            try:
                from mediapipe.python.solutions import hands as mp_hands_module
                from mediapipe.python.solutions import drawing_utils as mp_draw_module
                self.mp_hands = mp_hands_module
                self.mp_draw  = mp_draw_module
                self.hands    = mp_hands_module.Hands(
                    max_num_hands=max_hands,
                    min_detection_confidence=detection_conf,
                    min_tracking_confidence=tracking_conf,
                )
                self._use_legacy = True   # same call interface
            except Exception as e:
                raise RuntimeError(
                    f"Could not initialise MediaPipe Hands.\n"
                    f"Please run: pip install mediapipe==0.10.9\nOriginal error: {e}"
                )

    def find_hands(self, frame: np.ndarray, draw: bool = True):
        """
        Process a BGR frame and return (annotated_frame, list_of_landmark_lists).
        Each landmark list contains 21 (x_pixel, y_pixel) tuples.
        """
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
