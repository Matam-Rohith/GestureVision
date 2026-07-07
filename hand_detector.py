"""hand_detector.py — Pure OpenCV skin-based hand detector. No mediapipe needed."""

import cv2
import numpy as np


class HandDetector:
    """
    Detects a hand region using HSV skin-color segmentation and returns
    a 21-point landmark approximation compatible with the gesture classifier.
    No mediapipe required.
    """

    def __init__(self, max_hands: int = 1, detection_conf: float = 0.75, tracking_conf: float = 0.6):
        self.max_hands = max_hands
        # HSV skin color range (works for most skin tones in indoor light)
        self.lower_skin = np.array([0,  20,  70],  dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        self.kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _skin_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel, iterations=3)
        return mask

    def _largest_contour(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _build_landmarks(self, contour, frame_shape) -> list:
        """
        Approximate 21 landmark points from the contour bounding box.
        Returns list of (x, y) tuples that gesture_classifier can use.
        """
        x, y, w, h = cv2.boundingRect(contour)
        H, W = frame_shape[:2]

        # Palm center (landmark 0 = wrist)
        cx, cy = x + w // 2, y + h // 2

        # Generate 21 pseudo-landmarks arranged in a 5-finger fan
        # 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
        landmarks = [(cx, y + h)] # wrist at bottom of bounding box

        # 5 fingers, 4 points each (MCP, PIP, DIP, TIP)
        finger_x_offsets = [-int(w*0.35), -int(w*0.18), 0, int(w*0.18), int(w*0.35)]
        for fx in finger_x_offsets:
            base_x = cx + fx
            base_y = y + int(h * 0.6)
            for seg in range(4):
                lx = base_x
                ly = base_y - seg * (h // 6)
                lx = max(0, min(W - 1, lx))
                ly = max(0, min(H - 1, ly))
                landmarks.append((lx, ly))

        return landmarks[:21]

    def find_hands(self, frame: np.ndarray, draw: bool = True):
        """
        Returns (annotated_frame, list_of_landmark_lists).
        """
        mask    = self._skin_mask(frame)
        contour = self._largest_contour(mask)
        all_landmarks = []

        if contour is not None and cv2.contourArea(contour) > 5000:
            landmarks = self._build_landmarks(contour, frame.shape)
            all_landmarks.append(landmarks)

            if draw:
                cv2.drawContours(frame, [contour], -1, (0, 220, 100), 2)
                for (lx, ly) in landmarks:
                    cv2.circle(frame, (lx, ly), 4, (255, 80, 80), -1)
                # Draw wrist to each finger base
                for i in [1, 5, 9, 13, 17]:
                    if i < len(landmarks):
                        cv2.line(frame, landmarks[0], landmarks[i], (100, 200, 255), 1)

        return frame, all_landmarks
