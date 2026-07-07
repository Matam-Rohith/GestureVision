"""filters.py — All visual filter implementations using OpenCV + NumPy."""

import cv2
import numpy as np


def apply_normal(frame: np.ndarray) -> np.ndarray:
    """No filter — original feed."""
    return frame


def apply_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert to grayscale and back to BGR for display."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_sepia(frame: np.ndarray) -> np.ndarray:
    """Warm sepia tone effect."""
    kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    sepia = cv2.transform(frame.astype(np.float64), kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia


def apply_edges(frame: np.ndarray) -> np.ndarray:
    """Canny edge detection — white edges on black background."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_cartoon(frame: np.ndarray) -> np.ndarray:
    """Cartoonify: bilateral filter + edge overlay."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def apply_invert(frame: np.ndarray) -> np.ndarray:
    """Invert all pixel colors."""
    return cv2.bitwise_not(frame)


def apply_blur(frame: np.ndarray) -> np.ndarray:
    """Strong Gaussian blur."""
    return cv2.GaussianBlur(frame, (31, 31), 0)


# Map gesture name → filter function
FILTER_MAP = {
    "OPEN_PALM":  apply_normal,
    "ONE":        apply_grayscale,
    "TWO":        apply_sepia,
    "THREE":      apply_edges,
    "FOUR":       apply_cartoon,
    "FIST":       apply_invert,
    "THUMBS_UP":  apply_blur,
    "NONE":       apply_normal,
}


def get_filter(gesture: str):
    """Return the filter function for a given gesture string."""
    return FILTER_MAP.get(gesture, apply_normal)
