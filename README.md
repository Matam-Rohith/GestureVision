# 🖐️ GestureVision

> Real-time AI-powered camera application — control visual filters using hand gestures. No keyboard. No mouse. Just your hands.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-FF6F00?style=flat-square&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📸 Demo

| Gesture | Filter Applied |
|---|---|
| ✋ Open Palm (5 fingers) | Normal / No Filter |
| ☝️ 1 Finger (Index up) | Grayscale |
| ✌️ 2 Fingers | Sepia Tone |
| 🤟 3 Fingers | Canny Edge Detection |
| 🖖 4 Fingers | Cartoonify |
| 👊 Fist (0 fingers) | Invert Colors |
| 👍 Thumbs Up | Blur (Gaussian) |

---

## 🚀 Features

- 🖐️ **Real-time hand gesture detection** using MediaPipe Hands
- 🎨 **7 visual filters** switchable hands-free
- 📊 **FPS counter** and gesture label overlay on screen
- 🖥️ **Webcam live feed** processed at 30+ FPS
- 💡 Works in regular indoor lighting
- 🔁 Smooth filter transitions with debounce to prevent accidental switches

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core language |
| OpenCV | Video capture, image processing, filter rendering |
| MediaPipe | Hand landmark detection (21 keypoints) |
| NumPy | Array operations for filter math |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Matam-Rohith/GestureVision.git
cd GestureVision
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python gesture_vision.py
```

> Press **`Q`** to quit the application.

---

## 📁 Project Structure

```
GestureVision/
├── gesture_vision.py      # Main application entry point
├── hand_detector.py       # MediaPipe hand landmark detection module
├── filters.py             # All visual filter implementations
├── gesture_classifier.py  # Finger counting & gesture logic
├── requirements.txt       # Python dependencies
├── README.md
└── assets/
    └── demo.png           # Demo screenshot (add your own)
```

---

## 🧠 How It Works

```
Webcam Frame
    ↓
MediaPipe Hands → 21 Hand Landmarks (x, y, z)
    ↓
Gesture Classifier → Count raised fingers
    ↓
Filter Engine → Apply corresponding OpenCV filter
    ↓
Display on Screen with gesture label + FPS
```

MediaPipe detects **21 keypoints** per hand. The gesture classifier checks which fingertips are above their respective knuckle joints to count raised fingers and maps that to a filter.

---

## 🔧 Customization

Add your own filter in `filters.py`:
```python
def my_custom_filter(frame):
    # your OpenCV operations here
    return modified_frame
```
Then register it in `gesture_vision.py` under the `FILTER_MAP` dictionary.

---

## 📋 Requirements

```
opencv-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.21.0
```

---

## 👤 Author

**Matam Rohith**
- Portfolio: [rohith-portfolio-six.vercel.app](https://rohith-portfolio-six.vercel.app/)
- GitHub: [@Matam-Rohith](https://github.com/Matam-Rohith)
- LinkedIn: [matam-rohith](https://www.linkedin.com/in/matam-rohith/)

---

## 📄 License

MIT License — free to use, modify, and distribute.
