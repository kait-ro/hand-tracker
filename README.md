# Air Gesture & Mouse Control System

A lightweight, real-time computer vision application that turns your webcam into an air-gesture controller and an absolute-mode mouse pointer using MediaPipe and X11.

## Features

- **Open Palm Toggle:** Easily toggle swipe/scrolling modes on or off using an open hand gesture.
- **Victory Sign Toggle:** Switch absolute mouse pointer control on or off using a classic "V" / Victory hand gesture.
- **Butter-Smooth Cursor Tracking:** Features built-in deadzones and exponential smoothing parameters for precise and fluid mouse movement.
- **Status Bar Integration:** Automatically outputs operational status updates to `/tmp/swipe_status.txt` for integration with status bars like `i3blocks`.

---

## Prerequisites

- **Linux Operating System** running an **X11 Display Server** (e.g., i3wm, GNOME on Xorg). *Note: Wayland is not directly supported by `python-xlib`.*
- **Python 3.8+**
- A working **Webcam**.

---

## Installation & Setup

1. **Clone or download** the project files (`main.py` and `detector.py`) into a dedicated directory.

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt