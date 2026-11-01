import math
import os
import subprocess
import time
from collections import deque

import cv2
from Xlib import X, display
from Xlib.ext import xtest

from detector import heavy_subprogram_detect

# --- TOP CONFIGURATION PARAMETERS ---
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1200

traverse_point = deque(maxlen=15)

swipe_active = False
mouse_active = False

# Cooldown lock timestamp to prevent double-registering / rapid re-toggling
last_toggle_time = 0
TOGGLE_COOLDOWN = 1.5  # Seconds to wait before accepting another toggle

LOST_TOLERANCE = 5
missing_frames = 0

last_scroll_direction = None
RETURN_STROKE_COOLDOWN = 0.5

x11_display = display.Display()
root_window = x11_display.screen().root

SCROLL_COOLDOWN = 0.001
last_active_scroll_time = 0
last_scroll_time = 0

# Absolute Mode butter-smooth animation, deadzone, and precision variables
prev_cursor_pos = None
smooth_abs_x = None
smooth_abs_y = None
RE_SWIPE_VELOCITY_LIMIT = 100

STATUS_FILE = "/tmp/swipe_status.txt"

def write_status(text):
    try:
        with open(STATUS_FILE, "w") as f:
            f.write(text)
        subprocess.run(["pkill", "-SIGRTMIN+10", "i3blocks"], capture_output=True, check=False)
    except Exception:
        pass

def scroll_mouse(direction, multiplier=2):
    button = 4 if direction == "up" else 5
    for _ in range(multiplier):
        xtest.fake_input(x11_display, X.ButtonPress, button)
        xtest.fake_input(x11_display, X.ButtonRelease, button)
    x11_display.sync()

def move_mouse_pointer(index_pos, frame_w, frame_h):
    global prev_cursor_pos, smooth_abs_x, smooth_abs_y
    x, y = index_pos
    
    target_x = (x / frame_w) * SCREEN_WIDTH
    target_y = (y / frame_h) * SCREEN_HEIGHT
    
    if prev_cursor_pos is None:
        prev_cursor_pos = (x, y)
        smooth_abs_x = target_x
        smooth_abs_y = target_y
        return

    raw_dx = x - prev_cursor_pos[0]
    raw_dy = y - prev_cursor_pos[1]
    prev_cursor_pos = (x, y)
    
    if math.hypot(raw_dx, raw_dy) > RE_SWIPE_VELOCITY_LIMIT:
        smooth_abs_x = target_x
        smooth_abs_y = target_y
        return

    distance_to_target = math.hypot(target_x - smooth_abs_x, target_y - smooth_abs_y)
    
    DEADZONE = 1.5
    if distance_to_target < DEADZONE:
        return

    alpha = min(0.5, max(0.08, distance_to_target * 0.03))

    smooth_abs_x = smooth_abs_x * (1.0 - alpha) + target_x * alpha
    smooth_abs_y = smooth_abs_y * (1.0 - alpha) + target_y * alpha

    final_x = max(0, min(SCREEN_WIDTH, int(smooth_abs_x)))
    final_y = max(0, min(SCREEN_HEIGHT, int(smooth_abs_y)))

    xtest.fake_input(x11_display, X.MotionNotify, x=final_x, y=final_y)
    x11_display.sync()

def update_i3_status():
    if swipe_active and mouse_active:
        write_status("SWIPE: ON | MOUSE: ABSOLUTE")
    elif swipe_active:
        write_status("SWIPE: ON | MOUSE: OFF")
    elif mouse_active:
        write_status("SWIPE: OFF | MOUSE: ABSOLUTE")
    else:
        write_status("ALL: OFF")

def append_tracking_point(point):
    if point is None:
        return
    if traverse_point:
        previous_point = traverse_point[-1]
        smoothed_point = (
            int(previous_point[0] * 0.6 + point[0] * 0.4),
            int(previous_point[1] * 0.6 + point[1] * 0.4),
        )
    else:
        smoothed_point = point
    traverse_point.append(smoothed_point)

def draw_circles(frame, traverse_point):
    if not traverse_point:
        return
    points = list(traverse_point)
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (0, 255, 255), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 255), -1)
        cv2.circle(frame, point, 7, (0, 200, 255), 1)

def evaluate_swipe():
    global last_scroll_time, last_active_scroll_time, last_scroll_direction
    if len(traverse_point) < 10:
        return

    current_y = traverse_point[-1][1]
    past_y = traverse_point[-10][1]
    delta_y = current_y - past_y
    current_time = time.time()

    if current_time - last_scroll_time <= SCROLL_COOLDOWN:
        return

    current_direction = "up" if delta_y > 40 else ("down" if delta_y < -40 else None)

    if current_direction:
        if current_direction != last_scroll_direction:
            if current_time - last_active_scroll_time < RETURN_STROKE_COOLDOWN:
                traverse_point.clear()
                return
            last_scroll_direction = current_direction

        scroll_mouse(current_direction, multiplier=2)
        last_scroll_time = current_time
        last_active_scroll_time = current_time
    else:
        if current_time - last_active_scroll_time > 0.2:
            last_scroll_direction = None

def main():
    global swipe_active, mouse_active, last_toggle_time, missing_frames, prev_cursor_pos, smooth_abs_x, smooth_abs_y
    capture = cv2.VideoCapture(0)

    write_status("ALL: OFF")
    print("Tracker running. Open Palm = Toggle Swipe | Victory Sign = Toggle Mouse Control (Absolute).")

    try:
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            result = heavy_subprogram_detect(frame)
            current_time = time.time()

            if result == "PALM":
                if current_time - last_toggle_time > TOGGLE_COOLDOWN:
                    swipe_active = not swipe_active
                    last_toggle_time = current_time
                    traverse_point.clear()
                    update_i3_status()
                    print(f"[INFO] Swipe Mode: {'ACTIVATED' if swipe_active else 'DEACTIVATED'}")

            elif result == "VICTORY":
                if current_time - last_toggle_time > TOGGLE_COOLDOWN:
                    mouse_active = not mouse_active
                    last_toggle_time = current_time
                    prev_cursor_pos = None
                    smooth_abs_x = None
                    smooth_abs_y = None
                    update_i3_status()
                    print(f"[INFO] Mouse Mode (ABSOLUTE): {'ACTIVATED' if mouse_active else 'DEACTIVATED'}")

            elif isinstance(result, tuple):
                index_pos = result
                
                # Handle Swipe logic
                if swipe_active:
                    missing_frames = 0
                    append_tracking_point(index_pos)
                    draw_circles(frame, traverse_point)
                    evaluate_swipe()
                
                # Handle Absolute Mouse movement logic
                if mouse_active:
                    move_mouse_pointer(index_pos, w, h)
            else:
                if swipe_active:
                    missing_frames += 1
                    if missing_frames > LOST_TOLERANCE:
                        traverse_point.clear()
                if mouse_active:
                    prev_cursor_pos = None
                    smooth_abs_x = None
                    smooth_abs_y = None

            if cv2.waitKey(1) == 27:
                break
    finally:
        if os.path.exists(STATUS_FILE):
            os.remove(STATUS_FILE)
        cv2.destroyAllWindows()
        capture.release()

if __name__ == "__main__":
    main()