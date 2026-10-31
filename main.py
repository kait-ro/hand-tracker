import time
from collections import deque

import cv2
import numpy as np
from Xlib import X, display
from Xlib.ext import xtest

from detector import heavy_subprogram_detect

traverse_point = deque(maxlen=15)

# Refined thresholds for optimal balance between speed and stability
LOST_THRESHOLD = 12
REFRESH_INTERVAL = 5

last_scroll_direction = None
RETURN_STROKE_COOLDOWN = 0.5

# Expanded window size (31x31) helps single-point LK hold onto texture much better
LK_PARAMS = {
    "winSize": (31, 31),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
}

x11_display = display.Display()
SCROLL_COOLDOWN = 0.001
last_active_scroll_time = 0
last_scroll_time = 0


def scroll_mouse(direction, multiplier=2):
    button = 4 if direction == "up" else 5
    for _ in range(multiplier):
        xtest.fake_input(x11_display, X.ButtonPress, button)
        xtest.fake_input(x11_display, X.ButtonRelease, button)
    x11_display.sync()


def append_tracking_point(point):
    if point is None:
        return

    if traverse_point:
        previous_point = traverse_point[-1]
        smoothed_point = (
            int(previous_point[0] * 0.7 + point[0] * 0.3),
            int(previous_point[1] * 0.7 + point[1] * 0.3),
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

    current_direction = None
    if delta_y > 40:
        current_direction = "up"
    elif delta_y < -40:
        current_direction = "down"

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
    capture = cv2.VideoCapture(0)

    tracked_point = None
    prev_gray = None
    lost_frames = 0
    frames_since_refresh = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        need_detect = (
            tracked_point is None
            or lost_frames > LOST_THRESHOLD
            or frames_since_refresh >= REFRESH_INTERVAL
        )

        if need_detect:
            result = heavy_subprogram_detect(frame)
            
            if result == "PALM":
                print("Open palm detected. Shutting down...")
                break
                
            elif result is not None:
                tracked_point = result
                lost_frames = 0
                frames_since_refresh = 0
            else:
                tracked_point = None
                lost_frames += 1
        else:
            prev_pts = np.array([[tracked_point]], dtype=np.float32)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **LK_PARAMS
            )

            tracked_ok = status is not None and status[0][0] == 1
            if tracked_ok:
                x, y = next_pts[0][0]
                h, w = gray.shape
                if 0 <= x < w and 0 <= y < h:
                    tracked_point = (int(x), int(y))
                    lost_frames = 0
                    frames_since_refresh += 1
                else:
                    tracked_point = None
                    lost_frames += 1
            else:
                tracked_point = None
                lost_frames += 1

        if tracked_point is not None:
            cv2.circle(frame, tracked_point, 6, (0, 0, 255), -1)
            append_tracking_point(tracked_point)
            draw_circles(frame, traverse_point)
            evaluate_swipe()
        else:
            traverse_point.clear()

        prev_gray = gray

        # cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    main()