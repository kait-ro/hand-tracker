import time
from collections import deque

import cv2
from Xlib import X, display
from Xlib.ext import xtest

from detector import heavy_subprogram_detect

traverse_point = deque(maxlen=15)

last_scroll_direction = None
RETURN_STROKE_COOLDOWN = 0.5

# Grace period: allow this many consecutive missed frames during fast motion 
# before wiping the tracking queue
LOST_TOLERANCE = 5
missing_frames = 0

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
    global missing_frames
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)

        result = heavy_subprogram_detect(frame)

        if result == "PALM":
            print("Open palm detected. Shutting down...")
            break
        elif result is not None:
            missing_frames = 0  # Reset missing counter on successful detection
            append_tracking_point(result)
            draw_circles(frame, traverse_point)
            evaluate_swipe()
        else:
            missing_frames += 1
            # Only wipe the queue if the hand is truly lost for several consecutive frames
            if missing_frames > LOST_TOLERANCE:
                traverse_point.clear()

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    main()