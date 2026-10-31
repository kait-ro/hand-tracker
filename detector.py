import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download the required Gesture Recognizer model file if it doesn't exist
model_path = 'gesture_recognizer.task'
if not os.path.exists(model_path):
    print("Downloading gesture_recognizer.task model (this only happens once)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize the native Gesture Recognizer globally
recognizer = vision.GestureRecognizer.create_from_options(options)

INDEX_FINGER_TIP = 8


def heavy_subprogram_detect(frame):
    """Runs MediaPipe Gesture Recognizer.

    Returns "PALM" if an open palm is recognized natively,
    the index fingertip (x, y) tuple if hand landmarks are found,
    or None otherwise.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    recognition_result = recognizer.recognize(mp_image)

    if not recognition_result.hand_landmarks:
        return None

    # Check model-level recognized gestures for an Open Palm
    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        if top_gesture.category_name == "Open_Palm" and top_gesture.score > 0.5:
            return "PALM"

    first_hand_landmarks = recognition_result.hand_landmarks[0]
    h, w, _ = frame.shape

    tip = first_hand_landmarks[INDEX_FINGER_TIP]
    x, y = int(tip.x * w), int(tip.y * h)

    return (x, y)