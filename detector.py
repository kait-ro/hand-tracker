import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Download the required model file if it doesn't exist
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand_landmarker.task model (this only happens once)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path,
    )

# 2. Configure the new Tasks API
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=1, min_hand_detection_confidence=0.5
)

# Initialize globally to keep it in RAM
detector = vision.HandLandmarker.create_from_options(options)

# MediaPipe hand landmark index for the index fingertip.
INDEX_FINGER_TIP = 8



def heavy_subprogram_detect(frame):
    """
    returns "PALM" if an open palm gesture is detected.
    returns the index fingertip as an (x, y) pixel tuple, or None if no
    hand was found.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = detector.detect(mp_image)

    if not results.hand_landmarks:
        return None

    first_hand_landmarks = results.hand_landmarks[0]
    h, w, _ = frame.shape

    # Check for Open Palm Gesture
    # We check if the Index, Middle, Ring, and Pinky tips are higher than their PIP joints (knuckles).
    # In MediaPipe, y=0 is the top of the screen, so a lower 'y' means higher up physically.
    is_palm = True
    finger_tip_ids = [8, 12, 16, 20]
    finger_pip_ids = [6, 10, 14, 18]

    for tip, pip in zip(finger_tip_ids, finger_pip_ids):
        if first_hand_landmarks[tip].y > first_hand_landmarks[pip].y:
            is_palm = False
            break

    if is_palm:
        return "PALM"

    # If not a palm, return just the index fingertip for tracking
    tip = first_hand_landmarks[INDEX_FINGER_TIP]
    x, y = int(tip.x * w), int(tip.y * h)

    return (x, y)
