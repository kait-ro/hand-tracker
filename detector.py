import os
import urllib.request
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'gesture_recognizer.task'
if not os.path.exists(model_path):
    print("Downloading gesture_recognizer.task model (this only happens once)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/hand_landmarker.task",
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

recognizer = vision.GestureRecognizer.create_from_options(options)

INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
THUMB_TIP = 4

def heavy_subprogram_detect(frame, mouse_active=False):
    """Returns 'PALM', 'VICTORY', ('CLICK', (x,y)), or the index fingertip (x, y) coordinate tuple."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    recognition_result = recognizer.recognize(mp_image)

    if not recognition_result.hand_landmarks:
        return None

    first_hand_landmarks = recognition_result.hand_landmarks[0]
    h, w, _ = frame.shape

    # 1. Lax Geometric Open Palm Check
    finger_tip_ids = [8, 12, 16, 20]
    finger_pip_ids = [6, 10, 14, 18]
    open_fingers = 0
    for tip, pip in zip(finger_tip_ids, finger_pip_ids):
        if first_hand_landmarks[tip].y < first_hand_landmarks[pip].y:
            open_fingers += 1
            
    if open_fingers >= 4:
        return "PALM"

    # 2. Lax Model-Level Victory Sign Check
    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        if top_gesture.score > 0.45 and top_gesture.category_name == "Victory":
            return "VICTORY"

    index_tip = first_hand_landmarks[INDEX_FINGER_TIP]
    index_pos = (int(index_tip.x * w), int(index_tip.y * h))

    # 3. Middle and Thumb Pinch Check (Left Click) - ONLY checked if mouse is active
    if mouse_active:
        thumb = first_hand_landmarks[THUMB_TIP]
        middle = first_hand_landmarks[MIDDLE_FINGER_TIP]
        
        # Calculate normalized distance between thumb and middle finger
        pinch_dist = math.hypot(thumb.x - middle.x, thumb.y - middle.y)
        
        if pinch_dist < 0.05:  
            return ("CLICK", index_pos)

    # Return pure index fingertip tuple for tracking & mouse control
    return index_pos