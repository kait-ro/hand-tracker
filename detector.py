import os
import urllib.request
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

THUMB_TIP = 4
INDEX_FINGER_TIP = 8

def heavy_subprogram_detect(frame):
    """Returns 'THUMB_UP', 'PALM', or a dict containing index fingertip and thumb tip coordinates."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    recognition_result = recognizer.recognize(mp_image)

    if not recognition_result.hand_landmarks:
        return None

    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        if top_gesture.score > 0.5:
            if top_gesture.category_name == "Thumb_Up":
                return "THUMB_UP"
            elif top_gesture.category_name == "Open_Palm":
                return "PALM"

    first_hand_landmarks = recognition_result.hand_landmarks[0]
    h, w, _ = frame.shape
    
    index_tip = first_hand_landmarks[INDEX_FINGER_TIP]
    thumb_tip = first_hand_landmarks[THUMB_TIP]

    return {
        "index": (int(index_tip.x * w), int(index_tip.y * h)),
        "thumb": (int(thumb_tip.x * w), int(thumb_tip.y * h))
    }