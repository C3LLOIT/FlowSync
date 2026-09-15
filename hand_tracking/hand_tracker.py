import os
import urllib.request
import zipfile

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_ASSET_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def is_valid_task_file(path: str) -> bool:
    return zipfile.is_zipfile(path)


def ensure_model_asset(model_path: str = MODEL_ASSET_PATH) -> str:
    if os.path.exists(model_path) and is_valid_task_file(model_path):
        return model_path

    if os.path.exists(model_path):
        os.remove(model_path)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to download the hand landmark model: {exc}") from exc

    if not is_valid_task_file(model_path):
        raise RuntimeError("Downloaded hand_landmarker.task is not a valid task archive.")

    return model_path


class HandTracker:
    def __init__(self):
        model_path = ensure_model_asset()

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.35,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_count = 0
        self.fps = 30  # Match camera.py setting

    def detect(self, frame):
        detection_frame = self._prepare_frame(frame)
        rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Convert frame count to milliseconds timestamp
        timestamp_ms = int(self.frame_count * 1000 / self.fps)
        self.frame_count += 1

        return self.detector.detect_for_video(mp_image, timestamp_ms)

    @staticmethod
    def _prepare_frame(frame):
        """Improve local contrast and lift very dark frames for detection."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, green_red, blue_yellow = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            lightness
        )
        enhanced = cv2.cvtColor(
            cv2.merge((lightness, green_red, blue_yellow)),
            cv2.COLOR_LAB2BGR,
        )

        mean_brightness = float(np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)))
        if mean_brightness < 105.0:
            gamma = 0.75 if mean_brightness >= 65.0 else 0.62
            lookup = np.array(
                [((value / 255.0) ** gamma) * 255 for value in range(256)],
                dtype=np.uint8,
            )
            enhanced = cv2.LUT(enhanced, lookup)

        return enhanced