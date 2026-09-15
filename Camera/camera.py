import cv2
import mediapipe as mp

def get_frame(cap):
    success, frame = cap.read()

    if not success or frame is None or frame.size == 0:
        return None

    return frame

def open_camera():
    camera_candidates = [
        (0, cv2.CAP_DSHOW),
        (0, cv2.CAP_ANY),
        (1, cv2.CAP_DSHOW),
        (1, cv2.CAP_ANY),
        (2, cv2.CAP_DSHOW),
        (2, cv2.CAP_ANY),
    ]

    for index, backend in camera_candidates:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        for _ in range(3):
            success, frame = cap.read()
            if success and frame is not None and frame.size > 0:
                return cap
            if not success:
                break

        cap.release()
    return None