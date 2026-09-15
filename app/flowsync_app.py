import cv2

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

from Camera.camera import open_camera, get_frame
from hand_tracking.hand_tracker import HandTracker
from Landmarks.landmarks_processor import (
    check_finger_visibility,
    process_landmarks,
)
from Landmarks.landmarks_smoothing import LandmarkSmoother

from features.cv_pipeline import get_hand_features


def run():

    # Open camera
    cap = open_camera()

    if cap is None:
        print("Camera not opened.")
        return

    # Create hand tracker
    tracker = HandTracker()

    # Create landmark smoother
    smoothers = {}

    # Create display window
    cv2.namedWindow("FlowSync", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FlowSync", 1280, 720)

    while cap.isOpened():

        # Get frame from camera
        frame = get_frame(cap)

        if frame is None:
            continue

        # Mirror the camera
        frame = cv2.flip(frame, 1)

        # Detect hands
        result = tracker.detect(frame)

        # Process detected landmarks
        processed_landmarks = process_landmarks(result)

        if not processed_landmarks:
            smoothers.clear()

        active_smoothers = set()
        
        # Process each detected hand
        for hand_index, hand in enumerate(processed_landmarks):

            handedness = "hand_" + str(hand_index)
            if result and getattr(result, "handedness", None):
                categories = result.handedness[hand_index]
                if categories:
                    handedness = categories[0].category_name or handedness

            active_smoothers.add(handedness)

            smoother = smoothers.setdefault(
                handedness,
                LandmarkSmoother(alpha=0.5)
            )

            # Smooth landmark coordinates
            smoothed_landmarks = smoother.smooth(
                hand["landmarks"]
            )

            # Use the smoothed coordinates for gesture completeness as well.
            finger_status = check_finger_visibility(smoothed_landmarks)

            # Generate CV feature output
            output = get_hand_features(
                smoothed_landmarks,
                finger_status
            )

            if output is None:
                continue

            # CV output is ready for the AI/ML layer
            # output["feature_vector"] contains 79 features
            # output["finger_states"] contains finger states

        for smoother_id in set(smoothers) - active_smoothers:
            del smoothers[smoother_id]

        # Draw hand landmarks
        if result and getattr(result, "hand_landmarks", None):

            for hand_landmarks in result.hand_landmarks:

                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

        # Display frame
        cv2.imshow("FlowSync", frame)

        # Space or ESC → exit
        if cv2.waitKey(1) & 0xFF in (ord(" "), 27):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()