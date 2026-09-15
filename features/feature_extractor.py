import math
from features.feature_schema import (
    LANDMARK_FEATURES,
    DISTANCE_FEATURES,
    ANGLE_FEATURES
)


# ---------------------------------------------------------
# 1. LANDMARK NORMALIZATION
# ---------------------------------------------------------

def normalize_landmarks(landmarks):
    if not landmarks:
        return None

    wrist = landmarks.get("Wrist")
    middle_mcp = landmarks.get("Middle_MCP")

    if wrist is None or middle_mcp is None:
        return None

    # Calculate hand scale
    hand_scale = math.sqrt(
        (middle_mcp["x"] - wrist["x"]) ** 2 +
        (middle_mcp["y"] - wrist["y"]) ** 2 +
        (middle_mcp["z"] - wrist["z"]) ** 2
    )

    if hand_scale == 0:
        return None

    normalized = {}

    for name, point in landmarks.items():

        if point is None:
            normalized[name] = None
            continue

        normalized[name] = {
            "x": (point["x"] - wrist["x"]) / hand_scale,
            "y": (point["y"] - wrist["y"]) / hand_scale,
            "z": (point["z"] - wrist["z"]) / hand_scale
        }

    return normalized


# ---------------------------------------------------------
# 2. DISTANCE FEATURES
# ---------------------------------------------------------

def calculate_distance(point1, point2):

    if point1 is None or point2 is None:
        return None

    distance = math.sqrt(
        (point1["x"] - point2["x"]) ** 2 +
        (point1["y"] - point2["y"]) ** 2 +
        (point1["z"] - point2["z"]) ** 2
    )

    return distance


DISTANCE_PAIRS = {

    "thumb_tip_to_index_tip":
        ("Thumb_Tip", "Index_Tip"),

    "index_tip_to_middle_tip":
        ("Index_Tip", "Middle_Tip"),

    "middle_tip_to_ring_tip":
        ("Middle_Tip", "Ring_Tip"),

    "ring_tip_to_pinky_tip":
        ("Ring_Tip", "Pinky_Tip"),

    "wrist_to_index_tip":
        ("Wrist", "Index_Tip"),

    "wrist_to_middle_tip":
        ("Wrist", "Middle_Tip"),

    "wrist_to_ring_tip":
        ("Wrist", "Ring_Tip"),

    "wrist_to_pinky_tip":
        ("Wrist", "Pinky_Tip")
}


def extract_distance_features(landmarks):

    features = {}

    for feature_name, (point1_name, point2_name) in DISTANCE_PAIRS.items():

        point1 = landmarks.get(point1_name)
        point2 = landmarks.get(point2_name)

        features[feature_name] = calculate_distance(
            point1,
            point2
        )

    return features


# ---------------------------------------------------------
# 3. ANGLE FEATURES
# ---------------------------------------------------------

def calculate_angle(point1, point2, point3):

    if point1 is None or point2 is None or point3 is None:
        return None

    vector1 = (
        point1["x"] - point2["x"],
        point1["y"] - point2["y"],
        point1["z"] - point2["z"]
    )

    vector2 = (
        point3["x"] - point2["x"],
        point3["y"] - point2["y"],
        point3["z"] - point2["z"]
    )

    dot_product = (
        vector1[0] * vector2[0] +
        vector1[1] * vector2[1] +
        vector1[2] * vector2[2]
    )

    magnitude1 = math.sqrt(
        vector1[0] ** 2 +
        vector1[1] ** 2 +
        vector1[2] ** 2
    )

    magnitude2 = math.sqrt(
        vector2[0] ** 2 +
        vector2[1] ** 2 +
        vector2[2] ** 2
    )

    if magnitude1 == 0 or magnitude2 == 0:
        return None

    cosine_angle = dot_product / (magnitude1 * magnitude2)

    # Avoid floating-point errors
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    angle = math.degrees(
        math.acos(cosine_angle)
    )

    return angle


ANGLE_TRIPLETS = {

    "index_pip_angle": (
        "Index_MCP",
        "Index_PIP",
        "Index_DIP"
    ),

    "index_dip_angle": (
        "Index_PIP",
        "Index_DIP",
        "Index_Tip"
    ),

    "middle_pip_angle": (
        "Middle_MCP",
        "Middle_PIP",
        "Middle_DIP"
    ),

    "middle_dip_angle": (
        "Middle_PIP",
        "Middle_DIP",
        "Middle_Tip"
    ),

    "ring_pip_angle": (
        "Ring_MCP",
        "Ring_PIP",
        "Ring_DIP"
    ),

    "ring_dip_angle": (
        "Ring_PIP",
        "Ring_DIP",
        "Ring_Tip"
    ),

    "pinky_pip_angle": (
        "Pinky_MCP",
        "Pinky_PIP",
        "Pinky_DIP"
    ),

    "pinky_dip_angle": (
        "Pinky_PIP",
        "Pinky_DIP",
        "Pinky_Tip"
    )
}


def extract_angle_features(landmarks):

    features = {}

    for feature_name, (
        point1_name,
        point2_name,
        point3_name
    ) in ANGLE_TRIPLETS.items():

        point1 = landmarks.get(point1_name)
        point2 = landmarks.get(point2_name)
        point3 = landmarks.get(point3_name)

        features[feature_name] = calculate_angle(
            point1,
            point2,
            point3
        )

    return features


# ---------------------------------------------------------
# 4. FINGER STATE
# ---------------------------------------------------------

FINGER_ANGLE_PAIRS = {

    "Index": (
        "index_pip_angle",
        "index_dip_angle"
    ),

    "Middle": (
        "middle_pip_angle",
        "middle_dip_angle"
    ),

    "Ring": (
        "ring_pip_angle",
        "ring_dip_angle"
    ),

    "Pinky": (
        "pinky_pip_angle",
        "pinky_dip_angle"
    )
}


# These are initial geometric thresholds.
# They can later be tuned using real FlowSync data.

STRAIGHT_ANGLE_THRESHOLD = 160.0
BENT_ANGLE_THRESHOLD = 120.0


def determine_finger_state(
    finger,
    finger_status,
    angle_features,
    landmarks=None
):

    status = finger_status.get(finger)

    if status is None:
        return "unknown"

    # Missing or incomplete finger
    if not status["complete"]:
        return "unknown"

    if finger == "Thumb":
        if landmarks is None:
            return "unknown"

        thumb_angle = calculate_angle(
            landmarks.get("Thumb_CMC"),
            landmarks.get("Thumb_MCP"),
            landmarks.get("Thumb_Tip")
        )

        if thumb_angle is None:
            return "unknown"
        if thumb_angle >= 145.0:
            return "straight"
        if thumb_angle <= 115.0:
            return "bent"
        return "unknown"

    angle_names = FINGER_ANGLE_PAIRS.get(finger)

    if angle_names is None:
        return "unknown"

    pip_angle = angle_features.get(angle_names[0])
    dip_angle = angle_features.get(angle_names[1])

    if pip_angle is None or dip_angle is None:
        return "unknown"

    if (
        pip_angle >= STRAIGHT_ANGLE_THRESHOLD
        and
        dip_angle >= STRAIGHT_ANGLE_THRESHOLD
    ):
        return "straight"

    if (
        pip_angle <= BENT_ANGLE_THRESHOLD
        or
        dip_angle <= BENT_ANGLE_THRESHOLD
    ):
        return "bent"

    return "unknown"


def extract_finger_states(finger_status, angle_features, landmarks):

    finger_states = {}

    for finger in [
        "Thumb",
        "Index",
        "Middle",
        "Ring",
        "Pinky"
    ]:

        finger_states[finger] = determine_finger_state(
            finger,
            finger_status,
            angle_features,
            landmarks
        )

    return finger_states


# ---------------------------------------------------------
# 5. COMPLETE FEATURE EXTRACTION
# ---------------------------------------------------------

def extract_features(landmarks, finger_status):

    normalized = normalize_landmarks(landmarks)

    if normalized is None:
        return None

    distance_features = extract_distance_features(
        normalized
    )

    angle_features = extract_angle_features(
        normalized
    )

    finger_states = extract_finger_states(
        finger_status,
        angle_features,
        normalized
    )

    return {
        "landmarks": normalized,
        "distances": distance_features,
        "angles": angle_features,
        "finger_states": finger_states
    }


# ---------------------------------------------------------
# 6. FEATURE VECTOR
# ---------------------------------------------------------

def create_feature_vector(features):

    if features is None:
        return None

    feature_vector = []

    # -------------------------------------------------
    # 1. Normalized Landmark Coordinates
    # -------------------------------------------------

    for landmark_name in LANDMARK_FEATURES:

        landmark = features["landmarks"].get(landmark_name)

        if landmark is None:

            feature_vector.extend([
                0.0,
                0.0,
                0.0
            ])

        else:

            feature_vector.extend([
                landmark["x"],
                landmark["y"],
                landmark["z"]
            ])

    # -------------------------------------------------
    # 2. Distance Features
    # -------------------------------------------------

    for feature_name in DISTANCE_FEATURES:

        distance = features["distances"].get(feature_name)

        if distance is None:
            feature_vector.append(0.0)
        else:
            feature_vector.append(distance)

    # -------------------------------------------------
    # 3. Angle Features
    # -------------------------------------------------

    for feature_name in ANGLE_FEATURES:

        angle = features["angles"].get(feature_name)

        if angle is None:
            feature_vector.append(0.0)
        else:
            feature_vector.append(angle)

    return feature_vector