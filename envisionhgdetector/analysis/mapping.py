# MediaPipe pose indices; group order preserves the existing feature layout.
UPPER_LIMB_LANDMARKS = {
    "left_shoulder": 11,
    "left_elbow": 13,
    "left_wrist": 15,
    "right_shoulder": 12,
    "right_elbow": 14,
    "right_wrist": 16,
    "left_pinky": 17,
    "left_index": 19,
    "left_thumb": 21,
    "right_pinky": 18,
    "right_index": 20,
    "right_thumb": 22,
}
ARM_JOINT_INDICES = tuple(
    index for name, index in UPPER_LIMB_LANDMARKS.items()
    if name.endswith(("shoulder", "elbow", "wrist"))
)
LEFT_FINGER_INDICES = tuple(
    index for name, index in UPPER_LIMB_LANDMARKS.items()
    if name.startswith("left_") and name.endswith(("pinky", "index", "thumb"))
)
RIGHT_FINGER_INDICES = tuple(
    index for name, index in UPPER_LIMB_LANDMARKS.items()
    if name.startswith("right_") and name.endswith(("pinky", "index", "thumb"))
)
UPPER_LIMB_INDICES = tuple(UPPER_LIMB_LANDMARKS.values())

class JointNames:
    """Canonical DataFrame column names shared by gesture analysis helpers."""

    LEFT_WRIST = "L_Wrist"
    RIGHT_WRIST = "R_Wrist"
    LEFT_ELBOW = "L_Elbow"
    RIGHT_ELBOW = "R_Elbow"
    LEFT_SHOULDER = "L_Shoulder"
    RIGHT_SHOULDER = "R_Shoulder"
    NECK = "Neck"
    MID_HIP = "MidHip"
    LEFT_EYE = "L_Eye"
    RIGHT_EYE = "R_Eye"
    NOSE = "Nose"
    LEFT_HIP = "L_Hip"
    RIGHT_HIP = "R_Hip"


# Existing index assignments are preserved here. Neck and MidHip are legacy
# references to individual hips, not derived shoulder/hip midpoints.
joint_map = {
    JointNames.LEFT_WRIST: UPPER_LIMB_LANDMARKS["left_wrist"],
    JointNames.RIGHT_WRIST: UPPER_LIMB_LANDMARKS["right_wrist"],
    JointNames.LEFT_ELBOW: UPPER_LIMB_LANDMARKS["left_elbow"],
    JointNames.RIGHT_ELBOW: UPPER_LIMB_LANDMARKS["right_elbow"],
    JointNames.LEFT_SHOULDER: UPPER_LIMB_LANDMARKS["left_shoulder"],
    JointNames.RIGHT_SHOULDER: UPPER_LIMB_LANDMARKS["right_shoulder"],
    JointNames.NECK: 23,
    JointNames.MID_HIP: 24,
    JointNames.LEFT_EYE: 2,
    JointNames.RIGHT_EYE: 5,
    JointNames.NOSE: 0,
    JointNames.LEFT_HIP: 23,
    JointNames.RIGHT_HIP: 24,
}