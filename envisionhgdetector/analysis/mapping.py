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

# Define mapping from joint names to MediaPipe indices
joint_map = {
    'L_Wrist': 15,      # Left wrist
    'R_Wrist': 16,      # Right wrist
    'L_Elbow': 13,        # Left elbow
    'R_Elbow': 14,        # Right elbow
    'L_Shoulder': 11,   # Left shoulder
    'R_Shoulder': 12,   # Right shoulder
    'Neck': 23,        # Neck (approximated as top of spine)
    'MidHip': 24,      # Mid hip
    'L_Eye': 2,         # Left eye
    'R_Eye': 5,         # Right eye
    'Nose': 0,         # Nose
    'L_Hip': 23,        # Left hip
    'R_Hip': 24         # Right hip
}