# envisionhgdetector/envisionhgdetector/preprocessing.py
"""
Preprocessing module for gesture detection.
Supports three feature sets:
- Basic:    41 features (29 body + 6 visibility + 6 move-distinguishing)
- Extended: 61 features (29 body + 20 hand + 6 visibility + 6 move-distinguishing)  
- World:    92 features (23 upper body landmarks × 4: x, y, z, visibility)

ALIGNED WITH TRAINING DATA EXTRACTION v3.2!
"""

from enum import auto, Enum
from typing import List, Optional, Union, Tuple
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature set options: "basic", "extended", "world"
FEATURE_SET = "world"  # Best performing model uses world landmarks

# Feature counts
NUM_BASIC_FEATURES = 41     # 29 body + 6 visibility + 6 move-distinguishing
NUM_EXTENDED_FEATURES = 61  # 29 body + 20 hand + 6 visibility + 6 move-distinguishing
NUM_WORLD_FEATURES = 92     # 23 upper body landmarks × 4 (x, y, z, visibility)

# ============================================================================
# MEDIAPIPE SETUP
# ============================================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Body landmark names (33 landmarks from BlazePose)
BODY_LANDMARKS = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 
    'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 
    'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
    'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL',
    'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

# Upper body landmarks used for world features (23 landmarks, indices 0-22)
UPPER_BODY_LANDMARKS = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 
    'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 
    'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
    'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB'
]
UPPER_BODY_INDICES = list(range(23))

# Hand landmark names (21 landmarks per hand)
HAND_LANDMARKS = [
    'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 
    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 
    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 
    'PINKY_FINGER_MCP', 'PINKY_FINGER_PIP', 'PINKY_FINGER_DIP', 'PINKY_FINGER_TIP'
]

# Generate world feature names
WORLD_FEATURE_NAMES = []
for lm_name in UPPER_BODY_LANDMARKS:
    for dim in ['X', 'Y', 'Z', 'visibility']:
        WORLD_FEATURE_NAMES.append(f"{lm_name}_{dim}")


# ============================================================================
# FEATURE ENUMS
# ============================================================================

class FeatureWorld(Enum):
    """World landmarks: 23 upper body × 4 (x, y, z, visibility) = 92 features"""
    pass  # Features are dynamic based on WORLD_FEATURE_NAMES


class FeatureBasic(Enum):
    """Basic 41 features: 29 body + 6 visibility + 6 move-distinguishing"""
    rot_x = auto()
    rot_y = auto()
    rot_z = auto()
    nose_x = auto()
    nose_y = auto()
    nose_z = auto()
    norm_dist = auto()
    left_brow_left_eye_norm_dist = auto()
    right_brow_right_eye_norm_dist = auto()
    mouth_corners_norm_dist = auto()
    mouth_apperture_norm_dist = auto()
    left_right_wrist_norm_dist = auto()
    left_right_elbow_norm_dist = auto()
    left_elbow_midpoint_shoulder_norm_dist = auto()
    right_elbow_midpoint_shoulder_norm_dist = auto()
    left_wrist_midpoint_shoulder_norm_dist = auto()
    right_wrist_midpoint_shoulder_norm_dist = auto()
    left_shoulder_left_ear_norm_dist = auto()
    right_shoulder_right_ear_norm_dist = auto()
    left_thumb_left_index_norm_dist = auto()
    right_thumb_right_index_norm_dist = auto()
    left_thumb_left_pinky_norm_dist = auto()
    right_thumb_right_pinky_norm_dist = auto()
    x_left_wrist_x_left_elbow_norm_dist = auto()
    x_right_wrist_x_right_elbow_norm_dist = auto()
    y_left_wrist_y_left_elbow_norm_dist = auto()
    y_right_wrist_y_right_elbow_norm_dist = auto()
    left_index_finger_nose_norm_dist = auto()
    right_index_finger_nose_norm_dist = auto()
    # Visibility features (6)
    left_wrist_visibility = auto()
    right_wrist_visibility = auto()
    left_elbow_visibility = auto()
    right_elbow_visibility = auto()
    left_hand_detected = auto()
    right_hand_detected = auto()
    # Move-distinguishing features (6)
    min_hand_face_dist = auto()
    hand_face_contact = auto()
    min_hand_shoulder_y_diff = auto()
    left_wrist_above_shoulder = auto()
    right_wrist_above_shoulder = auto()
    hand_position_symmetry = auto()


class FeatureExtended(Enum):
    """Extended 61 features: 29 body + 20 hand + 6 visibility + 6 move-distinguishing"""
    rot_x = auto()
    rot_y = auto()
    rot_z = auto()
    nose_x = auto()
    nose_y = auto()
    nose_z = auto()
    norm_dist = auto()
    left_brow_left_eye_norm_dist = auto()
    right_brow_right_eye_norm_dist = auto()
    mouth_corners_norm_dist = auto()
    mouth_apperture_norm_dist = auto()
    left_right_wrist_norm_dist = auto()
    left_right_elbow_norm_dist = auto()
    left_elbow_midpoint_shoulder_norm_dist = auto()
    right_elbow_midpoint_shoulder_norm_dist = auto()
    left_wrist_midpoint_shoulder_norm_dist = auto()
    right_wrist_midpoint_shoulder_norm_dist = auto()
    left_shoulder_left_ear_norm_dist = auto()
    right_shoulder_right_ear_norm_dist = auto()
    left_thumb_left_index_norm_dist = auto()
    right_thumb_right_index_norm_dist = auto()
    left_thumb_left_pinky_norm_dist = auto()
    right_thumb_right_pinky_norm_dist = auto()
    x_left_wrist_x_left_elbow_norm_dist = auto()
    x_right_wrist_x_right_elbow_norm_dist = auto()
    y_left_wrist_y_left_elbow_norm_dist = auto()
    y_right_wrist_y_right_elbow_norm_dist = auto()
    left_index_finger_nose_norm_dist = auto()
    right_index_finger_nose_norm_dist = auto()
    # Hand features (20)
    left_wrist_thumb_tip_norm_dist = auto()
    left_wrist_index_tip_norm_dist = auto()
    left_wrist_middle_tip_norm_dist = auto()
    left_wrist_ring_tip_norm_dist = auto()
    left_wrist_pinky_tip_norm_dist = auto()
    left_thumb_index_angle = auto()
    left_index_middle_angle = auto()
    left_middle_ring_angle = auto()
    left_ring_pinky_angle = auto()
    left_hand_openness = auto()
    right_wrist_thumb_tip_norm_dist = auto()
    right_wrist_index_tip_norm_dist = auto()
    right_wrist_middle_tip_norm_dist = auto()
    right_wrist_ring_tip_norm_dist = auto()
    right_wrist_pinky_tip_norm_dist = auto()
    right_thumb_index_angle = auto()
    right_index_middle_angle = auto()
    right_middle_ring_angle = auto()
    right_ring_pinky_angle = auto()
    right_hand_openness = auto()
    # Visibility features (6)
    left_wrist_visibility = auto()
    right_wrist_visibility = auto()
    left_elbow_visibility = auto()
    right_elbow_visibility = auto()
    left_hand_detected = auto()
    right_hand_detected = auto()
    # Move-distinguishing features (6)
    min_hand_face_dist = auto()
    hand_face_contact = auto()
    min_hand_shoulder_y_diff = auto()
    left_wrist_above_shoulder = auto()
    right_wrist_above_shoulder = auto()
    hand_position_symmetry = auto()


class VideoSegment(Enum):
    """Video segment selection for processing."""
    BEGINNING = "beginning"
    MIDDLE = "middle"
    END = "end"
    LAST = "last"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_num_features(feature_set: Optional[str] = None) -> int:
    """Get the number of features based on feature set."""
    if feature_set is None:
        feature_set = FEATURE_SET
    
    if feature_set == "world":
        return NUM_WORLD_FEATURES
    elif feature_set == "extended":
        return NUM_EXTENDED_FEATURES
    return NUM_BASIC_FEATURES


def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """Calculate angle between three points (p1-p2-p3) in degrees."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


# ============================================================================
# WORLD LANDMARK EXTRACTION (92 features) - BEST PERFORMING
# ============================================================================

def extract_world_landmarks(results) -> Optional[List[float]]:
    """
    Extract upper body world landmarks with visibility.
    
    World landmarks are in metric scale (meters) relative to hip center.
    This is the BEST PERFORMING feature set!
    
    Returns:
        List of 92 features (23 landmarks × 4: x, y, z, visibility)
        or None if pose not detected
    """
    if not results.pose_world_landmarks:
        return None
    
    features = []
    for idx in UPPER_BODY_INDICES:
        if idx < len(results.pose_world_landmarks.landmark):
            lm = results.pose_world_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    
    return features


# ============================================================================
# HAND FEATURE EXTRACTION (for extended features)
# ============================================================================

def extract_hand_features(hand_landmarks, wrist_pos: Tuple[float, float], 
                         norm_dist: float, w: int, h: int) -> List[float]:
    """
    Extract 10 compact hand features from MediaPipe hand landmarks.
    """
    if hand_landmarks is None or norm_dist == 0:
        return [0.0] * 10
    
    lm = hand_landmarks.landmark
    
    wrist = (lm[0].x * w, lm[0].y * h)
    thumb_tip = (lm[4].x * w, lm[4].y * h)
    index_tip = (lm[8].x * w, lm[8].y * h)
    middle_tip = (lm[12].x * w, lm[12].y * h)
    ring_tip = (lm[16].x * w, lm[16].y * h)
    pinky_tip = (lm[20].x * w, lm[20].y * h)
    
    index_mcp = (lm[5].x * w, lm[5].y * h)
    middle_mcp = (lm[9].x * w, lm[9].y * h)
    ring_mcp = (lm[13].x * w, lm[13].y * h)
    pinky_mcp = (lm[17].x * w, lm[17].y * h)
    
    dists = [
        np.linalg.norm(np.array(wrist) - np.array(tip)) / norm_dist
        for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    ]
    
    angles = [
        calculate_angle(thumb_tip, wrist, index_tip) / 180.0,
        calculate_angle(index_tip, index_mcp, middle_tip) / 180.0,
        calculate_angle(middle_tip, middle_mcp, ring_tip) / 180.0,
        calculate_angle(ring_tip, ring_mcp, pinky_tip) / 180.0,
    ]
    
    extensions = []
    for tip, mcp in [(index_tip, index_mcp), (middle_tip, middle_mcp), 
                     (ring_tip, ring_mcp), (pinky_tip, pinky_mcp)]:
        tip_dist = np.linalg.norm(np.array(wrist) - np.array(tip))
        mcp_dist = np.linalg.norm(np.array(wrist) - np.array(mcp))
        extensions.append(tip_dist / (mcp_dist + 1e-6))
    openness = np.mean(extensions)
    
    return dists + angles + [openness]


# ============================================================================
# VISIBILITY AND MOVE-DISTINGUISHING FEATURES
# ============================================================================

def extract_visibility_features(results) -> List[float]:
    """Extract 6 visibility/confidence features."""
    visibility = [0.0] * 6
    
    if results.pose_landmarks:
        pose_lm = results.pose_landmarks.landmark
        visibility[0] = pose_lm[15].visibility if len(pose_lm) > 15 else 0.0
        visibility[1] = pose_lm[16].visibility if len(pose_lm) > 16 else 0.0
        visibility[2] = pose_lm[13].visibility if len(pose_lm) > 13 else 0.0
        visibility[3] = pose_lm[14].visibility if len(pose_lm) > 14 else 0.0
    
    visibility[4] = 1.0 if results.left_hand_landmarks else 0.0
    visibility[5] = 1.0 if results.right_hand_landmarks else 0.0
    
    return visibility


def extract_move_distinguishing_features(body: dict, nose_2d: Tuple[float, float], 
                                         norm_dist: float) -> List[float]:
    """Extract 6 move-distinguishing features."""
    features = [0.0] * 6
    
    lw = body.get('LEFT_WRIST')
    rw = body.get('RIGHT_WRIST')
    ls = body.get('LEFT_SHOULDER')
    rs = body.get('RIGHT_SHOULDER')
    li = body.get('LEFT_INDEX')
    ri = body.get('RIGHT_INDEX')

    l_dist = np.linalg.norm(np.array(li) - np.array(nose_2d)) / norm_dist if li else 99
    r_dist = np.linalg.norm(np.array(ri) - np.array(nose_2d)) / norm_dist if ri else 99
    min_dist = min(l_dist, r_dist)
    features[0] = min_dist if min_dist < 99 else 0.0
    features[1] = 1.0 if min_dist < 0.35 else 0.0 

    if lw and ls and rw and rs:
        features[2] = min((lw[1]-ls[1])/norm_dist, (rw[1]-rs[1])/norm_dist)

    features[3] = 1.0 if (lw and ls and lw[1] < ls[1]) else 0.0
    features[4] = 1.0 if (rw and rs and rw[1] < rs[1]) else 0.0

    if lw and rw and ls and rs:
        center_x = (ls[0] + rs[0]) / 2
        symm = abs(abs(lw[0]-center_x) - abs(rw[0]-center_x)) / norm_dist
        features[5] = max(0, 1.0 - symm)
    
    return features


# ============================================================================
# MAIN FEATURE EXTRACTION
# ============================================================================

def video_to_landmarks(
    video_path: Optional[Union[int, str]],
    max_num_frames: Optional[int] = None,
    video_segment: VideoSegment = VideoSegment.BEGINNING,
    end_padding: bool = True,
    drop_consecutive_duplicates: bool = False,
    feature_set: Optional[str] = None
) -> Tuple[List[List[float]], List[float]]:
    """
    Extract landmarks from video frames at 25 fps.
    
    Args:
        video_path: Path to video file or camera index (0 for webcam)
        max_num_frames: Maximum frames to process
        video_segment: Which part of video to process
        end_padding: Pad output to max_num_frames if needed
        drop_consecutive_duplicates: Skip duplicate frames
        feature_set: "basic" (41), "extended" (61), or "world" (92)

    Returns:
        Tuple of (features_list, timestamps)
    """
    if feature_set is None:
        feature_set = FEATURE_SET
    
    assert video_segment in VideoSegment
    video_path = video_path if video_path else 0

    valid_frame_count = 0
    prev_features: List[float] = []
    landmarks: List[List[float]] = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(fps / 25)) if fps > 0 else 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing frames")
    frame_timestamps = []
    frame_number = 0
    processed_frame_count = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5, 
        model_complexity=1
    ) as holistic:
        while cap.isOpened():
            ret, bgr_frame = cap.read()
            if not ret:
                if video_path == 0:
                    continue
                break

            if frame_number % frame_interval == 0:
                processed_frame_count += 1

            if frame_number % frame_interval != 0:
                pbar.update(1)
                frame_number += 1
                continue

            if max_num_frames and video_segment == VideoSegment.BEGINNING and valid_frame_count >= max_num_frames:
                break

            frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)

            h, w, _ = frame.shape

            # ================================================================
            # WORLD LANDMARKS (92 features) - BEST PERFORMING
            # ================================================================
            if feature_set == "world":
                if results.pose_world_landmarks:
                    frame_timestamps.append(processed_frame_count / 25.0)
                    
                    features = extract_world_landmarks(results)
                    
                    if features is not None:
                        if drop_consecutive_duplicates and prev_features and np.array_equal(
                                np.round(features, decimals=2),
                                np.round(prev_features, decimals=2)):
                            pbar.update(1)
                            frame_number += 1
                            continue

                        landmarks.append(features)
                        prev_features = features
                        valid_frame_count += 1
                
                pbar.update(1)
                frame_number += 1
                continue

            # ================================================================
            # BASIC/EXTENDED LANDMARKS (41/61 features)
            # ================================================================
            if results.face_landmarks and results.pose_landmarks:
                frame_timestamps.append(processed_frame_count / 25.0)
                
                # Head rotation calculation
                face_2d, face_3d = [], []
                nose_2d, nose_3d = None, None
                
                for idx, lm in enumerate(results.face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        if idx == 1:
                            nose_2d = (lm.x * w, lm.y * h)
                            nose_3d = (lm.x * w, lm.y * h, lm.z * 3000)
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * w
                cam_matrix = np.array([
                    [focal_length, 0, h / 2], 
                    [0, focal_length, w / 2], 
                    [0, 0, 1]
                ])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                xrot = angles[0] * w
                yrot = angles[1] * h
                zrot = angles[2] * 3000

                nose_x, nose_y, nose_z = nose_3d

                # Face landmarks
                chin, top_head = None, None
                left_inner_eye, left_brow = None, None
                right_inner_eye, right_brow = None, None
                left_mouth, right_mouth = None, None
                upper_lip, lower_lip = None, None

                for idx, lm in enumerate(results.face_landmarks.landmark):
                    if idx == 152: chin = (lm.x * w, lm.y * h)
                    elif idx == 10: top_head = (lm.x * w, lm.y * h)
                    elif idx == 133: left_inner_eye = (lm.x * w, lm.y * h)
                    elif idx == 443: left_brow = (lm.x * w, lm.y * h)
                    elif idx == 223: right_brow = (lm.x * w, lm.y * h)
                    elif idx == 362: right_inner_eye = (lm.x * w, lm.y * h)
                    elif idx == 87: left_mouth = (lm.x * w, lm.y * h)
                    elif idx == 308: right_mouth = (lm.x * w, lm.y * h)
                    elif idx == 13: upper_lip = (lm.x * w, lm.y * h)
                    elif idx == 14: lower_lip = (lm.x * w, lm.y * h)

                norm_dist = np.linalg.norm(np.array(chin) - np.array(top_head)) if chin and top_head else 1.0

                left_brow_left_eye_norm_dist = np.linalg.norm(np.array(left_inner_eye) - np.array(left_brow)) / norm_dist if left_inner_eye and left_brow else 0.0
                right_brow_right_eye_norm_dist = np.linalg.norm(np.array(right_inner_eye) - np.array(right_brow)) / norm_dist if right_inner_eye and right_brow else 0.0
                mouth_corners_norm_dist = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth)) / norm_dist if left_mouth and right_mouth else 0.0
                mouth_apperture_norm_dist = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip)) / norm_dist if upper_lip and lower_lip else 0.0

                # Body landmarks
                body = {}
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    if idx < len(BODY_LANDMARKS):
                        name = BODY_LANDMARKS[idx]
                        if name in ['LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
                                   'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_WRIST', 'RIGHT_WRIST',
                                   'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                                   'LEFT_EAR', 'RIGHT_EAR']:
                            body[name] = (lm.x * w, lm.y * h)

                lw = body.get('LEFT_WRIST')
                rw = body.get('RIGHT_WRIST')
                le = body.get('LEFT_ELBOW')
                re = body.get('RIGHT_ELBOW')
                ls = body.get('LEFT_SHOULDER')
                rs = body.get('RIGHT_SHOULDER')
                lear = body.get('LEFT_EAR')
                rear = body.get('RIGHT_EAR')
                lt = body.get('LEFT_THUMB')
                li = body.get('LEFT_INDEX')
                lp = body.get('LEFT_PINKY')
                rt = body.get('RIGHT_THUMB')
                ri = body.get('RIGHT_INDEX')
                rp = body.get('RIGHT_PINKY')

                left_right_wrist_norm_dist = np.linalg.norm(np.array(lw) - np.array(rw)) / norm_dist if lw and rw else 0.0
                left_right_elbow_norm_dist = np.linalg.norm(np.array(le) - np.array(re)) / norm_dist if le and re else 0.0
                left_elbow_midpoint_shoulder_norm_dist = np.linalg.norm(np.array(le) - np.array(ls)) / norm_dist if le and ls else 0.0
                right_elbow_midpoint_shoulder_norm_dist = np.linalg.norm(np.array(re) - np.array(rs)) / norm_dist if re and rs else 0.0
                left_wrist_midpoint_shoulder_norm_dist = np.linalg.norm(np.array(lw) - np.array(ls)) / norm_dist if lw and ls else 0.0
                right_wrist_midpoint_shoulder_norm_dist = np.linalg.norm(np.array(rw) - np.array(rs)) / norm_dist if rw and rs else 0.0
                left_shoulder_left_ear_norm_dist = np.linalg.norm(np.array(ls) - np.array(lear)) / norm_dist if ls and lear else 0.0
                right_shoulder_right_ear_norm_dist = np.linalg.norm(np.array(rs) - np.array(rear)) / norm_dist if rs and rear else 0.0
                left_thumb_left_index_norm_dist = np.linalg.norm(np.array(lt) - np.array(li)) / norm_dist if lt and li else 0.0
                right_thumb_right_index_norm_dist = np.linalg.norm(np.array(rt) - np.array(ri)) / norm_dist if rt and ri else 0.0
                left_thumb_left_pinky_norm_dist = np.linalg.norm(np.array(lt) - np.array(lp)) / norm_dist if lt and lp else 0.0
                right_thumb_right_pinky_norm_dist = np.linalg.norm(np.array(rt) - np.array(rp)) / norm_dist if rt and rp else 0.0
                x_left_wrist_x_left_elbow_norm_dist = (lw[0] - le[0]) / norm_dist if lw and le else 0.0
                x_right_wrist_x_right_elbow_norm_dist = (rw[0] - re[0]) / norm_dist if rw and re else 0.0
                y_left_wrist_y_left_elbow_norm_dist = (lw[1] - le[1]) / norm_dist if lw and le else 0.0
                y_right_wrist_y_right_elbow_norm_dist = (rw[1] - re[1]) / norm_dist if rw and re else 0.0
                left_index_finger_nose_norm_dist = np.linalg.norm(np.array(li) - np.array(nose_2d)) / norm_dist if li and nose_2d else 0.0
                right_index_finger_nose_norm_dist = np.linalg.norm(np.array(ri) - np.array(nose_2d)) / norm_dist if ri and nose_2d else 0.0

                # 29 basic body features
                features = [
                    xrot, yrot, zrot,
                    nose_x, nose_y, nose_z,
                    norm_dist,
                    left_brow_left_eye_norm_dist,
                    right_brow_right_eye_norm_dist,
                    mouth_corners_norm_dist,
                    mouth_apperture_norm_dist,
                    left_right_wrist_norm_dist,
                    left_right_elbow_norm_dist,
                    left_elbow_midpoint_shoulder_norm_dist,
                    right_elbow_midpoint_shoulder_norm_dist,
                    left_wrist_midpoint_shoulder_norm_dist,
                    right_wrist_midpoint_shoulder_norm_dist,
                    left_shoulder_left_ear_norm_dist,
                    right_shoulder_right_ear_norm_dist,
                    left_thumb_left_index_norm_dist,
                    right_thumb_right_index_norm_dist,
                    left_thumb_left_pinky_norm_dist,
                    right_thumb_right_pinky_norm_dist,
                    x_left_wrist_x_left_elbow_norm_dist,
                    x_right_wrist_x_right_elbow_norm_dist,
                    y_left_wrist_y_left_elbow_norm_dist,
                    y_right_wrist_y_right_elbow_norm_dist,
                    left_index_finger_nose_norm_dist,
                    right_index_finger_nose_norm_dist
                ]

                # Add hand features for extended mode
                if feature_set == "extended":
                    left_hand_features = extract_hand_features(
                        results.left_hand_landmarks, lw, norm_dist, w, h
                    )
                    right_hand_features = extract_hand_features(
                        results.right_hand_landmarks, rw, norm_dist, w, h
                    )
                    features.extend(left_hand_features)
                    features.extend(right_hand_features)

                # Add visibility features (6)
                visibility = extract_visibility_features(results)
                features.extend(visibility)

                # Add move-distinguishing features (6)
                move_features = extract_move_distinguishing_features(body, nose_2d, norm_dist)
                features.extend(move_features)

                # Handle duplicate dropping
                if drop_consecutive_duplicates and prev_features and np.array_equal(
                        np.round(features, decimals=2),
                        np.round(prev_features, decimals=2)):
                    pbar.update(1)
                    frame_number += 1
                    continue

                landmarks.append(features)
                prev_features = features
                valid_frame_count += 1
                
            pbar.update(1)
            frame_number += 1
                
        pbar.close()
        cap.release()

        if not landmarks:
            return [], []

        if max_num_frames and video_segment == VideoSegment.LAST:
            landmarks = landmarks[-max_num_frames:]
            frame_timestamps = frame_timestamps[-max_num_frames:]

        if max_num_frames and end_padding and len(landmarks) < max_num_frames:
            last = landmarks[-1]
            landmarks = landmarks + [last] * (max_num_frames - len(landmarks))
            if frame_timestamps:
                last_time = frame_timestamps[-1]
                time_step = 1.0 / 25.0
                for i in range(1, max_num_frames - len(frame_timestamps) + 1):
                    frame_timestamps.append(last_time + i * time_step)

        return landmarks, frame_timestamps


# ============================================================================
# VIDEO PROCESSOR CLASS
# ============================================================================

class VideoProcessor:
    """Handles video processing and feature extraction."""
    
    def __init__(self, seq_length: int = 25, feature_set: Optional[str] = None):
        """
        Initialize processor.
        
        Args:
            seq_length: Window size for sequences
            feature_set: "basic", "extended", or "world"
        """
        self.seq_length = seq_length
        self.feature_set = feature_set or FEATURE_SET
        self.mp_holistic = mp.solutions.holistic
        
    def process_video(self, video_path: str) -> Tuple[List[List[float]], List[float]]:
        """
        Process video and extract landmarks features.
        """
        features_list, timestamps = video_to_landmarks(
            video_path=video_path, 
            max_num_frames=None,
            video_segment=VideoSegment.BEGINNING,
            feature_set=self.feature_set
        )
        
        return features_list, timestamps


def create_sliding_windows(
    features: List[List[float]],
    seq_length: int,
    stride: int = 1
) -> np.ndarray:
    """Create sliding windows from feature sequence."""
    if len(features) < seq_length:
        return np.array([])
        
    windows = []
    for i in range(0, len(features) - seq_length + 1, stride):
        window = features[i:i + seq_length]
        windows.append(window)
    
    return np.array(windows)


# ============================================================================
# LEGACY SUPPORT
# ============================================================================

Feature = FeatureBasic if FEATURE_SET == "basic" else (FeatureExtended if FEATURE_SET == "extended" else FeatureWorld)