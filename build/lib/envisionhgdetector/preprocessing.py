# envisionhgdetector/envisionhgdetector/preprocessing.py

from enum import auto, Enum
from typing import List, Optional, Union, Dict
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple
from .config import Config
from tqdm import tqdm

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# envisionhgdetector/envisionhgdetector/preprocessing.py

################## NAMES FOR MEDIAPIPE

markerxyzbody = [] #body landmarks
markerxyzhands = [] #hand landmarks
markerxyzface = [] #face landmarks

#landmarks 33x that are used by Mediapipe (Blazepose)
markersbody = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
          'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
          'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
          'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
          'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

markershands = ['LEFT_WRIST', 'LEFT_THUMB_CMC', 'LEFT_THUMB_MCP', 'LEFT_THUMB_IP', 'LEFT_THUMB_TIP', 'LEFT_INDEX_FINGER_MCP',
              'LEFT_INDEX_FINGER_PIP', 'LEFT_INDEX_FINGER_DIP', 'LEFT_INDEX_FINGER_TIP', 'LEFT_MIDDLE_FINGER_MCP', 
               'LEFT_MIDDLE_FINGER_PIP', 'LEFT_MIDDLE_FINGER_DIP', 'LEFT_MIDDLE_FINGER_TIP', 'LEFT_RING_FINGER_MCP', 
               'LEFT_RING_FINGER_PIP', 'LEFT_RING_FINGER_DIP', 'LEFT_RING_FINGER_TIP', 'LEFT_PINKY_FINGER_MCP', 
               'LEFT_PINKY_FINGER_PIP', 'LEFT_PINKY_FINGER_DIP', 'LEFT_PINKY_FINGER_TIP',
              'RIGHT_WRIST', 'RIGHT_THUMB_CMC', 'RIGHT_THUMB_MCP', 'RIGHT_THUMB_IP', 'RIGHT_THUMB_TIP', 'RIGHT_INDEX_FINGER_MCP',
              'RIGHT_INDEX_FINGER_PIP', 'RIGHT_INDEX_FINGER_DIP', 'RIGHT_INDEX_FINGER_TIP', 'RIGHT_MIDDLE_FINGER_MCP', 
               'RIGHT_MIDDLE_FINGER_PIP', 'RIGHT_MIDDLE_FINGER_DIP', 'RIGHT_MIDDLE_FINGER_TIP', 'RIGHT_RING_FINGER_MCP', 
               'RIGHT_RING_FINGER_PIP', 'RIGHT_RING_FINGER_DIP', 'RIGHT_RING_FINGER_TIP', 'RIGHT_PINKY_FINGER_MCP', 
               'RIGHT_PINKY_FINGER_PIP', 'RIGHT_PINKY_FINGER_DIP', 'RIGHT_PINKY_FINGER_TIP']

# Initialize body markers
for mark in markersbody:
    for pos in ['X', 'Y', 'Z', 'visibility']:  # body markers include visibility score
        nm = pos + "_" + mark
        markerxyzbody.append(nm)

def num_there(s: str) -> bool:
    """
    Check if string contains any digits.
    
    Args:
        s: Input string
        
    Returns:
        True if string contains digits, False otherwise
    """
    return any(i.isdigit() for i in s)

def makegoginto_str(gogobj) -> List[str]:
    """
    Convert MediaPipe landmark object to string list.
    
    Args:
        gogobj: MediaPipe landmark object
        
    Returns:
        List of strings representing landmarks
    """
    gogobj = str(gogobj).strip("[]")
    gogobj = gogobj.split("\n")
    return gogobj[:-1]  # ignore last empty element

def listpositions(newsamplemarks: List[str]) -> List[str]:
    """
    Convert stringified position traces into clean numerical values.
    
    Args:
        newsamplemarks: List of landmark strings
        
    Returns:
        List of cleaned numerical values as strings
    """
    newsamplemarks = makegoginto_str(newsamplemarks)
    tracking_p = []
    for value in newsamplemarks:
        if num_there(value):
            stripped = value.split(':', 1)[1]
            stripped = stripped.strip()  # remove spaces
            tracking_p.append(stripped)
    return tracking_p

class LandmarkProcessor:
    """Helper class for processing MediaPipe landmarks."""
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize processor with frame dimensions.
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
        """
        self.w = frame_width
        self.h = frame_height
        
    def get_face_rotation(self, face_landmarks) -> Tuple[float, float, float]:
        """
        Calculate face rotation angles from landmarks.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            
        Returns:
            Tuple of (x_rotation, y_rotation, z_rotation)
        """
        face_3d = []
        face_2d = []
        nose_3d = None
        nose_2d = None
        
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in [33, 263, 1, 61, 291, 199]:
                x, y = int(lm.x * self.w), int(lm.y * self.h)
                if idx == 1:  # Nose landmark
                    nose_2d = (lm.x * self.w, lm.y * self.h)
                    nose_3d = (lm.x * self.w, lm.y * self.h, lm.z * 3000)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
        
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        
        # Camera matrix
        focal_length = 1 * self.w
        cam_matrix = np.array([
            [focal_length, 0, self.h / 2],
            [0, focal_length, self.w / 2],
            [0, 0, 1]
        ])
        
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        
        if not success:
            return 0.0, 0.0, 0.0, nose_2d, nose_3d
            
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        return (
            angles[0] * self.w,    # x rotation (up/down)
            angles[1] * self.h,    # y rotation (left/right)
            angles[2] * 3000,      # z rotation (tilt)
            nose_2d,
            nose_3d
        )
    
    def get_landmark_point(self, landmark, idx: int) -> Tuple[float, float]:
        """
        Get 2D point from landmark at given index.
        
        Args:
            landmark: MediaPipe landmark object
            idx: Index of desired landmark
            
        Returns:
            Tuple of (x, y) coordinates
        """
        lm = landmark.landmark[idx]
        return (lm.x * self.w, lm.y * self.h)
    
    def get_normalized_distance(self, point1: Tuple[float, float], 
                              point2: Tuple[float, float], 
                              norm_factor: float) -> float:
        """
        Calculate normalized distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            norm_factor: Normalization factor
            
        Returns:
            Normalized distance between points
        """
        return np.linalg.norm(np.array(point1) - np.array(point2)) / norm_factor

# Landmark indices
markersbody = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 
    'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 
    'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
    'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL',
    'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

class Feature(Enum):
    """Enumeration of features to extract."""
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

class VideoSegment(Enum):
    BEGINNING = "beginning"
    MIDDLE = "middle"
    END = "end"
    LAST = "last"

def video_to_landmarks(
    video_path: Optional[Union[int, str]],
    max_num_frames: Optional[int] = None,
    video_segment: VideoSegment = VideoSegment.BEGINNING,
    end_padding: bool = True,
    drop_consecutive_duplicates: bool = False
) -> List[List[float]]:
    """
    Extract landmarks from video frames, ensuring a maximum of 25 frames per second.
    If the video has a higher frame rate, frames are skipped to maintain this rate.

    Args:
        video_path: Path to the video file or camera index (0 for default camera).
        max_num_frames: Maximum number of frames to process. If None, all frames are processed
                        (subject to the 1/25 second constraint).
        video_segment: Which part of the video to process ('beginning', 'middle', 'end', 'last').
                       'last' takes the last max_num_frames.
        end_padding: If True and the number of extracted frames is less than max_num_frames,
                     the last frame's landmarks are repeated to pad the output.
        drop_consecutive_duplicates: If True, frames with very similar landmark features
                                     to the previous frame are skipped.

    Returns:
        A tuple containing:
            - A list of lists, where each inner list contains the flattened landmark features
              for a single frame.
            - A list of timestamps (in seconds) corresponding to the extracted frames.
    """
    assert video_segment in VideoSegment
    video_path = video_path if video_path else 0

    valid_frame_count = 0
    prev_features: List[float] = []
    landmarks: List[List[float]] = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(fps / 25)) if fps > 0 else 1  # Skip frames if FPS > 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing frames")
    frame_timestamps = []  # Store actual timestamps of processed frames
    frame_number = 0
    processed_frame_count = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
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

            #current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Get actual timestamp

            if max_num_frames and video_segment == VideoSegment.BEGINNING and valid_frame_count >= max_num_frames:
                break

            frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            resultsh = holistic.process(frame)

            h, w, _ = frame.shape
            face_3d = []
            face_2d = []

            if resultsh.face_landmarks and resultsh.pose_landmarks:
                frame_timestamps.append(processed_frame_count / 25.0)  # Store actual timestamp
                # Head rotation calculation
                for idx, lm in enumerate(resultsh.face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * w, lm.y * h)
                            nose_3d = (lm.x * w, lm.y * h, lm.z * 3000)
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # Camera matrix
                focal_length = 1 * w
                cam_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ])

                # Solve PnP
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotation
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get rotation degrees
                xrot = angles[0] * 360
                yrot = angles[1] * 360
                zrot = angles[2] * 360

                # Get nose coordinates
                nose_x = nose_3d[0]
                nose_y = nose_3d[1]
                nose_z = nose_3d[2]

                # Get normalized distance from chin to top head
                chin, top_head = None, None
                for idx, lm in enumerate(resultsh.face_landmarks.landmark):
                    if idx == 152:
                        chin = (lm.x * w, lm.y * h)
                    if idx == 10:
                        top_head = (lm.x * w, lm.y * h)
                norm_dist = np.linalg.norm(np.array(chin) - np.array(top_head)) if chin is not None and top_head is not None else 1.0

                # Get left brow and eye distance
                left_inner_eye, left_brow = None, None
                for idx, lm in enumerate(resultsh.face_landmarks.landmark):
                    if idx == 33:
                        left_inner_eye = (lm.x * w, lm.y * h)
                    if idx == 225:
                        left_brow = (lm.x * w, lm.y * h)
                left_brow_left_eye_norm_dist = np.linalg.norm(np.array(left_inner_eye) - np.array(left_brow)) / norm_dist if left_inner_eye is not None and left_brow is not None else 0.0

                # Get right brow and eye distance
                right_inner_eye, right_brow = None, None
                for idx, lm in enumerate(resultsh.face_landmarks.landmark):
                    if idx == 263:
                        right_inner_eye = (lm.x * w, lm.y * h)
                    if idx == 295:  # Corrected index for right brow
                        right_brow = (lm.x * w, lm.y * h)
                right_brow_right_eye_norm_dist = np.linalg.norm(np.array(right_inner_eye) - np.array(right_brow)) / norm_dist if right_inner_eye is not None and right_brow is not None else 0.0

                # Get mouth corners distance
                left_mouth, right_mouth = None, None
                for idx, lm in enumerate(resultsh.face_landmarks.landmark):
                    if idx == 61:  # Corrected index for left mouth corner
                        left_mouth = (lm.x * w, lm.y * h)
                    if idx == 291:  # Corrected index for right mouth corner
                        right_mouth = (lm.x * w, lm.y * h)
                mouth_corners_norm_dist = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth)) / norm_dist if left_mouth is not None and right_mouth is not None else 0.0

                # Get mouth aperture distance
                upper_lip, lower_lip = None, None
                for idx, lm in enumerate(resultsh.face_landmarks.landmark):
                    if idx == 13:
                        upper_lip = (lm.x * w, lm.y * h)
                    if idx == 14:
                        lower_lip = (lm.x * w, lm.y * h)
                mouth_apperture_norm_dist = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip)) / norm_dist if upper_lip is not None and lower_lip is not None else 0.0

                # Get body landmarks
                body_landmarks = {}
                for idx, lm in enumerate(resultsh.pose_landmarks.landmark):
                    if markersbody[idx] in ['LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
                                             'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_WRIST', 'RIGHT_WRIST',
                                             'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                                             'LEFT_EAR', 'RIGHT_EAR']:
                        body_landmarks[markersbody[idx]] = (lm.x * w, lm.y * h)

                # Calculate normalized distances
                left_wrist = body_landmarks.get('LEFT_WRIST')
                right_wrist = body_landmarks.get('RIGHT_WRIST')
                left_elbow = body_landmarks.get('LEFT_ELBOW')
                right_elbow = body_landmarks.get('RIGHT_ELBOW')
                left_shoulder = body_landmarks.get('LEFT_SHOULDER')
                right_shoulder = body_landmarks.get('RIGHT_SHOULDER')
                left_ear = body_landmarks.get('LEFT_EAR')
                right_ear = body_landmarks.get('RIGHT_EAR')
                left_thumb = body_landmarks.get('LEFT_THUMB')
                left_index_finger = body_landmarks.get('LEFT_INDEX')
                left_pinky = body_landmarks.get('LEFT_PINKY')
                right_thumb = body_landmarks.get('RIGHT_THUMB')
                right_index_finger = body_landmarks.get('RIGHT_INDEX')
                right_pinky = body_landmarks.get('RIGHT_PINKY')

                left_right_wrist_norm_dist = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist)) / norm_dist if left_wrist is not None and right_wrist is not None else 0.0
                left_right_elbow_norm_dist = np.linalg.norm(np.array(left_elbow) - np.array(right_elbow)) / norm_dist if left_elbow is not None and right_elbow is not None else 0.0
                left_elbow_midpoint_shoulder_norm_dist = np.linalg.norm(np.array(left_elbow) - np.array(left_shoulder)) / norm_dist if left_elbow is not None and left_shoulder is not None else 0.0
                right_elbow_midpoint_shoulder_norm_dist = np.linalg.norm(np.array(right_elbow) - np.array(right_shoulder)) / norm_dist if right_elbow is not None and right_shoulder is not None else 0.0
                left_wrist_midpoint_shoulder_norm_dist = np.linalg.norm(np.array(left_wrist) - np.array(left_shoulder)) / norm_dist if left_wrist is not None and left_shoulder is not None else 0.0
                right_wrist_midpoint_shoulder_norm_dist = np.linalg.norm(np.array(right_wrist) - np.array(right_shoulder)) / norm_dist if right_wrist is not None and right_shoulder is not None else 0.0
                left_shoulder_left_ear_norm_dist = np.linalg.norm(np.array(left_shoulder) - np.array(left_ear)) / norm_dist if left_shoulder is not None and left_ear is not None else 0.0
                right_shoulder_right_ear_norm_dist = np.linalg.norm(np.array(right_shoulder) - np.array(right_ear)) / norm_dist if right_shoulder is not None and right_ear is not None else 0.0
                left_thumb_left_index_norm_dist = np.linalg.norm(np.array(left_thumb) - np.array(left_index_finger)) / norm_dist if left_thumb is not None and left_index_finger is not None else 0.0
                right_thumb_right_index_norm_dist = np.linalg.norm(np.array(right_thumb) - np.array(right_index_finger)) / norm_dist if right_thumb is not None and right_index_finger is not None else 0.0
                left_thumb_left_pinky_norm_dist = np.linalg.norm(np.array(left_thumb) - np.array(left_pinky)) / norm_dist if left_thumb is not None and left_pinky is not None else 0.0
                right_thumb_right_pinky_norm_dist = np.linalg.norm(np.array(right_thumb) - np.array(right_pinky)) / norm_dist if right_thumb is not None and right_pinky is not None else 0.0
                x_left_wrist_x_left_elbow_norm_dist = (left_wrist[0] - left_elbow[0]) / norm_dist if left_wrist is not None and left_elbow is not None else 0.0
                x_right_wrist_x_right_elbow_norm_dist = (right_wrist[0] - right_elbow[0]) / norm_dist if right_wrist is not None and right_elbow is not None else 0.0
                y_left_wrist_y_left_elbow_norm_dist = (left_wrist[1] - left_elbow[1]) / norm_dist if left_wrist is not None and left_elbow is not None else 0.0
                y_right_wrist_y_right_elbow_norm_dist = (right_wrist[1] - right_elbow[1]) / norm_dist if right_wrist is not None and right_elbow is not None else 0.0
                left_index_finger_nose_norm_dist = np.linalg.norm(np.array(left_index_finger) - np.array(nose_2d)) / norm_dist if left_index_finger is not None and nose_2d is not None else 0.0
                right_index_finger_nose_norm_dist = np.linalg.norm(np.array(right_index_finger) - np.array(nose_2d)) / norm_dist if right_index_finger is not None and nose_2d is not None else 0.0

                # Collect all features in order
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

                if drop_consecutive_duplicates and prev_features and np.array_equal(
                        np.round(features, decimals=2),
                        np.round(prev_features, decimals=2)
                ):
                    continue

                landmarks.append(features)
                prev_features = features
                valid_frame_count += 1
                pbar.update(1)
                
        pbar.close()
        cap.release()

        if not landmarks:
            return []

        if max_num_frames and video_segment == VideoSegment.LAST:
            landmarks = landmarks[-max_num_frames:]

        if max_num_frames and end_padding and len(landmarks) < max_num_frames:
            last = landmarks[-1]
            landmarks = landmarks + [last] * (max_num_frames - len(landmarks))

        return landmarks, frame_timestamps
    
class VideoProcessor:
    """Handles video processing and feature extraction."""
    
    def __init__(self, seq_length: int = 25):
        """Initialize processor with window size."""
        self.seq_length = seq_length
        self.mp_holistic = mp.solutions.holistic
        
    def process_video(self, video_path: str) -> Tuple[List[List[float]], List[float]]:
        """
        Process video and extract landmarks features.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (features_list, timestamps)
        """
        # Directly use video_to_landmarks to extract features
        features_list, timestamps = video_to_landmarks(
            video_path=video_path, 
            max_num_frames=None,  # Process entire video
            video_segment=VideoSegment.BEGINNING
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
