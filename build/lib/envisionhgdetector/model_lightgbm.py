# envisionhgdetector/model_lightgbm.py

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
from .config import Config
import os

# Key joints for LightGBM (shoulders, elbows, wrists)
KEY_JOINT_INDICES = [11, 12, 13, 14, 15, 16]
KEY_JOINT_NAMES = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST']

# Finger landmark indices
FINGER_INDICES = [17, 18, 19, 20, 21, 22]
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16

class LightGBMGestureModel:
    """
    LightGBM-based gesture detection model.
    Optimized for real-time processing with MediaPipe pose landmarks.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize LightGBM model with configuration."""
        self.config = config or Config()
        
        # Find and load model
        model_path = self._find_model_path()
        self.load_model(model_path)
        
        # Initialize MediaPipe for ultra-fast processing
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Fastest model
            enable_segmentation=False,
            smooth_landmarks=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Buffers for windowing
        self.key_joints_buffer = deque(maxlen=self.window_size)
        self.left_fingers_buffer = deque(maxlen=self.window_size)
        self.right_fingers_buffer = deque(maxlen=self.window_size)
        
        # Motion detection
        self.motion_threshold = 0.02
        
        # Confidence threshold (can be set externally)
        self.confidence_threshold = 0.2
    
    def _find_model_path(self) -> str:
        """Find the LightGBM model file."""
        # Try config path first
        if hasattr(self.config, 'weights_path') and self.config.weights_path:
            if self.config.weights_path.endswith('.pkl') and os.path.exists(self.config.weights_path):
                return self.config.weights_path
        
        # Try default paths
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'model', 'lightgbm_gesture_model_v1.pkl'),
            os.path.join(os.path.dirname(__file__), 'lightgbm_gesture_model_v1.pkl'),
            'lightgbm_gesture_model_v1.pkl'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"LightGBM model not found. Tried paths: {possible_paths}. "
            f"Please ensure lightgbm_gesture_model_v1.pkl is in the model folder."
        )
    
    def load_model(self, model_path: str):
        """Load LightGBM model from joblib file."""
        print(f"Loading LightGBM model from {model_path}")
        
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.window_size = model_data['window_size']
            self.gesture_labels = model_data['gesture_labels']
            self.includes_fingers = model_data.get('includes_fingers', False)
            self.expected_features = model_data.get('feature_count', 80 if self.includes_fingers else 50)
            
            print(f"LightGBM model loaded successfully!")
            print(f"Window size: {self.window_size} frames")
            print(f"Available gestures: {self.gesture_labels}")
            print(f"Advanced features: {'ENABLED' if self.includes_fingers else 'DISABLED'}")
            print(f"Expected features: {self.expected_features}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load LightGBM model from {model_path}: {str(e)}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on input features.
        
        Args:
            features: Input features of shape (batch_size, feature_dim) or (feature_dim,)
            
        Returns:
            Model predictions with shape (batch_size, num_classes)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions (LightGBM returns probabilities)
        probabilities = self.model.predict(features_scaled)
        
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
            
        return probabilities
    
    def extract_features_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features from a single frame and return them for prediction.
        This is the main interface for real-time processing.
        
        Args:
            frame: Input video frame
            
        Returns:
            Features array ready for prediction, or None if not enough data
        """
        # Extract landmarks
        key_joints, left_fingers, right_fingers = self._extract_landmarks_fast(frame)
        
        if key_joints is None:
            return None
        
        # Add to buffers
        self.key_joints_buffer.append(key_joints)
        if self.includes_fingers:
            self.left_fingers_buffer.append(left_fingers if left_fingers is not None else np.zeros(9))
            self.right_fingers_buffer.append(right_fingers if right_fingers is not None else np.zeros(9))
        
        # Check if we have enough frames for a window
        if len(self.key_joints_buffer) < self.window_size:
            return None
        
        # Extract enhanced features from current window
        key_sequence = np.array(list(self.key_joints_buffer))
        left_fingers_sequence = np.array(list(self.left_fingers_buffer)) if self.includes_fingers else np.array([])
        right_fingers_sequence = np.array(list(self.right_fingers_buffer)) if self.includes_fingers else np.array([])
        
        enhanced_features = self._extract_enhanced_features(
            key_sequence, left_fingers_sequence, right_fingers_sequence
        )
        
        return enhanced_features
    
    def fast_motion_detection(self) -> bool:
        """Check if there's motion in recent frames."""
        if len(self.key_joints_buffer) < 2:
            return True
        
        current_joints = self.key_joints_buffer[-1]
        prev_joints = self.key_joints_buffer[-2]
        
        # Only check wrist movement (indices 12-17 in key joints)
        left_wrist_motion = np.linalg.norm(current_joints[12:15] - prev_joints[12:15])
        right_wrist_motion = np.linalg.norm(current_joints[15:18] - prev_joints[15:18])
        
        max_motion = max(left_wrist_motion, right_wrist_motion)
        return max_motion > self.motion_threshold
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold dynamically."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def standardize_gesture_name(self, gesture: str) -> str:
        """Standardize gesture names to consistent format."""
        if not gesture or gesture.lower() in ['no_gesture', 'nogesture', 'none', '']:
            return "NOGESTURE"
        
        # Convert to consistent format and remove underscores/spaces  
        standardized = gesture.upper().replace('_', '').replace(' ', '')
        return standardized
    
    def _extract_landmarks_fast(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract landmarks from frame using MediaPipe pose."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        # Process with MediaPipe pose
        pose_results = self.pose.process(rgb_image)
        
        if not pose_results.pose_world_landmarks:
            return None, None, None
        
        # Extract key joints
        key_landmarks = []
        landmarks = pose_results.pose_world_landmarks.landmark
        
        for idx in KEY_JOINT_INDICES:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                key_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                key_landmarks.extend([0.0, 0.0, 0.0])
        
        key_joints = np.array(key_landmarks, dtype=np.float32)
        
        # Extract finger landmarks if enabled
        left_fingers_relative = np.zeros(9, dtype=np.float32)
        right_fingers_relative = np.zeros(9, dtype=np.float32)
        
        if self.includes_fingers:
            left_fingers_relative, right_fingers_relative = self._extract_fingers_from_pose_landmarks(pose_results)
        
        return key_joints, left_fingers_relative, right_fingers_relative
    
    def _extract_fingers_from_pose_landmarks(self, pose_results) -> Tuple[np.ndarray, np.ndarray]:
        """Extract finger landmarks from pose landmarks, centered around wrists."""
        left_fingers_relative = np.zeros(9, dtype=np.float32)
        right_fingers_relative = np.zeros(9, dtype=np.float32)
        
        if not pose_results or not pose_results.pose_world_landmarks:
            return left_fingers_relative, right_fingers_relative
        
        landmarks = pose_results.pose_world_landmarks.landmark
        
        if len(landmarks) < 23:
            return left_fingers_relative, right_fingers_relative
        
        # Get wrist positions
        left_wrist = None
        right_wrist = None
        
        if len(landmarks) > LEFT_WRIST_IDX:
            left_wrist_landmark = landmarks[LEFT_WRIST_IDX]
            left_wrist = np.array([left_wrist_landmark.x, left_wrist_landmark.y, left_wrist_landmark.z], dtype=np.float32)
            
        if len(landmarks) > RIGHT_WRIST_IDX:
            right_wrist_landmark = landmarks[RIGHT_WRIST_IDX]
            right_wrist = np.array([right_wrist_landmark.x, right_wrist_landmark.y, right_wrist_landmark.z], dtype=np.float32)
        
        # Extract left fingers relative to left wrist (indices 17, 19, 21)
        if left_wrist is not None and len(landmarks) > 21:
            finger_indices = [17, 19, 21]  # left_pinky, left_index, left_thumb
            for i, finger_idx in enumerate(finger_indices):
                if len(landmarks) > finger_idx:
                    finger_landmark = landmarks[finger_idx]
                    finger_pos = np.array([finger_landmark.x, finger_landmark.y, finger_landmark.z], dtype=np.float32)
                    left_fingers_relative[i*3:(i+1)*3] = finger_pos - left_wrist
        
        # Extract right fingers relative to right wrist (indices 18, 20, 22)
        if right_wrist is not None and len(landmarks) > 22:
            finger_indices = [18, 20, 22]  # right_pinky, right_index, right_thumb
            for i, finger_idx in enumerate(finger_indices):
                if len(landmarks) > finger_idx:
                    finger_landmark = landmarks[finger_idx]
                    finger_pos = np.array([finger_landmark.x, finger_landmark.y, finger_landmark.z], dtype=np.float32)
                    right_fingers_relative[i*3:(i+1)*3] = finger_pos - right_wrist
        
        return left_fingers_relative, right_fingers_relative
    
    def _extract_enhanced_features(self, key_joints_sequence: np.ndarray, 
                                  left_fingers_sequence: np.ndarray, 
                                  right_fingers_sequence: np.ndarray) -> np.ndarray:
        """Extract enhanced features matching the training feature extraction."""
        if len(key_joints_sequence) == 0:
            return np.zeros(self.expected_features, dtype=np.float32)
        
        features = []
        
        # Current pose (18 values: 6 joints * 3 coords)
        current_pose = key_joints_sequence[-1]
        features.extend(current_pose)
        
        if len(key_joints_sequence) > 1:
            # Simple velocity (18 values)
            velocity = key_joints_sequence[-1] - key_joints_sequence[-2]
            features.extend(velocity)
            
            # Wrist speeds only (2 values)
            left_wrist_speed = np.linalg.norm(velocity[12:15])
            right_wrist_speed = np.linalg.norm(velocity[15:18])
            features.extend([left_wrist_speed, right_wrist_speed])
        else:
            features.extend([0.0] * 20)
        
        # Simple range over window for pose
        if len(key_joints_sequence) >= 3:
            # Range for wrists only (6 values)
            wrist_data = key_joints_sequence[:, 12:18]
            wrist_ranges = np.ptp(wrist_data, axis=0)
            features.extend(wrist_ranges)
        else:
            features.extend([0.0] * 6)
        
        # Advanced features (if enabled)
        if self.includes_fingers and self.expected_features >= 80:
            # Current finger positions (18 values: 2 hands * 9 coords each)
            current_left_fingers = left_fingers_sequence[-1] if len(left_fingers_sequence) > 0 else np.zeros(9)
            current_right_fingers = right_fingers_sequence[-1] if len(right_fingers_sequence) > 0 else np.zeros(9)
            features.extend(current_left_fingers)
            features.extend(current_right_fingers)
            
            # Shape features for each hand (6 values total)
            finger_distances = self._calculate_finger_distances(current_left_fingers, current_right_fingers)
            features.extend(finger_distances)
        
        # Pad to expected size
        while len(features) < self.expected_features:
            features.append(0.0)
        
        return np.array(features[:self.expected_features], dtype=np.float32)
    
    def _calculate_finger_distances(self, left_fingers: np.ndarray, right_fingers: np.ndarray) -> List[float]:
        """Calculate finger distances for shape features."""
        distances = []
        
        # Left hand distances
        if len(left_fingers) >= 9 and np.any(left_fingers):
            left_pinky_pos = left_fingers[0:3]
            left_index_pos = left_fingers[3:6]
            left_thumb_pos = left_fingers[6:9]
            
            distances.extend([
                np.linalg.norm(left_pinky_pos - left_thumb_pos),
                np.linalg.norm(left_index_pos - left_thumb_pos),
                np.linalg.norm(left_pinky_pos - left_index_pos)
            ])
        else:
            distances.extend([0.0, 0.0, 0.0])
        
        # Right hand distances
        if len(right_fingers) >= 9 and np.any(right_fingers):
            right_pinky_pos = right_fingers[0:3]
            right_index_pos = right_fingers[3:6]
            right_thumb_pos = right_fingers[6:9]
            
            distances.extend([
                np.linalg.norm(right_pinky_pos - right_thumb_pos),
                np.linalg.norm(right_index_pos - right_thumb_pos),
                np.linalg.norm(right_pinky_pos - right_index_pos)
            ])
        else:
            distances.extend([0.0, 0.0, 0.0])
        
        return distances