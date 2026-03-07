# envisionhgdetector/model_lightgbm.py
"""
LightGBM-based gesture detection model.
Updated for Config 13 (World landmarks, 100 features, 2-class).

Classes: NoGesture, Gesture (Move merged into NoGesture)
Features: 100 engineered features from 92 world landmarks
Window: 5 frames
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
from .config import Config

# Upper body landmark indices (23 landmarks, matching training)
UPPER_BODY_INDICES = list(range(23))

# Key joint indices for feature extraction
KEY_JOINT_INDICES = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16

# Visibility landmark indices
VISIBILITY_LANDMARKS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


def extract_lgbm_features(sequence: np.ndarray) -> np.ndarray:
    """
    Extract 100 features from a sequence of world landmarks for LightGBM.
    Matches Config 13 training exactly.

    Args:
        sequence: Array of shape (window_size, 92) - world landmarks

    Returns:
        100-dimensional feature vector
    """
    if len(sequence) == 0:
        return np.zeros(100, dtype=np.float32)

    n_frames = len(sequence)
    n_landmarks = 23

    try:
        seq_4d = sequence.reshape(n_frames, n_landmarks, 4)
    except ValueError:
        return np.zeros(100, dtype=np.float32)

    seq_3d = seq_4d[:, :, :3]
    visibility = seq_4d[:, :, 3]

    features = []

    key_joints = seq_3d[:, KEY_JOINT_INDICES, :]
    key_joints_flat = key_joints.reshape(n_frames, -1)

    # 1-18: Current pose
    current_pose = key_joints_flat[-1]
    features.extend(current_pose)

    # 19-38: Velocity + wrist speeds
    if n_frames > 1:
        velocity = key_joints_flat[-1] - key_joints_flat[-2]
        features.extend(velocity)
        left_wrist_speed = np.linalg.norm(velocity[12:15])
        right_wrist_speed = np.linalg.norm(velocity[15:18])
        features.extend([left_wrist_speed, right_wrist_speed])
    else:
        features.extend([0.0] * 20)

    # 39-44: Wrist ranges
    if n_frames >= 3:
        wrist_data = key_joints_flat[:, 12:18]
        wrist_ranges = np.ptp(wrist_data, axis=0)
        features.extend(wrist_ranges)
    else:
        features.extend([0.0] * 6)

    # 45-62: Finger features
    left_wrist = seq_3d[-1, LEFT_WRIST_IDX, :]
    right_wrist = seq_3d[-1, RIGHT_WRIST_IDX, :]

    left_fingers = np.zeros(9, dtype=np.float32)
    right_fingers = np.zeros(9, dtype=np.float32)

    if np.any(left_wrist):
        left_fingers[0:3] = seq_3d[-1, 17, :] - left_wrist
        left_fingers[3:6] = seq_3d[-1, 19, :] - left_wrist
        left_fingers[6:9] = seq_3d[-1, 21, :] - left_wrist

    if np.any(right_wrist):
        right_fingers[0:3] = seq_3d[-1, 18, :] - right_wrist
        right_fingers[3:6] = seq_3d[-1, 20, :] - right_wrist
        right_fingers[6:9] = seq_3d[-1, 22, :] - right_wrist

    features.extend(left_fingers)
    features.extend(right_fingers)

    # 63-68: Finger distances
    left_pinky_thumb = np.linalg.norm(left_fingers[0:3] - left_fingers[6:9]) if np.any(left_fingers) else 0.0
    left_index_thumb = np.linalg.norm(left_fingers[3:6] - left_fingers[6:9]) if np.any(left_fingers) else 0.0
    left_pinky_index = np.linalg.norm(left_fingers[0:3] - left_fingers[3:6]) if np.any(left_fingers) else 0.0
    right_pinky_thumb = np.linalg.norm(right_fingers[0:3] - right_fingers[6:9]) if np.any(right_fingers) else 0.0
    right_index_thumb = np.linalg.norm(right_fingers[3:6] - right_fingers[6:9]) if np.any(right_fingers) else 0.0
    right_pinky_index = np.linalg.norm(right_fingers[0:3] - right_fingers[3:6]) if np.any(right_fingers) else 0.0

    features.extend([left_pinky_thumb, left_index_thumb, left_pinky_index,
                     right_pinky_thumb, right_index_thumb, right_pinky_index])

    # 69-70: Wrist acceleration
    if n_frames > 2:
        vel_prev = key_joints_flat[-2] - key_joints_flat[-3]
        vel_curr = key_joints_flat[-1] - key_joints_flat[-2]
        accel = vel_curr - vel_prev
        left_wrist_accel = np.linalg.norm(accel[12:15])
        right_wrist_accel = np.linalg.norm(accel[15:18])
        features.extend([left_wrist_accel, right_wrist_accel])
    else:
        features.extend([0.0, 0.0])

    # 71-72: Trajectory smoothness
    if n_frames > 2:
        velocities = np.diff(key_joints_flat, axis=0)
        left_wrist_vels = velocities[:, 12:15]
        right_wrist_vels = velocities[:, 15:18]
        left_smoothness = np.std(np.linalg.norm(left_wrist_vels, axis=1))
        right_smoothness = np.std(np.linalg.norm(right_wrist_vels, axis=1))
        features.extend([left_smoothness, right_smoothness])
    else:
        features.extend([0.0, 0.0])

    # 73-74: Wrist height
    left_shoulder = seq_3d[-1, 11, :]
    right_shoulder = seq_3d[-1, 12, :]
    left_wrist_pos = seq_3d[-1, LEFT_WRIST_IDX, :]
    right_wrist_pos = seq_3d[-1, RIGHT_WRIST_IDX, :]

    left_wrist_height = left_wrist_pos[1] - left_shoulder[1] if np.any(left_shoulder) else 0.0
    right_wrist_height = right_wrist_pos[1] - right_shoulder[1] if np.any(right_shoulder) else 0.0
    features.extend([left_wrist_height, right_wrist_height])

    # 75: Wrist spread
    wrist_spread = np.linalg.norm(left_wrist_pos - right_wrist_pos) if np.any(left_wrist_pos) and np.any(right_wrist_pos) else 0.0
    features.append(wrist_spread)

    # 76-77: Arm extension
    left_arm_extension = np.linalg.norm(left_wrist_pos - left_shoulder) if np.any(left_shoulder) else 0.0
    right_arm_extension = np.linalg.norm(right_wrist_pos - right_shoulder) if np.any(right_shoulder) else 0.0
    features.extend([left_arm_extension, right_arm_extension])

    # 78: Total motion
    if n_frames > 1:
        total_motion = np.sum(np.linalg.norm(np.diff(seq_3d[:, LEFT_WRIST_IDX, :], axis=0), axis=1))
        total_motion += np.sum(np.linalg.norm(np.diff(seq_3d[:, RIGHT_WRIST_IDX, :], axis=0), axis=1))
        features.append(total_motion)
    else:
        features.append(0.0)

    # 79: Position symmetry
    if np.any(left_wrist_pos) and np.any(right_wrist_pos):
        body_center = (left_shoulder + right_shoulder) / 2
        left_rel = left_wrist_pos - body_center
        right_rel = right_wrist_pos - body_center
        symmetry = 1.0 / (1.0 + np.linalg.norm(left_rel + right_rel * np.array([-1, 1, 1])))
        features.append(symmetry)
    else:
        features.append(0.0)

    # 80: Motion asymmetry
    if n_frames > 1:
        left_motion = np.sum(np.linalg.norm(np.diff(seq_3d[:, LEFT_WRIST_IDX, :], axis=0), axis=1))
        right_motion = np.sum(np.linalg.norm(np.diff(seq_3d[:, RIGHT_WRIST_IDX, :], axis=0), axis=1))
        motion_asymmetry = abs(left_motion - right_motion) / (left_motion + right_motion + 1e-6)
        features.append(motion_asymmetry)
    else:
        features.append(0.0)

    # 81-92: Current visibility
    current_vis = visibility[-1, VISIBILITY_LANDMARKS]
    features.extend(current_vis)

    # 93-98: Mean visibility
    mean_vis = np.mean(visibility[:, [11, 12, 13, 14, 15, 16]], axis=0)
    features.extend(mean_vis)

    # 99-100: Min wrist visibility
    min_vis_left_wrist = np.min(visibility[:, 15])
    min_vis_right_wrist = np.min(visibility[:, 16])
    features.extend([min_vis_left_wrist, min_vis_right_wrist])

    assert len(features) == 100, f"Expected 100 features, got {len(features)}"
    return np.array(features, dtype=np.float32)


class LightGBMGestureModel:
    """
    LightGBM-based gesture detection model.
    Matches Config 13 training: 100 features from world landmarks.
    
    2-class model: NoGesture vs Gesture (Move merged into NoGesture)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize LightGBM model with configuration."""
        self.config = config or Config()
        
        # Default parameters (will be overwritten by model file)
        self.window_size = 5
        self.n_features = 100
        self.gesture_labels = ("NoGesture", "Gesture")
        
        # Find and load model
        model_path = self._find_model_path()
        if model_path:
            self.load_model(model_path)
        else:
            print("Warning: LightGBM model not found. Call load_model() manually.")
            self.model = None
            self.scaler = None
            self.label_encoder = None
        
        # Initialize MediaPipe Holistic for world landmarks
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Buffer for world landmarks (window_size frames)
        self.landmarks_buffer = deque(maxlen=self.window_size)
        
        # Backward compatibility aliases
        self.key_joints_buffer = self.landmarks_buffer  # Alias for old code
        self.left_fingers_buffer = deque(maxlen=self.window_size)  # Dummy for old code
        self.right_fingers_buffer = deque(maxlen=self.window_size)  # Dummy for old code
        self.includes_fingers = True  # Always true for Config 13
        self.expected_features = self.n_features  # Alias
        
        # Confidence threshold
        self.confidence_threshold = 0.5
    
    def _find_model_path(self) -> Optional[str]:
        """Find the LightGBM model file."""
        # Try config path first
        if hasattr(self.config, 'lightgbm_weights_path') and self.config.lightgbm_weights_path:
            if os.path.exists(self.config.lightgbm_weights_path):
                return self.config.lightgbm_weights_path
        
        # Try default paths
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'model', 'best_lightgbm_model.pkl'),
            os.path.join(os.path.dirname(__file__), 'model', 'lightgbm_gesture_model_v2.pkl'),
            os.path.join(os.path.dirname(__file__), 'model', 'lightgbm_gesture_model_v1.pkl'),
            os.path.join(os.path.dirname(__file__), 'best_lightgbm_model.pkl'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def load_model(self, model_path: str):
        """Load LightGBM model from joblib file."""
        print(f"Loading LightGBM model from {model_path}")
        
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.window_size = model_data.get('window_size', 5)
            self.n_features = model_data.get('n_features', 100)
            
            # IMPORTANT: Use label_encoder.classes_ for correct label order
            # LabelEncoder sorts alphabetically, so classes_ = ['Gesture', 'NoGesture']
            # This means: index 0 = Gesture, index 1 = NoGesture
            if self.label_encoder is not None:
                self.gesture_labels = tuple(self.label_encoder.classes_)
            else:
                # Fallback: alphabetical order (sklearn default)
                self.gesture_labels = ("Gesture", "NoGesture")
            
            # Update buffer size
            self.landmarks_buffer = deque(maxlen=self.window_size)
            
            print(f"✓ LightGBM model loaded successfully!")
            print(f"  Window size: {self.window_size} frames")
            print(f"  Features: {self.n_features}")
            print(f"  Classes (model order): {self.gesture_labels}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load LightGBM model from {model_path}: {str(e)}")
    
    def extract_world_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract world landmarks from frame using MediaPipe Holistic.
        
        Returns:
            Array of 92 features (23 landmarks × 4: x, y, z, visibility)
            or None if pose not detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process with MediaPipe
        results = self.holistic.process(rgb_frame)
        
        if not results.pose_world_landmarks:
            return None
        
        # Extract upper body world landmarks (23 × 4 = 92 features)
        features = []
        for idx in UPPER_BODY_INDICES:
            if idx < len(results.pose_world_landmarks.landmark):
                lm = results.pose_world_landmarks.landmark[idx]
                features.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def extract_sequence_features(self, sequence: np.ndarray) -> np.ndarray:
        return extract_lgbm_features(sequence)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on input features.
        
        Args:
            features: Input features of shape (batch_size, 100) or (100,)
            
        Returns:
            Probabilities with shape (batch_size, num_classes)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions - handle both Booster and Classifier objects
        if hasattr(self.model, 'predict_proba'):
            # LGBMClassifier - returns probabilities directly
            probabilities = self.model.predict_proba(features_scaled)
        else:
            # Raw Booster trained with multiclass objective
            # predict() already returns probabilities with shape (n_samples, n_classes)
            raw_output = self.model.predict(features_scaled)
            probabilities = np.array(raw_output)
            
            # Ensure 2D output
            if probabilities.ndim == 1:
                # Single sample, check if it's already probabilities
                if len(probabilities) == len(self.gesture_labels):
                    probabilities = probabilities.reshape(1, -1)
                else:
                    # Binary single value - apply sigmoid
                    gesture_probs = 1.0 / (1.0 + np.exp(-probabilities))
                    probabilities = np.column_stack([1 - gesture_probs, gesture_probs])
        
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
            
        return probabilities
    
    def extract_features_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features from a single frame for real-time prediction.
        
        Args:
            frame: Input video frame (BGR)
            
        Returns:
            100-dimensional feature array, or None if not enough data
        """
        # Extract world landmarks
        landmarks = self.extract_world_landmarks(frame)
        
        if landmarks is None:
            return None
        
        # Add to buffer
        self.landmarks_buffer.append(landmarks)
        
        # Check if we have enough frames
        if len(self.landmarks_buffer) < self.window_size:
            return None
        
        # Extract features from sequence
        sequence = np.array(list(self.landmarks_buffer))
        features = self.extract_sequence_features(sequence)
        
        return features
    
    def predict_frame(self, frame: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Predict gesture from a single frame.
        
        Args:
            frame: Input video frame (BGR)
            
        Returns:
            Tuple of (gesture_label, confidence) or (None, None) if not ready
        """
        features = self.extract_features_from_frame(frame)
        
        if features is None:
            return None, None
        
        # Get prediction
        probs = self.predict(features)[0]
        
        # Get class and confidence
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        
        # Apply threshold
        if confidence < self.confidence_threshold:
            return "NoGesture", 1.0 - probs[1]  # Return NoGesture confidence
        
        gesture_label = self.gesture_labels[class_idx]
        
        return gesture_label, confidence
    
    def reset_buffer(self):
        """Clear the landmark buffer."""
        self.landmarks_buffer.clear()
        # Clear backward compatibility buffers too
        if hasattr(self, 'left_fingers_buffer'):
            self.left_fingers_buffer.clear()
        if hasattr(self, 'right_fingers_buffer'):
            self.right_fingers_buffer.clear()
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def standardize_gesture_name(self, gesture: str) -> str:
        """Standardize gesture names to consistent format."""
        if not gesture or gesture.lower() in ['no_gesture', 'nogesture', 'none', '']:
            return "NOGESTURE"
        return gesture.upper().replace('_', '').replace(' ', '')
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'holistic') and self.holistic:
            self.holistic.close()