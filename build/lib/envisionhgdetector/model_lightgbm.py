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
UPPER_BODY_VIS = np.array([11, 12, 13, 14, 15, 16])

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
    
    # OLD IMPLEMENTATION
    # def extract_sequence_features(self, sequence: np.ndarray) -> np.ndarray:
    #     """
    #     Extract 100 features from a sequence of world landmarks.
    #     MATCHES TRAINING EXACTLY!
        
    #     Args:
    #         sequence: Array of shape (window_size, 92) - world landmarks
        
    #     Returns:
    #         100-dimensional feature vector
    #     """
    #     if len(sequence) == 0:
    #         return np.zeros(100, dtype=np.float32)
        
    #     n_frames = len(sequence)
    #     n_landmarks = 23
        
    #     # Reshape to (frames, landmarks, 4) where 4 = x, y, z, visibility
    #     try:
    #         seq_4d = sequence.reshape(n_frames, n_landmarks, 4)
    #     except ValueError:
    #         return np.zeros(100, dtype=np.float32)
        
    #     # Separate xyz and visibility
    #     seq_3d = seq_4d[:, :, :3]  # (frames, 23, 3)
    #     visibility = seq_4d[:, :, 3]  # (frames, 23)
        
    #     features = []
        
    #     # Key joints (shoulders, elbows, wrists)
    #     key_joints = seq_3d[:, KEY_JOINT_INDICES, :]  # (frames, 6, 3)
    #     key_joints_flat = key_joints.reshape(n_frames, -1)  # (frames, 18)
        
    #     # ===== FEATURES 1-18: Current pose =====
    #     current_pose = key_joints_flat[-1]
    #     features.extend(current_pose)
        
    #     # ===== FEATURES 19-38: Velocity =====
    #     if n_frames > 1:
    #         velocity = key_joints_flat[-1] - key_joints_flat[-2]
    #         features.extend(velocity)
            
    #         # Wrist speeds (2 values)
    #         left_wrist_speed = np.linalg.norm(velocity[12:15])
    #         right_wrist_speed = np.linalg.norm(velocity[15:18])
    #         features.extend([left_wrist_speed, right_wrist_speed])
    #     else:
    #         features.extend([0.0] * 20)
        
    #     # ===== FEATURES 39-44: Wrist ranges =====
    #     if n_frames >= 3:
    #         wrist_data = key_joints_flat[:, 12:18]
    #         wrist_ranges = np.ptp(wrist_data, axis=0)
    #         features.extend(wrist_ranges)
    #     else:
    #         features.extend([0.0] * 6)
        
    #     # ===== FEATURES 45-62: Finger features (18 values) =====
    #     left_wrist = seq_3d[-1, LEFT_WRIST_IDX, :]
    #     right_wrist = seq_3d[-1, RIGHT_WRIST_IDX, :]
        
    #     left_fingers = np.zeros(9, dtype=np.float32)
    #     right_fingers = np.zeros(9, dtype=np.float32)
        
    #     if np.any(left_wrist):
    #         left_fingers[0:3] = seq_3d[-1, 17, :] - left_wrist  # pinky
    #         left_fingers[3:6] = seq_3d[-1, 19, :] - left_wrist  # index
    #         left_fingers[6:9] = seq_3d[-1, 21, :] - left_wrist  # thumb
        
    #     if np.any(right_wrist):
    #         right_fingers[0:3] = seq_3d[-1, 18, :] - right_wrist  # pinky
    #         right_fingers[3:6] = seq_3d[-1, 20, :] - right_wrist  # index
    #         right_fingers[6:9] = seq_3d[-1, 22, :] - right_wrist  # thumb
        
    #     features.extend(left_fingers)
    #     features.extend(right_fingers)
        
    #     # ===== FEATURES 63-68: Finger distances (6 values) =====
    #     left_pinky_thumb = np.linalg.norm(left_fingers[0:3] - left_fingers[6:9]) if np.any(left_fingers) else 0.0
    #     left_index_thumb = np.linalg.norm(left_fingers[3:6] - left_fingers[6:9]) if np.any(left_fingers) else 0.0
    #     left_pinky_index = np.linalg.norm(left_fingers[0:3] - left_fingers[3:6]) if np.any(left_fingers) else 0.0
    #     right_pinky_thumb = np.linalg.norm(right_fingers[0:3] - right_fingers[6:9]) if np.any(right_fingers) else 0.0
    #     right_index_thumb = np.linalg.norm(right_fingers[3:6] - right_fingers[6:9]) if np.any(right_fingers) else 0.0
    #     right_pinky_index = np.linalg.norm(right_fingers[0:3] - right_fingers[3:6]) if np.any(right_fingers) else 0.0
        
    #     features.extend([left_pinky_thumb, left_index_thumb, left_pinky_index,
    #                     right_pinky_thumb, right_index_thumb, right_pinky_index])
        
    #     # ===== FEATURES 69-70: Wrist acceleration =====
    #     if n_frames > 2:
    #         vel_prev = key_joints_flat[-2] - key_joints_flat[-3]
    #         vel_curr = key_joints_flat[-1] - key_joints_flat[-2]
    #         accel = vel_curr - vel_prev
    #         left_wrist_accel = np.linalg.norm(accel[12:15])
    #         right_wrist_accel = np.linalg.norm(accel[15:18])
    #         features.extend([left_wrist_accel, right_wrist_accel])
    #     else:
    #         features.extend([0.0, 0.0])
        
    #     # ===== FEATURES 71-72: Trajectory smoothness =====
    #     if n_frames > 2:
    #         velocities = np.diff(key_joints_flat, axis=0)
    #         left_wrist_vels = velocities[:, 12:15]
    #         right_wrist_vels = velocities[:, 15:18]
    #         left_smoothness = np.std(np.linalg.norm(left_wrist_vels, axis=1))
    #         right_smoothness = np.std(np.linalg.norm(right_wrist_vels, axis=1))
    #         features.extend([left_smoothness, right_smoothness])
    #     else:
    #         features.extend([0.0, 0.0])
        
    #     # ===== FEATURES 73-74: Wrist height relative to shoulders =====
    #     left_shoulder = seq_3d[-1, 11, :]
    #     right_shoulder = seq_3d[-1, 12, :]
    #     left_wrist_pos = seq_3d[-1, LEFT_WRIST_IDX, :]
    #     right_wrist_pos = seq_3d[-1, RIGHT_WRIST_IDX, :]
        
    #     left_wrist_height = left_wrist_pos[1] - left_shoulder[1] if np.any(left_shoulder) else 0.0
    #     right_wrist_height = right_wrist_pos[1] - right_shoulder[1] if np.any(right_shoulder) else 0.0
    #     features.extend([left_wrist_height, right_wrist_height])
        
    #     # ===== FEATURE 75: Wrist spread =====
    #     wrist_spread = np.linalg.norm(left_wrist_pos - right_wrist_pos) if np.any(left_wrist_pos) and np.any(right_wrist_pos) else 0.0
    #     features.append(wrist_spread)
        
    #     # ===== FEATURES 76-77: Arm extension =====
    #     left_arm_extension = np.linalg.norm(left_wrist_pos - left_shoulder) if np.any(left_shoulder) else 0.0
    #     right_arm_extension = np.linalg.norm(right_wrist_pos - right_shoulder) if np.any(right_shoulder) else 0.0
    #     features.extend([left_arm_extension, right_arm_extension])
        
    #     # ===== FEATURE 78: Total motion =====
    #     if n_frames > 1:
    #         total_motion = np.sum(np.linalg.norm(np.diff(seq_3d[:, LEFT_WRIST_IDX, :], axis=0), axis=1))
    #         total_motion += np.sum(np.linalg.norm(np.diff(seq_3d[:, RIGHT_WRIST_IDX, :], axis=0), axis=1))
    #         features.append(total_motion)
    #     else:
    #         features.append(0.0)
        
    #     # ===== FEATURE 79: Position symmetry =====
    #     if np.any(left_wrist_pos) and np.any(right_wrist_pos):
    #         body_center = (left_shoulder + right_shoulder) / 2
    #         left_rel = left_wrist_pos - body_center
    #         right_rel = right_wrist_pos - body_center
    #         symmetry = 1.0 / (1.0 + np.linalg.norm(left_rel + right_rel * np.array([-1, 1, 1])))
    #         features.append(symmetry)
    #     else:
    #         features.append(0.0)
        
    #     # ===== FEATURE 80: Motion asymmetry =====
    #     if n_frames > 1:
    #         left_motion = np.sum(np.linalg.norm(np.diff(seq_3d[:, LEFT_WRIST_IDX, :], axis=0), axis=1))
    #         right_motion = np.sum(np.linalg.norm(np.diff(seq_3d[:, RIGHT_WRIST_IDX, :], axis=0), axis=1))
    #         motion_asymmetry = abs(left_motion - right_motion) / (left_motion + right_motion + 1e-6)
    #         features.append(motion_asymmetry)
    #     else:
    #         features.append(0.0)
        
    #     # ===== FEATURES 81-92: Current visibility (12 values) =====
    #     current_vis = visibility[-1, VISIBILITY_LANDMARKS]
    #     features.extend(current_vis)
        
    #     # ===== FEATURES 93-98: Mean visibility (6 values) =====
    #     mean_vis = np.mean(visibility[:, [11, 12, 13, 14, 15, 16]], axis=0)
    #     features.extend(mean_vis)
        
    #     # ===== FEATURES 99-100: Min wrist visibility (2 values) =====
    #     min_vis_left_wrist = np.min(visibility[:, 15])
    #     min_vis_right_wrist = np.min(visibility[:, 16])
    #     features.extend([min_vis_left_wrist, min_vis_right_wrist])
        
    #     assert len(features) == 100, f"Expected 100 features, got {len(features)}"
        
    #     return np.array(features, dtype=np.float32)
    
    def extract_sequence_features(
        self,
        video: np.ndarray,
        window_size: int = 5,
        stride: int = 2,
    ) -> np.ndarray:
        """Extract features from ALL windows of a video in one vectorized pass.

        Instead of looping over windows and calling extract_sequence_features()
        for each, this processes the entire video using numpy broadcasting.

        Args:
            video: Array of shape (n_frames, 92).
            window_size: Sliding window size.
            stride: Step between windows.

        Returns:
            Array of shape (n_windows, 100).
        """
        n_frames = len(video)
        if n_frames < window_size:
            return np.empty((0, 100), dtype=np.float32)

        n_landmarks = 23
        seq_4d = video.reshape(n_frames, n_landmarks, 4)
        seq_3d = seq_4d[:, :, :3]       # (n_frames, 23, 3)
        visibility = seq_4d[:, :, 3]     # (n_frames, 23)

        # Key joints: (n_frames, 6, 3) → (n_frames, 18)
        key_joints = seq_3d[:, KEY_JOINT_INDICES, :]
        kj_flat = key_joints.reshape(n_frames, -1)

        # Window start indices
        starts = np.arange(0, n_frames - window_size + 1, stride)
        n_windows = len(starts)
        if n_windows == 0:
            return np.empty((0, 100), dtype=np.float32)

        # End frame of each window (the "current" frame)
        ends = starts + window_size - 1  # (n_windows,)

        # Pre-allocate output
        out = np.zeros((n_windows, 100), dtype=np.float32)

        def window_mean(arr):
            '''
                mean over window for any per-frame array (n_frames, feat_dim)
                cumsum is implemented for performance
                extra frames (if remaining_frames < window_size) are dropped
                Returns (n_windows, feat_dim)    
            '''
            cumsum = np.vstack([np.zeros((1, arr.shape[1])), np.cumsum(arr, axis=0)])
            return (cumsum[ends + 1] - cumsum[starts]) / window_size

        # --- 1. Current pose (18) 
        out[:, 0:18] = window_mean(kj_flat)

        # --- 2. Velocity (20) --- mean velocity over window ---
        # NOTE: window_mean of per-frame velocity vectors = net displacement / window_size.
        # This is mathematically equivalent to (pos[end] - pos[start]) / window_size.
        # Direction changes within the window cancel out, so a hand moving right
        # then left shows ~zero mean velocity. The scalar speed features (36-37)
        # correctly average norms and capture sustained motion magnitude.
        frame_vels = np.zeros_like(kj_flat)
        frame_vels[1:] = kj_flat[1:] - kj_flat[:-1]   # (n_frames, 18)
        mean_vel = window_mean(frame_vels)              # (n_windows, 18)
        out[:, 18:36] = mean_vel
        lw_speed_seq = np.linalg.norm(frame_vels[:, 12:15], axis=1, keepdims=True)  # (n_frames, 1)
        rw_speed_seq = np.linalg.norm(frame_vels[:, 15:18], axis=1, keepdims=True)  # (n_frames, 1)
        out[:, 36] = window_mean(lw_speed_seq)[:, 0]  # left wrist mean speed
        out[:, 37] = window_mean(rw_speed_seq)[:, 0]  # right wrist mean speed

        # --- 3. Wrist ranges (6) ---
        # Sliding window view: (n_all_windows, window_size, 6) then take strided subset
        wrist_data = kj_flat[:, 12:18]  # (n_frames, 6)
        if window_size >= 3:
            wrist_windows = np.lib.stride_tricks.sliding_window_view(wrist_data, window_size, axis=0)  # (n_valid, 6, window_size)
            wrist_windows = wrist_windows[starts]  # (n_windows, 6, window_size)
            out[:, 38:44] = np.ptp(wrist_windows, axis=2)  # max - min per feature

        # --- 4. Finger features (18) --- mean over window ---
        # Per-frame finger positions relative to wrist
        l_wrist_seq = seq_3d[:, LEFT_WRIST_IDX, :]   # (n_frames, 3)
        r_wrist_seq = seq_3d[:, RIGHT_WRIST_IDX, :]

        l_has_seq = np.any(l_wrist_seq, axis=1, keepdims=True)  # (n_frames, 1)
        r_has_seq = np.any(r_wrist_seq, axis=1, keepdims=True)

        l_pinky_seq = np.where(l_has_seq, seq_3d[:, 17, :] - l_wrist_seq, 0.0)
        l_index_seq = np.where(l_has_seq, seq_3d[:, 19, :] - l_wrist_seq, 0.0)
        l_thumb_seq = np.where(l_has_seq, seq_3d[:, 21, :] - l_wrist_seq, 0.0)
        r_pinky_seq = np.where(r_has_seq, seq_3d[:, 18, :] - r_wrist_seq, 0.0)
        r_index_seq = np.where(r_has_seq, seq_3d[:, 20, :] - r_wrist_seq, 0.0)
        r_thumb_seq = np.where(r_has_seq, seq_3d[:, 22, :] - r_wrist_seq, 0.0)

        out[:, 44:47] = window_mean(l_pinky_seq)
        out[:, 47:50] = window_mean(l_index_seq)
        out[:, 50:53] = window_mean(l_thumb_seq)
        out[:, 53:56] = window_mean(r_pinky_seq)
        out[:, 56:59] = window_mean(r_index_seq)
        out[:, 59:62] = window_mean(r_thumb_seq)

        # --- 5. Finger distances (6) --- mean norm over window ---
        out[:, 62] = window_mean(np.linalg.norm(l_pinky_seq - l_thumb_seq, axis=1, keepdims=True))[:, 0]
        out[:, 63] = window_mean(np.linalg.norm(l_index_seq - l_thumb_seq, axis=1, keepdims=True))[:, 0]
        out[:, 64] = window_mean(np.linalg.norm(l_pinky_seq - l_index_seq, axis=1, keepdims=True))[:, 0]
        out[:, 65] = window_mean(np.linalg.norm(r_pinky_seq - r_thumb_seq, axis=1, keepdims=True))[:, 0]
        out[:, 66] = window_mean(np.linalg.norm(r_index_seq - r_thumb_seq, axis=1, keepdims=True))[:, 0]
        out[:, 67] = window_mean(np.linalg.norm(r_pinky_seq - r_index_seq, axis=1, keepdims=True))[:, 0]

        # --- 6. Wrist acceleration (2) --- mean norm over window ---
        frame_accels = np.zeros_like(kj_flat)
        frame_accels[2:] = kj_flat[2:] - 2 * kj_flat[1:-1] + kj_flat[:-2] # double derivate (n_frames, 18), first 2 frames stay 0
        out[:, 68] = window_mean(np.linalg.norm(frame_accels[:, 12:15], axis=1, keepdims=True))[:, 0]
        out[:, 69] = window_mean(np.linalg.norm(frame_accels[:, 15:18], axis=1, keepdims=True))[:, 0]

        # --- 7. Trajectory smoothness (2) --- std of speed over window ---
        if n_frames > 1 and window_size > 2:
            frame_vels_all = np.diff(kj_flat, axis=0)
            lw_speed = np.linalg.norm(frame_vels_all[:, 12:15], axis=1)
            rw_speed = np.linalg.norm(frame_vels_all[:, 15:18], axis=1)
            vel_win_size = window_size - 1
            if len(lw_speed) >= vel_win_size:
                lw_windows = np.lib.stride_tricks.sliding_window_view(lw_speed, vel_win_size)[starts]
                rw_windows = np.lib.stride_tricks.sliding_window_view(rw_speed, vel_win_size)[starts]
                out[:, 70] = np.std(lw_windows, axis=1)
                out[:, 71] = np.std(rw_windows, axis=1)

        # --- 8. Wrist height (y coordinate) (2) --- mean over window ---
        l_shoulder_seq = seq_3d[:, 11, :]
        r_shoulder_seq = seq_3d[:, 12, :]
        l_wrist_pos_seq = seq_3d[:, LEFT_WRIST_IDX, :]
        r_wrist_pos_seq = seq_3d[:, RIGHT_WRIST_IDX, :]

        l_sh_has_seq = np.any(l_shoulder_seq, axis=1, keepdims=True)
        r_sh_has_seq = np.any(r_shoulder_seq, axis=1, keepdims=True)

        l_height_seq = np.where(l_sh_has_seq, l_wrist_pos_seq[:, 1:2] - l_shoulder_seq[:, 1:2], 0.0)
        r_height_seq = np.where(r_sh_has_seq, r_wrist_pos_seq[:, 1:2] - r_shoulder_seq[:, 1:2], 0.0)
        out[:, 72] = window_mean(l_height_seq)[:, 0]
        out[:, 73] = window_mean(r_height_seq)[:, 0]

        # --- 9. Wrist spread (1) --- mean over window ---
        both_has_seq = np.any(l_wrist_seq, axis=1) & np.any(r_wrist_seq, axis=1)
        both_has_seq_2d = both_has_seq[:, None]
        spread_seq = np.where(both_has_seq_2d,
                            np.linalg.norm(l_wrist_pos_seq - r_wrist_pos_seq, axis=1, keepdims=True), 0.0)
        out[:, 74] = window_mean(spread_seq)[:, 0]

        # --- 10. Arm extension (2) --- mean over window ---
        l_ext_seq = np.where(l_sh_has_seq, np.linalg.norm(l_wrist_pos_seq - l_shoulder_seq, axis=1, keepdims=True), 0.0)
        r_ext_seq = np.where(r_sh_has_seq, np.linalg.norm(r_wrist_pos_seq - r_shoulder_seq, axis=1, keepdims=True), 0.0)
        out[:, 75] = window_mean(l_ext_seq)[:, 0]
        out[:, 76] = window_mean(r_ext_seq)[:, 0]

        # --- 11. Total motion (1) --- sum of wrist displacements over window ---
        if n_frames > 1:
            lw_disp = np.linalg.norm(np.diff(seq_3d[:, LEFT_WRIST_IDX, :], axis=0), axis=1)
            rw_disp = np.linalg.norm(np.diff(seq_3d[:, RIGHT_WRIST_IDX, :], axis=0), axis=1)
            total_disp = lw_disp + rw_disp # n_frames-1
            cs = np.concatenate([[0.0], np.cumsum(total_disp)])
            out[:, 77] = cs[ends] - cs[starts] # no need to do ends+1 since already offset by 1 due to total_disp being n_frames-1

        # --- 12. Position symmetry wrist (1) --- mean over window ---
        body_center_seq = (l_shoulder_seq + r_shoulder_seq) / 2
        l_rel_seq = l_wrist_pos_seq - body_center_seq
        r_rel_seq = r_wrist_pos_seq - body_center_seq
        mirror = np.array([-1, 1, 1], dtype=np.float32)
        sym_norm_seq = np.linalg.norm(l_rel_seq + r_rel_seq * mirror, axis=1, keepdims=True)
        sym_seq = np.where(both_has_seq_2d, 1.0 / (1.0 + sym_norm_seq), 0.0)
        out[:, 78] = window_mean(sym_seq)[:, 0]

        # --- 13. Motion asymmetry (1) --- mean over window ---
        if n_frames > 1:
            cs_lw = np.concatenate([[0.0], np.cumsum(lw_disp)])
            cs_rw = np.concatenate([[0.0], np.cumsum(rw_disp)])
            lm_sum = cs_lw[ends] - cs_lw[starts]
            rm_sum = cs_rw[ends] - cs_rw[starts]
            out[:, 79] = np.abs(lm_sum - rm_sum) / (lm_sum + rm_sum + 1e-6)

        # --- 14. Visibility (20) --- mean for current ---
        upper_vis = visibility[:, UPPER_BODY_VIS] # (n_frames, 6)
        out[:, 80:92] = window_mean(visibility[:, VISIBILITY_LANDMARKS])
        out[:, 92:98] = window_mean(upper_vis)
        
        # Min wrist visibility over window (2)
        lw_vis = visibility[:, 15]  # (n_frames,)
        rw_vis = visibility[:, 16]
        lw_vis_win = np.lib.stride_tricks.sliding_window_view(lw_vis, window_size)[starts]
        rw_vis_win = np.lib.stride_tricks.sliding_window_view(rw_vis, window_size)[starts]
        out[:, 98] = np.min(lw_vis_win, axis=1)
        out[:, 99] = np.min(rw_vis_win, axis=1)

        return out

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