# envisionhgdetector/model_combined.py
"""
Combined CNN + LightGBM gesture detection model.
Single MediaPipe pass feeds both models for efficient inference.

Both models run independently and output their results separately.
User can compare predictions from both models.

CNN: 3-class (NoGesture, Gesture, Move) - 25 frame window, 92 world landmarks
LightGBM: 2-class (NoGesture, Gesture) - 5 frame window, 100 engineered features

Output includes:
- CNN predictions: class, confidence, has_motion, gesture_prob, move_prob
- LightGBM predictions: class, confidence, gesture_prob, nogesture_prob
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from typing import Optional, List, Dict, Any, Tuple, Literal
from collections import deque
from dataclasses import dataclass
import tensorflow as tf

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# CONSTANTS
# ============================================================================

# Upper body landmark indices (23 landmarks)
UPPER_BODY_INDICES = list(range(23))

# Key joint indices for LightGBM features
KEY_JOINT_INDICES = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16

# Visibility landmark indices
VISIBILITY_LANDMARKS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
UPPER_BODY_VIS = np.array([11, 12, 13, 14, 15, 16])

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CombinedConfig:
    """Configuration for combined model inference."""
    
    # CNN settings
    cnn_seq_length: int = 25
    cnn_num_features: int = 92
    cnn_weights_path: Optional[str] = None
    cnn_motion_threshold: float = 0.5
    cnn_gesture_threshold: float = 0.5
    
    # LightGBM settings
    lgbm_window_size: int = 5
    lgbm_num_features: int = 100
    lgbm_weights_path: Optional[str] = None
    lgbm_threshold: float = 0.5
    
    # Post-processing (shared)
    min_gap_s: float = 0.5
    min_length_s: float = 0.5
    
    # Labels
    cnn_labels: Tuple[str, str, str] = ("NoGesture", "Gesture", "Move")
    lgbm_labels: Tuple[str, str] = ("NoGesture", "Gesture")


# ============================================================================
# LIGHTGBM FEATURE EXTRACTION (100 features)
# ============================================================================
# OLD IMPLEMENTATION
# def extract_lgbm_features(sequence: np.ndarray) -> np.ndarray:
#     """
#     Extract 100 features from a sequence of world landmarks for LightGBM.
    
#     Args:
#         sequence: Array of shape (window_size, 92) - world landmarks
    
#     Returns:
#         100-dimensional feature vector
#     """
#     if len(sequence) == 0:
#         return np.zeros(100, dtype=np.float32)
    
#     n_frames = len(sequence)
#     n_landmarks = 23
    
#     try:
#         seq_4d = sequence.reshape(n_frames, n_landmarks, 4)
#     except ValueError:
#         return np.zeros(100, dtype=np.float32)
    
#     seq_3d = seq_4d[:, :, :3]
#     visibility = seq_4d[:, :, 3]
    
#     features = []
    
#     key_joints = seq_3d[:, KEY_JOINT_INDICES, :]
#     key_joints_flat = key_joints.reshape(n_frames, -1)
    
#     # 1-18: Current pose
#     current_pose = key_joints_flat[-1]
#     features.extend(current_pose)
    
#     # 19-38: Velocity + wrist speeds
#     if n_frames > 1:
#         velocity = key_joints_flat[-1] - key_joints_flat[-2]
#         features.extend(velocity)
#         left_wrist_speed = np.linalg.norm(velocity[12:15])
#         right_wrist_speed = np.linalg.norm(velocity[15:18])
#         features.extend([left_wrist_speed, right_wrist_speed])
#     else:
#         features.extend([0.0] * 20)
    
#     # 39-44: Wrist ranges
#     if n_frames >= 3:
#         wrist_data = key_joints_flat[:, 12:18]
#         wrist_ranges = np.ptp(wrist_data, axis=0)
#         features.extend(wrist_ranges)
#     else:
#         features.extend([0.0] * 6)
    
#     # 45-62: Finger features
#     left_wrist = seq_3d[-1, LEFT_WRIST_IDX, :]
#     right_wrist = seq_3d[-1, RIGHT_WRIST_IDX, :]
    
#     left_fingers = np.zeros(9, dtype=np.float32)
#     right_fingers = np.zeros(9, dtype=np.float32)
    
#     if np.any(left_wrist):
#         left_fingers[0:3] = seq_3d[-1, 17, :] - left_wrist
#         left_fingers[3:6] = seq_3d[-1, 19, :] - left_wrist
#         left_fingers[6:9] = seq_3d[-1, 21, :] - left_wrist
    
#     if np.any(right_wrist):
#         right_fingers[0:3] = seq_3d[-1, 18, :] - right_wrist
#         right_fingers[3:6] = seq_3d[-1, 20, :] - right_wrist
#         right_fingers[6:9] = seq_3d[-1, 22, :] - right_wrist
    
#     features.extend(left_fingers)
#     features.extend(right_fingers)
    
#     # 63-68: Finger distances
#     left_pinky_thumb = np.linalg.norm(left_fingers[0:3] - left_fingers[6:9]) if np.any(left_fingers) else 0.0
#     left_index_thumb = np.linalg.norm(left_fingers[3:6] - left_fingers[6:9]) if np.any(left_fingers) else 0.0
#     left_pinky_index = np.linalg.norm(left_fingers[0:3] - left_fingers[3:6]) if np.any(left_fingers) else 0.0
#     right_pinky_thumb = np.linalg.norm(right_fingers[0:3] - right_fingers[6:9]) if np.any(right_fingers) else 0.0
#     right_index_thumb = np.linalg.norm(right_fingers[3:6] - right_fingers[6:9]) if np.any(right_fingers) else 0.0
#     right_pinky_index = np.linalg.norm(right_fingers[0:3] - right_fingers[3:6]) if np.any(right_fingers) else 0.0
    
#     features.extend([left_pinky_thumb, left_index_thumb, left_pinky_index,
#                     right_pinky_thumb, right_index_thumb, right_pinky_index])
    
#     # 69-70: Wrist acceleration
#     if n_frames > 2:
#         vel_prev = key_joints_flat[-2] - key_joints_flat[-3]
#         vel_curr = key_joints_flat[-1] - key_joints_flat[-2]
#         accel = vel_curr - vel_prev
#         left_wrist_accel = np.linalg.norm(accel[12:15])
#         right_wrist_accel = np.linalg.norm(accel[15:18])
#         features.extend([left_wrist_accel, right_wrist_accel])
#     else:
#         features.extend([0.0, 0.0])
    
#     # 71-72: Trajectory smoothness
#     if n_frames > 2:
#         velocities = np.diff(key_joints_flat, axis=0)
#         left_wrist_vels = velocities[:, 12:15]
#         right_wrist_vels = velocities[:, 15:18]
#         left_smoothness = np.std(np.linalg.norm(left_wrist_vels, axis=1))
#         right_smoothness = np.std(np.linalg.norm(right_wrist_vels, axis=1))
#         features.extend([left_smoothness, right_smoothness])
#     else:
#         features.extend([0.0, 0.0])
    
#     # 73-74: Wrist height
#     left_shoulder = seq_3d[-1, 11, :]
#     right_shoulder = seq_3d[-1, 12, :]
#     left_wrist_pos = seq_3d[-1, LEFT_WRIST_IDX, :]
#     right_wrist_pos = seq_3d[-1, RIGHT_WRIST_IDX, :]
    
#     left_wrist_height = left_wrist_pos[1] - left_shoulder[1] if np.any(left_shoulder) else 0.0
#     right_wrist_height = right_wrist_pos[1] - right_shoulder[1] if np.any(right_shoulder) else 0.0
#     features.extend([left_wrist_height, right_wrist_height])
    
#     # 75: Wrist spread
#     wrist_spread = np.linalg.norm(left_wrist_pos - right_wrist_pos) if np.any(left_wrist_pos) and np.any(right_wrist_pos) else 0.0
#     features.append(wrist_spread)
    
#     # 76-77: Arm extension
#     left_arm_extension = np.linalg.norm(left_wrist_pos - left_shoulder) if np.any(left_shoulder) else 0.0
#     right_arm_extension = np.linalg.norm(right_wrist_pos - right_shoulder) if np.any(right_shoulder) else 0.0
#     features.extend([left_arm_extension, right_arm_extension])
    
#     # 78: Total motion
#     if n_frames > 1:
#         total_motion = np.sum(np.linalg.norm(np.diff(seq_3d[:, LEFT_WRIST_IDX, :], axis=0), axis=1))
#         total_motion += np.sum(np.linalg.norm(np.diff(seq_3d[:, RIGHT_WRIST_IDX, :], axis=0), axis=1))
#         features.append(total_motion)
#     else:
#         features.append(0.0)
    
#     # 79: Position symmetry
#     if np.any(left_wrist_pos) and np.any(right_wrist_pos):
#         body_center = (left_shoulder + right_shoulder) / 2
#         left_rel = left_wrist_pos - body_center
#         right_rel = right_wrist_pos - body_center
#         symmetry = 1.0 / (1.0 + np.linalg.norm(left_rel + right_rel * np.array([-1, 1, 1])))
#         features.append(symmetry)
#     else:
#         features.append(0.0)
    
#     # 80: Motion asymmetry
#     if n_frames > 1:
#         left_motion = np.sum(np.linalg.norm(np.diff(seq_3d[:, LEFT_WRIST_IDX, :], axis=0), axis=1))
#         right_motion = np.sum(np.linalg.norm(np.diff(seq_3d[:, RIGHT_WRIST_IDX, :], axis=0), axis=1))
#         motion_asymmetry = abs(left_motion - right_motion) / (left_motion + right_motion + 1e-6)
#         features.append(motion_asymmetry)
#     else:
#         features.append(0.0)
    
#     # 81-92: Current visibility
#     current_vis = visibility[-1, VISIBILITY_LANDMARKS]
#     features.extend(current_vis)
    
#     # 93-98: Mean visibility
#     mean_vis = np.mean(visibility[:, [11, 12, 13, 14, 15, 16]], axis=0)
#     features.extend(mean_vis)
    
#     # 99-100: Min wrist visibility
#     min_vis_left_wrist = np.min(visibility[:, 15])
#     min_vis_right_wrist = np.min(visibility[:, 16])
#     features.extend([min_vis_left_wrist, min_vis_right_wrist])
    
#     return np.array(features[:100], dtype=np.float32)

def extract_lgbm_features(
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

# ============================================================================
# CNN MODEL BUILDER
# ============================================================================

def build_cnn_model(
    num_features: int = 92,
    seq_length: int = 25,
    num_gesture_classes: int = 2,
    conv_filters: Tuple[int, int, int] = (48, 96, 192),
    dense_units: int = 256,
    dropout_rate: float = 0.36,
    l2_weight: float = 0.0002
) -> tf.keras.Model:
    """Build CNN model matching training architecture."""
    from tensorflow.keras import layers, regularizers, Model
    
    inputs = layers.Input(shape=(seq_length, num_features), name="input")
    
    # Basic preprocessing (noise during training only)
    x = inputs  # No preprocessing layer needed for inference
    
    # Residual blocks
    for i, filters in enumerate(conv_filters):
        shortcut = x
        
        x = layers.Conv1D(filters, 3, padding="same",
                         kernel_regularizer=regularizers.l2(l2_weight))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SpatialDropout1D(0.1)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
        
        shortcut = layers.Conv1D(filters, 1, strides=2, padding="same",
                                kernel_regularizer=regularizers.l2(l2_weight))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
    
    # Head
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(dense_units, activation="relu",
                    kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Hierarchical output
    has_motion = layers.Dense(1, activation="sigmoid", name="has_motion")(x)
    gesture_probs = layers.Dense(num_gesture_classes, activation="softmax", name="gesture_probs")(x)
    outputs = layers.Concatenate(name="output")([has_motion, gesture_probs])
    
    return Model(inputs, outputs)


# ============================================================================
# COMBINED MODEL CLASS
# ============================================================================

class CombinedGestureModel:
    """
    Combined CNN + LightGBM gesture detection.
    Single MediaPipe pass feeds both models.
    """
    
    def __init__(self, config: Optional[CombinedConfig] = None):
        """
        Initialize combined model.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or CombinedConfig()
        
        # Initialize MediaPipe (single instance for both models)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark buffer (stores raw 92-feature world landmarks)
        # Sized for CNN (25 frames), LightGBM uses last 5
        self.landmarks_buffer = deque(maxlen=self.config.cnn_seq_length)
        
        # Load models
        self.cnn_model = None
        self.lgbm_model = None
        self.lgbm_scaler = None
        self.lgbm_label_encoder = None
        
        self._load_cnn_model()
        self._load_lgbm_model()
        
        print(f"\n✓ Combined model initialized")
        print(f"  CNN thresholds: motion={self.config.cnn_motion_threshold}, gesture={self.config.cnn_gesture_threshold}")
        print(f"  LightGBM threshold: {self.config.lgbm_threshold}")
        print(f"  CNN ready: {self.cnn_model is not None}")
        print(f"  LightGBM ready: {self.lgbm_model is not None}")
    
    def _find_model_path(self, model_type: str) -> Optional[str]:
        """Find model file path."""
        base_dir = os.path.dirname(__file__)
        
        if model_type == 'cnn':
            paths = [
                self.config.cnn_weights_path,
                os.path.join(base_dir, 'model', 'cnn_default.h5'),
                os.path.join(base_dir, 'model', 'R2_CNN_world_best.h5'),
                os.path.join(base_dir, 'model', 'best_model_final.h5'),
            ]
        else:  # lightgbm
            paths = [
                self.config.lgbm_weights_path,
                os.path.join(base_dir, 'model', 'lightgbm_default.pkl'),
                os.path.join(base_dir, 'model', 'best_lightgbm_model.pkl'),
                os.path.join(base_dir, 'model', 'lightgbm_gesture_model_v2.pkl'),
            ]
        
        for path in paths:
            if path and os.path.exists(path):
                return path
        return None
    
    def _load_cnn_model(self):
        """Load CNN model."""
        weights_path = self._find_model_path('cnn')
        
        if weights_path is None:
            print("Warning: CNN weights not found")
            return
        
        try:
            self.cnn_model = build_cnn_model(
                num_features=self.config.cnn_num_features,
                seq_length=self.config.cnn_seq_length
            )
            self.cnn_model.load_weights(weights_path)
            print(f"✓ CNN model loaded from {weights_path}")
        except Exception as e:
            print(f"Warning: Failed to load CNN model: {e}")
            self.cnn_model = None
    
    def _load_lgbm_model(self):
        """Load LightGBM model."""
        model_path = self._find_model_path('lightgbm')
        
        if model_path is None:
            print("Warning: LightGBM model not found")
            return
        
        try:
            model_data = joblib.load(model_path)
            self.lgbm_model = model_data['model']
            self.lgbm_scaler = model_data['scaler']
            self.lgbm_label_encoder = model_data.get('label_encoder')
            
            # Print class order for debugging
            if self.lgbm_label_encoder is not None:
                print(f"✓ LightGBM model loaded from {model_path}")
            else:
                print(f"✓ LightGBM model loaded from {model_path}")
                print(f"  Warning: No label_encoder found, assuming alphabetical order")
        except Exception as e:
            print(f"Warning: Failed to load LightGBM model: {e}")
            self.lgbm_model = None
    
    def extract_world_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract world landmarks from frame.
        
        Returns:
            Array of 92 features (23 landmarks × 4) or None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = self.holistic.process(rgb_frame)
        
        if not results.pose_world_landmarks:
            return None
        
        features = []
        for idx in UPPER_BODY_INDICES:
            if idx < len(results.pose_world_landmarks.landmark):
                lm = results.pose_world_landmarks.landmark[idx]
                features.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a single frame and get predictions from both models.
        
        Args:
            frame: BGR video frame
            
        Returns:
            Dictionary with separate predictions from CNN and LightGBM, or None if not ready
        """
        # Extract landmarks (single MediaPipe call)
        landmarks = self.extract_world_landmarks(frame)
        
        if landmarks is None:
            return None
        
        # Add to buffer
        self.landmarks_buffer.append(landmarks)
        
        # Check if we have enough frames for CNN
        if len(self.landmarks_buffer) < self.config.cnn_seq_length:
            return None
        
        results = {
            'cnn': None,
            'lightgbm': None
        }
        
        # Get full sequence for CNN (25 frames)
        cnn_sequence = np.array(list(self.landmarks_buffer))
        
        # Get last 5 frames for LightGBM
        lgbm_sequence = cnn_sequence[-self.config.lgbm_window_size:]
        
        # ===== CNN Prediction =====
        if self.cnn_model is not None:
            cnn_input = cnn_sequence.reshape(1, self.config.cnn_seq_length, self.config.cnn_num_features)
            cnn_pred = self.cnn_model.predict(cnn_input, verbose=0)[0]
            
            # Parse CNN output: [has_motion, gesture_prob, move_prob]
            cnn_has_motion = float(cnn_pred[0])
            cnn_gesture_prob = float(cnn_pred[1])
            cnn_move_prob = float(cnn_pred[2])
            
            # Determine class using thresholds
            if cnn_has_motion < self.config.cnn_motion_threshold:
                cnn_class = "NoGesture"
                cnn_confidence = 1 - cnn_has_motion
            elif cnn_gesture_prob > cnn_move_prob:
                cnn_class = "Gesture"
                cnn_confidence = cnn_has_motion * cnn_gesture_prob
            else:
                cnn_class = "Move"
                cnn_confidence = cnn_has_motion * cnn_move_prob
            
            results['cnn'] = {
                'class': cnn_class,
                'confidence': cnn_confidence,
                'has_motion': cnn_has_motion,
                'gesture_prob': cnn_gesture_prob,
                'move_prob': cnn_move_prob,
            }
        
        # ===== LightGBM Prediction =====
        if self.lgbm_model is not None:
            lgbm_features = extract_lgbm_features(lgbm_sequence)
            lgbm_features_scaled = self.lgbm_scaler.transform(lgbm_features.reshape(1, -1))
            
            # Handle both Booster and Classifier objects
            if hasattr(self.lgbm_model, 'predict_proba'):
                # LGBMClassifier - returns probabilities directly
                lgbm_probs = self.lgbm_model.predict_proba(lgbm_features_scaled)[0]
            else:
                # Raw Booster trained with multiclass objective
                # predict() already returns probabilities with shape (n_samples, n_classes)
                raw_output = self.lgbm_model.predict(lgbm_features_scaled)
                raw_output = np.array(raw_output)
                
                # For multiclass, output is already probabilities (n_samples, n_classes)
                if raw_output.ndim == 2:
                    lgbm_probs = raw_output[0]  # Get first sample
                elif raw_output.ndim == 1 and len(raw_output) == 2:
                    # Already probabilities for 2 classes
                    lgbm_probs = raw_output
                else:
                    # Single value - binary, apply sigmoid
                    raw_score = float(raw_output.flatten()[0])
                    gesture_prob = 1.0 / (1.0 + np.exp(-raw_score))
                    lgbm_probs = np.array([1 - gesture_prob, gesture_prob])
            
            # LightGBM: LabelEncoder sorts alphabetically!
            # So class order is: [Gesture_prob, NoGesture_prob] (alphabetical)
            lgbm_gesture_prob = float(lgbm_probs[0])      # Class 0 = Gesture
            lgbm_nogesture_prob = float(lgbm_probs[1])    # Class 1 = NoGesture
            
            # Determine class using threshold
            if lgbm_gesture_prob >= self.config.lgbm_threshold:
                lgbm_class = "Gesture"
                lgbm_confidence = lgbm_gesture_prob
            else:
                lgbm_class = "NoGesture"
                lgbm_confidence = lgbm_nogesture_prob
            
            results['lightgbm'] = {
                'class': lgbm_class,
                'confidence': lgbm_confidence,
                'nogesture_prob': lgbm_nogesture_prob,
                'gesture_prob': lgbm_gesture_prob,
            }
        
        return results
    
    def predict_video(
        self,
        video_path: str,
        target_fps: int = 25,
        return_all: bool = False
    ) -> Dict[str, Any]:
        """
        Process entire video and return predictions from both models.
        
        Args:
            video_path: Path to video file
            target_fps: Target FPS for processing
            return_all: If True, return predictions for all frames
            
        Returns:
            Dictionary with frame-by-frame predictions from both models
        """
        from tqdm import tqdm
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, int(original_fps / target_fps))
        
        self.reset_buffer()
        
        all_predictions = []
        frame_idx = 0
        processed_idx = 0
        
        # Progress bar
        pbar = tqdm(total=total_frames, desc="Processing frames", unit="frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                result = self.process_frame(frame)
                
                if result is not None:
                    result['frame_idx'] = frame_idx
                    result['time_s'] = frame_idx / original_fps
                    all_predictions.append(result)
                
                processed_idx += 1
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Build summary
        summary = {
            'video_path': video_path,
            'total_frames': frame_idx,
            'processed_frames': len(all_predictions),
            'original_fps': original_fps,
            'target_fps': target_fps,
            'predictions': all_predictions if return_all else None,
            'cnn_available': self.cnn_model is not None,
            'lgbm_available': self.lgbm_model is not None,
        }
        
        return summary
    
    def reset_buffer(self):
        """Clear the landmark buffer."""
        self.landmarks_buffer.clear()
    
    def set_thresholds(
        self,
        cnn_motion_threshold: Optional[float] = None,
        cnn_gesture_threshold: Optional[float] = None,
        lgbm_threshold: Optional[float] = None
    ):
        """Update thresholds for both models."""
        if cnn_motion_threshold is not None:
            self.config.cnn_motion_threshold = cnn_motion_threshold
        if cnn_gesture_threshold is not None:
            self.config.cnn_gesture_threshold = cnn_gesture_threshold
        if lgbm_threshold is not None:
            self.config.lgbm_threshold = lgbm_threshold
        print(f"Thresholds updated: CNN motion={self.config.cnn_motion_threshold}, "
              f"CNN gesture={self.config.cnn_gesture_threshold}, "
              f"LightGBM={self.config.lgbm_threshold}")
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'holistic') and self.holistic:
            self.holistic.close()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_combined_model(
    cnn_weights: Optional[str] = None,
    lgbm_weights: Optional[str] = None,
    cnn_motion_threshold: float = 0.5,
    cnn_gesture_threshold: float = 0.5,
    lgbm_threshold: float = 0.5
) -> CombinedGestureModel:
    """
    Load combined model with custom weights and thresholds.
    
    Args:
        cnn_weights: Path to CNN weights (.h5)
        lgbm_weights: Path to LightGBM model (.pkl)
        cnn_motion_threshold: Motion threshold for CNN
        cnn_gesture_threshold: Gesture threshold for CNN
        lgbm_threshold: Confidence threshold for LightGBM
        
    Returns:
        CombinedGestureModel instance
    """
    config = CombinedConfig(
        cnn_weights_path=cnn_weights,
        lgbm_weights_path=lgbm_weights,
        cnn_motion_threshold=cnn_motion_threshold,
        cnn_gesture_threshold=cnn_gesture_threshold,
        lgbm_threshold=lgbm_threshold
    )
    return CombinedGestureModel(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Combined CNN + LightGBM Gesture Detection")
    print("=" * 50)
    print("Both models run independently - compare results yourself!")
    
    # Load model
    model = CombinedGestureModel()
    
    # Process video
    # results = model.predict_video("test_video.mp4")
    # print(f"Processed {results['processed_frames']} frames")
    
    # Or process frame by frame
    # cap = cv2.VideoCapture(0)  # Webcam
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     
    #     result = model.process_frame(frame)
    #     if result:
    #         print(f"CNN: {result['cnn']['class']} ({result['cnn']['confidence']:.2f})")
    #         print(f"LightGBM: {result['lightgbm']['class']} ({result['lightgbm']['confidence']:.2f})")