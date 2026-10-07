"""Compute per-gesture kinematic features from world landmarks."""

import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass

from .mapping import joint_map
from .kinematics import calc_holds, compute_limb_kinematics
from .spatial import calc_mcneillian_space, calc_volume_size, calc_vert_height

@dataclass
class KinematicFeatures:
    """Data class to store comprehensive kinematic features for a gesture."""
    gesture_id: str
    video_id: str
    
    # Which hand was more active in this specific gesture
    active_hand: str  # 'L' or 'R'
    
    # Spatial features
    space_use: int
    mcneillian_max: float
    mcneillian_mode: int
    volume: float
    max_height: float
    
    # Temporal features
    duration: float
    hold_count: int
    hold_time: float
    hold_avg_duration: float
    
    # Submovement features
    hand_submovements: int
    hand_submovement_peaks: List[float]
    hand_mean_submovement_amplitude: float
    
    elbow_submovements: int
    elbow_mean_submovement_amplitude: float
    
    # Dynamic features
    hand_peak_speed: float
    hand_mean_speed: float
    hand_peak_acceleration: float
    hand_peak_deceleration: float
    hand_peak_jerk: float
    
    elbow_peak_speed: float
    elbow_mean_speed: float
    elbow_peak_acceleration: float
    elbow_peak_deceleration: float
    elbow_peak_jerk: float

def compute_kinematic_features(
    landmarks: np.ndarray,
    visibility: np.ndarray = None,
    fps: float = 25.0,
    gesture_id: str = "",
    video_id: str = ""
) -> KinematicFeatures:
    """
    Compute comprehensive kinematic features for a gesture using the more active hand.
    For each gesture, determines which hand was more active during that specific gesture.
    """
    # Convert landmarks to DataFrame format first
    df = pd.DataFrame()
    for joint in ['L_Wrist', 'R_Wrist', 'L_Elbow', 'R_Elbow', 'L_Shoulder', 'R_Shoulder', 
                 'Neck', 'MidHip', 'L_Eye', 'R_Eye', 'Nose']:
        df[joint] = [landmarks[i, joint_map[joint]] for i in range(len(landmarks))]
    
    # Analyze movement for this specific gesture
    left_hand = landmarks[:, joint_map['L_Wrist']]  # Left wrist
    right_hand = landmarks[:, joint_map['R_Wrist']]  # Right wrist
    
    # Calculate total movement (speed) for each hand
    left_speeds = np.linalg.norm(np.diff(left_hand, axis=0), axis=1)
    right_speeds = np.linalg.norm(np.diff(right_hand, axis=0), axis=1)
    
    # Apply visibility masking if available
    if visibility is not None:
        visibility_threshold = 0.5
        left_vis_mask = visibility[:-1, joint_map['L_Wrist']] >= visibility_threshold
        right_vis_mask = visibility[:-1, joint_map['R_Wrist']] >= visibility_threshold
        
        # Count frames where each hand is visible
        left_visible_frames = np.sum(visibility[:, joint_map['L_Wrist']] >= visibility_threshold)
        right_visible_frames = np.sum(visibility[:, joint_map['R_Wrist']] >= visibility_threshold)
        
        # Apply visibility masks
        left_speeds = left_speeds * left_vis_mask
        right_speeds = right_speeds * right_vis_mask
        
        # Normalize by number of visible frames to avoid bias
        left_total = np.sum(left_speeds) * (len(visibility) / max(left_visible_frames, 1))
        right_total = np.sum(right_speeds) * (len(visibility) / max(right_visible_frames, 1))
    else:
        left_total = np.sum(left_speeds)
        right_total = np.sum(right_speeds)
    
    # Select the more active hand for this specific gesture
    active_hand = 'L' if left_total > right_total else 'R'
    print(f"Gesture {gesture_id}: {active_hand} hand showed more movement")
    
    # Get keys for the active hand
    hand_key = 'L_Wrist' if active_hand == 'L' else 'R_Wrist'
    elbow_key = 'L_Elbow' if active_hand == 'L' else 'R_Elbow'
    
    # Calculate spatial features
    mcn_space = calc_mcneillian_space(df, visibility)
    space_use = mcn_space[0] if active_hand == 'L' else mcn_space[1]
    mcneillian_max = mcn_space[2] if active_hand == 'L' else mcn_space[3]
    mcneillian_mode = mcn_space[4] if active_hand == 'L' else mcn_space[5]
    
    # Calculate volume for active hand
    volume = calc_volume_size(df, active_hand)
    
    # Calculate max height for active hand
    max_heights = calc_vert_height(df, visibility)
    max_height = max_heights[0] if active_hand == 'L' else max_heights[1]
    
    # Compute kinematics for active arm
    hand = compute_limb_kinematics(np.array([p for p in df[hand_key]]), fps)
    elbow = compute_limb_kinematics(np.array([p for p in df[elbow_key]]), fps)
    
    # Calculate hold features using only active hand
    if active_hand == 'L':
        hold_peaks = hand.peaks
        other_peaks = np.array([])
    else:
        hold_peaks = np.array([])
        other_peaks = hand.peaks
        
    hold_count, hold_time, hold_avg = calc_holds(
        df, hold_peaks, other_peaks, fps, active_hand
    )
    
    # Safe computation helpers
    def safe_mean(arr): return float(np.mean(arr)) if len(arr) > 0 else 0.0
    def safe_max(arr): return float(np.max(arr)) if len(arr) > 0 else 0.0
    def safe_min(arr): return float(np.min(arr)) if len(arr) > 0 else 0.0
    def safe_norm(arr, axis=1): return np.linalg.norm(arr, axis=axis) if len(arr) > 0 else np.zeros(1)
    
    return KinematicFeatures(
        gesture_id=gesture_id,
        video_id=video_id,
        active_hand=active_hand,
        
        # Spatial features
        space_use=space_use,
        mcneillian_max=mcneillian_max,
        mcneillian_mode=mcneillian_mode,
        volume=volume,
        max_height=max_height,
        
        # Temporal features
        duration=len(landmarks) / fps,
        hold_count=hold_count,
        hold_time=hold_time,
        hold_avg_duration=hold_avg,
        
        # Hand submovements
        hand_submovements=len(hand.peaks),
        hand_submovement_peaks=hand.peak_heights.tolist() if len(hand.peak_heights) > 0 else [0],
        hand_mean_submovement_amplitude=safe_mean(hand.peak_heights),
        
        # Elbow submovements
        elbow_submovements=len(elbow.peaks),
        elbow_mean_submovement_amplitude=safe_mean(elbow.peak_heights),
        
        # Hand dynamics
        hand_peak_speed=safe_max(hand.speed),          # Changed from velocity to speed
        hand_mean_speed=safe_mean(hand.speed),         # Changed from velocity to speed
        hand_peak_acceleration=safe_max(safe_norm(hand.acceleration)),
        hand_peak_deceleration=safe_min(safe_norm(hand.acceleration)),
        hand_peak_jerk=safe_max(safe_norm(hand.jerk)),
        
        # Elbow dynamics
        elbow_peak_speed=safe_max(elbow.speed),        # Changed from velocity to speed
        elbow_mean_speed=safe_mean(elbow.speed),       # Changed from velocity to speed
        elbow_peak_acceleration=safe_max(safe_norm(elbow.acceleration)),
        elbow_peak_deceleration=safe_min(safe_norm(elbow.acceleration)),
        elbow_peak_jerk=safe_max(safe_norm(elbow.jerk))
    )
