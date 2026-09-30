"""Compare tracked gestures with dynamic time warping."""

import glob
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from shapedtw.shapedtw import shape_dtw
from shapedtw.shapeDescriptors import RawSubsequenceDescriptor

from .features import extract_upper_limb_features, remove_nans
from .gesture_kinematics import compute_kinematic_features


def compute_gesture_kinematics_dtw(
    tracked_folder: str,
    output_folder: str,
    fps: float = 25.0,
    landmark_pattern: str = "*_world_landmarks.npy"
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Compute DTW distances between all gesture pairs and extract kinematic features.
    
    Args:
        tracked_folder: Folder containing tracked landmark data
        output_folder: Folder to save DTW results
        fps: Frames per second of the video
        landmark_pattern: Pattern to match landmark files
        
    Returns:
        Tuple containing:
        - DTW distance matrix
        - List of gesture names
        - DataFrame of kinematic features
    """   
    os.makedirs(output_folder, exist_ok=True)
    
    # Load all landmark files
    landmark_files = glob.glob(os.path.join(tracked_folder, landmark_pattern))
    gesture_data = {}
    gesture_names = []
    kinematic_features = []
    
    for idx, lm_path in enumerate(landmark_files):
        landmarks = np.load(lm_path, allow_pickle=True)
        
        # Extract features for DTW
        features = extract_upper_limb_features(landmarks)
        features = remove_nans(features)
        
        gesture_data[idx] = features
        gesture_name = Path(lm_path).stem.replace('_world_landmarks', '')
        gesture_names.append(gesture_name)
        
        # Compute kinematic features
        video_id = gesture_name.split('_')[0]  # Assuming video ID is first part of filename
        kin_features = compute_kinematic_features(
            landmarks=landmarks,
            fps=fps,
            gesture_id=gesture_name,
            video_id=video_id
        )
        kinematic_features.append(kin_features)
    
    num_gestures = len(gesture_data)
    dtw_dist = np.zeros((num_gestures, num_gestures))
    
    # Compute DTW distances
    for i in range(num_gestures):
        for j in range(i + 1, num_gestures):
            try:
                result = shape_dtw(
                    x=gesture_data[i],
                    y=gesture_data[j],
                    subsequence_width=4,
                    shape_descriptor=RawSubsequenceDescriptor(),
                    multivariate_version="dependent"
                )
                distance = result.normalized_distance
                dtw_dist[i, j] = distance
                dtw_dist[j, i] = distance
            except Exception as e:
                print(f"Error computing DTW for gestures {gesture_names[i]} and {gesture_names[j]}: {e}")
                dtw_dist[i, j] = np.nan
                dtw_dist[j, i] = np.nan
    
    # Convert kinematic features to DataFrame
    features_df = pd.DataFrame([{
        'gesture_id': f.gesture_id,
        'video_id': f.video_id,
        'active_hand': f.active_hand,
        'space_use': f.space_use,
        'mcneillian_max': f.mcneillian_max,
        'mcneillian_mode': f.mcneillian_mode,
        'volume': f.volume,
        'max_height': f.max_height,
        'duration': f.duration,
        'hold_count': f.hold_count,
        'hold_time': f.hold_time,
        'hold_avg_duration': f.hold_avg_duration,
        'hand_submovements': f.hand_submovements,
        'hand_submovement_peak_max': max(f.hand_submovement_peaks) if f.hand_submovement_peaks else 0,
        'hand_submovement_peak_mean': sum(f.hand_submovement_peaks)/len(f.hand_submovement_peaks) if f.hand_submovement_peaks else 0,
        'hand_mean_submovement_amplitude': f.hand_mean_submovement_amplitude,
        'elbow_submovements': f.elbow_submovements,
        'elbow_mean_submovement_amplitude': f.elbow_mean_submovement_amplitude,
        'hand_peak_speed': f.hand_peak_speed,
        'hand_mean_speed': f.hand_mean_speed,
        'hand_peak_acceleration': f.hand_peak_acceleration,
        'hand_peak_deceleration': f.hand_peak_deceleration,
        'hand_peak_jerk': f.hand_peak_jerk,
        'elbow_peak_speed': f.elbow_peak_speed,
        'elbow_mean_speed': f.elbow_mean_speed,
        'elbow_peak_acceleration': f.elbow_peak_acceleration,
        'elbow_peak_deceleration': f.elbow_peak_deceleration,
        'elbow_peak_jerk': f.elbow_peak_jerk
    } for f in kinematic_features])
    
    # Save results
    matrix_path = os.path.join(output_folder, "dtw_distances.csv")
    features_path = os.path.join(output_folder, "kinematic_features.csv")
    
    np.savetxt(matrix_path, dtw_dist, delimiter=',')
    features_df.to_csv(features_path, index=False)
    
    return dtw_dist, gesture_names, features_df
