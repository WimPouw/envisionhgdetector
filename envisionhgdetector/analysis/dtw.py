"""Compare tracked gestures with dynamic time warping."""

import warnings
import cv2
from math import isfinite
import numpy as np
import pandas as pd
import umap.umap_ as umap
from pathlib import Path
from typing import List, Tuple
from shapedtw.shapedtw import shape_dtw
from shapedtw.shapeDescriptors import RawSubsequenceDescriptor

from .features import extract_upper_limb_features, prepare_upper_limb_landmarks
from .gesture_kinematics import compute_kinematic_features
from envisionhgdetector.utils import filter_segment_source_paths, return_path

def create_gesture_visualization(
    dtw_matrix: np.ndarray,
    gesture_names: List[str],
    output_folder: str,
) -> None:
    """Save UMAP coordinates in the dataset output root's analysis directory."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        metric="precomputed",
    )
    projection = reducer.fit_transform(dtw_matrix)
    viz_df = pd.DataFrame({
        "x": projection[:, 0],
        "y": projection[:, 1],
        "gesture": gesture_names,
    })
    viz_df.to_csv(return_path("gesture_visualization", output_folder), index=False)


def compute_gesture_kinematics_dtw(
    data_folder: str,
    max_gap: int = 3,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Compute DTW distances between all gesture pairs and extract kinematic features.
    Kinematics use the frame rate read from each segment video.
    
    Args:
        data_folder: Dataset output root containing videos/ and analysis/.
        max_gap: Maximum consecutive missing frames allowed per coordinate.
        
    Returns:
        Tuple containing:
        - DTW distance matrix
        - List of gesture names
        - DataFrame of kinematic features
    """
    data_folder = Path(data_folder)
    if not data_folder.is_dir():
        raise ValueError(f"Data folder does not exist: {data_folder}")

    videos_folder = Path(return_path("videos_folder", data_folder, create_parents=False))
    segment_videos = filter_segment_source_paths(
        sorted(path for path in videos_folder.rglob("*.mp4") if path.is_file()),
    )
    if not segment_videos:
        raise ValueError(f"No segment videos found in {data_folder}")

    # Check every input before loading landmarks or computing any features/distances.
    landmark_files = []
    missing_landmarks = []
    for video_path in segment_videos:
        landmark_path = Path(return_path(
            "world_landmarks", data_folder, video_path.stem,
            segment_video_path=video_path, create_parents=False,
        ))
        if not landmark_path.is_file():
            missing_landmarks.append(f"{video_path} -> {landmark_path}")
        landmark_files.append((video_path, landmark_path))
    if missing_landmarks:
        raise FileNotFoundError(
            "Missing retracked world landmarks for segment videos. "
            "Retrack all segments before running DTW:\n" + "\n".join(missing_landmarks)
        )
    segment_inputs = []
    for video_path, landmark_path in landmark_files:
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open segment video: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid FPS {fps!r} for segment video: {video_path}")
        finally:
            cap.release()
        segment_inputs.append((video_path.stem, landmark_path, fps))

    gesture_data = []
    gesture_names = []
    kinematic_features = []

    for gesture_name, lm_path, fps in segment_inputs:
        try:
            landmarks = np.load(str(lm_path), allow_pickle=True) # TODO - do we need pickle?
            landmarks = prepare_upper_limb_landmarks(landmarks, max_gap=max_gap)
            features = extract_upper_limb_features(landmarks, max_gap=max_gap)
            kin_features = compute_kinematic_features(
                landmarks=landmarks,
                fps=fps,
                gesture_id=gesture_name,
                video_id=gesture_name
            )
        except (ValueError, TypeError, IndexError, OSError) as exc:
            warnings.warn(f"Skipping gesture {gesture_name}: {exc}", UserWarning, stacklevel=2)
            continue

        # TODO this can get very slow with many gestures. Consider parallelizing or optimizing the DTW computation.
        gesture_data.append(features)
        gesture_names.append(gesture_name)
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
    matrix_path = return_path("dtw_distances", data_folder)
    features_path = return_path("kinematic_features", data_folder)
    
    np.savetxt(matrix_path, dtw_dist, delimiter=',')
    features_df.to_csv(features_path, index=False)
    
    return dtw_dist, gesture_names, features_df
