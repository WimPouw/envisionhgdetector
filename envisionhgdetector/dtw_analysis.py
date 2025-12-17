# envisionhgdetector/dtw_analysis.py
"""
Dynamic Time Warping (DTW) analysis utilities for gesture detection.
Computes similarity between gestures and creates visualizations.
"""

import glob
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import umap.umap_ as umap
from shapedtw.shapedtw import shape_dtw
from shapedtw.shapeDescriptors import RawSubsequenceDescriptor

from .kinematics import compute_kinematic_features
from .tracking import extract_upper_limb_features, remove_nans


class DTWAnalysisError(Exception):
    """Exception raised for errors during DTW analysis."""
    pass


def validate_folder_path(folder: str, name: str = "folder", must_exist: bool = True) -> str:
    """
    Validate a folder path.

    Args:
        folder: Path to validate
        name: Parameter name for error messages
        must_exist: If True, verify folder exists

    Returns:
        Validated path string

    Raises:
        DTWAnalysisError: If validation fails
    """
    if not isinstance(folder, str):
        raise DTWAnalysisError(
            f"{name} must be a string, got {type(folder).__name__}"
        )

    if not folder.strip():
        raise DTWAnalysisError(f"{name} path cannot be empty")

    if must_exist:
        if not os.path.exists(folder):
            raise DTWAnalysisError(f"{name} not found: {folder}")

        if not os.path.isdir(folder):
            raise DTWAnalysisError(f"{name} is not a directory: {folder}")

    return folder


def validate_fps(fps: Union[int, float]) -> float:
    """
    Validate frames per second value.

    Args:
        fps: FPS value to validate

    Returns:
        Validated float value

    Raises:
        DTWAnalysisError: If validation fails
    """
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        raise DTWAnalysisError(
            f"fps must be a number, got {type(fps).__name__}"
        )

    if fps <= 0:
        raise DTWAnalysisError(f"fps must be positive, got {fps}")

    return fps


def compute_gesture_kinematics_dtw(
    tracked_folder: str,
    output_folder: str,
    fps: float = 25.0,
    landmark_pattern: str = "*_world_landmarks.npy"
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Compute DTW distances between all gesture pairs and extract kinematic features.

    This function loads tracked landmark data, extracts upper limb features,
    computes pairwise DTW distances, and extracts kinematic features for
    each gesture.

    Args:
        tracked_folder: Folder containing tracked landmark data (.npy files)
        output_folder: Folder to save DTW results
        fps: Frames per second of the video (must be > 0)
        landmark_pattern: Pattern to match landmark files

    Returns:
        Tuple containing:
        - DTW distance matrix (symmetric, with zeros on diagonal)
        - List of gesture names (corresponding to matrix indices)
        - DataFrame of kinematic features for each gesture

    Raises:
        DTWAnalysisError: If input validation fails or no landmark files found
    """
    # Validate inputs
    tracked_folder = validate_folder_path(tracked_folder, "tracked_folder", must_exist=True)
    output_folder = validate_folder_path(output_folder, "output_folder", must_exist=False)
    fps = validate_fps(fps)

    # Create output folder
    try:
        os.makedirs(output_folder, exist_ok=True)
    except OSError as e:
        raise DTWAnalysisError(f"Cannot create output folder {output_folder}: {e}")

    # Load all landmark files
    landmark_files = glob.glob(os.path.join(tracked_folder, landmark_pattern))
    gesture_data = {}
    gesture_names = []
    kinematic_features = []

    print(f"Found {len(landmark_files)} landmark files")

    for idx, lm_path in enumerate(landmark_files):
        landmarks = np.load(lm_path, allow_pickle=True)

        # Extract features for DTW
        features = extract_upper_limb_features(landmarks)
        features = remove_nans(features)

        gesture_data[idx] = features
        gesture_name = Path(lm_path).stem.replace('_world_landmarks', '')
        gesture_names.append(gesture_name)

        # Compute kinematic features
        video_id = gesture_name.split('_')[0]
        kin_features = compute_kinematic_features(
            landmarks=landmarks,
            fps=fps,
            gesture_id=gesture_name,
            video_id=video_id
        )
        kinematic_features.append(kin_features)

    num_gestures = len(gesture_data)
    dtw_dist = np.zeros((num_gestures, num_gestures))

    print(f"Computing DTW distances for {num_gestures} gestures...")

    # Compute DTW distances (symmetric matrix)
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
                print(f"Error computing DTW for {gesture_names[i]} and {gesture_names[j]}: {e}")
                dtw_dist[i, j] = np.nan
                dtw_dist[j, i] = np.nan

    # Convert kinematic features to DataFrame
    features_df = _kinematic_features_to_dataframe(kinematic_features)

    # Save results
    matrix_path = os.path.join(output_folder, "dtw_distances.csv")
    features_path = os.path.join(output_folder, "kinematic_features.csv")

    np.savetxt(matrix_path, dtw_dist, delimiter=',')
    features_df.to_csv(features_path, index=False)

    print(f"DTW matrix saved to: {matrix_path}")
    print(f"Kinematic features saved to: {features_path}")

    return dtw_dist, gesture_names, features_df


def _kinematic_features_to_dataframe(kinematic_features: List) -> pd.DataFrame:
    """
    Convert list of KinematicFeatures objects to DataFrame.

    Args:
        kinematic_features: List of KinematicFeatures dataclass instances

    Returns:
        DataFrame with one row per gesture
    """
    return pd.DataFrame([{
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
        'hand_submovement_peak_mean': (
            sum(f.hand_submovement_peaks) / len(f.hand_submovement_peaks)
            if f.hand_submovement_peaks else 0
        ),
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


def create_gesture_visualization(
    dtw_matrix: np.ndarray,
    gesture_names: List[str],
    output_folder: str,
    n_neighbors: int = 15
) -> None:
    """
    Create UMAP visualization from DTW distances.

    UMAP (Uniform Manifold Approximation and Projection) is used to create
    a 2D embedding of gestures based on their DTW distances. Gestures that
    are kinematically similar will appear close together in the visualization.

    Args:
        dtw_matrix: Symmetric DTW distance matrix
        gesture_names: List of gesture names corresponding to matrix indices
        output_folder: Folder to save visualization data
        n_neighbors: Number of neighbors for UMAP (affects local vs global structure)

    Raises:
        DTWAnalysisError: If input validation fails
    """
    # Validate inputs
    if not isinstance(dtw_matrix, np.ndarray):
        raise DTWAnalysisError(
            f"dtw_matrix must be a numpy array, got {type(dtw_matrix).__name__}"
        )

    if dtw_matrix.size == 0:
        raise DTWAnalysisError("dtw_matrix cannot be empty")

    if dtw_matrix.ndim != 2:
        raise DTWAnalysisError(
            f"dtw_matrix must be 2D, got {dtw_matrix.ndim}D array"
        )

    if dtw_matrix.shape[0] != dtw_matrix.shape[1]:
        raise DTWAnalysisError(
            f"dtw_matrix must be square, got shape {dtw_matrix.shape}"
        )

    if not isinstance(gesture_names, list):
        raise DTWAnalysisError(
            f"gesture_names must be a list, got {type(gesture_names).__name__}"
        )

    if len(gesture_names) != dtw_matrix.shape[0]:
        raise DTWAnalysisError(
            f"gesture_names length ({len(gesture_names)}) must match "
            f"dtw_matrix dimension ({dtw_matrix.shape[0]})"
        )

    if len(gesture_names) < 2:
        raise DTWAnalysisError(
            "Need at least 2 gestures for UMAP visualization"
        )

    output_folder = validate_folder_path(output_folder, "output_folder", must_exist=False)

    if not isinstance(n_neighbors, int) or n_neighbors < 1:
        raise DTWAnalysisError(
            f"n_neighbors must be a positive integer, got {n_neighbors}"
        )

    # Create output folder
    try:
        os.makedirs(output_folder, exist_ok=True)
    except OSError as e:
        raise DTWAnalysisError(f"Cannot create output folder {output_folder}: {e}")

    # Handle NaN values in DTW matrix
    dtw_matrix_clean = np.nan_to_num(dtw_matrix, nan=np.nanmax(dtw_matrix))

    # Create UMAP projection using precomputed distances
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(gesture_names) - 1),
        metric='precomputed'
    )
    projection = reducer.fit_transform(dtw_matrix_clean)

    # Create visualization DataFrame
    viz_df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'gesture': gesture_names
    })

    # Save visualization data
    viz_path = os.path.join(output_folder, "gesture_visualization.csv")
    viz_df.to_csv(viz_path, index=False)

    print(f"Gesture visualization saved to: {viz_path}")
