"""Feature extraction helpers for gesture analysis."""

import numpy as np


def process_hand_fingers(landmarks, side, finger_indices):
    """Extract and center selected finger landmarks for one hand."""
    fingers = []
    for idx in finger_indices:
        feature = landmarks[:, idx]
        if not (np.any(np.isnan(feature)) or feature.size == 0):
            fingers.append(feature.reshape(-1, 3))

    if fingers:
        fingers = np.concatenate(fingers, axis=1)
        fingers_mean = np.mean(fingers, axis=0)
        return fingers - fingers_mean
    return None


def extract_upper_limb_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract and format upper limb features from world landmarks.
    
    Args:
        landmarks: Array of world landmarks in format [N, num_points, 3] 
        where 3 represents (x,y,z)
        
    Returns:
        Array of upper limb features containing coordinates for shoulders, elbows,
        wrists, and mean-centered fingers.
    """
    # Check if landmarks are the expected shape
    print(f"Debug: Landmarks shape is {landmarks.shape}")
    if landmarks.ndim != 3 or landmarks.shape[2] != 3:
        print(f"Debug: Landmarks shape is not as expected! Shape: {landmarks.shape}")
        raise ValueError("Landmarks must be a 3D array with shape [N, num_points, 3]")
    
    # Update the keypoint indices based on the 33 keypoints (0-32)
    keypoint_indices = {
        'left_shoulder': 11,  # Index 11 corresponds to left shoulder
        'right_shoulder': 12,  # Index 12 corresponds to right shoulder
        'left_elbow': 13,  # Index 13 corresponds to left elbow
        'right_elbow': 14,  # Index 14 corresponds to right elbow
        'left_wrist': 15,  # Index 15 corresponds to left wrist
        'right_wrist': 16  # Index 16 corresponds to right wrist
    }
    
    # Define finger indices separately for mean centering
    left_finger_indices = {
        'left_pinky': 17,  # Index 17 corresponds to left pinky
        'left_index': 19,  # Index 19 corresponds to left index
        'left_thumb': 21  # Index 21 corresponds to left thumb
    }
    
    right_finger_indices = {
        'right_pinky': 18,  # Index 18 corresponds to right pinky
        'right_index': 20,  # Index 20 corresponds to right index
        'right_thumb': 22  # Index 22 corresponds to right thumb
    }

    ordered_keypoints = [
        ('left_shoulder', 11),
        ('left_elbow', 13),
        ('left_wrist', 15),
        ('right_shoulder', 12), 
        ('right_elbow', 14),
        ('right_wrist', 16)
    ]
    
    # Initialize list to hold the extracted features
    all_features = []
    
    # Extract features in consistent order
    for key, index in ordered_keypoints:
        print(f"Debug: Extracting keypoint {key} at index {index}")
        feature = landmarks[:, index]
        if np.any(np.isnan(feature)) or feature.size == 0:
            print(f"Debug: No data for keypoint {key}, skipping")
        else:
            print(f"Debug: Data for keypoint {key}: {feature}")
            all_features.append(feature.reshape(-1, 3))

    # Process fingers with clear left/right separation
    left_fingers = process_hand_fingers(landmarks, 'left', [17, 19, 21])
    right_fingers = process_hand_fingers(landmarks, 'right', [18, 20, 22])
    
    if left_fingers is not None:
        all_features.append(left_fingers)
    if right_fingers is not None:
        all_features.append(right_fingers)

    features = np.concatenate(all_features, axis=1)
    print(f"Debug: Final feature array shape: {features.shape}")
    
    return features


def remove_nans(features):
    """Replace NaN values in a feature matrix with zeros."""
    return np.nan_to_num(features, nan=0.0)
