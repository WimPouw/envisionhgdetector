"""Feature extraction helpers for gesture analysis."""

import warnings
import numpy as np
from typing import Literal

from .mapping import UPPER_LIMB_INDICES, ARM_JOINT_INDICES, LEFT_FINGER_INDICES, RIGHT_FINGER_INDICES

def fill_missing_values(values: np.ndarray, policy: Literal["zero", "interpolate"] = "interpolate", max_gap: int = 3) -> np.ndarray:
    """Fill nonfinite values in one coordinate's equally spaced time series.

    "zero" replaces NaN and positive/negative infinity with zero.
    "interpolate" linearly fills internal gaps and copies the nearest valid
    value at either end. An entirely missing series cannot be interpolated.
    Interpolation rejects consecutive gaps longer than max_gap, including
    leading/trailing gaps. Returns a floating-point copy or raises an error if
    empty series.
    """
    if policy not in ("zero", "interpolate"):
        raise ValueError("policy must be 'zero' or 'interpolate'.")
    if max_gap < 1:
        raise ValueError("max_gap must be a atleast 1.")
    
    result = np.array(values, dtype=float, copy=True) # make a float copy of the input
    if result.ndim != 1:
        raise ValueError("values must be a one-dimensional time series.")
    if result.size == 0:
        raise ValueError("Cannot fill an empty time series.")

    valid = np.isfinite(result) # checks for NaN and positive/negative infinity
    if valid.all():
        return result
    if policy == "zero":
        result[~valid] = 0.0
        return result
    if not valid.any():
        raise ValueError("Cannot interpolate a time series with no finite values.")
    
    boundaries = np.diff(np.r_[False, ~valid, False].astype(int)) # append False to start and end to ensure gaps at the edges are detected, diff ensures length matches the original series
    lengths = np.flatnonzero(boundaries == -1) - np.flatnonzero(boundaries == 1) # gap lengths
    if lengths.max() > max_gap:
        raise ValueError(f"Missing gap of {lengths.max()} frames exceeds max_gap={max_gap}.")

    positions = np.arange(result.size)
    result[~valid] = np.interp(positions[~valid], positions[valid], result[valid]) # interpolate missing values using linear interpolation
    return result

def prepare_upper_limb_landmarks(landmarks: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Interpolate short gaps in upper-limb landmarks, preserving the input layout.

    Warn once per trajectory with the number of filled coordinate values.
    Empty inputs, entirely missing coordinates, and long gaps raise ValueError.
    Other landmarks are copied unchanged.
    """
    result = np.array(landmarks, dtype=float, copy=True)
    required_points = max(UPPER_LIMB_INDICES) + 1
    if result.ndim != 3 or result.shape[2] != 3 or result.shape[1] < required_points:
        raise ValueError(f"Expected landmarks with shape (N, at least {required_points}, 3); received {result.shape}.")
    if result.shape[0] == 0:
        raise ValueError("No frames available for upper-limb analysis.")
    
    missing_count = int((~np.isfinite(result[:, UPPER_LIMB_INDICES, :])).sum())
    if missing_count:
        warnings.warn(
            f"Filled {missing_count} missing upper-limb coordinate values; internal gaps will be interpolated and edge gaps copied from the nearest valid value (max_gap={max_gap} frames).",
            UserWarning, stacklevel=2,
        )
    for index in UPPER_LIMB_INDICES:
        for coordinate in range(3):
            try:
                result[:, index, coordinate] = fill_missing_values(
                    result[:, index, coordinate], max_gap=max_gap
                )
            except ValueError as exc:
                raise ValueError(f"Landmark {index}, coordinate {'xyz'[coordinate]}: {exc}") from exc
    return result

def process_hand_fingers(landmarks, finger_indices):
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

def extract_upper_limb_features(landmarks: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """
    Extract and format upper limb features from world landmarks.
    
    Args:
        landmarks: Array of world landmarks in format [N, num_points, 3] 
        where 3 represents (x,y,z)
        
    Returns:
        Array of upper limb features containing coordinates for shoulders, elbows,
        wrists, and mean-centered fingers.
    """
    landmarks = prepare_upper_limb_landmarks(landmarks, max_gap=max_gap)
    
    # Initialize list to hold the extracted features
    all_features = []
    
    # Extract features in consistent order
    for index in ARM_JOINT_INDICES:
        feature = landmarks[:, index]
        all_features.append(feature.reshape(-1, 3))

    # Process fingers with clear left/right separation
    left_fingers = process_hand_fingers(landmarks, LEFT_FINGER_INDICES)
    right_fingers = process_hand_fingers(landmarks, RIGHT_FINGER_INDICES)
    
    if left_fingers is not None:
        all_features.append(left_fingers)
    if right_fingers is not None:
        all_features.append(right_fingers)

    features = np.concatenate(all_features, axis=1)    
    return features
