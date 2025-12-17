# envisionhgdetector/segmentation.py
"""
Segmentation utilities for gesture detection.
Handles frame-by-frame annotation processing and segment creation.
"""

from typing import List
import numpy as np
import pandas as pd


class SegmentationError(Exception):
    """Exception raised for errors during segmentation."""
    pass


def validate_annotations_dataframe(annotations: pd.DataFrame, label_column: str) -> None:
    """
    Validate that annotations DataFrame has required structure.

    Args:
        annotations: DataFrame to validate
        label_column: Expected label column name

    Raises:
        SegmentationError: If validation fails
    """
    if not isinstance(annotations, pd.DataFrame):
        raise SegmentationError(
            f"annotations must be a pandas DataFrame, got {type(annotations).__name__}"
        )

    if annotations.empty:
        return  # Empty DataFrame is valid, will return empty segments

    if 'time' not in annotations.columns:
        raise SegmentationError(
            f"annotations DataFrame must contain 'time' column. "
            f"Found columns: {list(annotations.columns)}"
        )

    if label_column not in annotations.columns:
        raise SegmentationError(
            f"annotations DataFrame must contain '{label_column}' column. "
            f"Found columns: {list(annotations.columns)}"
        )


def validate_threshold(value: float, name: str, min_val: float = 0.0, max_val: float = None) -> float:
    """
    Validate and coerce a threshold value.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value (optional)

    Returns:
        Validated float value

    Raises:
        SegmentationError: If validation fails
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise SegmentationError(
            f"{name} must be a number, got {type(value).__name__}: {value}"
        )

    if min_val is not None and value < min_val:
        raise SegmentationError(
            f"{name} must be >= {min_val}, got {value}"
        )

    if max_val is not None and value > max_val:
        raise SegmentationError(
            f"{name} must be <= {max_val}, got {value}"
        )

    return value


def create_segments(
    annotations: pd.DataFrame,
    label_column: str,
    min_gap_s: float = 0.3,
    min_length_s: float = 0.5
) -> pd.DataFrame:
    """
    Create segments from frame-by-frame annotations, merging segments that are close in time.

    Args:
        annotations: DataFrame with predictions containing 'time' and label columns
        label_column: Name of the column containing gesture labels
        min_gap_s: Minimum gap between segments in seconds. Segments with gaps smaller
                  than this will be merged. Must be >= 0.
        min_length_s: Minimum segment length in seconds. Must be >= 0.

    Returns:
        DataFrame with columns: start_time, end_time, labelid, label, duration

    Raises:
        SegmentationError: If input validation fails
    """
    # Validate inputs
    validate_annotations_dataframe(annotations, label_column)
    min_gap_s = validate_threshold(min_gap_s, 'min_gap_s', min_val=0.0)
    min_length_s = validate_threshold(min_length_s, 'min_length_s', min_val=0.0)

    # Handle empty DataFrame
    if annotations.empty:
        return pd.DataFrame(
            columns=['start_time', 'end_time', 'labelid', 'label', 'duration']
        )

    is_gesture = annotations[label_column] == 'Gesture'
    is_move = annotations[label_column] == 'Move'
    is_any_gesture = is_gesture | is_move

    if not is_any_gesture.any():
        return pd.DataFrame(
            columns=['start_time', 'end_time', 'labelid', 'label', 'duration']
        )

    # Find state changes
    changes = np.diff(is_any_gesture.astype(int), prepend=0)
    start_idxs = np.where(changes == 1)[0]
    end_idxs = np.where(changes == -1)[0]

    if len(start_idxs) > len(end_idxs):
        end_idxs = np.append(end_idxs, len(annotations) - 1)

    # Create initial segments
    initial_segments = []
    for i in range(len(start_idxs)):
        start_idx = start_idxs[i]
        end_idx = end_idxs[i]

        start_time = annotations.iloc[start_idx]['time']
        end_time = annotations.iloc[end_idx]['time']

        segment_labels = annotations.loc[
            start_idx:end_idx,
            label_column
        ]
        current_label = segment_labels.mode()[0]

        # Only add segments with valid labels
        if current_label != 'NoGesture':
            initial_segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'label': current_label
            })

    if not initial_segments:
        return pd.DataFrame(
            columns=['start_time', 'end_time', 'labelid', 'label', 'duration']
        )

    # Sort segments by start time
    initial_segments.sort(key=lambda x: x['start_time'])

    # Merge close segments
    merged_segments = []
    current_segment = initial_segments[0]

    for next_segment in initial_segments[1:]:
        time_gap = next_segment['start_time'] - current_segment['end_time']

        # If segments are close enough and have the same label, merge them
        if (time_gap <= min_gap_s and
            current_segment['label'] == next_segment['label']):
            current_segment['end_time'] = next_segment['end_time']
        else:
            # Check if current segment meets minimum length requirement
            if (current_segment['end_time'] -
                current_segment['start_time']) >= min_length_s:
                merged_segments.append(current_segment)
            current_segment = next_segment

    # Add the last segment if it meets the minimum length requirement
    if (current_segment['end_time'] -
        current_segment['start_time']) >= min_length_s:
        merged_segments.append(current_segment)

    # Create final DataFrame with all required columns
    final_segments = []
    for idx, segment in enumerate(merged_segments, 1):
        final_segments.append({
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'labelid': idx,
            'label': segment['label'],
            'duration': segment['end_time'] - segment['start_time']
        })

    return pd.DataFrame(final_segments)


def get_prediction_at_threshold(
    row: pd.Series,
    motion_threshold: float = 0.6,
    gesture_threshold: float = 0.6
) -> str:
    """
    Apply thresholds to get final prediction from confidence scores.

    Args:
        row: DataFrame row containing confidence columns:
             - NoGesture_confidence
             - Gesture_confidence
             - Move_confidence
        motion_threshold: Minimum motion confidence to consider as having motion (0.0-1.0)
        gesture_threshold: Minimum gesture/move confidence to classify as that type (0.0-1.0)

    Returns:
        Predicted label: 'Gesture', 'Move', or 'NoGesture'

    Raises:
        SegmentationError: If required columns are missing or thresholds invalid
    """
    # Validate thresholds
    motion_threshold = validate_threshold(motion_threshold, 'motion_threshold', min_val=0.0, max_val=1.0)
    gesture_threshold = validate_threshold(gesture_threshold, 'gesture_threshold', min_val=0.0, max_val=1.0)

    # Validate required columns
    required_cols = ['NoGesture_confidence', 'Gesture_confidence', 'Move_confidence']
    missing_cols = [col for col in required_cols if col not in row.index]
    if missing_cols:
        raise SegmentationError(
            f"Row missing required confidence columns: {missing_cols}. "
            f"Found: {list(row.index)}"
        )

    has_motion = 1 - row['NoGesture_confidence']

    if has_motion >= motion_threshold:
        gesture_conf = row['Gesture_confidence']
        move_conf = row['Move_confidence']

        valid_gestures = []
        if gesture_conf >= gesture_threshold:
            valid_gestures.append(('Gesture', gesture_conf))
        if move_conf >= gesture_threshold:
            valid_gestures.append(('Move', move_conf))

        if valid_gestures:
            return max(valid_gestures, key=lambda x: x[1])[0]

    return 'NoGesture'
