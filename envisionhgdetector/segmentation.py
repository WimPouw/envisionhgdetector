# envisionhgdetector/segmentation.py
"""
Segmentation utilities for gesture detection.
Handles frame-by-frame annotation processing and segment creation.
"""

from typing import List
import numpy as np
import pandas as pd


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
                  than this will be merged
        min_length_s: Minimum segment length in seconds

    Returns:
        DataFrame with columns: start_time, end_time, labelid, label, duration
    """
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
        motion_threshold: Minimum motion confidence to consider as having motion
        gesture_threshold: Minimum gesture/move confidence to classify as that type

    Returns:
        Predicted label: 'Gesture', 'Move', or 'NoGesture'
    """
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
