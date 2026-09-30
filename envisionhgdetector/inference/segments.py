"""Build gesture segments from frame predictions."""

from typing import List

import numpy as np
import pandas as pd

from ..state import Labels, MoveMode, SegmentColumns

# TODO - what if we add frame index as well. will be useful for downstream processing
def create_segments(
    annotations: pd.DataFrame,
    label_column: str,
    min_gap_s: float,
    min_length_s: float
) -> pd.DataFrame:
    """
    Create segments from frame-by-frame annotations, merging segments that are close in time.
    
    Args:
        annotations: DataFrame with predictions
        label_column: Name of label column
        min_gap_s: Minimum gap between segments in seconds. Segments with gaps smaller 
                  than this will be merged
        min_length_s: Minimum segment length in seconds
        
    Returns:
        DataFrame with columns: start_time, end_time, labelid, label, duration.
        Input annotations must contain a ``timestamp`` column.
    """
    output_columns = ['start_time', 'end_time', 'labelid', 'label', 'duration']
    if annotations.empty:
        return pd.DataFrame(columns=output_columns)
    if 'timestamp' not in annotations.columns:
        raise ValueError("Annotations must contain 'timestamp'.")

    is_gesture = annotations[label_column] == Labels.GESTURE
    is_move = annotations[label_column] == Labels.MOVE
    is_any_gesture = is_gesture | is_move
    if not is_any_gesture.any():
        return pd.DataFrame(columns=output_columns)

    changes = np.diff(is_any_gesture.astype(int), prepend=0)
    start_idxs = np.where(changes == 1)[0]
    end_idxs = np.where(changes == -1)[0]
    if len(start_idxs) > len(end_idxs):
        end_idxs = np.append(end_idxs, len(annotations) - 1)

    initial_segments = []
    for start_idx, end_idx in zip(start_idxs, end_idxs):
        segment_labels = annotations.iloc[start_idx:end_idx + 1][label_column]
        current_label = segment_labels.mode()[0]
        if current_label != Labels.NOGESTURE:
            initial_segments.append({
                'start_time': annotations.iloc[start_idx]['timestamp'],
                'end_time': annotations.iloc[end_idx]['timestamp'],
                'label': current_label,
            })

    if not initial_segments:
        return pd.DataFrame(columns=output_columns)

    merged_segments = []
    current_segment = initial_segments[0]
    for next_segment in initial_segments[1:]:
        time_gap = next_segment['start_time'] - current_segment['end_time']
        same_label = current_segment['label'] == next_segment['label']
        if time_gap <= min_gap_s and same_label:
            current_segment['end_time'] = next_segment['end_time']
        else:
            if current_segment['end_time'] - current_segment['start_time'] >= min_length_s:
                merged_segments.append(current_segment)
            current_segment = next_segment

    if current_segment['end_time'] - current_segment['start_time'] >= min_length_s:
        merged_segments.append(current_segment)

    return pd.DataFrame([
        {
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'labelid': index,
            'label': segment['label'],
            'duration': segment['end_time'] - segment['start_time'],
        }
        for index, segment in enumerate(merged_segments, start=1)
    ], columns=output_columns)

def create_segments_from_labels(
    times: np.ndarray,
    labels: List[str],
    min_gap_s: float = 0.3,
    min_length_s: float = 0.5,
    move_mode: MoveMode | str = MoveMode.SEPARATE,
) -> pd.DataFrame:
    """Create canonical segments from timestamped labels.

    ``move_mode`` controls whether ``Move`` remains a separate label, is
    normalized to ``Gesture``, or is treated as ``NoGesture``.
    """
    columns = [
        SegmentColumns.START_TIME,
        SegmentColumns.END_TIME,
        SegmentColumns.PREDICTION,
        SegmentColumns.PREDICTION_ID,
        SegmentColumns.DURATION,
    ]
    if len(times) == 0 or len(labels) == 0:
        return pd.DataFrame(columns=columns)
    if len(times) != len(labels):
        raise ValueError("times and labels must have the same length.")

    mode = MoveMode(move_mode)
    normalized_labels = []
    for label in labels:
        value = label.value if isinstance(label, Labels) else str(label)
        if value == Labels.MOVE.value:
            if mode is MoveMode.AS_GESTURE:
                value = Labels.GESTURE.value
            elif mode is MoveMode.IGNORE:
                value = Labels.NOGESTURE.value
        normalized_labels.append(value)

    segments = []
    start_time = None
    current_label = None
    for index, (time_value, label) in enumerate(zip(times, normalized_labels)):
        is_active = label in (Labels.GESTURE.value, Labels.MOVE.value)
        if is_active and start_time is None:
            start_time = time_value
            current_label = label
        elif start_time is not None and (not is_active or label != current_label):
            end_time = times[index - 1]
            if end_time - start_time >= min_length_s:
                segments.append({
                    SegmentColumns.START_TIME: start_time,
                    SegmentColumns.END_TIME: end_time,
                    SegmentColumns.PREDICTION: current_label,
                    SegmentColumns.DURATION: end_time - start_time,
                })
            start_time = time_value if is_active else None
            current_label = label if is_active else None

    if start_time is not None:
        end_time = times[-1]
        if end_time - start_time >= min_length_s:
            segments.append({
                SegmentColumns.START_TIME: start_time,
                SegmentColumns.END_TIME: end_time,
                SegmentColumns.PREDICTION: current_label,
                SegmentColumns.DURATION: end_time - start_time,
            })

    if not segments:
        return pd.DataFrame(columns=columns)

    merged = [segments[0]]
    for segment in segments[1:]:
        current = merged[-1]
        gap = segment[SegmentColumns.START_TIME] - current[SegmentColumns.END_TIME]
        same_label = segment[SegmentColumns.PREDICTION] == current[SegmentColumns.PREDICTION]
        if gap <= min_gap_s and same_label:
            current[SegmentColumns.END_TIME] = segment[SegmentColumns.END_TIME]
            current[SegmentColumns.DURATION] = (
                current[SegmentColumns.END_TIME] - current[SegmentColumns.START_TIME]
            )
        else:
            merged.append(segment)

    for prediction_id, segment in enumerate(merged, start=1):
        segment[SegmentColumns.PREDICTION_ID] = prediction_id
    return pd.DataFrame(merged, columns=columns)

