"""Build gesture segments from frame predictions."""

from typing import Literal

import numpy as np
import pandas as pd

from envisionhgdetector.state import Labels, PredictionColumns, SegmentColumns

def create_segments(
    annotations: pd.DataFrame,
    min_gap_s: float,
    min_length_s: float,
    segments_policy: Literal["combine", "separate"] = "separate",
) -> pd.DataFrame:
    """
    Create segments from frame-by-frame annotations, merging segments that are close in time.
    
    Args:
        annotations: DataFrame with prediction, timestamp, and original frame_index columns
        min_gap_s: Minimum gap between segments in seconds. Segments with gaps smaller 
                  than this will be merged
        min_length_s: Minimum segment length in seconds
        segments_policy: "separate" splits Gesture/Move transitions; "combine" groups
            consecutive active rows and uses their majority label. Ties use
            the first label returned by pandas mode().

    Segment endpoints are inclusive and use the last active sample's timestamp.
    A segment containing one sample therefore has zero duration.
        
    Returns:
        DataFrame with start/end times, inclusive original frame indices,
        segment_idx, prediction, and duration. Frame indices are taken from
        frame_index, not from DataFrame row positions.
    """
    if segments_policy not in ("combine", "separate"):
        raise ValueError("policy must be 'combine' or 'separate'.")
    
    output_columns = [SegmentColumns.START_TIME, SegmentColumns.START_FRAME_IDX, SegmentColumns.END_TIME, SegmentColumns.END_FRAME_IDX, SegmentColumns.SEGMENT_IDX, SegmentColumns.PREDICTION, SegmentColumns.DURATION]
    if annotations.empty:
        print("Warning: Annotations DataFrame is empty. Returning empty segments DataFrame.")
        return pd.DataFrame(columns=output_columns)
    if PredictionColumns.TIMESTAMP not in annotations.columns:
        raise ValueError(f"Annotations must contain '{PredictionColumns.TIMESTAMP}'.")
    if PredictionColumns.FRAME_INDEX not in annotations.columns:
        raise ValueError(f"Annotations must contain '{PredictionColumns.FRAME_INDEX}'.")

    is_gesture = annotations[PredictionColumns.PREDICTION] == Labels.GESTURE
    is_move = annotations[PredictionColumns.PREDICTION] == Labels.MOVE
    is_any_gesture = is_gesture | is_move
    if not is_any_gesture.any():
        print("Warning: No gesture or move labels found in annotations. Returning empty segments DataFrame.")
        return pd.DataFrame(columns=output_columns)

    if segments_policy == "combine":
        changes = np.diff(is_any_gesture.astype(int), prepend=0)
        start_idxs = np.where(changes == 1)[0]
        # Falling edges point to the first inactive row.
        end_idxs = np.where(changes == -1)[0] - 1
        if len(start_idxs) > len(end_idxs):
            end_idxs = np.append(end_idxs, len(annotations) - 1)
    else:
        active = is_any_gesture.to_numpy()
        labels = annotations[PredictionColumns.PREDICTION].to_numpy()
        label_changes = labels[1:] != labels[:-1] # compare consecutive labels
        # Start indices are where we have an active label and either the previous label was inactive or the label changed.
        start_idxs = np.flatnonzero(active & np.r_[True, label_changes])
        # End indices are where we have an active label and either the next label is inactive or the label changes.
        end_idxs = np.flatnonzero(active & np.r_[label_changes, True])

    initial_segments = []
    for start_idx, end_idx in zip(start_idxs, end_idxs):
        segment_labels = annotations.iloc[start_idx:end_idx + 1][PredictionColumns.PREDICTION]
        current_label = segment_labels.mode()[0] if segments_policy == "combine" else segment_labels.iloc[0]
        initial_segments.append({
            SegmentColumns.START_TIME: annotations.iloc[start_idx][PredictionColumns.TIMESTAMP],
            SegmentColumns.START_FRAME_IDX: annotations[PredictionColumns.FRAME_INDEX].iloc[start_idx],
            SegmentColumns.END_TIME: annotations.iloc[end_idx][PredictionColumns.TIMESTAMP],
            SegmentColumns.END_FRAME_IDX: annotations[PredictionColumns.FRAME_INDEX].iloc[end_idx],
            SegmentColumns.PREDICTION: current_label,
        })

    if not initial_segments:
        print("Warning: No valid segments found after initial segmentation. Returning empty segments DataFrame.")
        return pd.DataFrame(columns=output_columns)

    merged_segments = []
    current_segment = initial_segments[0]
    for next_segment in initial_segments[1:]:
        time_gap = next_segment[SegmentColumns.START_TIME] - current_segment[SegmentColumns.END_TIME]
        same_label = current_segment[SegmentColumns.PREDICTION] == next_segment[SegmentColumns.PREDICTION]
        if time_gap <= min_gap_s and same_label:
            current_segment[SegmentColumns.END_TIME] = next_segment[SegmentColumns.END_TIME]
            current_segment[SegmentColumns.END_FRAME_IDX] = next_segment[SegmentColumns.END_FRAME_IDX]
        else:
            if current_segment[SegmentColumns.END_TIME] - current_segment[SegmentColumns.START_TIME] >= min_length_s:
                merged_segments.append(current_segment)
            current_segment = next_segment
    # last segment check
    if current_segment[SegmentColumns.END_TIME] - current_segment[SegmentColumns.START_TIME] >= min_length_s:
        merged_segments.append(current_segment)

    return pd.DataFrame([
        {
            SegmentColumns.START_TIME: segment[SegmentColumns.START_TIME],
            SegmentColumns.START_FRAME_IDX: segment[SegmentColumns.START_FRAME_IDX],
            SegmentColumns.END_TIME: segment[SegmentColumns.END_TIME],
            SegmentColumns.END_FRAME_IDX: segment[SegmentColumns.END_FRAME_IDX],
            SegmentColumns.SEGMENT_IDX: index,
            SegmentColumns.PREDICTION: segment[SegmentColumns.PREDICTION],
            SegmentColumns.DURATION: segment[SegmentColumns.END_TIME] - segment[SegmentColumns.START_TIME],
        }
        for index, segment in enumerate(merged_segments, start=1)
    ], columns=output_columns)
