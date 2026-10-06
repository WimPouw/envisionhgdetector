# Standard library imports
import os
import glob
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from scipy.spatial.distance import euclidean
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .state import Labels, PredictionColumns, Row, SegmentColumns
from .state import *
from .analysis.retracking import retrack_gesture_videos
from .analysis.visualization import create_gesture_visualization
from .analysis.features import process_hand_fingers, extract_upper_limb_features
from .analysis.kinematics import ArmKinematics, calculate_derivatives, compute_limb_kinematics, find_submovements, find_movepauses, calculate_distance, calc_holds
from .analysis.spatial import define_mcneillian_grid, get_mcneillian_mode, calc_mcneillian_space, calc_volume_size, calc_vert_height
from .analysis.gesture_kinematics import joint_map, KinematicFeatures, compute_kinematic_features
from .analysis.dtw import compute_gesture_kinematics_dtw
from .analysis.segments import cut_video_by_segments
from .rendering.label_video import label_video
from .dashboard.folders import setup_dashboard_folders
from .dashboard.gesture_space import create_dashboard

def valid_float(value: float) -> bool:
    return value >= 0 and value <= 1.0

def find_all_videos(folder: str | Path, pattern: str = ".mp4") -> List[str]:
    """Recursively return video paths below ``folder``."""
    return [
        str(path)
        for path in Path(folder).rglob("*")
        if path.is_file() and path.name.endswith(pattern)
    ]

def get_video_fps(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
	
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps

def create_elan_file(
    video_path: str, 
    output_path: str, 
    segments_df: pd.DataFrame, 
) -> None:
    """
    Create ELAN file from segments DataFrame.
    
    Supports combined model output by creating separate tiers for each model
    when a 'model' column is present (ex: CNN, LightGBM).
    
    Args:
        video_path: Path to the source video file
        output_path: Path to save the ELAN file
        segments_df: DataFrame containing segments with columns: start_time, end_time, label
                    Optional 'model' column for combined model (values: 'CNN', 'LightGBM')
    """
    # Check if this is combined model output (has 'model' column)
    has_model_column = 'model' in segments_df.columns
    
    if has_model_column:
        # Get unique models
        models = segments_df['model'].unique().tolist()
    else:
        models = [SegmentColumns.PREDICTION]  # Single tier for non-combined models
    
    # Create the basic ELAN file structure
    header = f'''<?xml version="1.0" encoding="UTF-8"?>
    <ANNOTATION_DOCUMENT AUTHOR="" DATE="{time.strftime('%Y-%m-%d-%H-%M-%S')}" FORMAT="3.0" VERSION="3.0"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
        <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
            <MEDIA_DESCRIPTOR MEDIA_URL="file://{os.path.abspath(video_path)}"
                MIME_TYPE="video/mp4" RELATIVE_MEDIA_URL=""/>
            <PROPERTY NAME="lastUsedAnnotationId">0</PROPERTY>
        </HEADER>
        <TIME_ORDER>
    '''

    # Create time slots for ALL segments (across all tiers)
    time_slots = []
    time_slot_id = 1
    time_slot_refs = {}  # Store references: (start_ms, end_ms) -> (start_slot, end_slot)

    for _, segment in segments_df.iterrows():
        # Convert time to milliseconds
        start_ms = int(segment[SegmentColumns.START_TIME] * 1000)
        end_ms = int(segment[SegmentColumns.END_TIME] * 1000)

        # TODO - why would we see duplicate time slots? Shouldn't segments be unique? Check if this is necessary.
        # Only create new time slots if we haven't seen these times before
        if start_ms not in time_slot_refs:
            time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{start_ms}"/>')
            time_slot_refs[start_ms] = f"ts{time_slot_id}"
            time_slot_id += 1
        
        if end_ms not in time_slot_refs:
            time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{end_ms}"/>')
            time_slot_refs[end_ms] = f"ts{time_slot_id}"
            time_slot_id += 1

    # Sort time slots by time value for cleaner output
    time_slots_sorted = sorted(time_slots, key=lambda x: int(x.split('TIME_VALUE="')[1].split('"')[0]))
    
    # Add time slots to header
    header += '\n'.join(time_slots_sorted) + '\n    </TIME_ORDER>\n'

    # Create tiers for each model
    annotation_id = 1
    tiers_content = ""
    
    for model in models:
        # Filter segments for this model (or use all if no model column)
        if has_model_column:
            model_segments = segments_df[segments_df['model'] == model]
            tier_id = model  # Use model name as tier ID (e.g., "CNN", "LightGBM")
        else:
            model_segments = segments_df
            tier_id = SegmentColumns.PREDICTION 
        
        if model_segments.empty:
            continue
            
        # Start tier
        tiers_content += f'    <TIER DEFAULT_LOCALE="en" LINGUISTIC_TYPE_REF="default" TIER_ID="{tier_id}">\n'
        
        annotations = []
        for _, segment in model_segments.iterrows():
            start_ms = int(segment[SegmentColumns.START_TIME] * 1000)
            end_ms = int(segment[SegmentColumns.END_TIME] * 1000)
            start_slot = time_slot_refs[start_ms]
            end_slot = time_slot_refs[end_ms]
            
            annotation = f'''        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{annotation_id}" TIME_SLOT_REF1="{start_slot}" TIME_SLOT_REF2="{end_slot}">
                <ANNOTATION_VALUE>{segment[SegmentColumns.PREDICTION]}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>'''
            
            annotations.append(annotation)
            annotation_id += 1
        
        tiers_content += '\n'.join(annotations) + '\n    </TIER>\n'

    # Combine header and tiers
    header += tiers_content

    # Add linguistic type definitions
    footer = '''    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="default" TIME_ALIGNABLE="true"/>
    <LOCALE LANGUAGE_CODE="en"/>
    <CONSTRAINT DESCRIPTION="Time subdivision of parent annotation's time interval, no time gaps allowed within this interval" STEREOTYPE="Time_Subdivision"/>
    <CONSTRAINT DESCRIPTION="Symbolic subdivision of a parent annotation. Annotations cannot be time-aligned" STEREOTYPE="Symbolic_Subdivision"/>
    <CONSTRAINT DESCRIPTION="1-1 association with a parent annotation" STEREOTYPE="Symbolic_Association"/>
    <CONSTRAINT DESCRIPTION="Time alignable annotations within the parent annotation's time interval, gaps are allowed" STEREOTYPE="Included_In"/>
</ANNOTATION_DOCUMENT>'''

    # Write the complete ELAN file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header + footer)
    
    # Print info about tiers created
    if has_model_column:
        print(f"Created ELAN file with {len(models)} tiers: {', '.join(models)}")

def get_prediction(
    no_gesture_confidence: float,
    gesture_confidence: float,
    move_confidence: float,
    motion_threshold: float,
    gesture_threshold: float
) -> str:
    """Apply motion and gesture thresholds to confidence values."""
    # TODO code seems messy. fix this
    has_motion = 1 - no_gesture_confidence
    prediction = Labels.NOGESTURE
    
    if has_motion >= motion_threshold:
        gesture_conf = gesture_confidence
        move_conf = move_confidence
        
        valid_gestures = []
        if gesture_conf >= gesture_threshold:
            valid_gestures.append((Labels.GESTURE, gesture_conf))
        if move_conf >= gesture_threshold:
            valid_gestures.append((Labels.MOVE, move_conf))
            
        if valid_gestures:
            prediction = max(valid_gestures, key=lambda x: x[1])[0]
            
    return prediction

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
