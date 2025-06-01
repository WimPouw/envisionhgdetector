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
from scipy.ndimage import gaussian_filter1d
import umap.umap_ as umap
from shapedtw.shapedtw import shape_dtw
from shapedtw.shapeDescriptors import RawSubsequenceDescriptor
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from scipy import signal
from scipy.spatial.distance import euclidean
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import statistics
from tqdm import tqdm

def create_segments(
    annotations: pd.DataFrame,
    label_column: str,
    min_gap_s: float = 0.3,
    min_length_s: float = 0.5
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
    """Apply thresholds to get final prediction."""
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

def create_elan_file(
    video_path: str, 
    segments_df: pd.DataFrame, 
    output_path: str, 
    fps: float, 
    include_ground_truth: bool = False
) -> None:
    """
    Create ELAN file from segments DataFrame
    
    Args:
        video_path: Path to the source video file
        segments_df: DataFrame containing segments with columns: start_time, end_time, label
        output_path: Path to save the ELAN file
        fps: Video frame rate
        include_ground_truth: Whether to include ground truth tier (not implemented)
    """
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

    # Create time slots
    time_slots = []
    time_slot_id = 1
    time_slot_refs = {}  # Store references for annotations

    for _, segment in segments_df.iterrows():
        # Convert time to milliseconds
        start_ms = int(segment['start_time'] * 1000)
        end_ms = int(segment['end_time'] * 1000)
        
        # Store start time slot
        time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{start_ms}"/>')
        time_slot_refs[start_ms] = f"ts{time_slot_id}"
        time_slot_id += 1
        
        # Store end time slot
        time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{end_ms}"/>')
        time_slot_refs[end_ms] = f"ts{time_slot_id}"
        time_slot_id += 1

    # Add time slots to header
    header += '\n'.join(time_slots) + '\n    </TIME_ORDER>\n'

    # Create predicted annotations tier
    annotations = []
    annotation_id = 1
    
    header += '    <TIER DEFAULT_LOCALE="en" LINGUISTIC_TYPE_REF="default" TIER_ID="PREDICTED">\n'
    
    for _, segment in segments_df.iterrows():
        start_ms = int(segment['start_time'] * 1000)
        end_ms = int(segment['end_time'] * 1000)
        start_slot = time_slot_refs[start_ms]
        end_slot = time_slot_refs[end_ms]
        
        annotation = f'''        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{annotation_id}" TIME_SLOT_REF1="{start_slot}" TIME_SLOT_REF2="{end_slot}">
                <ANNOTATION_VALUE>{segment['label']}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>'''
        
        annotations.append(annotation)
        annotation_id += 1
    
    header += '\n'.join(annotations) + '\n    </TIER>\n'

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

def label_video(
    video_path: str,
    segments: pd.DataFrame,
    output_path: str,
    predictions_df: Optional[pd.DataFrame] = None,
    valid_timestamps: Optional[List[float]] = None,
    motion_threshold: float = None,
    gesture_threshold: float = None,
    window_duration: float = 10.0,
    target_fps: float = 25.0
) -> None:
    """
    Label a video with predicted gestures based on segments.
    Creates output at target_fps regardless of input fps.
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / input_fps
    
    # Calculate frame sampling
    frame_interval = max(1, round(input_fps / target_fps)) if input_fps > target_fps else 1
    
    # Create VideoWriter object at target FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    # Color mapping for labels
    color_map = {
        'NoGesture': (50, 50, 50),      # Dark gray
        'Gesture': (0, 204, 204),        # Vibrant teal
        'Move': (255, 94, 98)            # Soft coral red
    }
    
    # Fixed y-axis parameters for absolute scale
    y_min = 0.0
    y_max = 1.0
    
    # Determine graph dimensions
    graph_width = int(width * 0.3)
    graph_height = int(height * 0.2)
    graph_margin = 10

    # Check if we have predictions
    has_predictions = predictions_df is not None and not predictions_df.empty
    
    if has_predictions:
        # Ensure time column exists
        if 'time' not in predictions_df.columns:
            has_predictions = False
            print("Warning: predictions_df doesn't have a 'time' column")
            
    if has_predictions:
        # Get confidence data
        times = predictions_df['time'].values
        predictions_start_time = times.min() if len(times) > 0 else None
        gesture_conf = predictions_df['Gesture_confidence'].values if 'Gesture_confidence' in predictions_df.columns else None
        move_conf = predictions_df['Move_confidence'].values if 'Move_confidence' in predictions_df.columns else None
        motion_conf = predictions_df['has_motion'].values if 'has_motion' in predictions_df.columns else None
        
    # Prepare segment lookup
    def get_label_at_time(time: float) -> str:
        if segments.empty:
            return 'NoGesture'
            
        matching_segments = segments[
            (segments['start_time'] <= time) & 
            (segments['end_time'] >= time)
        ]
        return matching_segments['label'].iloc[0] if len(matching_segments) > 0 else 'NoGesture'
    
    # Calculate total output frames at target FPS
    output_frames = int(video_duration * target_fps)
    
    progress_bar = tqdm(total=output_frames, desc="Labeling video", unit="frames")

    # Process frames at the target rate
    for output_frame_idx in range(output_frames):
        # Calculate which input frame to read
        output_time = output_frame_idx / target_fps
        input_frame_idx = int(output_time * input_fps)
        
        # Ensure we don't exceed video bounds
        if input_frame_idx >= total_frames:
            break
            
        # Seek to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, input_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break

        # Get the label at current time
        try:
            current_label = get_label_at_time(output_time)
        except Exception as e:
            print(f"Error getting label at time {output_time}: {str(e)}")
            current_label = 'NoGesture'
        
        # Add text label to frame
        cv2.putText(
            frame, 
            current_label, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            color_map.get(current_label, (255, 255, 255)), 
            2
        )
        
        # Add moving window confidence graph if predictions are available
        if has_predictions and predictions_start_time is not None and output_time >= predictions_start_time:
            # Create a blank sub-image for the graph with black semi-transparent background
            graph_pos_x = width - graph_width - graph_margin
            graph_pos_y = graph_margin
            
            # Draw background with semi-transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (graph_pos_x - 35, graph_pos_y - 5), 
                         (graph_pos_x + graph_width + 5, graph_pos_y + graph_height + 25), 
                         (0, 0, 0), 
                         -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # Calculate window bounds
            min_time = min(times) if len(times) > 0 else 0
            max_time = max(times) if len(times) > 0 else output_time + window_duration

            # For beginning of video
            if output_time < min_time + (window_duration * 0.2):
                window_start = min_time
                window_end = min(max_time, min_time + window_duration)
            # For end of video
            elif output_time > max_time - (window_duration * 0.2):
                window_end = max_time
                window_start = max(min_time, max_time - window_duration)
            # For middle of video (standard sliding window)
            else:
                window_start = max(min_time, output_time - (window_duration * 0.8))
                window_end = min(max_time, window_start + window_duration)

            # Add a safeguard
            if window_end <= window_start:
                window_start = max(0, output_time - (window_duration * 0.5))
                window_end = window_start + window_duration
            
            # Add title with timestamp info
            cv2.putText(
                frame,
                f"Confidence: {window_start:.1f}s - {window_end:.1f}s",
                (graph_pos_x, graph_pos_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
            
            # Draw axes
            cv2.line(frame, 
                    (graph_pos_x, graph_pos_y + graph_height), 
                    (graph_pos_x + graph_width, graph_pos_y + graph_height), 
                    (255, 255, 255), 1)  # X-axis
            cv2.line(frame, 
                    (graph_pos_x, graph_pos_y), 
                    (graph_pos_x, graph_pos_y + graph_height), 
                    (255, 255, 255), 1)  # Y-axis
            
            # Add Y-axis ticks and grid lines
            tick_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
            for tick in tick_positions:
                tick_y = graph_pos_y + graph_height - int(tick * graph_height)
                cv2.line(frame, 
                        (graph_pos_x - 3, tick_y), 
                        (graph_pos_x, tick_y), 
                        (180, 180, 180), 1)
                cv2.putText(frame, f"{tick:.1f}", 
                          (graph_pos_x - 25, tick_y + 4), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
                cv2.line(frame, 
                        (graph_pos_x, tick_y), 
                        (graph_pos_x + graph_width, tick_y), 
                        (50, 50, 50), 1, cv2.LINE_AA)
            
            # Draw threshold lines
            if motion_threshold is not None:
                motion_y = graph_pos_y + graph_height - int(motion_threshold * graph_height)
                for x in range(graph_pos_x, graph_pos_x + graph_width, 8):
                    cv2.line(frame, (x, motion_y), (x+4, motion_y), (200, 200, 200), 1)
                cv2.putText(frame, f"M:{motion_threshold:.1f}", 
                          (graph_pos_x + graph_width + 2, motion_y + 4), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
            if gesture_threshold is not None:
                gesture_y = graph_pos_y + graph_height - int(gesture_threshold * graph_height)
                for x in range(graph_pos_x, graph_pos_x + graph_width, 8):
                    cv2.line(frame, (x, gesture_y), (x+4, gesture_y), (128, 150, 150), 1)
                cv2.putText(frame, f"G:{gesture_threshold:.1f}", 
                          (graph_pos_x + graph_width + 2, gesture_y + 4), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 150, 150), 1)
            
            # Find indices within the time window
            mask = (times >= window_start) & (times <= window_end)
            if np.any(mask):
                window_times = times[mask]
                
                # Plot confidence lines
                if gesture_conf is not None:
                    window_gesture = gesture_conf[mask]
                    prev_point = None
                    
                    for i, (t, conf) in enumerate(zip(window_times, window_gesture)):
                        x = graph_pos_x + int(((t - window_start) / window_duration) * graph_width)
                        conf_clamped = max(min(conf, y_max), y_min)
                        y = graph_pos_y + graph_height - int((conf_clamped - y_min) / (y_max - y_min) * graph_height)
                        
                        if prev_point:
                            cv2.line(frame, prev_point, (x, y), (0, 204, 204), 1, cv2.LINE_AA)
                        prev_point = (x, y)
                
                if move_conf is not None:
                    window_move = move_conf[mask]
                    prev_point = None
                    
                    for i, (t, conf) in enumerate(zip(window_times, window_move)):
                        x = graph_pos_x + int(((t - window_start) / window_duration) * graph_width)
                        conf_clamped = max(min(conf, y_max), y_min)
                        y = graph_pos_y + graph_height - int((conf_clamped - y_min) / (y_max - y_min) * graph_height)
                        
                        if prev_point:
                            cv2.line(frame, prev_point, (x, y), (255, 94, 98), 1, cv2.LINE_AA)
                        prev_point = (x, y)
                
                if motion_conf is not None:
                    window_motion = motion_conf[mask]
                    prev_point = None
                    
                    for i, (t, conf) in enumerate(zip(window_times, window_motion)):
                        x = graph_pos_x + int(((t - window_start) / window_duration) * graph_width)
                        conf_clamped = max(min(conf, y_max), y_min)
                        y = graph_pos_y + graph_height - int((conf_clamped - y_min) / (y_max - y_min) * graph_height)
                        
                        if prev_point:
                            cv2.line(frame, prev_point, (x, y), (200, 200, 200), 1, cv2.LINE_AA)
                        prev_point = (x, y)
            
            # Add current time indicator
            x_current = graph_pos_x + int(((output_time - window_start) / window_duration) * graph_width)
            if graph_pos_x <= x_current <= graph_pos_x + graph_width:
                cv2.line(frame, 
                        (x_current, graph_pos_y), 
                        (x_current, graph_pos_y + graph_height), 
                        (255, 255, 100), 2)
            
            # Add legend
            legend_y = graph_pos_y + graph_height + 15
            cv2.putText(frame, "G", (graph_pos_x + 5, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 204, 204), 1)
            cv2.putText(frame, "M", (graph_pos_x + 25, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 94, 98), 1)
            cv2.putText(frame, "Motion", (graph_pos_x + 45, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        out.write(frame)
        progress_bar.update(1)

    progress_bar.close()
    cap.release()
    out.release()
    
    print(f"Video labeled at {target_fps}fps saved to {output_path}")

# a function that allows you to cut the videos by segments
def cut_video_by_segments(
    output_folder: str,
    segments_pattern: str = "*_segments.csv",
    labeled_video_prefix: str = "labeled_",
    output_subfolder: str = "gesture_segments"
) -> Dict[str, List[str]]:
    """
    Extracts video segments and corresponding features from labeled videos based on segments.csv files.
    
    Args:
        output_folder: Path to the folder containing segments.csv files and labeled videos
        segments_pattern: Pattern to match segment CSV files
        labeled_video_prefix: Prefix of labeled video files
        output_subfolder: Name of subfolder to store segmented videos
        
    Returns:
        Dictionary mapping original video names to lists of generated segment paths
    """
    # Create subfolder for segments if it doesn't exist
    segments_folder = os.path.join(output_folder, output_subfolder)
    os.makedirs(segments_folder, exist_ok=True)
    
    # Get all segment CSV files
    segment_files = glob.glob(os.path.join(output_folder, segments_pattern))
    results = {}
    
    for segment_file in segment_files:
        try:
            # Get original video name from segments file name
            base_name = os.path.basename(segment_file).replace('_segments.csv', '')
            labeled_video = os.path.join(output_folder, f"{labeled_video_prefix}{base_name}")
            features_path = os.path.join(output_folder, f"{base_name}_features.npy")
            
            # Check if labeled video and features exist
            if not os.path.exists(labeled_video):
                print(f"Warning: Labeled video not found for {base_name}")
                continue
            if not os.path.exists(features_path):
                print(f"Warning: Features file not found for {base_name}")
                continue
                
            # Read segments file
            segments_df = pd.read_csv(segment_file)
            
            if segments_df.empty:
                print(f"No segments found in {segment_file}")
                continue
            
            # Create subfolder for this video's segments
            video_segments_folder = os.path.join(segments_folder, base_name)
            os.makedirs(video_segments_folder, exist_ok=True)
            
            # Load video and get fps
            video = VideoFileClip(labeled_video)
            fps = video.fps
            
            # Load features
            features = np.load(features_path)
            
            segment_paths = []
            
            # Process each segment
            for idx, segment in segments_df.iterrows():
                start_time = segment['start_time']
                end_time = segment['end_time']
                label = segment['label']
                
                # Calculate frame indices
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                
                # Create segment filenames
                segment_filename = f"{base_name}_segment_{idx+1}_{label}_{start_time:.2f}_{end_time:.2f}.mp4"
                features_filename = f"{base_name}_segment_{idx+1}_{label}_{start_time:.2f}_{end_time:.2f}_features.npy"
                
                segment_path = os.path.join(video_segments_folder, segment_filename)
                features_path = os.path.join(video_segments_folder, features_filename)
                
                # Extract and save video segment
                try:
                    # Cut video
                    segment_clip = video.subclipped(start_time, end_time)
                    segment_clip.write_videofile(
                        segment_path,
                        codec='libx264',
                        audio=False
                    )
                    segment_clip.close()
                    
                    # Cut and save features
                    if start_frame < len(features) and end_frame <= len(features):
                        segment_features = features[start_frame:end_frame]
                        np.save(features_path, segment_features)
                        print(f"Created segment and features: {segment_filename}")
                    else:
                        print(f"Warning: Frame indices {start_frame}:{end_frame} out of bounds for features array of length {len(features)}")
                    
                    segment_paths.append(segment_path)
                    
                except Exception as e:
                    print(f"Error creating segment {segment_filename}: {str(e)}")
                    continue
            
            # Clean up
            video.close()
            
            results[base_name] = segment_paths
            print(f"Completed processing segments for {base_name}")
            
        except Exception as e:
            print(f"Error processing {segment_file}: {str(e)}")
            continue
    
    return results

def create_sliding_windows(
    features: List[List[float]],
    seq_length: int,
    stride: int = 1,
    input_fps: Optional[float] = None,
    target_fps: float = 25.0
) -> np.ndarray:
    """
    Create sliding windows from feature sequence.
    
    Args:
        features: List of feature vectors
        seq_length: Length of each window
        stride: Step size between windows (default: 1)
        input_fps: Original video FPS (if provided, will adjust stride)
        target_fps: Target FPS for analysis
        
    Returns:
        NumPy array of windowed features
    """
    if len(features) < seq_length:
        return np.array([])
    
    # If input_fps is provided and different from target, adjust stride
    if input_fps is not None and input_fps > target_fps:
        # Don't override stride here - it's already been sampled at ~25fps
        # The features are already at the target rate from video_to_landmarks
        pass
    
    windows = []
    for i in range(0, len(features) - seq_length + 1, stride):
        window = features[i:i + seq_length]
        if len(window) == seq_length:
            windows.append(window)
    
    return np.array(windows)


def create_gesture_visualization(
    dtw_matrix: np.ndarray,
    gesture_names: List[str],
    output_folder: str
) -> None:
    """
    Create UMAP visualization from DTW distances.
    
    Args:
        dtw_matrix: DTW distance matrix
        gesture_names: List of gesture names
        output_folder: Folder to save visualization
    """   
    # Create UMAP projection
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        metric='precomputed'
    )
    projection = reducer.fit_transform(dtw_matrix)
    
    # Create visualization DataFrame
    viz_df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'gesture': gesture_names
    })
    
    # Save visualization data
    viz_path = os.path.join(output_folder, "gesture_visualization.csv")
    viz_df.to_csv(viz_path, index=False)


import numpy as np

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

def process_hand_fingers(landmarks, side, finger_indices):
    """Helper function to process fingers for one hand"""
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


def remove_nans(features):
    """
    Remove NaN values from the feature matrix by replacing them with zeros.
    Args:
        features: 2D numpy array (gesture)
    Returns:
        Cleaned features (2D numpy array)
    """
    return np.nan_to_num(features, nan=0.0)

# Define mapping from joint names to MediaPipe indices
joint_map = {
    'L_Hand': 15,      # Left wrist
    'R_Hand': 16,      # Right wrist
    'LElb': 13,        # Left elbow
    'RElb': 14,        # Right elbow
    'LShoulder': 11,   # Left shoulder
    'RShoulder': 12,   # Right shoulder
    'Neck': 23,        # Neck (approximated as top of spine)
    'MidHip': 24,      # Mid hip
    'LEye': 2,         # Left eye
    'REye': 5,         # Right eye
    'Nose': 0,         # Nose
    'LHip': 23,        # Left hip
    'RHip': 24         # Right hip
}

class ArmKinematics(NamedTuple):
    """Container for arm kinematic measurements."""
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    speed: np.ndarray
    peaks: np.ndarray
    peak_heights: np.ndarray

@dataclass
class KinematicFeatures:
    """Data class to store comprehensive kinematic features for a gesture."""
    gesture_id: str
    video_id: str
    
    # Which hand was more active in this specific gesture
    active_hand: str  # 'L' or 'R'
    
    # Spatial features
    space_use: int
    mcneillian_max: float
    mcneillian_mode: int
    volume: float
    max_height: float
    
    # Temporal features
    duration: float
    hold_count: int
    hold_time: float
    hold_avg_duration: float
    
    # Submovement features
    hand_submovements: int
    hand_submovement_peaks: List[float]
    hand_mean_submovement_amplitude: float
    
    elbow_submovements: int
    elbow_mean_submovement_amplitude: float
    
    # Dynamic features
    hand_peak_speed: float
    hand_mean_speed: float
    hand_peak_acceleration: float
    hand_peak_deceleration: float
    hand_peak_jerk: float
    
    elbow_peak_speed: float
    elbow_mean_speed: float
    elbow_peak_acceleration: float
    elbow_peak_deceleration: float
    elbow_peak_jerk: float

def calculate_derivatives(positions: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate velocity, acceleration and jerk from position data."""
    if not isinstance(positions, np.ndarray) or positions.size == 0:
        raise ValueError("positions must be a non-empty numpy array")
    if fps <= 0:
        raise ValueError("fps must be positive")
    # Calculate time step
    dt = 1/fps
    # smooth positions
    positions = gaussian_filter1d(positions, sigma=2, axis=0)
    
    # Calculate velocity (first derivative)
    velocity = np.gradient(positions, dt, axis=0)
    
    # Calculate acceleration (second derivative)
    acceleration = np.gradient(velocity, dt, axis=0)
    
    # Calculate jerk (third derivative)
    jerk = np.gradient(acceleration, dt, axis=0)
    
    return velocity, acceleration, jerk

def compute_limb_kinematics(positions: np.ndarray, fps: float) -> ArmKinematics:
    """
    Compute comprehensive kinematics for a limb segment.
    
    Args:
        positions: Array of 3D positions over time
        fps: Frames per second
        
    Returns:
        ArmKinematics object containing computed measures
    """
    # Calculate derivatives
    velocity, acceleration, jerk = calculate_derivatives(positions, fps)
    
    # Calculate speed (magnitude of velocity)
    speed = np.linalg.norm(velocity, axis=1)
    
    # Find submovements
    peaks, peak_heights = find_submovements(speed, fps)
    
    return ArmKinematics(
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        speed=speed,
        peaks=peaks,
        peak_heights=peak_heights
    )

def find_submovements(speed_profile: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find submovements in a speed profile using peak detection.
    
    Args:
        speed_profile: Array of speeds over time
        fps: Frames per second
        
    Returns:
        Tuple of (peaks indices, peak heights)
    """
    # Apply Savitzky-Golay smoothing
    if len(speed_profile) >= 15:
        smoothed = signal.savgol_filter(speed_profile, 15, 5)
    else:
        window = len(speed_profile) - 1 if len(speed_profile) % 2 == 0 else len(speed_profile)
        smoothed = signal.savgol_filter(speed_profile, window, 5)
    
    # Find peaks with prominence and distance constraints
    peaks, properties = signal.find_peaks(
        smoothed,
        distance=5,  # Minimum distance between peaks (in frames)
        height=0,  # Include height to get peak heights
        prominence=0.1  # Add minimum prominence to avoid noise
    )
    
    # Get peak heights from the smoothed signal
    peak_heights = smoothed[peaks] if len(peaks) > 0 else np.array([0])  # Return [0] instead of empty array
    
    return peaks, peak_heights

def compute_limb_kinematics(positions: np.ndarray, fps: float) -> ArmKinematics:
    """
    Compute kinematics for a limb segment.
    """
    # Calculate derivatives
    velocity, acceleration, jerk = calculate_derivatives(positions, fps)
    
    # Calculate speed (magnitude of velocity)
    speed = np.linalg.norm(velocity, axis=1)
    
    # Find submovements (ensure we handle no peaks case)
    peaks, peak_heights = find_submovements(speed, fps)
    
    # If no peaks were found, use zero-arrays of appropriate shape
    if len(peaks) == 0:
        peaks = np.array([0])
        peak_heights = np.array([0])
    
    return ArmKinematics(
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        speed=speed,
        peaks=peaks,
        peak_heights=peak_heights
    )

def define_mcneillian_grid(df, frame):
    """Define the grid based on original implementation."""
    bodycent = df['Neck'][frame][1] - (df['Neck'][frame][1] - df['MidHip'][frame][1])/2
    face_width = (df['LEye'][frame][0] - df['REye'][frame][0])*2
    # Use shoulders instead of hips for more reliable body width
    body_width = df['LShoulder'][frame][0] - df['RShoulder'][frame][0]

    # Center-center boundaries
    cc_xmin = df['RShoulder'][frame][0]
    cc_xmax = df['LShoulder'][frame][0]
    cc_len = cc_xmax - cc_xmin
    cc_ymin = bodycent - cc_len/2
    cc_ymax = bodycent + cc_len/2

    # Center boundaries
    c_xmin = df['RShoulder'][frame][0] - body_width/2
    c_xmax = df['LShoulder'][frame][0] + body_width/2
    c_len = c_xmax - c_xmin
    c_ymin = bodycent - c_len/2
    c_ymax = bodycent + c_len/2

    # Periphery boundaries
    p_ymax = df['LEye'][frame][1] + (df['LEye'][frame][1] - df['Nose'][frame][1])
    p_ymin = bodycent - (p_ymax - bodycent)
    p_xmin = c_xmin - face_width
    p_xmax = c_xmax + face_width

    return cc_xmin, cc_xmax, cc_ymin, cc_ymax, c_xmin, c_xmax, c_ymin, c_ymax, p_xmin, p_xmax, p_ymin, p_ymax

def get_mcneillian_mode(spaces):
    """Convert subsection codes to main sections and calculate mode."""
    mainspace = []
    for space in spaces:
        if space > 40:
            mainspace.append(4)
        elif space > 30:
            mainspace.append(3)
        else:
            mainspace.append(space)
    
    return statistics.mode(mainspace)

def calc_mcneillian_space(df, visibility=None, visibility_threshold=0.5):
    """Calculate McNeillian space features using original implementation approach."""
    Space_L = []
    Space_R = []
    
    for frame in range(len(df['MidHip'])):
        try:
            # Get grid boundaries
            cc_xmin, cc_xmax, cc_ymin, cc_ymax, c_xmin, c_xmax, c_ymin, c_ymax, p_xmin, p_xmax, p_ymin, p_ymax = \
                define_mcneillian_grid(df, frame)
            
            # Process left hand if visible
            if visibility is None or visibility[frame, 15] >= visibility_threshold:
                left_hand = df['L_Hand'][frame]
                x, y = left_hand[0], left_hand[1]
                
                # Assign zone with subsections
                if cc_xmin < x < cc_xmax and cc_ymin < y < cc_ymax:
                    Space_L.append(1)
                elif c_xmin < x < c_xmax and c_ymin < y < c_ymax:
                    Space_L.append(2)
                elif p_xmin < x < p_xmax and p_ymin < y < p_ymax:
                    # Periphery subsections
                    if cc_xmax < x:  # Right side
                        if cc_ymax < y:
                            Space_L.append(31)
                        elif cc_ymin < y:
                            Space_L.append(32)
                        else:
                            Space_L.append(33)
                    elif cc_xmin < x:  # Center
                        if c_ymax < y:
                            Space_L.append(38)
                        else:
                            Space_L.append(34)
                    else:  # Left side
                        if cc_ymax < y:
                            Space_L.append(37)
                        elif cc_ymin < y:
                            Space_L.append(36)
                        else:
                            Space_L.append(35)
                else:  # Extra-periphery subsections
                    if c_xmax < x:  # Right side
                        if cc_ymax < y:
                            Space_L.append(41)
                        elif cc_ymin < y:
                            Space_L.append(42)
                        else:
                            Space_L.append(43)
                    elif cc_xmin < x:  # Center
                        if c_ymax < y:
                            Space_L.append(48)
                        else:
                            Space_L.append(44)
                    else:  # Left side
                        if c_ymax < y:
                            Space_L.append(47)
                        elif c_ymin < y:
                            Space_L.append(46)
                        else:
                            Space_L.append(45)
            
            # Process right hand similarly
            if visibility is None or visibility[frame, 16] >= visibility_threshold:
                right_hand = df['R_Hand'][frame]
                x, y = right_hand[0], right_hand[1]
                
                # Same zone assignment logic for right hand
                if cc_xmin < x < cc_xmax and cc_ymin < y < cc_ymax:
                    Space_R.append(1)
                elif c_xmin < x < c_xmax and c_ymin < y < c_ymax:
                    Space_R.append(2)
                elif p_xmin < x < p_xmax and p_ymin < y < p_ymax:
                    if cc_xmax < x:
                        if cc_ymax < y:
                            Space_R.append(31)
                        elif cc_ymin < y:
                            Space_R.append(32)
                        else:
                            Space_R.append(33)
                    elif cc_xmin < x:
                        if c_ymax < y:
                            Space_R.append(38)
                        else:
                            Space_R.append(34)
                    else:
                        if cc_ymax < y:
                            Space_R.append(37)
                        elif cc_ymin < y:
                            Space_R.append(36)
                        else:
                            Space_R.append(35)
                else:
                    if c_xmax < x:
                        if cc_ymax < y:
                            Space_R.append(41)
                        elif cc_ymin < y:
                            Space_R.append(42)
                        else:
                            Space_R.append(43)
                    elif cc_xmin < x:
                        if c_ymax < y:
                            Space_R.append(48)
                        else:
                            Space_R.append(44)
                    else:
                        if c_ymax < y:
                            Space_R.append(47)
                        elif c_ymin < y:
                            Space_R.append(46)
                        else:
                            Space_R.append(45)
                            
        except Exception as e:
            print(f"Error in frame {frame}: {str(e)}")
    
    # Ensure we have data
    if not Space_L:
        Space_L = [1]
    if not Space_R:
        Space_R = [1]
    
    # Calculate statistics using original method
    space_use_L = len(set(Space_L))
    space_use_R = len(set(Space_R))
    
    mcneillian_maxL = 4 if max(Space_L) > 40 else (3 if max(Space_L) > 30 else max(Space_L))
    mcneillian_maxR = 4 if max(Space_R) > 40 else (3 if max(Space_R) > 30 else max(Space_R))
    
    mcneillian_modeL = get_mcneillian_mode(Space_L)
    mcneillian_modeR = get_mcneillian_mode(Space_R)
    
    return (space_use_L, space_use_R, mcneillian_maxL, mcneillian_maxR, 
            mcneillian_modeL, mcneillian_modeR)

def calc_volume_size(df, hand):
    """
    Calculate the volumetric size of the gesture space, adapted for MediaPipe landmarks.
    
    Args:
        df: DataFrame with pose keypoints
        hand: Which hand to analyze ('L', 'R', or 'B' for both)
        
    Returns:
        Volume/area of the gesture space
    """
    # Initialize boundaries from first frame
    if hand == 'B':
        x_max = max([df['R_Hand'][0][0], df['L_Hand'][0][0]])
        x_min = min([df['R_Hand'][0][0], df['L_Hand'][0][0]])
        y_max = max([df['R_Hand'][0][1], df['L_Hand'][0][1]])  # Fixed y coordinate selection
        y_min = min([df['R_Hand'][0][1], df['L_Hand'][0][1]])  # Fixed y coordinate selection
        if len(df['R_Hand'][0]) > 2:  # If 3D
            z_max = max([df['R_Hand'][0][2], df['L_Hand'][0][2]])
            z_min = min([df['R_Hand'][0][2], df['L_Hand'][0][2]])
    else:
        hand_str = hand + '_Hand'
        x_min = x_max = df[hand_str][0][0]
        y_min = y_max = df[hand_str][0][1]  # Fixed y coordinate selection
        if len(df[hand_str][0]) > 2:  # If 3D
            z_min = z_max = df[hand_str][0][2]

    # Process all frames to find extremes
    hand_list = ['R_Hand', 'L_Hand'] if hand == 'B' else [hand + '_Hand']
    
    for frame in range(len(df)):
        for hand_idx in hand_list:
            curr_pos = df[hand_idx][frame]
            x_min = min(x_min, curr_pos[0])
            x_max = max(x_max, curr_pos[0])
            y_min = min(y_min, curr_pos[1])
            y_max = max(y_max, curr_pos[1])
            if len(curr_pos) > 2:  # If 3D
                z_min = min(z_min, curr_pos[2])
                z_max = max(z_max, curr_pos[2])

    # Calculate volume/area
    if len(df[hand_list[0]][0]) > 2:  # If 3D
        vol = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    else:  # If 2D
        vol = (x_max - x_min) * (y_max - y_min)
    
    return vol

def calc_vert_height(df, visibility=None, visibility_threshold=0.5):
    """
    Calculate vertical height separately and independently for each hand.
    Corrected to handle coordinate system properly and normalize heights.
    """
    H_L = []
    H_R = []
    
    for frame in range(len(df['MidHip'])):  # Iterate using range instead of index
        # Get reference points for normalization
        try:
            # Note: MediaPipe uses y-down coordinate system, so smaller y values are higher
            mid_hip_y = df['MidHip'][frame][1]
            neck_y = df['Neck'][frame][1]
            nose_y = df['Nose'][frame][1]
            left_eye_y = df['LEye'][frame][1]
            right_eye_y = df['REye'][frame][1]
            
            # Calculate body-scaled reference heights
            body_height = mid_hip_y - neck_y
            head_height = neck_y - nose_y
            
            # Process left hand if visible
            if visibility is None or visibility[frame, 15] >= visibility_threshold:
                left_hand_y = df['L_Hand'][frame][1]
                
                # Normalize height relative to body proportions
                if left_hand_y >= mid_hip_y:  # Below hip
                    H_L.append(0)
                elif left_hand_y >= neck_y:  # Between hip and neck
                    height_ratio = (mid_hip_y - left_hand_y) / body_height
                    H_L.append(1 + height_ratio)
                elif left_hand_y >= nose_y:  # Between neck and nose
                    height_ratio = (neck_y - left_hand_y) / head_height
                    H_L.append(2 + height_ratio)
                elif left_hand_y >= left_eye_y:  # Between nose and eye
                    height_ratio = (nose_y - left_hand_y) / (nose_y - left_eye_y)
                    H_L.append(3 + height_ratio)
                else:  # Above eye
                    H_L.append(5)
            else:
                H_L.append(0)
                
            # Process right hand if visible
            if visibility is None or visibility[frame, 16] >= visibility_threshold:
                right_hand_y = df['R_Hand'][frame][1]
                
                # Normalize height relative to body proportions
                if right_hand_y >= mid_hip_y:  # Below hip
                    H_R.append(0)
                elif right_hand_y >= neck_y:  # Between hip and neck
                    height_ratio = (mid_hip_y - right_hand_y) / body_height
                    H_R.append(1 + height_ratio)
                elif right_hand_y >= nose_y:  # Between neck and nose
                    height_ratio = (neck_y - right_hand_y) / head_height
                    H_R.append(2 + height_ratio)
                elif right_hand_y >= right_eye_y:  # Between nose and eye
                    height_ratio = (nose_y - right_hand_y) / (nose_y - right_eye_y)
                    H_R.append(3 + height_ratio)
                else:  # Above eye
                    H_R.append(5)
            else:
                H_R.append(0)
                
        except Exception as e:
            print(f"Error in frame {frame}: {str(e)}")
            H_L.append(0)
            H_R.append(0)
    
    # Calculate maximum heights with proper normalization
    max_height_L = max(H_L) if H_L else 0
    max_height_R = max(H_R) if H_R else 0
    
    return max_height_L, max_height_R

def find_movepauses(velocity_array):
    """
    Find moments when velocity is below a threshold.
    
    Args:
        velocity_array: Array of velocities
        
    Returns:
        Array of indices for pause moments
    """
    # We are using a 0.15m/s threshold, but this can be adjusted
    pause_ix = []
    for index, velpoint in enumerate(velocity_array):
        if velpoint < 0.15:
            pause_ix.append(index)
    if len(pause_ix) == 0:
        pause_ix = 0
    return pause_ix

def calculate_distance(positions, fps):
    """Calculate distance and velocity between consecutive positions."""
    distances = []
    velocities = []
    
    for i in range(1, len(positions)):
        dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
        distances.append(dist)
        velocities.append(dist * fps)  # Convert to units/second
        
    return distances, velocities

def calc_holds(df, subslocs_L, subslocs_R, FPS, hand):
    """Calculate hold features with safety checks."""
    try:
        # Initialize with safe defaults
        if not isinstance(subslocs_L, (list, np.ndarray)) or len(subslocs_L) == 0:
            subslocs_L = np.array([0])
        if not isinstance(subslocs_R, (list, np.ndarray)) or len(subslocs_R) == 0:
            subslocs_R = np.array([0])
            
        # Calculate hold features with safety checks
        _, RE_S = calculate_distance(df["RElb"], FPS)
        GERix = find_movepauses(RE_S)
        _, RH_S = calculate_distance(df["R_Hand"], FPS)
        GRix = find_movepauses(RH_S)
        GFRix = GRix  # Default to hand if no finger data

        # Initialize empty lists for holds
        GR = []
        GL = []

        # Process right side holds
        if isinstance(GERix, list) and isinstance(GRix, list):
            for handhold in GRix:
                for elbowhold in GERix:
                    if handhold == elbowhold:
                        GR.append(handhold)

        # Process left side
        _, LE_S = calculate_distance(df["LElb"], FPS)
        GELix = find_movepauses(LE_S)
        _, LH_S = calculate_distance(df["L_Hand"], FPS)
        GLix = find_movepauses(LH_S)
        GFLix = GLix  # Default to hand if no finger data

        if isinstance(GELix, list) and isinstance(GLix, list):
            for handhold in GLix:
                for elbowhold in GELix:
                    if handhold == elbowhold:
                        GL.append(handhold)

        # Initialize holds with safe defaults
        hold_count = 0
        hold_time = 0
        hold_avg = 0

        # Process holds based on hand selection
        if ((hand == 'B' and GL and GR) or 
            (hand == 'L' and GL) or 
            (hand == 'R' and GR)):

            full_hold = []
            if hand == 'B':
                for left_hold in GL:
                    for right_hold in GR:
                        if left_hold == right_hold:
                            full_hold.append(left_hold)
            elif hand == 'L':
                full_hold = GL
            elif hand == 'R':
                full_hold = GR

            if full_hold:
                # Cluster holds
                hold_cluster = [[full_hold[0]]]
                clustercount = 0
                holdcount = 1

                for idx in range(1, len(full_hold)):
                    if full_hold[idx] != hold_cluster[clustercount][holdcount - 1] + 1:
                        clustercount += 1
                        holdcount = 1
                        hold_cluster.append([full_hold[idx]])
                    else:
                        hold_cluster[clustercount].append(full_hold[idx])
                        holdcount += 1

                # Filter holds based on initial movement
                try:
                    if hand == 'B':
                        initial_move = min(np.concatenate((subslocs_L, subslocs_R)))
                    elif hand == 'L':
                        initial_move = min(subslocs_L)
                    else:
                        initial_move = min(subslocs_R)

                    hold_cluster = [cluster for cluster in hold_cluster if cluster[0] >= initial_move]
                except:
                    pass  # Keep all clusters if filtering fails

                # Calculate statistics
                hold_durations = []
                for cluster in hold_cluster:
                    if len(cluster) >= 3:
                        hold_count += 1
                        hold_time += len(cluster)
                        hold_durations.append(len(cluster))

                # Calculate final metrics with safety checks
                hold_time = hold_time / FPS if FPS > 0 else 0
                hold_avg = statistics.mean(hold_durations) if hold_durations else 0

        return hold_count, hold_time, hold_avg

    except Exception as e:
        print(f"Error in calc_holds: {str(e)}")
        return 0, 0, 0  # Return safe defaults if anything fails

def compute_kinematic_features(
    landmarks: np.ndarray,
    visibility: np.ndarray = None,
    fps: float = 25.0,
    gesture_id: str = "",
    video_id: str = ""
) -> KinematicFeatures:
    """
    Compute comprehensive kinematic features for a gesture using the more active hand.
    For each gesture, determines which hand was more active during that specific gesture.
    """
    # Convert landmarks to DataFrame format first
    df = pd.DataFrame()
    for joint in ['L_Hand', 'R_Hand', 'LElb', 'RElb', 'LShoulder', 'RShoulder', 
                 'Neck', 'MidHip', 'LEye', 'REye', 'Nose']:
        df[joint] = [landmarks[i, joint_map[joint]] for i in range(len(landmarks))]
    
    # Analyze movement for this specific gesture
    left_hand = landmarks[:, 15]  # Left wrist
    right_hand = landmarks[:, 16]  # Right wrist
    
    # Calculate total movement (speed) for each hand
    left_speeds = np.linalg.norm(np.diff(left_hand, axis=0), axis=1)
    right_speeds = np.linalg.norm(np.diff(right_hand, axis=0), axis=1)
    
    # Apply visibility masking if available
    if visibility is not None:
        visibility_threshold = 0.5
        left_vis_mask = visibility[:-1, 15] >= visibility_threshold
        right_vis_mask = visibility[:-1, 16] >= visibility_threshold
        
        # Count frames where each hand is visible
        left_visible_frames = np.sum(visibility[:, 15] >= visibility_threshold)
        right_visible_frames = np.sum(visibility[:, 16] >= visibility_threshold)
        
        # Apply visibility masks
        left_speeds = left_speeds * left_vis_mask
        right_speeds = right_speeds * right_vis_mask
        
        # Normalize by number of visible frames to avoid bias
        left_total = np.sum(left_speeds) * (len(visibility) / max(left_visible_frames, 1))
        right_total = np.sum(right_speeds) * (len(visibility) / max(right_visible_frames, 1))
    else:
        left_total = np.sum(left_speeds)
        right_total = np.sum(right_speeds)
    
    # Select the more active hand for this specific gesture
    active_hand = 'L' if left_total > right_total else 'R'
    print(f"Gesture {gesture_id}: {active_hand} hand showed more movement")
    
    # Get keys for the active hand
    hand_key = 'L_Hand' if active_hand == 'L' else 'R_Hand'
    elbow_key = 'LElb' if active_hand == 'L' else 'RElb'
    joint_idx = 15 if active_hand == 'L' else 16
    
    # Calculate spatial features
    mcn_space = calc_mcneillian_space(df, visibility)
    space_use = mcn_space[0] if active_hand == 'L' else mcn_space[1]
    mcneillian_max = mcn_space[2] if active_hand == 'L' else mcn_space[3]
    mcneillian_mode = mcn_space[4] if active_hand == 'L' else mcn_space[5]
    
    # Calculate volume for active hand
    volume = calc_volume_size(df, active_hand)
    
    # Calculate max height for active hand
    max_heights = calc_vert_height(df, visibility)
    max_height = max_heights[0] if active_hand == 'L' else max_heights[1]
    
    # Compute kinematics for active arm
    hand = compute_limb_kinematics(np.array([p for p in df[hand_key]]), fps)
    elbow = compute_limb_kinematics(np.array([p for p in df[elbow_key]]), fps)
    
    # Calculate hold features using only active hand
    if active_hand == 'L':
        hold_peaks = hand.peaks
        other_peaks = np.array([])
    else:
        hold_peaks = np.array([])
        other_peaks = hand.peaks
        
    hold_count, hold_time, hold_avg = calc_holds(
        df, hold_peaks, other_peaks, fps, active_hand
    )
    
    # Safe computation helpers
    def safe_mean(arr): return float(np.mean(arr)) if len(arr) > 0 else 0.0
    def safe_max(arr): return float(np.max(arr)) if len(arr) > 0 else 0.0
    def safe_min(arr): return float(np.min(arr)) if len(arr) > 0 else 0.0
    def safe_norm(arr, axis=1): return np.linalg.norm(arr, axis=axis) if len(arr) > 0 else np.zeros(1)
    
    return KinematicFeatures(
        gesture_id=gesture_id,
        video_id=video_id,
        active_hand=active_hand,
        
        # Spatial features
        space_use=space_use,
        mcneillian_max=mcneillian_max,
        mcneillian_mode=mcneillian_mode,
        volume=volume,
        max_height=max_height,
        
        # Temporal features
        duration=len(landmarks) / fps,
        hold_count=hold_count,
        hold_time=hold_time,
        hold_avg_duration=hold_avg,
        
        # Hand submovements
        hand_submovements=len(hand.peaks),
        hand_submovement_peaks=hand.peak_heights.tolist() if len(hand.peak_heights) > 0 else [0],
        hand_mean_submovement_amplitude=safe_mean(hand.peak_heights),
        
        # Elbow submovements
        elbow_submovements=len(elbow.peaks),
        elbow_mean_submovement_amplitude=safe_mean(elbow.peak_heights),
        
        # Hand dynamics
        hand_peak_speed=safe_max(hand.speed),          # Changed from velocity to speed
        hand_mean_speed=safe_mean(hand.speed),         # Changed from velocity to speed
        hand_peak_acceleration=safe_max(safe_norm(hand.acceleration)),
        hand_peak_deceleration=safe_min(safe_norm(hand.acceleration)),
        hand_peak_jerk=safe_max(safe_norm(hand.jerk)),
        
        # Elbow dynamics
        elbow_peak_speed=safe_max(elbow.speed),        # Changed from velocity to speed
        elbow_mean_speed=safe_mean(elbow.speed),       # Changed from velocity to speed
        elbow_peak_acceleration=safe_max(safe_norm(elbow.acceleration)),
        elbow_peak_deceleration=safe_min(safe_norm(elbow.acceleration)),
        elbow_peak_jerk=safe_max(safe_norm(elbow.jerk))
    )

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

def create_dashboard(data_folder: str, assets_folder: str = "./assets"):
    """
    Create and run the gesture space visualization dashboard.
    
    Args:
        data_folder: Path to folder containing visualization data and videos
        assets_folder: Path to Dash assets folder (will store videos)
    """
    # Create assets folder if it doesn't exist
    os.makedirs(assets_folder, exist_ok=True)
    
    # Import visualization data
    df = pd.read_csv(os.path.join(data_folder, "gesture_visualization.csv"))
    
    # Adjusted: Copy tracked videos to assets from the retracked folder
    retracked_folder = os.path.join(os.path.dirname(data_folder), "retracked", "tracked_videos")
    if not os.path.exists(retracked_folder):
        raise FileNotFoundError(f"Tracked videos folder not found at {retracked_folder}")
    
    for video in os.listdir(retracked_folder):
        if video.endswith("_tracked.mp4"):
            source = os.path.join(retracked_folder, video)
            dest = os.path.join(assets_folder, video)
            if not os.path.exists(dest):
                import shutil
                shutil.copy2(source, dest)
    
    app = Dash(__name__)
    
    # App layout
    app.layout = html.Div([ 
        html.H1("ASL Gesture Kinematic Space Visualization", 
                style={'text-align': 'center'}), 
        
        html.H3("This dashboard shows a gesture kinematic space generated by computing dynamic time warping "
                "distances between ASL gesture kinematic 3D timeseries. Gestures that are closer together "
                "in the space are more kinematically similar.", 
                style={'text-align': 'center'}), 
        
        html.Div([ 
            html.Div([ 
                dcc.Graph( 
                    id='gesture-space', 
                    figure={}, 
                    style={'height': '80vh'} 
                ) 
            ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}), 
            
            html.Div([ 
                html.H4("Gesture Video", style={'text-align': 'center'}), 
                html.Video( 
                    id='gesture-video', 
                    controls=True, 
                    autoPlay=True, 
                    loop=True, 
                    style={'width': '100%'} 
                ) 
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px'}) 
        ]), 
        
        html.Div(id='selected-gesture', 
                 style={'text-align': 'center', 'padding': '20px'}) 
    ]) 
    
    @app.callback( 
        [Output('gesture-space', 'figure'), 
         Output('gesture-video', 'src'), 
         Output('selected-gesture', 'children')], 
        [Input('gesture-space', 'clickData')] 
    ) 
    def update_graph(click_data): 
        # Create scatter plot 
        fig = px.scatter( 
            df, 
            x='x', 
            y='y', 
            hover_data=['gesture'], 
            template='plotly_dark', 
            labels={'x': 'UMAP Dimension 1', 
                   'y': 'UMAP Dimension 2'}, 
            title='Gesture Kinematic Space' 
        ) 
        
        fig.update_traces( 
            marker=dict(size=15), 
            marker_color='#00CED1', 
            opacity=0.7 
        ) 
        
        fig.update_layout( 
            hoverlabel=dict( 
                bgcolor="white", 
                font_size=16, 
                font_family="Rockwell" 
            ) 
        ) 
        
        # Handle video selection
        video_src = ''
        gesture_info = "Click on any point to view the gesture video"
        
        if click_data is not None:
            selected = click_data['points'][0]
            gesture = selected['customdata'][0]
            video_src = f'assets/{gesture}_tracked.mp4'
            gesture_info = f"Selected Gesture: {gesture}"
        
        return fig, video_src, gesture_info
    
    return app

def find_all_videos(folder: str, pattern: str = "*.mp4") -> List[str]:
    """
    Recursively find all video files in a folder and its subfolders.
    
    Args:
        folder: Root folder to search
        pattern: File pattern to match
        
    Returns:
        List of full paths to video files
    """
    videos = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.mp4'):
                videos.append(os.path.join(root, file))
    return videos

def retrack_gesture_videos(
    input_folder: str,
    output_folder: str,
    video_pattern: str = "*.mp4"
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Retrack gesture videos using MediaPipe world landmarks and save visualization.
    Now also tracks and saves visibility scores separately.
    
    Args:
        input_folder: Folder containing input videos
        output_folder: Folder to save tracked data
        video_pattern: Pattern to match video files
        
    Returns:
        Dictionary mapping video names to tuples of (landmarks, visibility scores)
    """
    os.makedirs(output_folder, exist_ok=True)
    tracked_folder = os.path.join(output_folder, "tracked_videos")
    os.makedirs(tracked_folder, exist_ok=True)
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    tracked_data = {}
    
    # Find all videos recursively
    video_paths = find_all_videos(input_folder, video_pattern)
    
    # Process each video
    for video_path in video_paths:
        video_name = Path(video_path).stem
        print(f"Processing {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        out_path = os.path.join(tracked_folder, f"{video_name}_tracked.mp4")
        out = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
        
        # Store world landmarks, visibility, and frame indices
        world_landmarks = []
        visibility_scores = []
        frame_indices = []
        
        with mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True
        ) as pose:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                if results.pose_world_landmarks:
                    # Extract world landmarks
                    frame_landmarks = [coord for landmark in results.pose_world_landmarks.landmark 
                                    for coord in (landmark.x, landmark.y, landmark.z)]
                    
                    # Extract visibility scores separately
                    frame_visibility = [landmark.visibility for landmark in results.pose_world_landmarks.landmark]
                    
                    world_landmarks.append(frame_landmarks)
                    visibility_scores.append(frame_visibility)
                    frame_indices.append(frame_idx)
                    
                    # Draw pose on frame
                    annotated_frame = frame.copy()
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                else:
                    # For frames without landmarks, just write the original frame
                    annotated_frame = frame
                    
                out.write(annotated_frame)
                frame_idx += 1
                
            cap.release()
            out.release()
        
        if world_landmarks:
            # Convert landmarks to numpy array
            landmarks_array = np.array(world_landmarks)
            visibility_array = np.array(visibility_scores)
            frame_indices = np.array(frame_indices)
            
            # Reshape landmarks to (frames, num_keypoints, 3)
            num_landmarks = landmarks_array.shape[1] // 3
            landmarks_array = landmarks_array.reshape(-1, num_landmarks, 3)
            
            # Create full arrays with all frames
            full_landmarks = np.zeros((total_frames, num_landmarks, 3))
            full_visibility = np.zeros((total_frames, num_landmarks))
            
            # Fill detected frames
            full_landmarks[frame_indices] = landmarks_array
            full_visibility[frame_indices] = visibility_array
            
            # Fill missing frames with nearest neighbor
            missing_indices = np.setdiff1d(np.arange(total_frames), frame_indices)
            
            if len(missing_indices) > 0:
                print(f"Filling {len(missing_indices)} missing frames with nearest neighbor values")
                
                for missing_idx in missing_indices:
                    # Find nearest detected frame
                    nearest_idx = frame_indices[np.abs(frame_indices - missing_idx).argmin()]
                    full_landmarks[missing_idx] = full_landmarks[nearest_idx]
                    full_visibility[missing_idx] = full_visibility[nearest_idx]
            
            # Apply smoothing (Gaussian filter) to landmarks
            smoothed = np.zeros_like(full_landmarks)
            for i in range(full_landmarks.shape[1]):  # Iterate over keypoints
                smoothed[:, i] = gaussian_filter1d(
                    full_landmarks[:, i], 
                    sigma=1  # Adjust sigma as needed
                )
            
            # Save smoothed landmarks
            landmarks_save_path = os.path.join(output_folder, f"{video_name}_world_landmarks.npy")
            np.save(landmarks_save_path, smoothed)
            
            # Save visibility scores
            visibility_save_path = os.path.join(output_folder, f"{video_name}_visibility.npy")
            np.save(visibility_save_path, full_visibility)
            
            tracked_data[video_name] = (smoothed, full_visibility)
    
    return tracked_data

def setup_dashboard_folders(data_folder: str, assets_folder: str) -> None:
    """
    Set up necessary folders for the dashboard.
    
    Args:
        data_folder: Path to analysis data folder
        assets_folder: Path to Dash assets folder
    """
    # Create assets folder if it doesn't exist
    os.makedirs(assets_folder, exist_ok=True)
    
    # Adjust path to tracked videos to point to the retracked directory
    retracked_folder = os.path.join(os.path.dirname(data_folder), "retracked", "tracked_videos")
    if not os.path.exists(retracked_folder):
        raise FileNotFoundError(f"Tracked videos folder not found at {retracked_folder}")
        
    # Copy videos if they don't exist in assets
    for video in os.listdir(retracked_folder):
        if video.endswith("_tracked.mp4"):
            source = os.path.join(retracked_folder, video)
            dest = os.path.join(assets_folder, video)
            if not os.path.exists(dest):
                import shutil
                print(f"Copying {video} to assets folder...")
                shutil.copy2(source, dest)
                
    # Correct path to visualization data from the analysis folder
    viz_path = os.path.join(data_folder, "gesture_visualization.csv")
    if not os.path.exists(viz_path):
        raise FileNotFoundError(f"Visualization data not found at {viz_path}")
        
    print(f"Dashboard folders set up successfully:")
    print(f"- Assets folder: {assets_folder}")
    print(f"- Data folder: {data_folder}")
    print(f"- {len(os.listdir(assets_folder))} videos in assets")


