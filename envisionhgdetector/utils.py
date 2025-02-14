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
    output_path: str 
) -> None:
    """
    Label a video with predicted gestures based on segments
    
    Args:
        video_path: Path to input video
        segments: DataFrame containing video segments 
            (must have columns: start_time, end_time, label)
        output_path: Path to save labeled video
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Color mapping for labels
    color_map = {
        'NoGesture': (50, 50, 50),      # Dark gray
        'Gesture': (0, 204, 204),        # Vibrant teal
        'Move': (255, 94, 98)            # Soft coral red
    }
    
    # Prepare segment lookup
    def get_label_at_time(time: float) -> str:
        matching_segments = segments[
            (segments['start_time'] <= time) & 
            (segments['end_time'] >= time)
        ]
        return matching_segments['label'].iloc[0] if len(matching_segments) > 0 else 'NoGesture'
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_idx in range(frame_count):
        # Calculate current time
        current_time = frame_idx / fps
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get label for this time
        label = get_label_at_time(current_time)
        
        # Add text label to frame
        cv2.putText(
            frame, 
            label, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            color_map.get(label, (255, 255, 255)), 
            2
        )
        
        out.write(frame)
    
    # Release video objects
    cap.release()
    out.release()

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
    stride: int = 1
) -> np.ndarray:
    """Create sliding windows from feature sequence."""
    if len(features) < seq_length:
        return np.array([])
        
    windows = []
    for i in range(0, len(features) - seq_length + 1, stride):
        window = features[i:i + seq_length]
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
    
    # Initialize list to hold the extracted features
    all_features = []
    
    # Extract basic features (shoulders, elbows, wrists)
    for key, index in keypoint_indices.items():
        print(f"Debug: Extracting keypoint {key} at index {index}")
        feature = landmarks[:, index]  # Shape (N, 3) for each joint (x, y, z)
        
        # Check for missing data
        if np.any(np.isnan(feature)) or feature.size == 0:
            print(f"Debug: No data for keypoint {key}, skipping")
        else:
            print(f"Debug: Data for keypoint {key}: {feature}")
            all_features.append(feature.reshape(-1, 3))  # Keep the x, y, z separate
    
    # Mean center left hand fingers
    left_fingers = []
    for key, index in left_finger_indices.items():
        print(f"Debug: Extracting left finger keypoint {key} at index {index}")
        feature = landmarks[:, index]
        
        # Check for missing data
        if np.any(np.isnan(feature)) or feature.size == 0:
            print(f"Debug: No data for left finger keypoint {key}, skipping")
        else:
            print(f"Debug: Data for left finger keypoint {key}: {feature}")
            left_fingers.append(feature.reshape(-1, 3))  # Keep x, y, z separate
    
    # Mean center the left fingers if data is present
    if left_fingers:
        left_fingers = np.concatenate(left_fingers, axis=1)  # Shape (N, 9) for 3 fingers (3 x 3)
        left_fingers_mean = np.mean(left_fingers, axis=0)
        left_fingers_centered = left_fingers - left_fingers_mean  # Center the coordinates
        all_features.append(left_fingers_centered)
    
    # Mean center right hand fingers
    right_fingers = []
    for key, index in right_finger_indices.items():
        print(f"Debug: Extracting right finger keypoint {key} at index {index}")
        feature = landmarks[:, index]
        
        # Check for missing data
        if np.any(np.isnan(feature)) or feature.size == 0:
            print(f"Debug: No data for right finger keypoint {key}, skipping")
        else:
            print(f"Debug: Data for right finger keypoint {key}: {feature}")
            right_fingers.append(feature.reshape(-1, 3))  # Keep x, y, z separate
    
    # Mean center the right fingers if data is present
    if right_fingers:
        right_fingers = np.concatenate(right_fingers, axis=1)  # Shape (N, 9) for 3 fingers (3 x 3)
        right_fingers_mean = np.mean(right_fingers, axis=0)
        right_fingers_centered = right_fingers - right_fingers_mean  # Center the coordinates
        all_features.append(right_fingers_centered)
    
    # Ensure the features are a consistent length
    features = np.concatenate(all_features, axis=1)  # Shape (N, num_features)
    print(f"Debug: Final feature array shape: {features.shape}")
    
    # If features array is empty, return an empty array or handle error
    if features.size == 0:
        print("Debug: Features array is empty, returning empty array")
        return np.array([])

    # Return the features with shape (time_steps, num_features)
    return features



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
    
    # Spatial features
    space_use_left: int
    space_use_right: int
    mcneillian_max_left: float
    mcneillian_max_right: float
    mcneillian_mode_left: int
    mcneillian_mode_right: int
    volume_both: float
    volume_right: float
    volume_left: float
    max_height_right: float
    max_height_left: float
    
    # Temporal features
    duration: float
    hold_count: int
    hold_time: float
    hold_avg_duration: float
    
    # Submovement features - Hand
    hand_submovements_left: int
    hand_submovements_right: int
    hand_submovements_combined: int
    hand_submovement_peaks_left: List[float]
    hand_submovement_peaks_right: List[float]
    hand_submovement_peaks_combined: List[float]
    hand_mean_submovement_amplitude_left: float
    hand_mean_submovement_amplitude_right: float
    hand_mean_submovement_amplitude_combined: float
    
    # Submovement features - Elbow
    elbow_submovements_left: int
    elbow_submovements_right: int
    elbow_submovements_combined: int
    elbow_mean_submovement_amplitude_left: float
    elbow_mean_submovement_amplitude_right: float
    elbow_mean_submovement_amplitude_combined: float
    
    # Dynamic features - Hand
    hand_peak_velocity_right: float
    hand_peak_velocity_left: float
    hand_mean_velocity_right: float
    hand_mean_velocity_left: float
    hand_peak_acceleration_right: float
    hand_peak_acceleration_left: float
    hand_peak_deceleration_right: float
    hand_peak_deceleration_left: float
    hand_peak_jerk_right: float
    hand_peak_jerk_left: float
    
    # Dynamic features - Elbow
    elbow_peak_velocity_right: float
    elbow_peak_velocity_left: float
    elbow_mean_velocity_right: float
    elbow_mean_velocity_left: float
    elbow_peak_acceleration_right: float
    elbow_peak_acceleration_left: float
    elbow_peak_deceleration_right: float
    elbow_peak_deceleration_left: float
    elbow_peak_jerk_right: float
    elbow_peak_jerk_left: float

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
    """Define the grid based on a single frame, adapted for MediaPipe landmarks."""
    # Use MidHip and Neck for body center
    bodycent = df['Neck'][frame][1] - (df['Neck'][frame][1] - df['MidHip'][frame][1])/2
    
    # Use eye distance for face width (same as before)
    face_width = (df['LEye'][frame][0] - df['REye'][frame][0])*2
    
    # Use shoulder width instead of hip width since it's more reliable in MediaPipe
    body_width = df['LShoulder'][frame][0] - df['RShoulder'][frame][0]
    
    # Center-center boundaries (use shoulders scaled inward slightly)
    scale_factor = 0.7  # Adjust this to tune the center-center zone
    cc_xmin = df['RShoulder'][frame][0] + (body_width * (1-scale_factor)/2)
    cc_xmax = df['LShoulder'][frame][0] - (body_width * (1-scale_factor)/2)
    cc_len = cc_xmax - cc_xmin
    cc_ymin = bodycent - cc_len/2
    cc_ymax = bodycent + cc_len/2
    
    # Center boundaries (use full shoulder width)
    c_xmin = df['RShoulder'][frame][0] - body_width/2
    c_xmax = df['LShoulder'][frame][0] + body_width/2
    c_len = c_xmax - c_xmin
    c_ymin = bodycent - c_len/2
    c_ymax = bodycent + c_len/2
    
    # Periphery boundaries
    p_ymax = df['LEye'][frame][1] + (df['LEye'][frame][1] - df['Nose'][frame][1])
    p_ymin = bodycent - (p_ymax - bodycent)  # symmetrical around body center
    p_xmin = c_xmin - face_width
    p_xmax = c_xmax + face_width
    
    return cc_xmin, cc_xmax, cc_ymin, cc_ymax, c_xmin, c_xmax, c_ymin, c_ymax, p_xmin, p_xmax, p_ymin, p_ymax

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

def calc_mcneillian_space(df, hand_idx):
    """Calculate McNeillian space features."""
    # Initialize with empty lists for both hands
    Space_L = []
    Space_R = []
    
    if hand_idx == 'B':
        hands = ['L_Hand','R_Hand']
    else:
        hands = [hand_idx + '_Hand']
    
    # compare, at each frame, each hand to the (sub)section limits
    for hand in hands:
        Space = []
        for frame in range(len(df)):
            cc_xmin, cc_xmax, cc_ymin, cc_ymax, c_xmin, c_xmax, c_ymin, c_ymax, p_xmin, p_xmax, p_ymin, p_ymax = \
            define_mcneillian_grid(df, frame)
            
            try:
                # Get hand position
                hand_pos = df[hand][frame]
                
                # centre-centre
                if cc_xmin < hand_pos[0] < cc_xmax and cc_ymin < hand_pos[1] < cc_ymax:
                    Space.append(1)
                # centre
                elif c_xmin < hand_pos[0] < c_xmax and c_ymin < hand_pos[1] < c_ymax:
                    Space.append(2)
                # periph
                elif p_xmin < hand_pos[0] < p_xmax and p_ymin < hand_pos[1] < p_ymax:
                    if cc_xmax < hand_pos[0]:
                        if cc_ymax < hand_pos[1]:
                            Space.append(31)
                        elif cc_ymin < hand_pos[1]:
                            Space.append(32)
                        else:
                            Space.append(33)
                    elif cc_xmin < hand_pos[0]:
                        if c_ymax < hand_pos[1]:
                            Space.append(38)
                        else:
                            Space.append(34)
                    else:
                        if cc_ymax < hand_pos[1]:
                            Space.append(37)
                        elif cc_ymin < hand_pos[1]:
                            Space.append(36)
                        else:
                            Space.append(35)
                else:  # extra periphery
                    if c_xmax < hand_pos[0]:
                        if cc_ymax < hand_pos[1]:
                            Space.append(41)
                        elif cc_ymin < hand_pos[1]:
                            Space.append(42)
                        else:
                            Space.append(43)
                    elif cc_xmin < hand_pos[0]:
                        if c_ymax < hand_pos[1]:
                            Space.append(48)
                        else:
                            Space.append(44)
                    else:
                        if c_ymax < hand_pos[1]:
                            Space.append(47)
                        elif c_ymin < hand_pos[1]:
                            Space.append(46)
                        else:
                            Space.append(45)
            except:
                # If there's any error processing a frame, append a default value
                Space.append(1)  # Default to center-center
                
        if hand == 'L_Hand':
            Space_L = Space
        else:
            Space_R = Space

    # Add safety for empty spaces
    if not Space_L:
        Space_L = [1]  # Default to center-center
    if not Space_R:
        Space_R = [1]  # Default to center-center

    # Calculate features with safety checks
    if hand_idx == 'L' or hand_idx == 'B':
        space_use_L = len(set(Space_L))
        mcneillian_maxL = 4 if max(Space_L) > 40 else (3 if max(Space_L) > 30 else max(Space_L))
        try:
            mcneillian_modeL = statistics.mode(Space_L)
        except:
            mcneillian_modeL = Space_L[0]  # Use first value if mode fails
    else:
        space_use_L = 0
        mcneillian_maxL = 0
        mcneillian_modeL = 0

    if hand_idx == 'R' or hand_idx == 'B':
        space_use_R = len(set(Space_R))
        mcneillian_maxR = 4 if max(Space_R) > 40 else (3 if max(Space_R) > 30 else max(Space_R))
        try:
            mcneillian_modeR = statistics.mode(Space_R)
        except:
            mcneillian_modeR = Space_R[0]  # Use first value if mode fails
    else:
        space_use_R = 0
        mcneillian_maxR = 0
        mcneillian_modeR = 0

    return space_use_L, space_use_R, mcneillian_maxL, mcneillian_maxR, mcneillian_modeL, mcneillian_modeR

def calc_vert_height(df, hand):
    """
    Calculate vertical height features.
    
    Args:
        df: DataFrame with pose keypoints 
        hand: Which hand to analyze ('L', 'R', or 'B' for both)
        
    Returns:
        Maximum height value
    """
    # Vertical amplitude
    # H: 0 = below midline;
    #    1 = between midline and middle-upper body;
    #    2 = above middle-upper body, but below shoulders;
    #    3 = between shoulders nad middle of face;
    #    4 = between middle of face and top of head;
    #    5 = above head

    H = []
    for index, frame in df.iterrows():
        SP_mid = ((df.loc[index, "Neck"][1] - df.loc[index, "MidHip"][1]) / 2) + df.loc[index, "MidHip"][1]
        Mid_up = ((df.loc[index, "Nose"][1] - df.loc[index, "Neck"][1]) / 2) + df.loc[index, "Neck"][1]
        Eye_mid = (df.loc[index, "REye"][1] + df.loc[index, "LEye"][1] / 2)  # mean of the two eyes vert height
        Head_TP = ((df.loc[index, "Nose"][1] - Eye_mid) * 2) + df.loc[index, "Nose"][1]

        if hand == "B":
            hand_height = max([df.loc[index, "R_Hand"][1], df.loc[index, "L_Hand"][1]])
        else:
            hand_str = hand + "_Hand"
            hand_height = df.loc[index][hand_str][1]

        if hand_height > SP_mid:
            if hand_height > Mid_up:
                if hand_height > df.loc[index, "Neck"][1]:
                    if hand_height > df.loc[index, "Nose"][1] :
                        if hand_height > Head_TP:
                            H.append(5)
                        else:
                            H.append(4)
                    else:
                        H.append(3)
                else:
                    H.append(2)
            else:
                H.append(1)
        else:
            H.append(0)
    MaxHeight = max(H)
    return MaxHeight

def find_movepauses(velocity_array):
    """
    Find moments when velocity is below a threshold.
    
    Args:
        velocity_array: Array of velocities
        
    Returns:
        Array of indices for pause moments
    """
    # We are using a 0.015m/s threshold, but this can be adjusted
    pause_ix = []
    for index, velpoint in enumerate(velocity_array):
        if velpoint < 0.015:
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
    fps: float = 25.0,
    gesture_id: str = "",
    video_id: str = ""
) -> KinematicFeatures:
    """
    Compute comprehensive kinematic features from landmark data.
    
    Args:
        landmarks: Numpy array of shape (frames, joints, 3) containing 3D landmark positions
        fps: Frames per second of the video
        gesture_id: Identifier for the gesture
        video_id: Identifier for the video
        
    Returns:
        KinematicFeatures object containing all computed features
    """
    # Convert landmarks to DataFrame for compatibility with existing functions
    df = pd.DataFrame()
    
    # Extract relevant joint positions
    for joint in ['L_Hand', 'R_Hand', 'LElb', 'RElb', 'LShoulder', 'RShoulder', 
                 'Neck', 'MidHip', 'LEye', 'REye', 'Nose']:
        df[joint] = [landmarks[i, joint_map[joint]] for i in range(len(landmarks))]
        
    # Calculate McNeillian space features
    space_use_L, space_use_R, mcneillian_maxL, mcneillian_maxR, mcneillian_modeL, mcneillian_modeR = \
        calc_mcneillian_space(df, 'B')
        
    # Calculate vertical height features
    max_height_R = calc_vert_height(df, "R")
    max_height_L = calc_vert_height(df, "L")
    
    # Calculate volume features
    volume_both = calc_volume_size(df, 'B')
    volume_right = calc_volume_size(df, 'R')
    volume_left = calc_volume_size(df, 'L')
    
    # Compute kinematics for each arm segment
    r_hand = compute_limb_kinematics(np.array([p for p in df['R_Hand']]), fps)
    l_hand = compute_limb_kinematics(np.array([p for p in df['L_Hand']]), fps)
    r_elbow = compute_limb_kinematics(np.array([p for p in df['RElb']]), fps)
    l_elbow = compute_limb_kinematics(np.array([p for p in df['LElb']]), fps)
    
    # Compute combined hand movement
    combined_hand_speed = np.linalg.norm(r_hand.velocity + l_hand.velocity, axis=1)
    combined_hand_peaks, combined_hand_heights = find_submovements(combined_hand_speed, fps)
    
    # Compute combined elbow movement
    combined_elbow_speed = np.linalg.norm(r_elbow.velocity + l_elbow.velocity, axis=1)
    combined_elbow_peaks, combined_elbow_heights = find_submovements(combined_elbow_speed, fps)
    
    # Calculate hold features using hand peaks
    hold_count, hold_time, hold_avg = calc_holds(df, l_hand.peaks, r_hand.peaks, fps, 'B')
    
    # Safe mean calculation helper
   # Define safe computation helpers
    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    def safe_max(arr):
        return float(np.max(arr)) if len(arr) > 0 else 0.0

    def safe_min(arr):
        return float(np.min(arr)) if len(arr) > 0 else 0.0

    def safe_norm(arr, axis=1):
        if len(arr) > 0:
            return np.linalg.norm(arr, axis=axis)
        return np.zeros(1)

    return KinematicFeatures(
        gesture_id=gesture_id,
        video_id=video_id,
        space_use_left=space_use_L,
        space_use_right=space_use_R,
        mcneillian_max_left=mcneillian_maxL,
        mcneillian_max_right=mcneillian_maxR,
        mcneillian_mode_left=mcneillian_modeL,
        mcneillian_mode_right=mcneillian_modeR,
        volume_both=volume_both,
        volume_right=volume_right,
        volume_left=volume_left,
        max_height_right=max_height_R,
        max_height_left=max_height_L,
        duration=len(landmarks) / fps,
        hold_count=hold_count,
        hold_time=hold_time,
        hold_avg_duration=hold_avg,
        
        # Hand submovements
        hand_submovements_left=len(l_hand.peaks),
        hand_submovements_right=len(r_hand.peaks),
        hand_submovements_combined=len(combined_hand_peaks),
        hand_submovement_peaks_left=l_hand.peak_heights.tolist() if len(l_hand.peak_heights) > 0 else [0],
        hand_submovement_peaks_right=r_hand.peak_heights.tolist() if len(r_hand.peak_heights) > 0 else [0],
        hand_submovement_peaks_combined=combined_hand_heights.tolist() if len(combined_hand_heights) > 0 else [0],
        hand_mean_submovement_amplitude_left=safe_mean(l_hand.peak_heights),
        hand_mean_submovement_amplitude_right=safe_mean(r_hand.peak_heights),
        hand_mean_submovement_amplitude_combined=safe_mean(combined_hand_heights),
        
        # Elbow submovements
        elbow_submovements_left=len(l_elbow.peaks),
        elbow_submovements_right=len(r_elbow.peaks),
        elbow_submovements_combined=len(combined_elbow_peaks),
        elbow_mean_submovement_amplitude_left=safe_mean(l_elbow.peak_heights),
        elbow_mean_submovement_amplitude_right=safe_mean(r_elbow.peak_heights),
        elbow_mean_submovement_amplitude_combined=safe_mean(combined_elbow_heights),
        
        # Hand dynamics
        hand_peak_velocity_right=safe_max(safe_norm(r_hand.velocity)),
        hand_peak_velocity_left=safe_max(safe_norm(l_hand.velocity)),
        hand_mean_velocity_right=safe_mean(safe_norm(r_hand.velocity)),
        hand_mean_velocity_left=safe_mean(safe_norm(l_hand.velocity)),
        hand_peak_acceleration_right=safe_max(safe_norm(r_hand.acceleration)),
        hand_peak_acceleration_left=safe_max(safe_norm(l_hand.acceleration)),
        hand_peak_deceleration_right=safe_min(safe_norm(r_hand.acceleration)),
        hand_peak_deceleration_left=safe_min(safe_norm(l_hand.acceleration)),
        hand_peak_jerk_right=safe_max(safe_norm(r_hand.jerk)),
        hand_peak_jerk_left=safe_max(safe_norm(l_hand.jerk)),
        
        # Elbow dynamics
        elbow_peak_velocity_right=safe_max(safe_norm(r_elbow.velocity)),
        elbow_peak_velocity_left=safe_max(safe_norm(l_elbow.velocity)),
        elbow_mean_velocity_right=safe_mean(safe_norm(r_elbow.velocity)),
        elbow_mean_velocity_left=safe_mean(safe_norm(l_elbow.velocity)),
        elbow_peak_acceleration_right=safe_max(safe_norm(r_elbow.acceleration)),
        elbow_peak_acceleration_left=safe_max(safe_norm(l_elbow.acceleration)),
        elbow_peak_deceleration_right=safe_min(safe_norm(r_elbow.acceleration)),
        elbow_peak_deceleration_left=safe_min(safe_norm(l_elbow.acceleration)),
        elbow_peak_jerk_right=safe_max(safe_norm(r_elbow.jerk)),
        elbow_peak_jerk_left=safe_max(safe_norm(l_elbow.jerk))
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
        'space_use_left': f.space_use_left,
        'space_use_right': f.space_use_right,
        'mcneillian_max_left': f.mcneillian_max_left,
        'mcneillian_max_right': f.mcneillian_max_right,
        'mcneillian_mode_left': f.mcneillian_mode_left,
        'mcneillian_mode_right': f.mcneillian_mode_right,
        'volume_both': f.volume_both,
        'volume_right': f.volume_right,
        'volume_left': f.volume_left,
        'max_height_right': f.max_height_right,
        'max_height_left': f.max_height_left,
        'duration': f.duration,
        'hold_count': f.hold_count,
        'hold_time': f.hold_time,
        'hold_avg_duration': f.hold_avg_duration,
        'hand_submovements_left': f.hand_submovements_left,
        'hand_submovements_right': f.hand_submovements_right,
        'hand_submovements_combined': f.hand_submovements_combined,
        'hand_mean_submovement_amplitude_left': f.hand_mean_submovement_amplitude_left,
        'hand_mean_submovement_amplitude_right': f.hand_mean_submovement_amplitude_right,
        'hand_mean_submovement_amplitude_combined': f.hand_mean_submovement_amplitude_combined,
        'elbow_submovements_left': f.elbow_submovements_left,
        'elbow_submovements_right': f.elbow_submovements_right,
        'elbow_submovements_combined': f.elbow_submovements_combined,
        'elbow_mean_submovement_amplitude_left': f.elbow_mean_submovement_amplitude_left,
        'elbow_mean_submovement_amplitude_right': f.elbow_mean_submovement_amplitude_right,
        'elbow_mean_submovement_amplitude_combined': f.elbow_mean_submovement_amplitude_combined,
        'hand_peak_velocity_right': f.hand_peak_velocity_right,
        'hand_peak_velocity_left': f.hand_peak_velocity_left,
        'hand_mean_velocity_right': f.hand_mean_velocity_right,
        'hand_mean_velocity_left': f.hand_mean_velocity_left,
        'hand_peak_acceleration_right': f.hand_peak_acceleration_right,
        'hand_peak_acceleration_left': f.hand_peak_acceleration_left,
        'hand_peak_deceleration_right': f.hand_peak_deceleration_right,
        'hand_peak_deceleration_left': f.hand_peak_deceleration_left,
        'hand_peak_jerk_right': f.hand_peak_jerk_right,
        'hand_peak_jerk_left': f.hand_peak_jerk_left,
        'elbow_peak_velocity_right': f.elbow_peak_velocity_right,
        'elbow_peak_velocity_left': f.elbow_peak_velocity_left,
        'elbow_mean_velocity_right': f.elbow_mean_velocity_right,
        'elbow_mean_velocity_left': f.elbow_mean_velocity_left,
        'elbow_peak_acceleration_right': f.elbow_peak_acceleration_right,
        'elbow_peak_acceleration_left': f.elbow_peak_acceleration_left,
        'elbow_peak_deceleration_right': f.elbow_peak_deceleration_right,
        'elbow_peak_deceleration_left': f.elbow_peak_deceleration_left,
        'elbow_peak_jerk_right': f.elbow_peak_jerk_right,
        'elbow_peak_jerk_left': f.elbow_peak_jerk_left
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
) -> Dict[str, np.ndarray]:
    """
    Retrack gesture videos using MediaPipe world landmarks and save visualization.
    Missing frames are filled with nearest neighbor values.
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
        
        # Store world landmarks and frame indices
        world_landmarks = []
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
                    world_landmarks.append(frame_landmarks)
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
            # Convert to numpy array
            landmarks_array = np.array(world_landmarks)
            frame_indices = np.array(frame_indices)
            
            # Reshape landmarks to (frames, num_keypoints, 3)
            num_landmarks = landmarks_array.shape[1] // 3
            landmarks_array = landmarks_array.reshape(-1, num_landmarks, 3)
            
            # Create full array with all frames
            full_landmarks = np.zeros((total_frames, num_landmarks, 3))
            
            # Fill detected frames
            full_landmarks[frame_indices] = landmarks_array
            
            # Fill missing frames with nearest neighbor
            missing_indices = np.setdiff1d(np.arange(total_frames), frame_indices)
            
            if len(missing_indices) > 0:
                print(f"Filling {len(missing_indices)} missing frames with nearest neighbor values")
                
                for missing_idx in missing_indices:
                    # Find nearest detected frame
                    nearest_idx = frame_indices[np.abs(frame_indices - missing_idx).argmin()]
                    full_landmarks[missing_idx] = full_landmarks[nearest_idx]
            
            # Apply smoothing (Gaussian filter)
            smoothed = np.zeros_like(full_landmarks)
            for i in range(full_landmarks.shape[1]):  # Iterate over keypoints
                smoothed[:, i] = gaussian_filter1d(
                    full_landmarks[:, i], 
                    sigma=1  # Adjust sigma as needed
                )
            
            # Save smoothed landmarks
            save_path = os.path.join(output_folder, f"{video_name}_world_landmarks.npy")
            np.save(save_path, smoothed)
            
            tracked_data[video_name] = smoothed
    
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


