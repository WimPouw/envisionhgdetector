import numpy as np
import pandas as pd
from typing import Dict, List
import os
import cv2
import pandas as pd
import numpy as np
import time
from typing import Dict, Optional
import glob
from moviepy.video.io.VideoFileClip import VideoFileClip

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

# functions for label videos and elan
import os
import cv2
import pandas as pd
import numpy as np
import time
from typing import Dict, Optional

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