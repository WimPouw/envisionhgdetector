import numpy as np
import pandas as pd
from typing import Dict, List
import os
import cv2
import pandas as pd
import numpy as np
import time
from typing import Dict, Optional

def create_segments(
    annotations: pd.DataFrame,
    label_column: str,
    min_gap_s: float = 0.3,
    min_length_s: float = 0.5
) -> pd.DataFrame:
    """
    Create segments from frame-by-frame annotations.
    
    Args:
        annotations: DataFrame with predictions
        label_column: Name of label column
        min_gap_s: Minimum gap between segments in seconds
        min_length_s: Minimum segment length in seconds
    """
    is_gesture = annotations[label_column] == 'Gesture'
    is_move = annotations[label_column] == 'Move'
    is_any_gesture = is_gesture | is_move
    
    if not is_any_gesture.any():
        return pd.DataFrame(
            columns=['start_time', 'end_time', 'labelid', 'label']
        )
    
    # Find state changes
    changes = np.diff(is_any_gesture.astype(int), prepend=0)
    start_idxs = np.where(changes == 1)[0]
    end_idxs = np.where(changes == -1)[0]
    
    if len(start_idxs) > len(end_idxs):
        end_idxs = np.append(end_idxs, len(annotations) - 1)
    
    segments = []
    i = 0
    
    while i < len(start_idxs):
        start_idx = start_idxs[i]
        end_idx = end_idxs[i]
        
        start_time = annotations.iloc[start_idx]['time']
        end_time = annotations.iloc[end_idx]['time']
        
        segment_labels = annotations.loc[
            start_idx:end_idx,
            label_column
        ]
        current_label = segment_labels.mode()[0]
        
        # Check segment duration
        if end_time - start_time >= min_length_s:
            if current_label != 'NoGesture':
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'labelid': len(segments) + 1,
                    'label': current_label,
                    'duration': end_time - start_time
                })
        
        i += 1
    
    return pd.DataFrame(segments)

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