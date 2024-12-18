
import numpy as np
import pandas as pd
from typing import Dict, List

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