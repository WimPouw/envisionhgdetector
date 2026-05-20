import os
import random
import uuid
import pandas as pd
import glob as glob
from utils import *

# find_no_gesture_intervals -- see if you can merge these
def get_non_gesture_intervals(stroke_df, video_duration, min_gap=0.01):
    """
    Find intervals where no gesture occurs
    
    Args:
        stroke_df: DataFrame with stroke segments
        video_duration: Total duration of video in seconds
        min_gap: Minimum gap size to consider (seconds)
    
    Returns:
        List of (start, end) tuples for valid non-gesture intervals
    """
    # Convert milliseconds to seconds and sort segments
    segments = stroke_df[['start_segment', 'end_segment']].values / 1000
    segments = sorted(segments, key=lambda x: x[0])
    
    # Find gaps between strokes
    gaps = []
    last_end = 0
    
    for start, end in segments:
        if start - last_end >= min_gap:
            gaps.append({
                'start_time': last_end,
                'end_time': start
            })
        last_end = end
    
    # Add final gap if exists
    if video_duration - last_end >= min_gap:
        gaps.append({
            'start_time': last_end,
            'end_time': video_duration
        })
        
    return gaps

def process_video(annotation_file_path, video_dir, vars):
    gesture_dir = vars['gesture_dir']
    no_gesture_dir = vars['no_gesture_dir']
    gesture_label = vars['gesture_label']
    no_gesture_label = vars['no_gesture_label']

    # Column names
    columns = ['tier', 'empty', 'start_segment', 'end_segment', 'label']
    df = pd.read_csv(annotation_file_path)
    df.columns = columns
    stroke_df = df[df['label'] == 'stroke'].copy()
    
    # Get corresponding video file
    base_name = os.path.basename(annotation_file_path).replace('.csv', '')
    video_file = os.path.join(video_dir, f"{base_name}.mp4")
    if not os.path.exists(video_file):
        print(f"Warning: Video file not found for {annotation_file_path}")
        return
    
    video_duration = get_video_info(video_file)['duration']
    if video_duration == 0:
        print(f"Warning: Could not get duration for video {video_file}")
        return
    
    valid_gestures = []
    for idx, row in stroke_df.iterrows():
        # Time in milliseconds, convert to seconds
        gesture_duration = (row['end_segment'] - row['start_segment']) / 1000
        gesture_start = row['start_segment'] / 1000
        gesture_end = row['end_segment'] / 1000
        unique_id = str(uuid.uuid4())[:8]  # Using first 8 characters of UUID

        # Save stroke segment with original video name
        gesture_output = os.path.join(gesture_dir, f'{base_name}_{unique_id}_{gesture_label}.mp4')
        
        if extract_clip_with_padding(video_file, gesture_output, gesture_start, gesture_end, video_duration, padding=1.0):
            print(f'Saved gesture segment: {gesture_output}')
            valid_gestures.append({
                'gesture_idx': unique_id,
                'duration': gesture_duration,
                'start_time': gesture_start,
                'end_time': gesture_end
            })
        else:
            print(f"Failed to extract gesture segment: {gesture_output}")
        
    # Find and save matching non-gesture segment
    gaps = get_non_gesture_intervals(stroke_df, video_duration)
    if not gaps:
        print(f"No valid non-gesture intervals found for video {video_file}")
        return
    
    for gesture in valid_gestures:
        no_gesture_clip = find_matching_no_gesture_clip(gaps, gesture['duration'])
        if no_gesture_clip:
            # Remove the chosen no-gesture interval from the remaining gaps to avoid duplicates
            gaps = consume_gap(gaps, no_gesture_clip['start_time'], no_gesture_clip['end_time'])
            
            no_gesture_output = os.path.join(no_gesture_dir, f'{base_name}_{gesture["gesture_idx"]}_{no_gesture_label}.mp4')
            if extract_clip_with_padding(video_file, no_gesture_output, no_gesture_clip['start_time'], no_gesture_clip['end_time'], video_duration, padding=1.0):
                print(f'Saved no-gesture segment: {no_gesture_output}')
            else:
                print(f"Failed to extract no-gesture segment: {no_gesture_output}")
        else:
            print(f"Could not find suitable non-gesture interval for gesture index {gesture['gesture_idx']} with duration {gesture['duration']:.2f}s")
    
def extract_clips_tedm3d(video_dir, vars):
    annotation_files = glob.glob(os.path.join(video_dir, '*.csv'))

    for annotation_file in annotation_files:
        print(f"Processing {annotation_file}...")
        process_video(annotation_file, video_dir, vars)