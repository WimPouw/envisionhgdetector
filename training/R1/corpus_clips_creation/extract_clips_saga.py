import os
import glob
import csv
import re
from utils import *

def parse_csv_annotation(csv_file):
    """Parse the CSV annotation file and return list of gestures"""
    gestures = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Detect column names for time and gesture
            reader = csv.reader(f)
            header = next(reader)
            
            # Find column indices
            start_col = -1
            end_col = -1
            gesture_col = -1
            
            for i, col_name in enumerate(header):
                if 'begin time' in col_name.lower() or 'start' in col_name.lower():
                    start_col = i
                elif 'end time' in col_name.lower() or 'stop' in col_name.lower():
                    end_col = i
                elif 'gesture' in col_name.lower() or 'phrase' in col_name.lower():
                    gesture_col = i
            
            if start_col == -1 or end_col == -1 or gesture_col == -1:
                print(f"Could not identify required columns in {csv_file}")
                print(f"Header: {header}")
                return []
            
            # Reset file pointer and skip header
            f.seek(0)
            next(reader)
            
            # Parse rows
            for row in reader:
                try:
                    if len(row) > max(start_col, end_col, gesture_col):
                        start_time = int(row[start_col]) / 1000.0  # Convert ms to seconds
                        end_time = int(row[end_col]) / 1000.0      # Convert ms to seconds
                        gesture_type = row[gesture_col].strip()
                        
                        if gesture_type and end_time > start_time:
                            gestures.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                'type': gesture_type
                            })
                except ValueError as e:
                    print(f"Error parsing row: {row}, Error: {e}")
                    continue
                
    except Exception as e:
        print(f"Error processing CSV file {csv_file}: {e}")
    
    return gestures

def clean_gesture_name(gesture_name):
    """Clean gesture name for use in filenames"""
    # Replace non-alphanumeric characters with underscore
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', gesture_name)
    # Remove consecutive underscores
    clean_name = re.sub(r'_+', '_', clean_name)
    # Remove leading and trailing underscores
    clean_name = clean_name.strip('_')
    # Ensure name is not empty
    if not clean_name:
        clean_name = "unknown"
    return clean_name

def process_video(video_path, annotations_folder, vars):
    """Process a single video file with its corresponding annotation file"""
    gesture_dir = vars['gesture_dir']
    no_gesture_dir = vars['no_gesture_dir']
    no_gesture_label = vars['no_gesture_label']
    
    # Extract video ID from filename (e.g., "V7" from "V7.mp4")
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # Get the annotation file path
    annotation_file = os.path.join(annotations_folder, f"{video_id}.csv")
    
    if not os.path.exists(annotation_file):
        print(f"No annotation file found for {video_path}")
        return
    
    # Get video duration
    video_duration = get_video_info(video_path)['duration']
    
    if video_duration == 0:
        print(f"Could not determine duration for {video_path}")
        return
        
    print(f"Video duration: {video_duration:.2f} seconds")
    
    # Parse annotation file
    gestures = parse_csv_annotation(annotation_file)
    print(f"Found {len(gestures)} gestures in {video_id}")
    
    # Filter valid gestures (within video duration)
    valid_gestures = [g for g in gestures if g['end_time'] <= video_duration and g['start_time'] >= 0]
    
    # Process each gesture
    for idx, gesture in enumerate(valid_gestures):
        # Clean gesture type for filename
        safe_type = clean_gesture_name(gesture['type'])
        
        # Output filename using DATASET_VIDEOID_INDEX_TYPE format
        gesture_output = os.path.join(
            gesture_dir, 
            f'SAGA_{video_id}_{idx:04d}_{safe_type}.mp4'
        )
        
        if not os.path.exists(gesture_output):
            print(f"Extracting gesture clip {idx} from {video_id}: "
                    f"{gesture['start_time']:.2f}s - {gesture['end_time']:.2f}s "
                    f"(Type: {gesture['type']}) with 1s padding")
                    
            if extract_clip_with_padding(video_path, gesture_output, 
                                        gesture['start_time'], gesture['end_time'], 
                                        video_duration, padding=1.0):
                print(f"Successfully extracted gesture clip: {gesture_output}")
            else:
                print(f"Failed to extract gesture clip: {gesture_output}")

    # create non-gesture clips
    gaps = find_no_gesture_intervals(valid_gestures, video_duration)
    for gesture_clip in valid_gestures:
        no_gesture_clip = find_matching_no_gesture_clip(gaps, gesture_clip['duration'])
        if no_gesture_clip:
            gaps = consume_gap(gaps, no_gesture_clip['start_time'], no_gesture_clip['end_time'])
            no_gesture_clip_output_path = os.path.join(no_gesture_dir, f"SAGAplus_{video_id}_{gesture_clip['gesture_idx']:04d}_{no_gesture_label}.mp4")
            extract_clip_with_padding(video_path, no_gesture_clip_output_path, no_gesture_clip['start_time'], no_gesture_clip['end_time'], video_duration, padding=1.0)
        else:
            print(f"Could not find suitable no-gesture interval for gesture index {gesture_clip['gesture_idx']} with duration {gesture_clip['duration']:.2f}s")
      
def extract_clips_saga(video_dir, vars):
    # Configuration paths
    videos_folder = f'{video_dir}/OriginalVideos/'
    annotations_folder = f'{video_dir}/OriginalCodings/'

    # Get list of videos
    video_extensions = ['.mp4', '.avi', '.mov']
    video_list = []
    
    for ext in video_extensions:
        video_list.extend(glob.glob(os.path.join(videos_folder, f'*{ext}')))
    
    print(f"Found {len(video_list)} videos to process")
    
    # Process each video
    for video_path in sorted(video_list):
        print(f"\nProcessing video: {video_path}")
        process_video(video_path, annotations_folder, vars)