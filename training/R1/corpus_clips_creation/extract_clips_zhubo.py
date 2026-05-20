import glob
import os
import json
from utils import *

def process_video(video_path, vars):
    """Process a single video file"""
    gesture_dir = vars['gesture_dir']
    no_gesture_dir = vars['no_gesture_dir']
    no_gesture_label = vars['no_gesture_label']

    # Get the video identifier (filename without extension)
    base_filename = os.path.splitext(os.path.basename(video_path))[0].replace('.h264', '')
    
    # Extract speaker ID and gesture ID from the filename
    # Assuming filename format like "123-456.h264.mp4" where 123 is speakerID and 456 is gestureID
    try:
        filename_parts = base_filename.split('-')
        speaker_id = filename_parts[0]
        gesture_base_id = filename_parts[1] if len(filename_parts) > 1 else "unknown"
    except Exception as e:
        print(f"Error parsing filename {base_filename}: {e}")
        speaker_id = "unknown"
        gesture_base_id = "unknown"
    
    # Get the actions file path
    actions_file = video_path.replace('.h264.mp4', '.actions.json')
    
    try:
        # Get video information
        video_info = get_video_info(video_path)
        fps = video_info['fps']
        duration = video_info['duration']
        
        if fps == 0 or duration == 0:
            print(f"Could not get video info for {video_path}")
            return
            
        print(f"Video info: {duration:.2f}s, {fps:.2f}fps")
        
        # Load actions
        with open(actions_file, 'r') as f:
            actions = json.load(f)
        
        gestures = []
        # Process gesture clips with 1 second padding
        for idx, action in enumerate(actions):
            # Calculate times with 1 second padding
            start_time = max(0, (action['start_frame'] / fps) - 1)  # Add 1 sec before, but don't go below 0
            end_time = min(duration, (action['end_frame'] / fps) + 1)  # Add 1 sec after, but don't exceed duration
            
            # Create the new filename using the requested format: ZHUBO_SPEAKERID_GESTUREID_NA
            # Where NA is the index of the gesture in the actions file
            output_file = os.path.join(gesture_dir, f'ZHUBO_{speaker_id}_{gesture_base_id}_{idx:04d}.mp4')
            
            if not os.path.exists(output_file):  # Only extract if file doesn't exist
                print(f"Extracting gesture clip {idx} from ZHUBO_{speaker_id}_{gesture_base_id}: {start_time:.2f}s - {end_time:.2f}s (with padding)")
                if extract_clip_with_padding(video_path, output_file, start_time, end_time, duration):
                    print(f"Successfully extracted gesture clip: {output_file}")
                gestures.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })
            else:
                print(f"Skipping existing gesture clip: {output_file}")
        
        valid_gestures = [g for g in gestures if g['end_time'] <= duration and g['start_time'] >= 0]

        # Extract matching no-gesture clips
        gaps = find_no_gesture_intervals(valid_gestures, duration)
        no_gesture_clip = find_matching_no_gesture_clip(gaps, end_time - start_time)
        if no_gesture_clip:
            no_gesture_output = os.path.join(no_gesture_dir, f'ZHUBO_{speaker_id}_{gesture_base_id}_{idx:04d}_{no_gesture_label}.mp4')
            if not os.path.exists(no_gesture_output):
                print(f"Extracting matching no-gesture clip: {no_gesture_clip['start_time']:.2f}s - {no_gesture_clip['end_time']:.2f}s")
                if extract_clip_with_padding(video_path, no_gesture_output, no_gesture_clip['start_time'], no_gesture_clip['end_time'], duration):
                    print(f"Successfully extracted no-gesture clip: {no_gesture_output}")
                else:
                    print(f"Failed to extract no-gesture clip: {no_gesture_output}")
            else:
                print(f"Skipping existing no-gesture clip: {no_gesture_output}")
        else:
            print(f"Could not find suitable no-gesture interval for gesture index {idx} with duration {(end_time - start_time):.2f}s")
                
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")

def extract_clips_zhubo(video_dir, vars):
    # Get list of videos
    video_list = glob.glob(os.path.join(video_dir, '*.h264.mp4'))
    
    # Process each video
    for video_path in video_list:
        print(f"\nProcessing video: {video_path}")
        process_video(video_path, vars)