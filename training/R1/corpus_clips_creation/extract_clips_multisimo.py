import glob
import os
from utils import *

def parse_gesture_file(gesture_file):
    """Parse the gesture annotation file and return list of actions"""
    actions = []
    with open(gesture_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                try:
                    start_frame = int(parts[2]) / 1000.0  # Convert to seconds
                    end_frame = int(parts[3]) / 1000.0    # Convert to seconds
                    gesture_type = parts[4] if len(parts) > 4 else "Unknown"
                    
                    if end_frame > start_frame:
                        actions.append({
                            'start_time': start_frame,
                            'end_time': end_frame,
                            'type': gesture_type
                        })
                except ValueError as e:
                    print(f"Error parsing line: {line.strip()}, Error: {e}")
                    continue
    return actions

def process_video(video_path, vars):
    """Process a single video file"""
    gesture_dir = vars['gesture_dir']
    no_gesture_dir = vars['no_gesture_dir']
    moves_dir = vars['moves_dir']
    gesture_label = vars['gesture_label']
    no_gesture_label = vars['no_gesture_label']
    move_label = vars['move_label']

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    gesture_file = os.path.join(os.path.dirname(video_path), f"{video_id}.txt")
    
    if not os.path.exists(gesture_file):
        print(f"No gesture file found for {video_path}")
        return
    
    # Get video duration
    video_duration = get_video_info(video_path)['duration']
    
    print(f"Video duration: {video_duration:.2f} seconds")
    
    # Load and validate actions
    actions = parse_gesture_file(gesture_file)
    valid_actions = [a for a in actions if a['end_time'] <= video_duration and a['start_time'] >= 0]
    
    # Get non-gesture intervals
    nogesture_intervals = find_no_gesture_intervals(valid_actions, video_duration)
    
    # Separate gestures and moves
    gestures = [a for a in valid_actions if a['type'].upper() != 'N/A']
    moves = [a for a in valid_actions if a['type'].upper() == 'N/A']
    
    # Process regular gesture clips and matching non-gesture clips
    for idx, action in enumerate(gestures):
        gesture_duration = action['end_time'] - action['start_time']
        
        # Process gesture clip
        safe_type = "".join(c if c.isalnum() else "_" for c in action["type"])
        gesture_output = os.path.join(
            gesture_dir, 
            f'multisimo_{gesture_label}_{video_id}_{idx:04d}_{safe_type}.mp4'
        )
        
        if not os.path.exists(gesture_output) or extract_clip_with_padding(video_path, gesture_output, action['start_time'], action['end_time'], video_duration):
            print(f"Extracting gesture clip {idx} from {video_id}: "
                    f"{action['start_time']:.2f}s - {action['end_time']:.2f}s "
                    f"(Type: {action['type']})")
                    
    # Find and extract matching non-gesture clip
    gaps = find_matching_no_gesture_clip(nogesture_intervals, gesture_duration)
    for idx, action in enumerate(gestures):
        no_gesture_clip = find_matching_no_gesture_clip(gaps, action['end_time'] - action['start_time'])
        if no_gesture_clip:
            nogesture_output = os.path.join(
                no_gesture_dir, 
                f'multisimo_{no_gesture_label}_{video_id}_{idx:04d}.mp4'
            )
            
            print(f"Extracting matching non-gesture clip {idx} from {video_id}: "
                    f"{no_gesture_clip['start_time']:.2f}s - {no_gesture_clip['end_time']:.2f}s")
                    
            if extract_clip_with_padding(video_path, nogesture_output, 
                            no_gesture_clip['start_time'], 
                            no_gesture_clip['end_time'], video_duration):
                print(f"Successfully extracted non-gesture clip: {nogesture_output}")
            else:
                print(f"Failed to extract non-gesture clip: {nogesture_output}")
        else:
            print(f"Could not find suitable non-gesture interval of duration {gesture_duration:.2f}s")
            

    # Process move clips (N/A gestures)
    for idx, action in enumerate(moves):
        output_file = os.path.join(
            moves_dir, 
            f'multisimo_{move_label}_{video_id}_{idx:04d}.mp4'
        )
        
        if not os.path.exists(output_file):
            print(f"Extracting move clip {idx} from {video_id}: "
                    f"{action['start_time']:.2f}s - {action['end_time']:.2f}s")
                    
            if extract_clip_with_padding(video_path, output_file, action['start_time'], action['end_time'], video_duration):
                print(f"Successfully extracted move clip: {output_file}")
            else:
                print(f"Failed to extract move clip: {output_file}")


def extract_clips_multisimo(video_dir, vars):
    video_extensions = ['.mp4', '.avi', '.mov']
    video_list = []
    for ext in video_extensions:
        video_list.extend(glob.glob(os.path.join(video_dir, f'*{ext}')))
    
    for video_path in video_list:
        print(f"\nProcessing video: {video_path}")
        process_video(video_path, vars)