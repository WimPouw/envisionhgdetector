import subprocess
import json
import os
import random

def extract_clip_with_padding(input_file_path, output_file_path, start_time, end_time, video_duration, padding=1.0):
    """Extract clip using ffmpeg with padding before and after"""
    try:
        # Add padding and ensure we don't go out of bounds
        padded_start = max(0, start_time - padding)
        padded_end = min(video_duration, end_time + padding)
        clip_duration = padded_end - padded_start
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(padded_start),
            '-i', input_file_path,
            '-t', str(clip_duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            output_file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Checks if video is > 1KB to ensure it's not an empty file
        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 1024:
            return True
        else:
            print(f"Failed to create valid output file: {output_file_path}")
            print(f"FFmpeg output: {result.stderr}")
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            return False
            
    except Exception as e:
        print(f"Error extracting clip: {e}")
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        return False

def find_no_gesture_intervals(gestures, duration):
    """Find time intervals where no gestures occur
    Input:
    - gestures: list of dicts with 'start_time' and 'end_time' in seconds
    - duration: total video duration in seconds
    Output:    
    - list of dicts with 'start_time' and 'end_time' for no-gesture intervals
    """
    # Sort gestures by start time
    sorted_gestures = sorted(gestures, key=lambda x: x['start_time'])
    
    # Find gaps between gestures
    gaps = []
    last_end = 0
    
    for gesture in sorted_gestures:
        start = gesture['start_time']
        end = gesture['end_time']
        
        if start > last_end:
            gaps.append({
                'start_time': last_end,
                'end_time': start
            })
        last_end = max(last_end, end)
    
    # Add final gap if there is one
    if last_end < duration:
        gaps.append({
            'start_time': last_end,
            'end_time': duration
        })
    
    return gaps

def consume_gap(remaining_gaps, chosen_start, chosen_end):
    """Remove the chosen no-gesture interval from the remaining gaps
        This ensures no duplicates"""
    new_gaps = []
    for gap in remaining_gaps:
        if chosen_start >= gap['end_time'] or chosen_end <= gap['start_time']:
            new_gaps.append(gap)  # unused gap
        else:
            # remove the chosen interval from gap
            if gap['start_time'] < chosen_start:
                new_gaps.append({'start_time': gap['start_time'], 'end_time': chosen_start})
            if chosen_end < gap['end_time']:
                new_gaps.append({'start_time': chosen_end, 'end_time': gap['end_time']})
    return new_gaps

def find_valid_gaps(gaps, gesture_duration, decreasing_factor=0.9):
    """Find gaps that can accommodate the gesture duration
        If no gaps are long enough, reduce the required duration by a factor and check again
        If no gaps are long enough, return empty list"""
    valid_gaps = []
    while gesture_duration > 0 and not valid_gaps:
        valid_gaps = [gap 
                      for gap in gaps 
                      if (gap['end_time'] - gap['start_time']) >= gesture_duration]
        gesture_duration = gesture_duration * decreasing_factor
    return valid_gaps

def find_matching_no_gesture_clip(gaps, gesture_duration):
    """Find a random gap that can accommodate the gesture duration"""
    # list of gaps long enough for the gesture duration
    valid_gaps = find_valid_gaps(gaps, gesture_duration, decreasing_factor=0.9)
    if not valid_gaps:
        return None
    
    gap = random.choice(valid_gaps)
    max_start = gap['end_time'] - gesture_duration
    random_start = random.uniform(gap['start_time'], max_start)
    
    return {'start_time': random_start, 'end_time': random_start + gesture_duration}

def get_video_info(video_path):
    """Get video information using ffprobe"""
    try:
        # Get duration and fps using ffprobe
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-select_streams', 'v:0', 
            '-show_entries', 'stream=duration,r_frame_rate', 
            '-of', 'json', 
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        
        # Extract duration and fps
        duration = float(info['streams'][0]['duration'])
        
        # r_frame_rate is usually in the format "num/den"
        fps_str = info['streams'][0]['r_frame_rate']
        if '/' in fps_str:
            num, den = map(float, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
            
        return {'duration': duration, 'fps': fps}
    
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {'duration': 0, 'fps': 0}

def check_ffmpeg():
    """Check if ffmpeg is available for video processing."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
