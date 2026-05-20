import os
import glob
import subprocess
import random
import re
from moviepy import VideoFileClip

def find_no_gesture_intervals(gestures, duration):
    """Find time intervals where no gestures occur"""
    # Convert to milliseconds for consistency
    duration_ms = duration * 1000
    
    # Sort gestures by start time
    sorted_gestures = sorted(gestures, key=lambda x: x['start_time'])
    
    # Find gaps between gestures
    gaps = []
    last_end = 0
    
    for gesture in sorted_gestures:
        start = gesture['start_time'] * 1000  # Convert to ms
        end = gesture['end_time'] * 1000
        
        if start > last_end:
            gaps.append({
                'start_time': last_end / 1000,  # Convert back to seconds
                'end_time': start / 1000
            })
        last_end = max(last_end, end)
    
    # Add final gap if there is one
    if last_end < duration_ms:
        gaps.append({
            'start_time': last_end / 1000,
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

def find_matching_no_gesture_clip(gaps, gesture_duration):
    """Find a random gap that can accommodate the gesture duration"""
    valid_gaps = [gap 
                  for gap in gaps 
                  if (gap['end_time'] - gap['start_time']) >= gesture_duration]
    
    if not valid_gaps:
        return None
    
    gap = random.choice(valid_gaps)
    max_start = gap['end_time'] - gesture_duration
    random_start = random.uniform(gap['start_time'], max_start)
    
    return {'start_time': random_start, 'end_time': random_start + gesture_duration}

def parse_gesture_file(file_path): # file is txt
    gestures = []
    seen_intervals = set()  # Track unique time intervals
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split('\t') if p.strip()]
            try:
                # Expected format is like: "R.G.Left.Phrase Router 12490 13510 beat"
                if len(parts) >= 5:
                    hand_info = parts[0]  # e.g., "R.G.Left.Phrase"
                    speaker = parts[1]    # e.g., "Router"
                    start_time = float(parts[2])
                    end_time = float(parts[3])
                    gesture_type = parts[4]  # e.g., "beat"
                    
                    # Skip annotations with duplicate start and end times -- why would there be duplicates in the first place
                    interval_key = (start_time, end_time)
                    if interval_key in seen_intervals:
                        continue
                    
                    seen_intervals.add(interval_key)
                    
                    if end_time > start_time and start_time >= 0:
                        gestures.append({
                            'type': gesture_type,
                            'start_time': start_time / 1000.0,  # Convert to seconds
                            'end_time': end_time / 1000.0,
                            'hand': hand_info
                        })
            except (ValueError, IndexError) as e:
                print(f"Error parsing line: {line.strip()} - {e}")
                continue
    
    print(f"First few parsed gestures: {gestures[:3] if gestures else 'None'}")
    return gestures

def extract_clip(input_file, output_file, start_time, end_time):
    try:
        duration = end_time - start_time
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', input_file,
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            output_file
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(output_file)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting clip: {e}")
        return False


def process_video(annotation_file, video_dir, gesture_dir, no_gesture_dir):
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(annotation_file))[0]
    # Pattern 1: If annotation file is "GrobV9.txt", look for "V9K2_left.mp4"
    # Pattern 2: Direct match - if annotation is "V9.txt", look for "V9K2_left.mp4"
    # Pattern 3: Extract just the V number (V9 from GrobV9)
    match = re.search(r'([V|v]\d+)', base_name) # Note if .txt can be part of path. ensure new code matches old code behaviour
    if not match:
        print(f"Base name {base_name} does not match expected pattern.")
        return
    
    base_name = f"{match.group(1)}K2"
    video_file = os.path.join(video_dir, f"{base_name}_left.mp4")
    
    if not os.path.exists(video_file): # Note: confirm this matches previous code behavior
        print(f"Video file not found for annotation {annotation_file}: {video_file}")
        return

    # Get video duration -- other files use ffmpeg -- try consistency
    with VideoFileClip(video_file) as video:
        duration = video.duration
    
    # Parse gestures from the annotation file
    gestures = parse_gesture_file(annotation_file)
    print(f"Found {len(gestures)} unique gestures in {annotation_file}")
    # Find gaps where no gestures occur
    gaps = find_no_gesture_intervals(gestures, duration)

    extracted_gesture_clips = []
    for idx, gesture in enumerate(gestures):
        gesture_end_time = gesture['end_time']
        gesture_start_time = gesture['start_time']
        gesture_duration = gesture_end_time - gesture_start_time
        if not gesture_end_time <= duration:
            print(f"Gesture {gesture['type']} extends beyond video duration.")
            continue
        # Sanitize gesture type for filename
        safe_type = "".join(c if c.isalnum() else "_" for c in gesture['type'])
        gesture_output = os.path.join(gesture_dir, f"SAGAplus_{base_name}_{idx:04d}_{safe_type}.mp4")
        # if clip exists or extraction is successful
        if os.path.exists(gesture_output) or extract_clip(video_file, gesture_output, gesture_start_time, gesture_end_time):
            print(f"Gesture clip already exist, skipping: {gesture_output}")
            extracted_gesture_clips.append({
                'gesture_idx': idx,
                'duration': gesture_duration,
                'output': gesture_output
            })

    for gesture_clip in extracted_gesture_clips:
        no_gesture_clip = find_matching_no_gesture_clip(gaps, gesture_clip['duration'])
        if no_gesture_clip:
            gaps = consume_gap(gaps, no_gesture_clip['start_time'], no_gesture_clip['end_time'])
            no_gesture_clip_output_path = os.path.join(no_gesture_dir, f"SAGAplus_{base_name}_{gesture_clip['gesture_idx']:04d}_no_gesture.mp4")
            extract_clip(video_file, no_gesture_clip_output_path, no_gesture_clip['start_time'], no_gesture_clip['end_time'])
        else:
            print(f"Could not find suitable no-gesture interval for gesture index {gesture_clip['gesture_idx']} with duration {gesture_clip['duration']:.2f}s")
        
def main():
    annotation_dir = "./annotations" # assumes .txt files
    video_dir = "./VideosCentered" # assumes .mp4 files
    output_dir = "./SAGAplusClips"
    
    # Print available files in directories for debugging
    print(f"Checking directories:")
    print(f"Annotation directory ({annotation_dir}):")
    if os.path.exists(annotation_dir):
        print(f"  - Contains {len(os.listdir(annotation_dir))} files")
    else:
        print(f"  - Directory doesn't exist!")
        return
    
    print(f"Video directory ({video_dir}):")
    if os.path.exists(video_dir):
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"  - Contains {len(video_files)} MP4 files")
        print(f"  - First few video files: {video_files[:5] if video_files else 'None'}")
    else:
        print(f"  - Directory doesn't exist!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Create output directories if they don't exist
    gesture_dir = os.path.join(output_dir, 'Gesture')
    no_gesture_dir = os.path.join(output_dir, 'NoGesture')
    os.makedirs(gesture_dir, exist_ok=True)
    os.makedirs(no_gesture_dir, exist_ok=True)
    
    # Find all txt files in the annotations directory
    txt_files = glob.glob(f"{annotation_dir}/*.txt")
    if not txt_files:
        print(f"No .txt files found in {annotation_dir}")
        return
        
    print(f"Found {len(txt_files)} text files to process")
    for txt_file in txt_files:
        print(f"\nProcessing: {txt_file}")
        process_video(txt_file, video_dir, gesture_dir, no_gesture_dir)

if __name__ == "__main__":
    main()