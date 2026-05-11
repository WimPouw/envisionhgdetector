import os
import glob
import subprocess
import random
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

def find_matching_no_gesture_clip(gaps, gesture_duration):
    """Find a random gap that can accommodate the gesture duration"""
    valid_gaps = [gap for gap in gaps 
                  if (gap['end_time'] - gap['start_time']) >= gesture_duration]
    
    if not valid_gaps:
        return None
        
    gap = random.choice(valid_gaps)
    max_start = gap['end_time'] - gesture_duration
    
    # Pick a random start time within the gap
    random_start = random.uniform(gap['start_time'], max_start)
    return {
        'start_time': random_start,
        'end_time': random_start + gesture_duration
    }

def parse_gesture_file(file_path):
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
                    
                    # Skip annotations with duplicate start and end times
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

def process_video(txt_file, video_dir, output_dir):
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(txt_file))[0]
    
    # Try different naming patterns for video files
    potential_video_files = []
    
    # Pattern 1: If annotation file is "GrobV9.txt", look for "V9K2_left.mp4"
    if base_name.startswith("Grob"):
        video_id = base_name[4:]  # Remove "Grob" prefix
        potential_video_files.append(os.path.join(video_dir, f"{video_id}K2_left.mp4"))
    
    # Pattern 2: Direct match - if annotation is "V9.txt", look for "V9K2_left.mp4"
    potential_video_files.append(os.path.join(video_dir, f"{base_name}K2_left.mp4"))
    
    # Pattern 3: Extract just the V number (V9 from GrobV9)
    import re
    match = re.search(r'([V|v]\d+)', base_name)
    if match:
        video_id = match.group(1)
        potential_video_files.append(os.path.join(video_dir, f"{video_id}K2_left.mp4"))
    
    # Try each potential video file
    video_file = None
    for potential_file in potential_video_files:
        if os.path.exists(potential_file):
            video_file = potential_file
            break
            
    # If no matching file was found, list available files for debugging
    if video_file is None:
        print(f"No matching video file found for {txt_file}")
        print(f"Looked for: {', '.join(potential_video_files)}")
        print(f"Available files in {video_dir}:")
        for file in os.listdir(video_dir):
            if file.endswith(".mp4"):
                print(f"  - {file}")
        return
    
    if not os.path.exists(video_file):
        print(f"No matching video file found for {txt_file}")
        print(f"Looked for: {video_file}")
        return
    
    # Create output directories if they don't exist
    gesture_dir = os.path.join(output_dir, 'Gesture')
    no_gesture_dir = os.path.join(output_dir, 'NoGesture')
    os.makedirs(gesture_dir, exist_ok=True)
    os.makedirs(no_gesture_dir, exist_ok=True)
    
    # Get video duration
    with VideoFileClip(video_file) as video:
        duration = video.duration
    
    # Parse gestures from the annotation file
    gestures = parse_gesture_file(txt_file)
    print(f"Found {len(gestures)} unique gestures in {txt_file}")
    
    # Find gaps where no gestures occur
    gaps = find_no_gesture_intervals(gestures, duration)
    
    # Extract video_id from the matched video file path for consistent naming
    video_basename = os.path.basename(video_file)
    video_id = os.path.splitext(video_basename)[0].replace('_left', '')
    
    for idx, gesture in enumerate(gestures):
        gesture_duration = gesture['end_time'] - gesture['start_time']
        
        if gesture['end_time'] <= duration:
            # Sanitize gesture type for filename
            safe_type = "".join(c if c.isalnum() else "_" for c in gesture['type'])
            gesture_output = os.path.join(gesture_dir, 
                                         f"SAGAplus_{video_id}_{idx:04d}_{safe_type}.mp4")
            
            if not os.path.exists(gesture_output):
                print(f"Extracting gesture {gesture['type']} from {base_name}: "
                      f"{gesture['start_time']:.2f}s - {gesture['end_time']:.2f}s")
                
                if extract_clip(video_file, gesture_output, 
                               gesture['start_time'], gesture['end_time']):
                    print(f"Successfully extracted gesture: {gesture_output}")
                    
                    # Find and extract matching no-gesture clip
                    no_gesture = find_matching_no_gesture_clip(gaps, gesture_duration)
                    if no_gesture:
                        no_gesture_output = os.path.join(no_gesture_dir,
                                                        f"SAGAplus_{video_id}_{idx:04d}_no_gesture.mp4")
                        
                        print(f"Extracting matching no-gesture clip: "
                              f"{no_gesture['start_time']:.2f}s - {no_gesture['end_time']:.2f}s")
                        
                        if extract_clip(video_file, no_gesture_output,
                                       no_gesture['start_time'], no_gesture['end_time']):
                            print(f"Successfully extracted no-gesture clip: {no_gesture_output}")
                        else:
                            print(f"Failed to extract no-gesture clip")
                    else:
                        print(f"Could not find suitable no-gesture interval of duration {gesture_duration:.2f}s")
                else:
                    print(f"Failed to extract gesture clip")

def main():
    annotation_dir = "./annotations"
    video_dir = "./VideosCentered"
    output_dir = "./VideosCut"
    
    # Print available files in directories for debugging
    print(f"Checking directories:")
    print(f"Annotation directory ({annotation_dir}):")
    if os.path.exists(annotation_dir):
        print(f"  - Contains {len(os.listdir(annotation_dir))} files")
    else:
        print(f"  - Directory doesn't exist!")
    
    print(f"Video directory ({video_dir}):")
    if os.path.exists(video_dir):
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"  - Contains {len(video_files)} MP4 files")
        print(f"  - First few video files: {video_files[:5] if video_files else 'None'}")
    else:
        print(f"  - Directory doesn't exist!")
    
    if not os.path.exists(annotation_dir):
        print(f"Input directory {annotation_dir} does not exist!")
        return
        
    if not os.path.exists(video_dir):
        print(f"Video directory {video_dir} does not exist!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all txt files in the annotations directory
    txt_files = glob.glob(f"{annotation_dir}/*.txt")
    if not txt_files:
        print(f"No .txt files found in {annotation_dir}")
        return
        
    print(f"Found {len(txt_files)} text files to process")
    for txt_file in txt_files:
        print(f"\nProcessing: {txt_file}")
        process_video(txt_file, video_dir, output_dir)

if __name__ == "__main__":
    main()