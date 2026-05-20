import os
import glob
import re
from utils import *

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

def process_video(annotation_file, video_dir, vars):
    gesture_dir = vars['gesture_dir']
    no_gesture_dir = vars['no_gesture_dir']
    no_gesture_label = vars['no_gesture_label']

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
    # with VideoFileClip(video_file) as video:
    #     duration = video.duration
    duration = get_video_info(video_file)['duration']
    print(f"Video duration for {video_file}: {duration:.2f} seconds")
    
    # Parse gestures from the annotation file
    gestures = parse_gesture_file(annotation_file)
    valid_gestures = [g for g in gestures if g['end_time'] <= duration and g['start_time'] >= 0]
    print(f"Found {len(gestures)} unique gestures in {annotation_file}")
    # Find gaps where no gestures occur
    gaps = find_no_gesture_intervals(valid_gestures, duration)

    extracted_gesture_clips = []
    for idx, gesture in enumerate(valid_gestures):
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
        if os.path.exists(gesture_output) or extract_clip_with_padding(video_file, gesture_output, gesture_start_time, gesture_end_time, get_video_info(video_file), padding=1.0):
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
            extract_clip_with_padding(video_file, no_gesture_clip_output_path, no_gesture_clip['start_time'], no_gesture_clip['end_time'], get_video_info(video_file), padding=1.0)
        else:
            print(f"Could not find suitable no-gesture interval for gesture index {gesture_clip['gesture_idx']} with duration {gesture_clip['duration']:.2f}s")
        
def extract_clips_saga_plus(corpus_dir, vars):
    annotation_dir = f"{corpus_dir}/annotations" # assumes .txt files
    video_dir = f"{corpus_dir}/VideosCentered" # assumes .mp4 files
    # Find all txt files in the annotations directory
    txt_files = glob.glob(f"{annotation_dir}/*.txt")
    if not txt_files:
        print(f"No .txt files found in {annotation_dir}")
        return
        
    print(f"Found {len(txt_files)} text files to process")
    for txt_file in txt_files:
        print(f"\nProcessing: {txt_file}")
        process_video(txt_file, video_dir, vars)