import os
import glob
import subprocess
import random
from utils import *

offsets = {"ad00": 106404, "ad01": 112595, "ad02": 72247, "ad03": 113641,
           "ad04": 124305, "ad05": 178690, "ad06": 63204, "ad07": 57814,
           "ad09": 96351, "ad10": 176260, "ad11": 106606, "ad12": 149395,
           "ad14": 14011, "ad15": 9900, "ad16": 36607, "ad17": 40368}

def parse_gesture_file(file_path, offset):
    gestures = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split('\t') if p.strip()]
            try:
                if len(parts) >= 3:
                    gesture_type = parts[0]
                    start_time = int(float(parts[1])) + offset
                    end_time = int(float(parts[2])) + offset
                    
                    if end_time > start_time and start_time >= 0:
                        gestures.append({
                            'type': gesture_type,
                            'start_time': start_time / 1000.0,
                            'end_time': end_time / 1000.0
                        })
            except (ValueError, IndexError) as e:
                continue
    return gestures

def process_video(annotation_file, vars):
    gesture_dir = vars['gesture_dir']
    no_gesture_dir = vars['no_gesture_dir']
    no_gesture_label = vars['no_gesture_label']

    base_name = os.path.splitext(os.path.basename(annotation_file))[0].replace('_final', '')
    video_id = base_name[:4]
    
    if video_id not in offsets:
        print(f"No offset found for video {video_id}")
        return
        
    video_file = os.path.join(os.path.dirname(annotation_file), f"{base_name}_speakerview480480.mp4")
    
    if not os.path.exists(video_file):
        print(f"No matching video file found for {annotation_file}")
        return
    
    video_duration = get_video_info(video_file)['duration']
    
    gestures = parse_gesture_file(annotation_file, offsets[video_id])
    print(f"Found {len(gestures)} gestures in {annotation_file}")
    valid_gestures = [g for g in gestures if g['end_time'] <= video_duration and g['start_time'] >= 0]
    
    # Find gaps where no gestures occur
    gaps = find_no_gesture_intervals(valid_gestures, video_duration)
    
    for idx, gesture in enumerate(valid_gestures):        
        # Process gesture clip
        safe_type = "".join(c if c.isalnum() else "_" for c in gesture['type'])
        gesture_output = os.path.join(gesture_dir, 
                                    f"ECOLANG_{base_name}_{idx:04d}_{safe_type}.mp4")
        
        if not os.path.exists(gesture_output) or \
        extract_clip_with_padding(video_file, gesture_output, gesture['start_time'], gesture['end_time'], video_duration, padding=1.0):
            print(f"Extracting gesture {gesture['type']} from {base_name}: "
                    f"{gesture['start_time']:.2f}s - {gesture['end_time']:.2f}s")

    gaps = find_no_gesture_intervals(valid_gestures, video_duration)
    for gesture_clip in valid_gestures:
        no_gesture_clip = find_matching_no_gesture_clip(gaps, gesture_clip['duration'])
        if no_gesture_clip:
            gaps = consume_gap(gaps, no_gesture_clip['start_time'], no_gesture_clip['end_time'])
            no_gesture_clip_output_path = os.path.join(no_gesture_dir, f"ECOLANG_{base_name}_{gesture_clip['gesture_idx']:04d}_{no_gesture_label}.mp4")
            extract_clip_with_padding(video_file, no_gesture_clip_output_path, no_gesture_clip['start_time'], no_gesture_clip['end_time'], video_duration, padding=1.0)
        else:
            print(f"Could not find suitable no-gesture interval for gesture index {gesture_clip['gesture_idx']} with duration {gesture_clip['duration']:.2f}s") 

def extract_clips_ecolang(video_dir, vars):    
    txt_files = glob.glob(f"{video_dir}/*final.txt")
    if not txt_files:
        print(f"No .txt files found in {video_dir}")
        return
        
    print(f"Found {len(txt_files)} text files to process")
    for txt_file in txt_files:
        print(f"\nProcessing: {txt_file}")
        process_video(txt_file, vars)
