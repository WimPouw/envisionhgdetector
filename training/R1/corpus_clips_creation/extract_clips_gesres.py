import os
import shutil
import re
import subprocess
import sys
from utils import check_ffmpeg
import pandas as pd
import random
from utils import *

# Mapping of lexemes to gesture types based on your provided data
LEXEME_TO_TYPE = {
    "NA": "adaptor",
    "beat": "beat",
    "hold": "beat",
    "progress": "beat",
    "chop": "beat", 
    "stab": "beat",
    "fling": "beat",
    "sweep": "deictic",
    "point": "deictic",
    "wave": "emblem",
    "count": "emblem",
    "air quotes": "emblem",
    "refuse": "emblem",
    "attention": "emblem",
    "neck cut": "emblem",
    "number": "emblem",
    "finger snap": "emblem",
    "x-sweep": "emblem",
    "calm": "emblem",
    "ladder": "metaphoric",
    "sphere": "metaphoric",
    "container": "metaphoric",
    "gather": "metaphoric",
    "invitation": "metaphoric",
    "separate": "metaphoric",
    "show": "metaphoric",
    "reveal": "metaphoric",
    "present": "metaphoric",
    "juggle": "metaphoric",
    "throw": "metaphoric",
    "space": "metaphoric",
    "rise": "metaphoric",
    "circular sway": "metaphoric",
    "emerge": "metaphoric",
    "path": "metaphoric",
    "span": "metaphoric",
    "air clap": "metaphoric",
    "cover": "metaphoric",
    "grab": "metaphoric",
    "give": "metaphoric",
    "insert": "metaphoric",
    "increase": "metaphoric",
    "regress": "metaphoric",
    "unleash": "metaphoric",
    "bring in": "metaphoric",
    "shake": "metaphoric",
    "stir": "metaphoric",
    "weigh": "metaphoric",
    "cup flip": "metaphoric",
    "hand flip": "metaphoric",
    "soft hand clap": "metaphoric",
    "clasp": "metaphoric",
    "fingers touch": "metaphoric",
    "hand hold": "metaphoric",
    "self": "metaphoric",
    "dismiss": "metaphoric",
    "block": "metaphoric",
    "push": "metaphoric",
    "steps": "metaphoric",
    "down": "metaphoric",
    "up": "metaphoric",
    "set down": "metaphoric",
    "moving": "metaphoric",
    "expanding": "metaphoric",
    "shrinking": "metaphoric",
    "erratic": "metaphoric",
    "symmetric": "metaphoric",
    "forward": "metaphoric",
    "backward": "metaphoric",
    "left": "metaphoric",
    "right": "metaphoric",
    "to-fro": "metaphoric",
    "front-back": "metaphoric",
    "circular": "metaphoric",
    "arc": "metaphoric",
    "dome": "metaphoric",
    "top down": "metaphoric",
    "upward": "metaphoric",
    "distance": "metaphoric",
    "knock": "iconic",
    "wave": "iconic"
}

def get_video_dimensions(video_path):
    """Get video dimensions using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        import json
        data = json.loads(result.stdout)
        
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                return int(stream['width']), int(stream['height'])
        
        return None, None
    except Exception as e:
        print(f"Error getting video dimensions: {e}")
        return None, None

def split_video_horizontally(input_path, output_left, output_right):
    """
    Split video horizontally into left and right halves using ffmpeg.
    """
    try:
        # Get video dimensions
        width, height = get_video_dimensions(input_path)
        if width is None or height is None:
            raise Exception("Could not determine video dimensions")
        
        half_width = width // 2
        
        # Split left half (Front view)
        cmd_left = [
            'ffmpeg', '-i', input_path,
            '-vf', f'crop={half_width}:{height}:0:0',
            '-c:a', 'copy',  # Copy audio stream
            '-y',  # Overwrite output files
            output_left
        ]
        
        # Split right half (Side view)
        cmd_right = [
            'ffmpeg', '-i', input_path,
            '-vf', f'crop={half_width}:{height}:{half_width}:0',
            '-c:a', 'copy',  # Copy audio stream
            '-y',  # Overwrite output files
            output_right
        ]
        
        print(f"  Splitting left half to: {os.path.basename(output_left)}")
        subprocess.run(cmd_left, check=True, capture_output=True)
        
        print(f"  Splitting right half to: {os.path.basename(output_right)}")
        subprocess.run(cmd_right, check=True, capture_output=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  Error splitting video: {e}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def get_gesture_type_from_lexeme(lexeme_filename):
    """
    Extract gesture type from lexeme filename using the mapping.
    For files like '2358_Lecturer3_reveal.mp4', extract 'reveal' and map to type.
    """
    # Extract the lexeme (last part before .mp4)
    base_name = os.path.splitext(lexeme_filename)[0]
    parts = base_name.split('_')
    
    if len(parts) >= 3:
        lexeme = parts[-1]  # Last part is the lexeme
        
        # Direct mapping
        if lexeme in LEXEME_TO_TYPE:
            return LEXEME_TO_TYPE[lexeme]
        
        # Try to find partial matches for compound lexemes
        for key, gesture_type in LEXEME_TO_TYPE.items():
            if key in lexeme or lexeme in key:
                return gesture_type
        
        # If no match found, try to categorize based on common patterns
        if any(word in lexeme.lower() for word in ['point', 'sweep']):
            return 'deictic'
        elif any(word in lexeme.lower() for word in ['beat', 'chop', 'stab', 'hold']):
            return 'beat'
        elif lexeme.lower() in ['na', 'adaptor']:
            return 'adaptor'
        else:
            return 'metaphoric'  # Default fallback
    
    return 'unknown'

def rename_and_split_files(source_dir, destination_dir, file_mapping_log=None):
    """
    Rename files from GESres dataset format to the new naming convention.
    Split Clinician1 videos into front and side views.
    
    Args:
        source_dir: Directory containing original files
        destination_dir: Directory where renamed files will be copied
        file_mapping_log: Optional file to log the mapping of old to new names
    """
    
    # Check if ffmpeg is available for video splitting
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        print("WARNING: ffmpeg not found. Clinician1 videos will be copied without splitting.")
        print("To enable video splitting, please install ffmpeg:")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        print("  - macOS: brew install ffmpeg")
        print("  - Linux: sudo apt-get install ffmpeg (Ubuntu/Debian)")
        print()
    
    mapping_log = []
    
    # Pattern to match files like: 1_Politician1_adaptor.mp4
    pattern = r'(\d+)_([^_]+)_(.+)\.mp4'
    
    for filename in os.listdir(source_dir):
        if filename.endswith('.mp4'):
            match = re.match(pattern, filename)
            
            if match:
                number, speaker, lexeme = match.groups()
                
                # Get gesture type from lexeme
                gesture_type = get_gesture_type_from_lexeme(f"dummy_{speaker}_{lexeme}.mp4")
                
                source_path = os.path.join(source_dir, filename)
                
                # Handle Clinician1 specially - split into front and side views
                if speaker == "Clinician1" and ffmpeg_available:
                    # Create filenames for front and side views
                    front_filename = f"GESres_Clinician1Front_{number}_gesture_{gesture_type}.mp4"
                    side_filename = f"GESres_Clinician1Side_{number}_gesture_{gesture_type}.mp4"
                    
                    front_path = os.path.join(destination_dir, front_filename)
                    side_path = os.path.join(destination_dir, side_filename)
                    
                    print(f"Processing Clinician1 video: {filename}")
                    
                    if split_video_horizontally(source_path, front_path, side_path):
                        print(f"Successfully split: {filename}")
                        print(f"  -> {front_filename} (left half - front view)")
                        print(f"  -> {side_filename} (right half - side view)")
                        mapping_log.append((filename, front_filename, lexeme, gesture_type, "split_left"))
                        mapping_log.append((filename, side_filename, lexeme, gesture_type, "split_right"))
                    else:
                        print(f"Failed to split {filename}, copying original instead.")
                        # Fallback: copy original file with modified name
                        fallback_filename = f"GESres_{speaker}_{number}_gesture_{gesture_type}.mp4"
                        dest_path = os.path.join(destination_dir, fallback_filename)
                        try:
                            shutil.copy2(source_path, dest_path)
                            mapping_log.append((filename, fallback_filename, lexeme, gesture_type, "copy_fallback"))
                        except Exception as e:
                            print(f"Error copying {filename}: {e}")
                
                else:
                    # Regular processing for other speakers or when ffmpeg is not available
                    if speaker == "Clinician1" and not ffmpeg_available:
                        print(f"Copying Clinician1 file without splitting: {filename}")
                    
                    new_filename = f"GESres_{speaker}_{number}_gesture_{gesture_type}.mp4"
                    dest_path = os.path.join(destination_dir, new_filename)
                    
                    try:
                        shutil.copy2(source_path, dest_path)
                        print(f"Renamed: {filename} -> {new_filename}")
                        mapping_log.append((filename, new_filename, lexeme, gesture_type, "copy"))
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        
            else:
                print(f"Skipping file (doesn't match pattern): {filename}")
    
    # Save mapping log if requested
    if file_mapping_log:
        with open(file_mapping_log, 'w') as f:
            f.write("Original_Filename,New_Filename,Lexeme,Gesture_Type,Operation\n")
            for orig, new, lexeme, gtype, operation in mapping_log:
                f.write(f"{orig},{new},{lexeme},{gtype},{operation}\n")
        print(f"Mapping log saved to: {file_mapping_log}")
    
    print(f"Processing complete. {len([x for x in mapping_log if x[4] != 'split_right'])} source files processed.")

def preview_renaming(source_dir, num_examples=10):
    """
    Preview what the renaming would look like without actually renaming files.
    """
    print("PREVIEW MODE - No files will be modified")
    print("=" * 50)
    
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        print("Note: ffmpeg not available - Clinician1 videos would be copied without splitting")
        print()
    
    pattern = r'(\d+)_([^_]+)_(.+)\.mp4'
    count = 0
    
    for filename in os.listdir(source_dir):
        if filename.endswith('.mp4') and count < num_examples:
            match = re.match(pattern, filename)
            
            if match:
                number, speaker, lexeme = match.groups()
                gesture_type = get_gesture_type_from_lexeme(f"dummy_{speaker}_{lexeme}.mp4")
                
                if speaker == "Clinician1" and ffmpeg_available:
                    front_filename = f"GESres_Clinician1Front_{number}_gesture_{gesture_type}.mp4"
                    side_filename = f"GESres_Clinician1Side_{number}_gesture_{gesture_type}.mp4"
                    print(f"{filename} -> {front_filename} + {side_filename}")
                    print(f"  Lexeme: {lexeme} -> Type: {gesture_type} (SPLIT INTO FRONT + SIDE)")
                else:
                    new_filename = f"GESres_{speaker}_{number}_gesture_{gesture_type}.mp4"
                    print(f"{filename} -> {new_filename}")
                    if speaker == "Clinician1":
                        print(f"  Lexeme: {lexeme} -> Type: {gesture_type} (NO SPLIT - ffmpeg unavailable)")
                    else:
                        print(f"  Lexeme: {lexeme} -> Type: {gesture_type}")
                print()
                count += 1

def extract_no_gesture_segments(csv_file, full_videos_dir, output_dir):
    """
    Extract no-gesture segments from full videos based on CSV annotations.
    
    Args:
        csv_file: Path to GESRes_dataset.csv
        full_videos_dir: Directory containing full videos
        output_dir: Directory to save extracted no-gesture segments
        target_duration: Target duration for each no-gesture segment in seconds
    """    
    if not check_ffmpeg():
        print("ERROR: ffmpeg is required for video extraction. Please install ffmpeg first.")
        return

    # Read CSV data
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} annotations from CSV")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Video mapping from CSV ID to actual file names -- just adding mp4? why not do it directly
    video_mapping = {
        "1Politician_id1_AOC_vid1_speechCongress2019": "1Politician_id1_AOC_vid1_speechCongress2019.mp4",
        "1Politician_id3_Boebert_vid1": "1Politician_id3_Boebert_vid1.mp4",
        "2Clinician_id1_vid1": "2Clinician_id1_vid1.mp4",
        # Ignore 2Clinician_id3_vid1 as requested
        "3Educator_id1_Psych_vid1": "3Educator_id1_psych_vid1.mp4",
        "3Educator_id2_Politics_vid1": "3Educator_id2_politics_vid1.mp4",
        "3Educator_id3_law": "3Educator_id3_law.mp4"
    }
    
    extraction_log = []
    
    for csv_id, video_filename in video_mapping.items():
        video_path = os.path.join(full_videos_dir, video_filename)
        
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            continue
            
        print(f"\nProcessing {csv_id}...")
        
        # Get gesture annotations for this video
        video_gestures = df[df['id'] == csv_id].copy()
        if len(video_gestures) == 0:
            print(f"  No gesture data found for {csv_id}")
            continue
            
        # Sort gestures by start time
        video_gestures = video_gestures.sort_values('start_t')
        
        # Get video duration using ffprobe
        
        video_duration = get_video_info(video_path)['duration']
        if video_duration == 0:
            print(f"  Warning: Could not determine duration for {video_path}, skipping...")
            continue
        print(f"  Video duration: {video_duration:.2f} seconds")
        
        # Find no-gesture gaps
        gaps = []
        
        # Gap at the beginning (if first gesture doesn't start at 0)
        first_gesture_start = video_gestures.iloc[0]['start_t']
        if first_gesture_start > target_duration:
            gaps.append((0, first_gesture_start))
        
        # Gaps between gestures
        for i in range(len(video_gestures) - 1):
            current_end = video_gestures.iloc[i]['end_t']
            next_start = video_gestures.iloc[i + 1]['start_t']
            gap_duration = next_start - current_end
            
            if gap_duration >= target_duration:
                gaps.append((current_end, next_start))
        
        # Gap at the end
        last_gesture_end = video_gestures.iloc[-1]['end_t']
        if video_duration - last_gesture_end >= target_duration:
            gaps.append((last_gesture_end, video_duration))
        
        print(f"  Found {len(gaps)} potential no-gesture segments")
        
        # Extract segments from gaps
        segment_count = 0
        speaker_name = csv_id.split('_')[0]  # e.g., "1Politician" -> "1Politician"
        
        for gap_start, gap_end in gaps:
            gap_duration = gap_end - gap_start
            
            # If gap is longer than target_duration, extract multiple segments
            num_segments = max(1, int(gap_duration // target_duration))
            
            for seg_idx in range(min(num_segments, 3)):  # Limit to 3 segments per gap
                # Random start time within the gap (leaving some buffer)
                buffer = 0.5  # 0.5 second buffer
                available_start = gap_start + buffer
                available_end = gap_end - target_duration - buffer
                
                if available_end <= available_start:
                    # Not enough space for a full segment
                    continue
                
                segment_start = random.uniform(available_start, available_end)
                segment_end = segment_start + target_duration
                
                segment_count += 1
                
                # Map CSV ID to speaker identifiers (simplified format)
                speaker_mapping = {
                    "1Politician_id1_AOC_vid1_speechCongress2019": "P1",
                    "1Politician_id3_Boebert_vid1": "P3", 
                    "2Clinician_id1_vid1": "C1",
                    "3Educator_id1_Psych_vid1": "E1",
                    "3Educator_id2_Politics_vid1": "E2",
                    "3Educator_id3_law": "E3"
                }
                
                if csv_id not in speaker_mapping:
                    print(f"    Warning: No mapping found for {csv_id}, skipping...")
                    continue
                    
                speaker_id = speaker_mapping[csv_id]
                
                # Format segment number as 4-digit number
                segment_num_str = f"{segment_count:04d}"
                
                # Handle Clinician1 specially (split into main and mirror)
                if csv_id == "2Clinician_id1_vid1":
                    # Main video (left half - front view)
                    main_filename = f"GESres_{speaker_id}_{segment_num_str}_nogesture_NA.mp4"
                    main_output = os.path.join(output_dir, main_filename)
                    
                    # Mirror video (right half - side view)
                    mirror_filename = f"GESres_{speaker_id}_{segment_num_str}_nogesture_NA_mirror.mp4"
                    mirror_output = os.path.join(output_dir, mirror_filename)
                    
                    # Get video dimensions for splitting
                    width, height = get_video_dimensions(video_path)
                    if width and height:
                        half_width = width // 2
                        
                        # Extract main view (left half - front view)
                        cmd_main = [
                            'ffmpeg', '-i', video_path,
                            '-ss', str(segment_start),
                            '-t', str(target_duration),
                            '-vf', f'crop={half_width}:{height}:0:0',
                            '-c:a', 'copy',
                            '-y', main_output
                        ]
                        
                        # Extract mirror view (right half - side view)
                        cmd_mirror = [
                            'ffmpeg', '-i', video_path,
                            '-ss', str(segment_start),
                            '-t', str(target_duration),
                            '-vf', f'crop={half_width}:{height}:{half_width}:0',
                            '-c:a', 'copy',
                            '-y', mirror_output
                        ]
                        
                        try:
                            subprocess.run(cmd_main, check=True, capture_output=True)
                            subprocess.run(cmd_mirror, check=True, capture_output=True)
                            print(f"    Extracted and split segment {segment_num_str}: {segment_start:.2f}s - {segment_end:.2f}s")
                            extraction_log.append((csv_id, main_filename, segment_start, segment_end, "main_nogesture"))
                            extraction_log.append((csv_id, mirror_filename, segment_start, segment_end, "mirror_nogesture"))
                        except subprocess.CalledProcessError as e:
                            print(f"    Error extracting segment {segment_num_str}: {e}")
                    
                else:
                    # Regular extraction for other speakers
                    segment_filename = f"GESres_{speaker_id}_{segment_num_str}_nogesture_NA.mp4"
                    segment_output = os.path.join(output_dir, segment_filename)
                    
                    cmd = [
                        'ffmpeg', '-i', video_path,
                        '-ss', str(segment_start),
                        '-t', str(target_duration),
                        '-c:v', 'libx264',
                        '-c:a', 'aac',
                        '-y', segment_output
                    ]
                    
                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                        print(f"    Extracted segment {segment_num_str}: {segment_start:.2f}s - {segment_end:.2f}s")
                        extraction_log.append((csv_id, segment_filename, segment_start, segment_end, "nogesture"))
                    except subprocess.CalledProcessError as e:
                        print(f"    Error extracting segment {segment_num_str}: {e}")
    
    # Save extraction log
    log_file = os.path.join(output_dir, "no_gesture_extraction_log.csv")
    with open(log_file, 'w') as f:
        f.write("Source_Video,Output_Filename,Start_Time,End_Time,Type\n")
        for source, output, start, end, seg_type in extraction_log:
            f.write(f"{source},{output},{start:.2f},{end:.2f},{seg_type}\n")
    
    print(f"\nExtraction complete! {len(extraction_log)} segments extracted.")
    print(f"Log saved to: {log_file}")

def extract_clips_gesres(corpus_dir, vars):
    print("GESres Dataset Tool")
    print("=" * 30)
    print("1. Rename and split gesture videos")
    print("2. Extract no-gesture segments from full videos")
    
    choice = input("Choose option (1 or 2): ")
    
    if choice == "1":
        # Original functionality
        SOURCE_DIRECTORY = f"{corpus_dir}/01Gesture_videos/"
        LOG_FILE = "file_renaming_log.csv"
        
        if os.path.exists(SOURCE_DIRECTORY):
            print("Preview of renaming (first 10 files):")
            preview_renaming(SOURCE_DIRECTORY, 10)
            
            response = input("\nDo you want to proceed with renaming and splitting? (y/n): ")
            
            if response.lower() == 'y':
                rename_and_split_files(SOURCE_DIRECTORY, vars['output_dir'], LOG_FILE)
            else:
                print("Operation cancelled.")
        else:
            print(f"Error: Source directory '{SOURCE_DIRECTORY}' does not exist.")
    
    elif choice == "2":
        # New no-gesture extraction functionality
        CSV_FILE = f"{corpus_dir}/GESRes_dataset.csv"
        FULL_VIDEOS_DIR = f"{corpus_dir}/02Full_videos/"
        
        if not os.path.exists(CSV_FILE):
            print(f"Error: CSV file '{CSV_FILE}' not found.")
            sys.exit(1)
            
        if not os.path.exists(FULL_VIDEOS_DIR):
            print(f"Error: Full videos directory '{FULL_VIDEOS_DIR}' not found.")
            sys.exit(1)
        
        extract_no_gesture_segments(CSV_FILE, FULL_VIDEOS_DIR, vars['output_dir'])
        
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")