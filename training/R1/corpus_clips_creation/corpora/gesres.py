import os
import shutil
import re
import subprocess
import sys
from corpus import Corpus
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

class GesRes(Corpus):
    def __init__(self, name: str, directory: str, defaults: dict):
        super().__init__(name, directory, defaults)
        self.lexeme_mapping = LEXEME_TO_TYPE
        self.csv_file = self.directory / "GESRes_dataset.csv"
        self.full_videos_dir = self.directory / "02Full_videos"

    # def extract(self):
    #     video_files = list(self.directory.glob('*.mp4'))
    #     for video_file in video_files:
    #         print(f"\nProcessing video: {video_file}")
    #         self.process_annotation_file(video_file)

    # def process_annotation_file(self, annotation_file_path: Path):
    #     # Pattern to match files like: 1_Politician1_adaptor.mp4
    #     pattern = r'(\d+)_([^_]+)_(.+)\.mp4'

    #     match = re.match(pattern, annotation_file_path.name)
    #     if not match:
    #         logging.error(f"Corpus {self.name}: Filename does not match expected pattern: {annotation_file_path.name}")
            
    #     number, speaker, lexeme = match.groups()
        
    #     # Get gesture type from lexeme
    #     gesture_type = get_gesture_type_from_lexeme(f"dummy_{speaker}_{lexeme}.mp4")
        
    #     source_path = os.path.join(self.directory, annotation_file_path.name)
        
    #     # Handle Clinician1 specially - split into front and side views
    #     if speaker == "Clinician1":
    #         # Create filenames for front and side views
    #         front_filename = f"GESres_Clinician1Front_{number}_gesture_{gesture_type}.mp4"
    #         side_filename = f"GESres_Clinician1Side_{number}_gesture_{gesture_type}.mp4"
            
    #         front_path = os.path.join(self.gesture_output_dir, front_filename)
    #         side_path = os.path.join(self.gesture_output_dir, side_filename)
            
    #         print(f"Processing Clinician1 video: {annotation_file_path.name}")
    #         if split_video_horizontally(source_path, front_path, side_path):
    #             return
        
    #     output_path = return_file_output_path(self.gesture_output_dir, self.name, speaker, number, self.gesture_label, gesture_type)
    #     try:
    #         shutil.copy2(source_path, output_path)
    #     except Exception as e:
    #         print(f"Error processing {annotation_file_path.name}: {e}")
                    
    def extract(self):
        try:
            df = pd.read_csv(self.csv_file)
            print(f"Loaded {len(df)} annotations from CSV")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return

        special_speaker = "2Clinician_id3_vid1" # requires special handling to split into front and side views 
        speaker_mapping = {
            "1Politician_id1_AOC_vid1_speechCongress2019": "P1",
            "1Politician_id3_Boebert_vid1": "P3", 
            "2Clinician_id1_vid1": "C1",
            "3Educator_id1_Psych_vid1": "E1",
            "3Educator_id2_Politics_vid1": "E2",
            "3Educator_id3_law": "E3"
        }

# def get_gesture_type_from_lexeme(lexeme_filename):
#     """
#     Extract gesture type from lexeme filename using the mapping.
#     For files like '2358_Lecturer3_reveal.mp4', extract 'reveal' and map to type.
#     """
#     # Extract the lexeme (last part before .mp4)
#     base_name = os.path.splitext(lexeme_filename)[0]
#     parts = base_name.split('_')
    
#     if len(parts) >= 3:
#         lexeme = parts[-1]  # Last part is the lexeme
        
#         # Direct mapping
#         if lexeme in LEXEME_TO_TYPE:
#             return LEXEME_TO_TYPE[lexeme]
        
#         # Try to find partial matches for compound lexemes
#         for key, gesture_type in LEXEME_TO_TYPE.items():
#             if key in lexeme or lexeme in key:
#                 return gesture_type
        
#         # If no match found, try to categorize based on common patterns
#         if any(word in lexeme.lower() for word in ['point', 'sweep']):
#             return 'deictic'
#         elif any(word in lexeme.lower() for word in ['beat', 'chop', 'stab', 'hold']):
#             return 'beat'
#         elif lexeme.lower() in ['na', 'adaptor']:
#             return 'adaptor'
#         else:
#             return 'metaphoric'  # Default fallback
    
#     return 'unknown'

def process_annotation_file(self, video_path: Path):
    """
    Extract no-gesture segments from full videos based on CSV annotations.
    
    Args:
        csv_file: Path to GESRes_dataset.csv
        full_videos_dir: Directory containing full videos
        output_dir: Directory to save extracted no-gesture segments
        target_duration: Target duration for each no-gesture segment in seconds
    """    
        video_path = os.path.join(self.directory, video_filename)
        
        if not os.path.exists(video_path):
            logging.error(f"Corpus {self.name}: Video file not found: {video_path}")
            return
                    
        # Get gesture annotations for this video
        video_gestures = df[df['id'] == csv_id].copy()
        if len(video_gestures) == 0:
            logging.error(f"Corpus {self.name}: No gesture data found for {csv_id}")
            return
            
        # Sort gestures by start time
        video_gestures = video_gestures.sort_values('start_t')
        
        # Get video duration using ffprobe
        
        video_duration = get_video_info(video_path)['duration']
        if video_duration == 0:
            logging.error(f"Corpus {self.name}: Could not determine duration for {video_path}, skipping...")
            return
        print(f"  Video duration: {video_duration:.2f} seconds")
        
        
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
                        
                        # try:
                        #     subprocess.run(cmd_main, check=True, capture_output=True)
                        #     subprocess.run(cmd_mirror, check=True, capture_output=True)
                        #     print(f"    Extracted and split segment {segment_num_str}: {segment_start:.2f}s - {segment_end:.2f}s")
                        #     extraction_log.append((csv_id, main_filename, segment_start, segment_end, "main_nogesture"))
                        #     extraction_log.append((csv_id, mirror_filename, segment_start, segment_end, "mirror_nogesture"))
                        # except subprocess.CalledProcessError as e:
                        #     print(f"    Error extracting segment {segment_num_str}: {e}")
                    
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
    


def extract_clips_gesres(corpus_dir, vars):
    print("GESres Dataset Tool")
    print("=" * 30)
    print("1. Rename and split gesture videos")
    print("2. Extract no-gesture segments from full videos")
    
    choice = input("Choose option (1 or 2): ")
    
    if choice == "1":
        # Original functionality
        SOURCE_DIRECTORY = f"{corpus_dir}/01Gesture_videos/"
        
        if os.path.exists(SOURCE_DIRECTORY):
                        
            rename_and_split_files(SOURCE_DIRECTORY, vars['output_dir'], LOG_FILE)
    
    elif choice == "2":
        # New no-gesture extraction functionality
        CSV_FILE = f"{corpus_dir}/GESRes_dataset.csv"
        FULL_VIDEOS_DIR = f"{corpus_dir}/02Full_videos/"
        
        extract_no_gesture_segments(CSV_FILE, FULL_VIDEOS_DIR, vars['output_dir'])
        
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")