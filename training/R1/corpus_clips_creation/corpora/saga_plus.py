import os
import re
import logging
from pathlib import Path
import threading
from utils import ClipInfo, return_file_output_path, get_video_info
from corpora.corpus import Corpus

class SagaPlus(Corpus):
    def __init__(self, name: str, directory: Path, defaults: dict):
        super().__init__(name, directory, defaults)
        self.videos_folder = self.directory / 'VideosCentered'
        self.annotations_folder = self.directory / 'annotations'

    def extract(self, cancel_event: threading.Event):
        logging.info(f"Corpus {self.name}: Starting extraction process")
        txt_files = list(self.annotations_folder.glob("*.txt"))
        logging.info(f"Corpus {self.name} - Found {len(txt_files)} text files to process")
            
        for idx, txt_file in enumerate(txt_files):
            if cancel_event.is_set():
                logging.info("Cancellation event detected. Skipping remaining tasks.")
                print("Cancellation event detected. Skipping remaining tasks.")
                break
            self.process_annotation_file(txt_file)
            print(f"Corpus {self.name} - Processed {idx + 1}/{len(txt_files)} annotation files")

    def extract_gesture_clips(self, annotation_file_path: str, video_duration: float, base_name: str): # file is txt
        extracted_gestures = []
        seen_intervals = set()  # Track unique time intervals
        
        with open(annotation_file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                parts = [p.strip() for p in line.strip().split('\t') if p.strip()]
                try:
                    # Expected format is like: "R.G.Left.Phrase Router 12490 13510 beat"
                    if len(parts) >= 5:
                        hand_info = parts[0]  # e.g., "R.G.Left.Phrase"
                        speaker = parts[1]    # e.g., "Router"
                        start_time = float(parts[2]) / 1000.0  # Convert to seconds
                        end_time = float(parts[3]) / 1000.0    # Convert to seconds
                        gesture_type = parts[4]  # e.g., "beat"
                        gesture_type_clean = self.clean_gesture_name(gesture_type)
                        
                        # Skip annotations with duplicate start and end times -- why would there be duplicates in the first place
                        interval_key = (start_time, end_time)
                        if interval_key in seen_intervals:
                            continue
                        
                        seen_intervals.add(interval_key)
                        
                        gesture_output_path = return_file_output_path(self.gesture_output_dir, self.name, base_name, idx, self.gesture_label, gesture_type_clean)
                        if end_time > start_time and start_time >= 0 and end_time <= video_duration:  # Convert video duration to milliseconds
                            clip_info = ClipInfo(
                                id=idx,
                                label=self.gesture_label,
                                type=gesture_type_clean,
                                start=start_time,
                                end=end_time,
                                output_path=gesture_output_path
                            )
                            extracted_gestures.append(clip_info)

                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line.strip()} - {e}")
                    continue
        
        return extracted_gestures
    
    def process_annotation_file(self, annotation_file_path: Path):
        base_name = annotation_file_path.stem
        # Pattern 1: If annotation file is "GrobV9.txt", look for "V9K2_left.mp4"
        # Pattern 2: Direct match - if annotation is "V9.txt", look for "V9K2_left.mp4"
        # Pattern 3: Extract just the V number (V9 from GrobV9)
        match = re.search(r'([V|v]\d+)', base_name) 
        if not match:
            logging.error(f"Corpus {self.name} - Base name {base_name} does not match expected pattern.")
            return
        
        base_name = f"{match.group(1)}K2"
        video_file_path = self.videos_folder / f"{base_name}_left.mp4"
        
        if not os.path.exists(video_file_path):
            logging.error(f"Corpus {self.name} - Video file not found for annotation {annotation_file_path}: {video_file_path}")
            return

        video_duration = get_video_info(video_file_path)['duration']
        
        gesture_clips = self.extract_gesture_clips(annotation_file_path, video_duration, base_name)
        if not gesture_clips:
            logging.error(f"Corpus {self.name} - No valid gesture clips extracted from {annotation_file_path}")
            return
        
        gaps = self.find_gaps_between_gestures(gesture_clips, video_duration)
        if not gaps:
            logging.error(f"Corpus: {self.name} - No valid gaps between gestures found for video {video_file_path}")
            return
        no_gesture_clips = self.extract_no_gesture_clips(gesture_clips, gaps, base_name)

        all_clips = gesture_clips + no_gesture_clips
        self.save_clips_info(all_clips, base_name, self.name)
        self.render_clips(video_file_path, all_clips, video_duration)