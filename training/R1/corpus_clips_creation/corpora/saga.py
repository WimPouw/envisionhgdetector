import os
import re
import csv
import logging
from pathlib import Path
import threading
from utils import ClipInfo, return_file_output_path, get_video_info
from corpora.corpus import Corpus

class Saga(Corpus):
    def __init__(self, name: str, directory: Path, defaults: dict):
        super().__init__(name, directory, defaults)
        self.videos_folder = self.directory / 'OriginalVideos'
        self.annotations_folder = self.directory / 'OriginalCodings'

    def extract(self, cancel_event: threading.Event):
        logging.info(f"Corpus {self.name}: Starting extraction process")
        video_list = list(self.videos_folder.glob(f'*.mp4'))
        logging.info(f"Corpus {self.name}: Found {len(video_list)} videos to process")
        
        for idx, video_path in enumerate(video_list):
            if cancel_event.is_set():
                logging.info("Cancellation event detected. Skipping remaining tasks.")
                print("Cancellation event detected. Skipping remaining tasks.")
                break
            self.process_annotation_file(video_path)
            print(f"Corpus {self.name} - Processed {idx + 1}/{len(video_list)} videos")

    def extract_gesture_clips(self, annotation_file_path: Path, video_duration: float, base_name: str):
        """Parse the CSV annotation file and return list of gestures"""
        extracted_gestures = []
        with open(annotation_file_path, 'r', encoding='utf-8') as f:
            # Detect column names for time and gesture
            reader = csv.reader(f)
            header = next(reader)
            
            # Find column indices
            start_col = -1
            end_col = -1
            gesture_col = -1
            
            for i, col_name in enumerate(header):
                if 'begin time' in col_name.lower() or 'start' in col_name.lower():
                    start_col = i
                elif 'end time' in col_name.lower() or 'stop' in col_name.lower():
                    end_col = i
                elif 'gesture' in col_name.lower() or 'phrase' in col_name.lower():
                    gesture_col = i
            
            if start_col == -1 or end_col == -1 or gesture_col == -1:
                logging.error(f"Corpus {self.name} - Could not identify required columns in {annotation_file_path}. Expected columns for start time, end time, and gesture type.")
                return []
            
            # Reset file pointer and skip header
            f.seek(0)
            next(reader)
            
            # Parse rows
            for idx, row in enumerate(reader):
                if len(row) > max(start_col, end_col, gesture_col):
                    start_time = int(row[start_col]) / 1000.0  # Convert ms to seconds
                    end_time = int(row[end_col]) / 1000.0      # Convert ms to seconds
                    gesture_type = row[gesture_col].strip()
                    safe_gesture_type = self.clean_gesture_name(gesture_type).lower()
                    safe_gesture_type = ''.join(p.capitalize() for p in safe_gesture_type.split('_'))

                    if safe_gesture_type == "Move":
                        gesture_output_path = return_file_output_path(self.move_output_dir, self.name, base_name, idx, self.move_label, safe_gesture_type)
                        label = self.move_label
                    else:                
                        gesture_output_path = return_file_output_path(self.gesture_output_dir, self.name, base_name, idx, self.gesture_label, safe_gesture_type)
                        label = self.gesture_label

                    if gesture_type  \
                    and end_time > start_time \
                    and start_time >= 0 \
                    and end_time <= video_duration:
                        clip_info = ClipInfo(
                            id=idx,
                            label=label,
                            type=safe_gesture_type,
                            start=start_time,
                            end=end_time,
                            output_path=gesture_output_path
                        )
                        extracted_gestures.append(clip_info)

        return extracted_gestures

    def process_annotation_file(self, video_path: Path):
        """Process a single video file with its corresponding annotation file"""
        base_name = video_path.stem
        match = re.search(r'(\D+)(\d+)', base_name)
        if match:
            prefix, number = match.groups()
            base_name =  f"{prefix}{int(number)}"  # This handles V07 -> V7 conversion. While maintaining V10 as is.

        annotation_file_path = self.annotations_folder / f"{base_name}.csv"

        if self.check_clips_info_exists(base_name, self.name):
            logging.info(f"Corpus {self.name} - Clips info already exists for {base_name}, skipping processing.")
            return # Skip processing if clips info already exists for this file

        if not os.path.exists(annotation_file_path):
            logging.error(f"Corpus: {self.name} - Annotation file not found for video {base_name}. Expected at {annotation_file_path}")
            return
        
        video_duration = get_video_info(video_path)['duration']
        if video_duration == 0:
            logging.error(f"Corpus: {self.name} - Could not determine duration for {video_path}")
            return
                
        gesture_clips = self.extract_gesture_clips(annotation_file_path, video_duration, base_name)
        if not gesture_clips:
            logging.error(f"Corpus: {self.name} - No valid gesture clips extracted from {annotation_file_path}")
            return

        # create non-gesture clips
        gaps = self.find_gaps_between_gestures(gesture_clips, video_duration)
        if not gaps:
            logging.error(f"Corpus: {self.name} - No valid gaps between gestures found for video {video_path}")
            return
        no_gesture_clips = self.extract_no_gesture_clips(gesture_clips, gaps, base_name)

        all_clips = gesture_clips + no_gesture_clips
        self.save_clips_info(all_clips, base_name, self.name)
        self.render_clips(video_path, all_clips, video_duration)