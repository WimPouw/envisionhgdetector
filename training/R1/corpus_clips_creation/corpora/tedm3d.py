import os
import logging
import threading
import pandas as pd
import glob as glob
from pathlib import Path
from utils import ClipInfo, return_file_output_path, get_video_info
from corpora.corpus import Corpus

class TedM3D(Corpus):
    def __init__(self, name: str, directory: str, defaults: dict):
        super().__init__(name, directory, defaults)

    def extract(self, cancel_event: threading.Event):
        logging.info(f"Corpus {self.name}: Starting extraction process")
        annotation_files = list(self.directory.glob('*.csv'))
        logging.info(f"Corpus {self.name} - Found {len(annotation_files)} annotation files in {self.directory}")
        
        for idx, annotation_file in enumerate(annotation_files):
            if cancel_event.is_set():
                logging.info("Cancellation event detected. Skipping remaining tasks.")
                print("Cancellation event detected. Skipping remaining tasks.")
                break
            self.process_annotation_file(annotation_file)
            print(f"Corpus {self.name} - Processed {idx + 1}/{len(annotation_files)} annotation files")

    def extract_gesture_clips(self, annotation_file_path: Path, video_duration: float, base_name: str):
        columns = ['tier', 'empty', 'start_segment', 'end_segment', 'label']
        df = pd.read_csv(str(annotation_file_path))
        df.columns = columns
        df = df[df['label'] == 'stroke'].copy()

        extracted_gestures = []
        for idx, row in df.iterrows():
            # Time in milliseconds, convert to seconds
            gesture_start = row['start_segment'] / 1000
            gesture_end = row['end_segment'] / 1000

            if not gesture_start >= 0 \
                or not gesture_end <= video_duration \
                or not gesture_end > gesture_start:
                continue

            gesture_output_path = return_file_output_path(self.gesture_output_dir, self.name, base_name, idx, self.gesture_label, None)
            clip_info = ClipInfo(
                id=idx,
                label=self.gesture_label,
                start=gesture_start,
                end=gesture_end,
                output_path=gesture_output_path,
            ) 
            extracted_gestures.append(clip_info)
            
        return extracted_gestures

    def process_annotation_file(self, annotation_file_path: Path):        
        # Get corresponding video file
        base_name = annotation_file_path.stem
        video_file_path = self.directory / f"{base_name}.mp4"

        if self.check_clips_info_exists(base_name, self.name):
            logging.info(f"Corpus {self.name} - Clips info already exists for {base_name}, skipping processing.")
            return # Skip processing if clips info already exists for this file

        if not os.path.exists(video_file_path):
            logging.error(f"Corpus: {self.name} - Video file not found for {annotation_file_path}. Expected at {video_file_path}")
            return
        
        video_duration = get_video_info(video_file_path).get('duration', 0)
        if video_duration == 0:
            logging.error(f"Corpus: {self.name} - Could not get duration for video {video_file_path}")
            return
        
        # extract clips
        gesture_clips = self.extract_gesture_clips(annotation_file_path, video_duration, base_name)
        if not gesture_clips:
            logging.error(f"Corpus: {self.name} - No valid gesture clips extracted from {annotation_file_path}")
            return
        
        gaps = self.find_gaps_between_gestures(gesture_clips, video_duration)
        if not gaps:
            logging.error(f"Corpus: {self.name} - No valid gaps between gestures found for video {video_file_path}")
            return
        no_gesture_clips = self.extract_no_gesture_clips(gesture_clips, gaps, base_name)

        all_clips = gesture_clips + no_gesture_clips
        self.save_clips_info(all_clips, base_name, self.name)
        self.render_clips(video_file_path, all_clips, video_duration)