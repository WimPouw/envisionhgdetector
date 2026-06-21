import os
import logging
import threading
import pandas as pd
from pathlib import Path
from corpora.corpus import Corpus
from utils import ClipInfo, get_video_info, return_file_output_path, split_video_horizontally

class GesRes(Corpus):
    def __init__(self, name: str, directory: Path, defaults: dict):
        super().__init__(name, directory, defaults)
        self.csv_file = self.directory / "GESRes_dataset.csv"
        self.full_videos_dir = self.directory / "02Full_videos"
        # Note: 2Clinician_id1_vid1 requires split into front and side views
        # all special_speakers are present in valid_speakers list
        self.special_speakers = ["2Clinician_id1_vid1"]
        # Note: several IDs in csv dont have a corresponding video file
        # Note: Ignoring 2clinician_id3_vid1 since it has 2 speakers in the entire video
        self.valid_speakers = [
            "1Politician_id1_AOC_vid1_speechCongress2019",
            "1Politician_id3_Boebert_vid1", 
            "3Educator_id1_Psych_vid1",
            "3Educator_id2_Politics_vid1",
            "3Educator_id3_law",
            "2Clinician_id1_vid1"
        ]

    def extract(self, cancel_event: threading.Event):
        logging.info(f"Corpus {self.name}: Starting extraction process")
        try:
            df = pd.read_csv(self.csv_file, quotechar='"', skipinitialspace=True, on_bad_lines='warn')
            df.columns = df.columns.str.strip().str.strip('"') # Clean column names -- remove trailing/leading spaces and quotes
            logging.info(f"Corpus {self.name}: Columns in CSV file: {df.columns.tolist()}")
            df = df[['id', 'start_t', 'end_t', 'type']]
            df[['id', 'type']] = df[['id', 'type']].apply(lambda col: col.str.strip().str.strip('"')) # Clean content -- remove trailing/leading spaces and quotes
            logging.info(f"Corpus {self.name}: Successfully read CSV file with {len(df)} rows")
        except Exception as e:
            logging.error(f"Corpus {self.name}: Error reading CSV file: {e}")
            return

        unique_ids = df['id'].unique()
        for idx, video_id in enumerate(self.valid_speakers):
            if cancel_event.is_set():
                logging.info("Cancellation event detected. Skipping remaining tasks.")
                print("Cancellation event detected. Skipping remaining tasks.")
                break
            if video_id not in unique_ids:
                logging.error(f"Corpus {self.name}: ID {video_id} not found in {unique_ids}, skipping...")
                continue

            video_gestures = df[df['id'] == video_id].copy()
            if video_id in self.special_speakers:
                self.process_annotation_file_special(video_gestures, video_id)
            else:
                self.process_annotation_file(video_gestures, video_id)
            print(f"Corpus {self.name} - Processed {idx + 1}/{len(self.valid_speakers)} annotation files")

    def extract_gesture_clips(self, video_gestures: pd.DataFrame, video_duration: float, video_id: str):
        extracted_gestures = []
        for idx, row in video_gestures.iterrows():
            gesture_start = row['start_t']
            gesture_end = row['end_t']
            type = row['type'].lower()
            type = type.replace('(', '').replace(')', '')
            type = ''.join(p.capitalize() for p in type.split()) 

            if type == "Adaptor":
                label = self.move_label
                gesture_output_path = return_file_output_path(self.move_output_dir, self.name, video_id, idx, label, type)
            else:
                label = self.gesture_label
                gesture_output_path = return_file_output_path(self.gesture_output_dir, self.name, video_id, idx, label, type)

            if not gesture_start >= 0 \
                or not gesture_end <= video_duration \
                or not gesture_end > gesture_start:
                continue

            clip_info = ClipInfo(
                id=idx,
                label=label,
                type=type,
                start=gesture_start,
                end=gesture_end,
                output_path=gesture_output_path,
            )
            extracted_gestures.append(clip_info)
        return extracted_gestures

    def process_annotation_file(self, video_gestures: pd.DataFrame, video_id: str):
        if self.check_clips_info_exists(video_id, self.name):
            logging.info(f"Corpus {self.name} - Clips info already exists for {video_id}, skipping processing.")
            return # Skip processing if clips info already exists for this file
        
        # Get gesture annotations for this video
        if len(video_gestures) == 0:
            logging.error(f"Corpus {self.name}: No gesture data found for {video_id}")
            return
        
        video_path = self.full_videos_dir /  f"{video_id}.mp4"
        if not os.path.exists(video_path):
            logging.error(f"Corpus {self.name}: Video file not found: {video_path}")
            return
        video_duration = get_video_info(video_path).get('duration', 0)
        if video_duration == 0:
            logging.error(f"Corpus {self.name}: Could not determine duration for {video_path}, skipping...")
            return
        
        # Sort gestures by start time
        video_gestures = video_gestures.sort_values('start_t')
        # extract clips
        gesture_clips = self.extract_gesture_clips(video_gestures, video_duration, video_id)
        if not gesture_clips:
            logging.error(f"Corpus {self.name} - No valid gesture clips extracted from {video_id}")
            return
        
        gaps = self.find_gaps_between_gestures(gesture_clips, video_duration)
        if not gaps:
            logging.error(f"Corpus: {self.name} - No valid gaps between gestures found for video {video_path}")
            return
        no_gesture_clips = self.extract_no_gesture_clips(gesture_clips, gaps, video_id)

        all_clips = gesture_clips + no_gesture_clips
        self.save_clips_info(all_clips, video_id, self.name)
        self.render_clips(video_path, all_clips, video_duration)

    def process_annotation_file_special(self, video_gestures: pd.DataFrame, video_id: str):
        # Get gesture annotations for this video
        if len(video_gestures) == 0:
            logging.error(f"Corpus {self.name}: No gesture data found for video {video_id}")
            return
        
        input_video_path = self.full_videos_dir / f"{video_id}.mp4"
        video_path_front = self.full_videos_dir / f"{video_id}_front.mp4"
        video_path_side = self.full_videos_dir / f"{video_id}_side.mp4"
        base_name_front = video_path_front.stem
        base_name_side = video_path_side.stem

        if self.check_clips_info_exists(base_name_front, self.name) and self.check_clips_info_exists(base_name_side, self.name):
            logging.info(f"Corpus {self.name} - Clips info already exists for {base_name_front} and {base_name_side}, skipping processing.")
            return # Skip processing if clips info already exists for this file
        

        if not os.path.exists(input_video_path):
            logging.error(f"Corpus {self.name}: Special speaker video file not found: {input_video_path}")
            return
        
        split_video_horizontally(input_video_path, video_path_front, video_path_side)
        if not os.path.exists(video_path_front) or not os.path.exists(video_path_side):
            logging.error(f"Corpus {self.name}: Video file not found: {video_path_front} or {video_path_side}")
            return

        video_duration = get_video_info(video_path_front).get('duration', 0)
        if video_duration == 0:
            logging.error(f"Corpus {self.name}: Could not determine duration for {video_path_front}, skipping...")
            return
        
        # Sort gestures by start time
        video_gestures = video_gestures.sort_values('start_t')
        gesture_clips_front = self.extract_gesture_clips(video_gestures, video_duration, video_path_front.stem)
        gesture_clips_side = self.extract_gesture_clips(video_gestures, video_duration, video_path_side.stem)

        if not gesture_clips_front:
            logging.error(f"Corpus {self.name} - No valid front gesture clips extracted from {video_id}")
            return
        if not gesture_clips_side:
            logging.error(f"Corpus {self.name} - No valid side gesture clips extracted from {video_id}")
            return        


        # gaps will be identical for both views since they have same timestamps
        gaps = self.find_gaps_between_gestures(gesture_clips_front, video_duration)
        if not gaps:
            logging.error(f"Corpus: {self.name} - No valid gaps between gestures found for video {video_path_front}")
            return
        no_gesture_clips_front = self.extract_no_gesture_clips(gesture_clips_front, gaps, base_name_front)
        no_gesture_clips_side = self.extract_no_gesture_clips(gesture_clips_side, gaps, base_name_side)

        all_clips_front = gesture_clips_front + no_gesture_clips_front
        all_clips_side = gesture_clips_side + no_gesture_clips_side

        self.save_clips_info(all_clips_front, base_name_front, self.name)
        self.render_clips(video_path_front, all_clips_front, video_duration)
        self.save_clips_info(all_clips_side, base_name_side, self.name)
        self.render_clips(video_path_side, all_clips_side, video_duration)
