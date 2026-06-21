import os
import logging
import threading
from pathlib import Path
from typing import List
from corpora.corpus import Corpus
from utils import ClipInfo, get_video_info, return_file_output_path

class Ecolang(Corpus):
    def __init__(self, name: str, directory: Path, defaults: dict):
        super().__init__(name, directory, defaults)

    def extract(self, cancel_event: threading.Event):
        logging.info(f"Corpus {self.name}: Starting extraction process")
        txt_files =  list(self.directory.glob("*final.txt"))
        if not txt_files:
            logging.error(f"Corpus {self.name} - No .txt files found in {self.directory}")
            return
            
        logging.info(f"Corpus {self.name} - Found {len(txt_files)} text files to process")
        for idx, txt_file in enumerate(txt_files):
            if cancel_event.is_set():
                logging.info("Cancellation event detected. Skipping remaining tasks.")
                print("Cancellation event detected. Skipping remaining tasks.")
                break
            self.process_annotation_file(txt_file)
            print(f"Corpus {self.name} - Processed {idx + 1}/{len(txt_files)} annotation files")
        

    def extract_gesture_clips(self, annotation_file_path: Path, video_duration: float, base_name: str) -> List[ClipInfo]: # file is txt
        # offsets in milliseconds
        offsets = {"ad00": 106404, "ad01": 112595, "ad02": 72247, "ad03": 113641,
           "ad04": 124305, "ad05": 178690, "ad06": 63204, "ad07": 57814,
           "ad09": 96351, "ad10": 176260, "ad11": 106606, "ad13": 149395,
           "ad14": 14011, "ad15": 9900, "ad16": 36607, "ad17": 40368, "ad18": 19942, "ad19": 14048, "ad21": 44016} 

        extracted_gestures = []
        video_id = base_name[:4]
        if video_id not in offsets:
            logging.error(f"Corpus {self.name} - No offset found for video {video_id}")
            return extracted_gestures
        
        with open(annotation_file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                parts = [p.strip() for p in line.strip().split('\t') if p.strip()]
                try:
                    if len(parts) >= 3:
                        gesture_type = parts[0]
                        safe_type = "".join(c if c.isalnum() else "_" for c in gesture_type).lower()
                        safe_type = ''.join(p.capitalize() for p in safe_type.split('_'))
                        start_time = (int(float(parts[1])) + offsets[video_id]) / 1000.0  # Convert to seconds
                        end_time = (int(float(parts[2])) + offsets[video_id]) / 1000.0    # Convert to seconds

                        if safe_type == "Objman":
                            gesture_output_path = return_file_output_path(self.move_output_dir, self.name, base_name, idx, self.move_label, safe_type)
                            label = self.move_label
                        else:
                            gesture_output_path = return_file_output_path(self.gesture_output_dir, self.name, base_name, idx, self.gesture_label, safe_type)
                            label = self.gesture_label
                        
                        if end_time > start_time and start_time >= 0 and end_time <= video_duration:
                            clip_info = ClipInfo(
                                id=idx,
                                label=label,
                                type=safe_type,
                                start=start_time,
                                end=end_time,
                                output_path=gesture_output_path
                            )
                            extracted_gestures.append(clip_info)

                except (ValueError, IndexError) as e:
                    logging.error(f"Corpus: {self.name} - Error parsing line: {line.strip()} - {e}")
                    continue
        return extracted_gestures

    def process_annotation_file(self, annotation_file: Path):
        base_name = annotation_file.stem.replace('_final', '')
        if self.check_clips_info_exists(base_name, self.name):
            logging.info(f"Corpus {self.name} - Clips info already exists for {base_name}, skipping processing.")
            return # Skip processing if clips info already exists for this file
        
        video_file_path = self.directory / f"{base_name}_speakerview480480.mp4"
        if not os.path.exists(video_file_path):
            video_file_path = self.directory / f"{base_name}_final.mp4" # naming convention for test videos
            if not os.path.exists(video_file_path):
                logging.error(f"Corpus:{self.name} - No matching video file found for {annotation_file}")
                return
        
        video_duration = get_video_info(video_file_path)['duration']
        
        gesture_clips = self.extract_gesture_clips(annotation_file, video_duration, base_name)
        if not gesture_clips:
            logging.error(f"Corpus {self.name} - No valid gesture clips extracted from {annotation_file}")
            return
        gaps = self.find_gaps_between_gestures(gesture_clips, video_duration)
        if not gaps:
            logging.error(f"Corpus: {self.name} - No valid gaps between gestures found for video {video_file_path}")
            return
        no_gesture_clips = self.extract_no_gesture_clips(gesture_clips, gaps, base_name)

        all_clips = gesture_clips + no_gesture_clips
        self.save_clips_info(all_clips, base_name, self.name)
        self.render_clips(video_file_path, all_clips, video_duration)
