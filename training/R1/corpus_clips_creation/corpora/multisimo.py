import os
import logging
from pathlib import Path
import threading
from corpora.corpus import Corpus
from utils import ClipInfo, return_file_output_path, get_video_info

class Multisimo(Corpus):
    def __init__(self, name: str, directory: Path, defaults: dict):
        super().__init__(name, directory, defaults)

    def extract(self, cancel_event: threading.Event):
        logging.info(f"Corpus {self.name}: Starting extraction process")
        video_list = list(self.directory.glob(f"*.mp4"))
        logging.info(f"Corpus {self.name}: Found {len(video_list)} videos to process")
        
        for idx, video_path in enumerate(video_list):
            if cancel_event.is_set():
                logging.info("Cancellation event detected. Skipping remaining tasks.")
                print("Cancellation event detected. Skipping remaining tasks.")
                break
            self.process_annotation_file(video_path)
            print(f"Corpus {self.name} - Processed {idx + 1}/{len(video_list)} videos")

    def extract_gesture_clips(self, annotation_file: Path, video_duration: float, base_name: str):
        """Parse the gesture annotation file and return list of actions"""
        extracted_clips = []
        with open(annotation_file, 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    try:
                        start_frame = int(parts[2]) / 1000.0  # Convert to seconds
                        end_frame = int(parts[3]) / 1000.0    # Convert to seconds
                        gesture_type = parts[4] if len(parts) > 4 else "Unknown"
                        safe_type = "".join(c if c.isalnum() else "_" for c in gesture_type).lower()
                        safe_type = ''.join(p.capitalize() for p in safe_type.split('_'))
                        
                        if end_frame > start_frame and start_frame >= 0 and end_frame <= video_duration:
                            if safe_type == "NA":
                                label = self.move_label
                                output_path = return_file_output_path(self.move_output_dir, self.name, base_name, idx, label, safe_type)
                            else:
                                label = self.gesture_label
                                output_path = return_file_output_path(self.gesture_output_dir, self.name, base_name, idx, label, safe_type)

                            clip_info = ClipInfo(
                                id=idx,
                                label=label,
                                type=safe_type,
                                start=start_frame,
                                end=end_frame,       
                                output_path=output_path
                            )
                            extracted_clips.append(clip_info)
                    except ValueError as e:
                        logging.error(f"Corpus {self.name}: Error parsing line: {line.strip()}, Error: {e}")
                        continue
        return extracted_clips

    def process_annotation_file(self, video_path: Path):
        """Process a single video file"""
        base_name = video_path.stem
        if self.check_clips_info_exists(base_name, self.name):
            logging.info(f"Corpus {self.name} - Clips info already exists for {base_name}, skipping processing.")
            return # Skip processing if clips info already exists for this file
        
        annotation_file_path = self.directory / f"{base_name}.txt"
        if not os.path.exists(annotation_file_path):
            logging.error(f"Corpus {self.name}: No annotation file found for {video_path}")
            return
        
        video_duration = get_video_info(video_path)['duration']
                
        gesture_clips = self.extract_gesture_clips(annotation_file_path, video_duration, base_name)
        if not gesture_clips:
            logging.error(f"Corpus {self.name} - No valid gesture clips extracted from {annotation_file_path}")
            return
        
        gaps = self.find_gaps_between_gestures(gesture_clips, video_duration)
        if not gaps:
            logging.error(f"Corpus {self.name}: No valid gaps between gestures found for video {video_path}")
            return
        no_gesture_clips = self.extract_no_gesture_clips(gesture_clips, gaps, base_name)

        all_clips = gesture_clips + no_gesture_clips
        self.save_clips_info(all_clips, base_name, self.name)
        self.render_clips(video_path, all_clips, video_duration)