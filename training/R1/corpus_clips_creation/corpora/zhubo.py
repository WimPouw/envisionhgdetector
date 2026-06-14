import json
import logging
from pathlib import Path
import threading
from corpora.corpus import Corpus
from utils import ClipInfo, return_file_output_path, get_video_info

class Zhubo(Corpus):
    def __init__(self, name: str, directory: Path, defaults: dict):
        super().__init__(name, directory, defaults)
    
    def extract(self, cancel_event: threading.Event):
        logging.info(f"Corpus {self.name}: Starting extraction process")
        video_list = list(self.directory.glob('*.h264.mp4'))
        logging.info(f"Corpus {self.name}: Found {len(video_list)} videos to process")
        for idx, video_path in enumerate(video_list):
            if cancel_event.is_set():
                logging.info("Cancellation event detected. Skipping remaining tasks.")
                print("Cancellation event detected. Skipping remaining tasks.")
                break
            self.process_annotation_file(video_path)
            print(f"Corpus {self.name} - Processed {idx + 1}/{len(video_list)} videos")

    def extract_gesture_clips(self, annotation_file: Path, video_duration: float, base_name: str, fps: float):  
        with open(annotation_file, 'r') as f:
            actions = json.load(f)
        
        extracted_gestures = []
        # Process gesture clips with 1 second padding
        for idx, action in enumerate(actions):
            # Calculate times with 1 second padding
            start_time = max(0, (action['start_frame'] / fps) - 1)  # Add 1 sec before, but don't go below 0
            end_time = min(video_duration, (action['end_frame'] / fps) + 1)  # Add 1 sec after, but don't exceed duration
            
            # Create the new filename using the requested format: ZHUBO_SPEAKERID_GESTUREID_NA
            # Where NA is the index of the gesture in the actions file
            output_file = return_file_output_path(self.gesture_output_dir, self.name, base_name, idx, self.gesture_label, "NA")
            extracted_gestures.append(ClipInfo(
                id=idx,
                label=self.gesture_label,
                start=start_time,
                end=end_time,
                output_path=output_file
            ))
        return extracted_gestures

    def process_annotation_file(self, video_file_path: Path):
        """Process a single video file"""
        # Get the video identifier (filename without extension)
        base_name = video_file_path.stem.replace('.h264', '')
        
        # Extract speaker ID and gesture ID from the filename
        # Assuming filename format like "123-456.h264.mp4" where 123 is speakerID and 456 is gestureID
        try:
            filename_parts = base_name.split('-')
            speaker_id = filename_parts[0]
            gesture_base_id = filename_parts[1] if len(filename_parts) > 1 else "unknown"
        except Exception as e:
            logging.error(f"Corpus {self.name} - Error parsing filename {base_name}: {e}")
            speaker_id = "unknown"
            gesture_base_id = "unknown"
        
        # Get the actions file path
        annotation_file = self.directory / f"{base_name}.actions.json"
        
        # Get video information
        video_info = get_video_info(video_file_path)
        fps = video_info['fps']
        video_duration = video_info['duration']
        
        if fps == 0 or video_duration == 0:
            logging.error(f"Corpus {self.name} - Could not get video info for {video_file_path}")
            return
            
        gesture_clips = self.extract_gesture_clips(annotation_file, video_duration, base_name, fps)
        if not gesture_clips:
            logging.error(f"Corpus {self.name} - No valid gesture clips extracted from {annotation_file}")
            return
        
        gaps = self.find_gaps_between_gestures(gesture_clips, video_duration)
        if not gaps:
            logging.error(f"Corpus {self.name}: No valid gaps between gestures found for video {video_file_path}")
            return
        no_gesture_clips = self.extract_no_gesture_clips(gesture_clips, gaps, base_name)

        all_clips = gesture_clips + no_gesture_clips
        self.save_clips_info(all_clips, base_name, self.name)
        self.render_clips(video_file_path, all_clips, video_duration)