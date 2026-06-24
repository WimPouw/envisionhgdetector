import json
import re
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from utils import ClipInfo, GapInfo, return_file_output_path, extract_clip_with_padding

import random
random.seed(42) # set a seed for reproducibility

class Corpus(ABC):
    def __init__(self, name: str, directory: Path, defaults: dict):
        self.name = name
        self.directory = directory
        self.decreasing_factor = defaults.get('decreasing_factor', 0.9)
        self.clips_info_dir = Path(defaults.get('clips_info_directory', 'ClipsInfo'))
        self.gesture_output_dir = Path(defaults.get('gesture_output_directory', 'GestureClips'))
        self.no_gesture_output_dir = Path(defaults.get('no_gesture_output_directory', 'NoGestureClips'))
        self.move_output_dir = Path(defaults.get('move_output_directory', 'MoveClips'))
        self.gesture_label = defaults.get('gesture_label', 'Gesture')
        self.no_gesture_label = defaults.get('no_gesture_label', 'NoGesture')
        self.move_label = defaults.get('move_label', 'Move')

    @abstractmethod
    def extract(self):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def extract_gesture_clips(self, annotation_file_path: Path, video_duration: float, base_name: str):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def process_annotation_file(self, annotation_file_path: Path):
        raise NotImplementedError("Subclasses must implement this method")

    def clean_gesture_name(self, gesture_name: str):
        """Clean gesture name for use in filenames"""
        # Replace non-alphanumeric characters with underscore
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', gesture_name)
        # Remove consecutive underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        # Remove leading and trailing underscores
        clean_name = clean_name.strip('_')
        # Ensure name is not empty
        if not clean_name:
            clean_name = "unknown"
        return clean_name

    def render_clips(self, video_file_path: Path, clips: List[ClipInfo], video_duration: float):
        for clip in clips:
            extract_clip_with_padding(video_file_path, clip.output_path, clip.start, clip.end, video_duration, padding=1.0)

    def save_clips_info(self, clips: List[ClipInfo], base_name: str, corpus_name: str):
        all_clips_info = []
        for clip in clips:
            all_clips_info.append({
                'id': clip.id,
                'label': clip.label,
                'type': clip.type,
                'start': clip.start,
                'end': clip.end,
                'output_path': str(clip.output_path)
            })
        with open(self.clips_info_dir / f'{corpus_name}_{base_name}_clips_info.json', 'w+') as f:
            json.dump(all_clips_info, f, indent=4)

    def check_clips_info_exists(self, base_name: str, corpus_name: str):
        file_path = self.clips_info_dir / f'{corpus_name}_{base_name}_clips_info.json'
        if not file_path.exists(): # check if file exists
            return False
        with open(file_path, 'r') as f: # check if file is empty
            if len(json.load(f)) <= 0:
                return False

        return True

    def consume_gap(self, gaps: List[GapInfo], chosen_start: float, chosen_end: float):
        """Remove the chosen interval from the gaps
            This ensures no duplicates"""
        new_gaps = []
        for gap in gaps:
            if chosen_start >= gap.end or chosen_end <= gap.start:
                new_gaps.append(gap)  # unused gap
            else:
                # remove the chosen interval from gap
                if gap.start < chosen_start:
                    new_gaps.append(GapInfo(start=gap.start, end=chosen_start))
                if chosen_end < gap.end:
                    new_gaps.append(GapInfo(start=chosen_end, end=gap.end))
        return new_gaps
    
    def find_matching_no_gesture_clip(self, id: int, gaps: list, gesture_duration: float, output_path: Path, type: str=None):
        """Find a gap that can accommodate the gesture clip duration
            If no gaps are long enough, reduce the required duration by a factor and check again
            If no gaps are long enough, return None"""
        valid_gaps = []
        while gesture_duration > 0 and not valid_gaps:
            valid_gaps = [gap 
                        for gap in gaps 
                        if (gap.end - gap.start) >= gesture_duration]
            gesture_duration = gesture_duration * self.decreasing_factor

        if not valid_gaps:
            return None
        
        gap = random.choice(valid_gaps)
        max_start = gap.end - gesture_duration
        random_start = random.uniform(gap.start, max_start)
        
        return ClipInfo(
            id=id,
            label=self.no_gesture_label,
            start=random_start,
            end=random_start + gesture_duration,
            output_path=output_path,
            type=type
            )
    
    def extract_no_gesture_clips(self, gesture_clips: List[ClipInfo], gaps: List[GapInfo], base_name: str):
        # No-gesture clips are extracted in process_video after finding valid gaps
        no_gesture_clips = []
        for idx, clip in enumerate(gesture_clips):
            no_gesture_output = return_file_output_path(self.no_gesture_output_dir, self.name, base_name, idx, self.no_gesture_label) 
            no_gesture_clip = self.find_matching_no_gesture_clip(idx, gaps, (clip.end - clip.start), no_gesture_output)
            if no_gesture_clip is None:
                logging.error(f"Corpus: {self.name} - Could not find suitable non-gesture interval for gesture index {clip.id} with duration {(clip.end - clip.start):.2f}s")
            gaps = self.consume_gap(gaps, no_gesture_clip.start, no_gesture_clip.end)
            no_gesture_clips.append(no_gesture_clip)

        return no_gesture_clips

    def find_gaps_between_gestures(self, gesture_clips: List[ClipInfo], video_duration: float):
        """Find time intervals where no gestures occur"""
        gaps = []
        last_end = 0
        
        for clip in gesture_clips:
            start = clip.start
            end = clip.end
            
            if start > last_end:
                gaps.append(GapInfo(
                    start=last_end,
                    end=start
                ))
            last_end = max(last_end, end)
        
        # Add final gap if there is one
        if last_end < video_duration:
            gaps.append(GapInfo(
                start=last_end,
                end=video_duration
            ))
        
        return gaps
    
    
