
from pathlib import Path
from typing import Optional
from .segments import cut_by_segments_batch, cut_by_segments
from .pose_extraction import extract_pose_from_videos_batch, extract_pose_from_video

class PostProcessing:
    def __init__(self, data_folder: str):
        '''
        Args:
            data_folder (str): Path to the folder where inference results are saved and analysis results will be saved.
        '''
        self.data_folder = Path(data_folder)
        if not self.data_folder.exists():
            raise ValueError(f"Data folder does not exist: {self.data_folder}")

    def cut_by_segments_batch(self, 
            videos_folder: Optional[str] = None,
            video_pattern: Optional[str] = ".mp4"):
        '''
        Cuts all videos in the videos_folder (matching the video_pattern), and feature files in data folder into segments based on the matching segment files in data_folder.
        Args:
            videos_folder (Optional[str]): Path to the folder containing original videos. If None, uses data_folder which stores labeled videos.
            video_pattern (Optional[str]): Pattern to match video files (default is ".mp4").
        '''
        cut_by_segments_batch(
            data_folder=self.data_folder,
            videos_folder=videos_folder,
            video_pattern=video_pattern,
        )

    def cut_by_segments(self, video_path: str, video_name: Optional[str] = None):
        '''
        Cuts a video and feature file into segments based on the matching segment file in data_folder.
        Saves the segmented video and features in the data_folder.
        Args:
            video_path (str): Path to the video file to be segmented.
            video_name (Optional[str]): Name of the video (without extension). If None, the name is derived from the video_path.
        '''
        if video_name is None:
            video_name = Path(video_path).stem
        cut_by_segments(
            video_name=video_name,
            video_path=video_path,
            data_folder=self.data_folder
        )

    def extract_pose_from_videos_batch(self, 
            videos_folder: Optional[str] = None,
            video_pattern: Optional[str] = ".mp4"):
        '''
        Extracts pose from all videos in the videos_folder (matching the video_pattern) and saves the results in the output_folder.
        Args:
            videos_folder (Optional[str]): Path to the folder containing original videos. If None, uses data_folder which stores labeled videos.
            video_pattern (Optional[str]): Pattern to match video files (default is ".mp4").
        '''
        extract_pose_from_videos_batch(
            videos_folder=videos_folder,
            data_folder=self.data_folder,
            video_pattern=video_pattern,
        )
    
    def extract_pose_from_video(self, video_path: str, video_name: Optional[str] = None):
        '''
        Extracts pose from a single video and saves the results in the output_folder.
        Args:
            video_path (str): Path to the input video file.
            video_name (Optional[str]): Name of the video (without extension). If None, the name is derived from the video_path.
        '''
        if video_name is None:
            video_name = Path(video_path).stem
        extract_pose_from_video(
            video_name=video_name,
            video_path=video_path,
            data_folder=self.data_folder
        )
            