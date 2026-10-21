
from pathlib import Path
from typing import Literal, Optional
from .segments import cut_by_segments_batch, cut_by_segments
from .pose_extraction import extract_pose_from_videos_batch, extract_pose_from_video
from .dtw import compute_gesture_kinematics_dtw, create_gesture_visualization

class PostProcessing:
    def __init__(self, data_folder: str):
        '''
        Args:
            data_folder (str): Dataset output root containing videos/ and analysis/.
        '''
        self.data_folder = Path(data_folder)
        if not self.data_folder.exists():
            raise ValueError(f"Data folder does not exist: {self.data_folder}")

    def cut_by_segments_batch(self, 
            videos_folder: Optional[str] = None,
            video_pattern: Optional[str] = ".mp4",
            video_scope: Literal["full", "segments"] = "full"):
        '''
        Cuts all videos in the videos_folder (matching the video_pattern), and feature files in data folder into segments based on the matching segment files in data_folder.
        Args:
            videos_folder (Optional[str]): Input folder. If None, searches data_folder for labeled full videos or segment videos.
            video_pattern (Optional[str]): Pattern to match video files (default is ".mp4").
            video_scope: "full" excludes segment folders; "segments" includes only segment folders.
        '''
        cut_by_segments_batch(
            data_folder=self.data_folder,
            videos_folder=videos_folder,
            video_pattern=video_pattern,
            video_scope=video_scope,
        )

    def compute_gesture_kinematics_dtw(self, max_gap: int = 3):
        """Compare retracked segments and save the DTW NPY matrix and kinematics CSV.

        Args:
            max_gap: Maximum consecutive missing frames allowed per coordinate.

        Returns:
            DTW distance matrix, gesture names, and kinematic features DataFrame.
            Analysis files are saved under data_folder/analysis/. Every segment video must have
            retracked world landmarks before analysis starts.
            Each segment video's frame rate is read automatically for kinematics.
        """
        dtw_dist, gesture_names, features_df =  compute_gesture_kinematics_dtw(
            data_folder=self.data_folder,
            max_gap=max_gap,
        )
        create_gesture_visualization(
            dtw_matrix=dtw_dist,
            gesture_names=gesture_names,
            output_folder=self.data_folder,
        )
        return dtw_dist, gesture_names, features_df

    def cut_by_segments(self, video_path: str, video_name: Optional[str] = None,
            video_scope: Literal["full", "segments"] = "full"):
        '''
        Cuts a video and feature file into segments based on the matching segment file in data_folder.
        Saves the segmented video and features in the data_folder.
        Args:
            video_path (str): Path to the video file to be segmented.
            video_name (Optional[str]): Name of the video (without extension). If None, the name is derived from the video_path.
            video_scope: Required path scope; mismatches raise ValueError.
        '''
        if video_name is None:
            video_name = Path(video_path).stem
        cut_by_segments(
            video_name=video_name,
            video_path=video_path,
            data_folder=self.data_folder,
            video_scope=video_scope,
        )

    def extract_pose_from_videos_batch(self, 
            videos_folder: Optional[str] = None,
            video_pattern: Optional[str] = ".mp4",
            blank_frame: bool = False,
            video_scope: Literal["full", "segments"] = "segments"):
        '''
        Extracts pose from all videos in the videos_folder (matching the video_pattern) and saves the results in the output_folder.
        Args:
            videos_folder (Optional[str]): Input folder. If None, searches data_folder for labeled full videos or segment videos.
            video_pattern (Optional[str]): Pattern to match video files (default is ".mp4").
            blank_frame (bool): If True, draws detected poses on a blank background.
            video_scope: "full" excludes segment folders; "segments" includes only segment folders.
        '''
        extract_pose_from_videos_batch(
            videos_folder=videos_folder,
            data_folder=self.data_folder,
            video_pattern=video_pattern,
            blank_frame=blank_frame,
            video_scope=video_scope,
        )
    
    def extract_pose_from_video(self, video_path: str, video_name: Optional[str] = None,
            blank_frame: bool = False,
            video_scope: Literal["full", "segments"] = "segments"):
        '''
        Extracts pose from a single video and saves the results in the output_folder.
        Args:
            video_path (str): Path to the input video file.
            video_name (Optional[str]): Name of the video (without extension). If None, the name is derived from the video_path.
            blank_frame (bool): If True, draws detected poses on a blank background.
            video_scope: Required path scope; mismatches raise ValueError.
        '''
        if video_name is None:
            video_name = Path(video_path).stem
        extract_pose_from_video(
            video_name=video_name,
            video_path=video_path,
            data_folder=self.data_folder,
            blank_frame=blank_frame,
            video_scope=video_scope,
        )
