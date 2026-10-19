"""Cut videos into gesture segments."""
import shutil
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
from moviepy.video.io.VideoFileClip import VideoFileClip
from envisionhgdetector.state import SegmentColumns
from envisionhgdetector.utils import filter_segment_source_paths, filter_video_paths, return_path

def cut_by_segments_batch(
    data_folder: str,
    video_pattern: str,
    videos_folder: str | None,
    video_scope: Literal["full", "segments"] = "full",
):
    '''
    Batch: Cuts videos and features into segments based on the matching segments files.

    Args:
        videos_folder: Input folder. If None, searches data_folder for labeled full videos or segment videos.
        data_folder: 
            - Path where segments and features files is stored. 
            - Path where segmented video and features will be saved.
        video_pattern: Pattern to match video files (ex: .mp4).
        video_scope: "full" excludes segment folders; "segments" includes only segment folders.
    '''
    filter_video_paths([], video_scope)  # Validate before discovering or processing videos.
    data_folder = Path(data_folder)
    if not data_folder.exists():
        raise ValueError(f"data folder does not exist: {data_folder}")

    remove_suffix = None
    if videos_folder is None:
        # Full videos are labeled; generated segment videos have no labeled suffix.
        remove_suffix = "_labeled" if video_scope == "full" else None
        search_pattern = f"*{remove_suffix}.mp4" if remove_suffix else f"*{video_pattern}"
        video_paths = [
            path
            for path in data_folder.rglob(search_pattern)
            if path.is_file()
        ]
    else:
        # use videos in the provided videos_folder
        videos_folder = Path(videos_folder)
        if not videos_folder.exists():
            raise ValueError(f"videos folder does not exist: {videos_folder}")
        video_paths = [
            path
            for path in videos_folder.rglob(f"*{video_pattern}")
            if path.is_file()
        ]

    if video_scope == "segments":
        video_paths = filter_segment_source_paths(video_paths)
    for video_path in filter_video_paths(video_paths, video_scope):
        video_name = video_path.stem
        if remove_suffix and video_name.endswith(remove_suffix):
            video_name = video_name[:-len(remove_suffix)]
        try:
            cut_by_segments(
                video_name=video_name,
                video_path=str(video_path),
                data_folder=str(data_folder),
                video_scope=video_scope,
            )
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            continue

def cut_by_segments(
    video_name: str,
    video_path: str,
    data_folder: str,
    video_scope: Literal["full", "segments"] = "full",
):
    """
    Cuts a single video and features file into segments based on the matching the segment file.
    
    Args:
        video_name: Name of the video (without extension)
        video_path: Path to the input video file
        video_scope: Required path scope; mismatches raise ValueError.
        data_folder: 
            - Path where segments and features files is stored. 
            - Path where segmented video and features will be saved.
    """
    if not filter_video_paths([Path(video_path)], video_scope):
        raise ValueError(f"Video path does not match video_scope={video_scope!r}: {video_path}")
    segments_csv_path = return_path("segments", data_folder, video_name)
    features_npy_path = return_path("features", data_folder, video_name)
    if segments_csv_path is None:
        raise ValueError(f"Could not find segments CSV for video: {video_name}")
    if features_npy_path is None:
        raise ValueError(f"Could not find features NPY for video: {video_name}")

    segments_csv_path = Path(segments_csv_path)
    features_npy_path = Path(features_npy_path)
    video_path = Path(video_path)
    if not segments_csv_path.exists():
        raise ValueError(f"Segments CSV file does not exist: {segments_csv_path}")
    if not features_npy_path.exists():
        raise ValueError(f"Features NPY file does not exist: {features_npy_path}")
    if not video_path.exists():
        raise ValueError(f"Video file does not exist: {video_path}")
    
    segments_df = pd.read_csv(segments_csv_path)
    if segments_df.empty:
        raise ValueError(f"No segments found in {segments_csv_path}")

    print(f"Cutting video: {video_path} into segments based on {segments_csv_path}")
    _cut_video(video_path=str(video_path), segments_df=segments_df, output_folder=data_folder, video_name=video_name)

    print(f"Cutting features: {features_npy_path} into segments based on {segments_csv_path}")
    _cut_features(segments_df=segments_df, features_npy_path=features_npy_path, output_folder=data_folder, video_name=video_name)

def _cut_video(video_path: str, segments_df: pd.DataFrame, output_folder: Path, video_name: str):
    video_obj = None
    try:
        video_obj = VideoFileClip(video_path)
        for _, segment in segments_df.iterrows():
            segment_idx =segment[SegmentColumns.SEGMENT_IDX] 
            start_time = segment[SegmentColumns.START_TIME]
            end_time = segment[SegmentColumns.END_TIME]
            prediction = segment[SegmentColumns.PREDICTION]

            # extract video segments
            segment_path = return_path(
                output_type="segment_videos",
                output_folder=output_folder,
                video_name=video_name,
                segment_idx=segment_idx,
                prediction=prediction,
                start_time=start_time,
                end_time=end_time 
            )
            if segment_path is None:
                print(f"Could not determine output path for segment video: {video_name}, segment_idx: {segment_idx}")
                return
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / f"{video_name}.mp4"
                segment_clip = None
                try:
                    # Segment timestamps identify inclusive frame endpoints.
                    exclusive_end_time = min(end_time + 1 / video_obj.fps, video_obj.duration)
                    segment_clip = video_obj.subclipped(start_time, exclusive_end_time)
                    segment_clip.write_videofile(
                        str(temp_path),
                        codec='libx264',
                        audio=False
                    )
                    shutil.copy2(temp_path, segment_path)
                except Exception as e:
                    print(f"Error creating video segment {segment_path}: {e}")
                finally:
                    if segment_clip is not None:
                        segment_clip.close()
                    
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
    finally:
        if video_obj is not None:
            video_obj.close()
        
def _cut_features(segments_df: pd.DataFrame, features_npy_path: Path, output_folder: Path, video_name: str):
    features = np.load(features_npy_path)
    for _, segment in segments_df.iterrows():
        try:
            segment_idx =segment[SegmentColumns.SEGMENT_IDX] 
            start_time = segment[SegmentColumns.START_TIME]
            end_time = segment[SegmentColumns.END_TIME]
            prediction = segment[SegmentColumns.PREDICTION]
            start_frame = segment[SegmentColumns.START_FRAME_IDX]
            end_frame = segment[SegmentColumns.END_FRAME_IDX]

            # extract features segments
            features_path = return_path(
                output_type="segment_features",
                output_folder=output_folder,
                video_name=video_name,
                segment_idx=segment_idx,
                prediction=prediction,
                start_time=start_time,
                end_time=end_time 
            )

            if 0 <= start_frame <= end_frame < len(features):
                segment_features = features[start_frame:end_frame + 1]
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir) / f"{video_name}.npy"
                    np.save(temp_path, segment_features)
                    shutil.copy2(temp_path, features_path)
                print(f"Created features segment: {features_path}")
            else:
                print(f"Warning: Frame indices {start_frame}:{end_frame} out of bounds for features array of length {len(features)}")
        except Exception as e:
            print(f"Error creating features segment for {video_name}: {e}")