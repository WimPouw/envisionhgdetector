"""Cut videos into gesture segments."""
import numpy as np
import pandas as pd
from uuid import uuid4
from pathlib import Path
from typing import Optional, Literal
from moviepy.video.io.VideoFileClip import VideoFileClip
from envisionhgdetector.state import SegmentColumns, PredictionColumns, Labels

def create_segments(
    annotations: pd.DataFrame,
    min_gap_s: float,
    min_length_s: float,
    segments_policy: Literal["combine", "separate"] = "separate",
) -> pd.DataFrame:
    """
    Create segments from frame-by-frame annotations, merging segments that are close in time.
    
    Args:
        annotations: DataFrame with prediction, timestamp, and original frame_index columns
        min_gap_s: Minimum gap between segments in seconds. Segments with gaps smaller 
                  than this will be merged
        min_length_s: Minimum segment length in seconds
        segments_policy: "separate" splits Gesture/Move transitions; "combine" groups
            consecutive active rows and uses their majority label. Ties use
            the first label returned by pandas mode().

    Segment endpoints are inclusive and use the last active sample's timestamp.
    A segment containing one sample therefore has zero duration.
        
    Returns:
        DataFrame with start/end times, inclusive original frame indices,
        segment_idx, prediction, and duration. Frame indices are taken from
        frame_index, not from DataFrame row positions.
    """
    if segments_policy not in ("combine", "separate"):
        raise ValueError("policy must be 'combine' or 'separate'.")
    
    output_columns = [SegmentColumns.START_TIME, SegmentColumns.START_FRAME_IDX, SegmentColumns.END_TIME, SegmentColumns.END_FRAME_IDX, SegmentColumns.SEGMENT_IDX, SegmentColumns.PREDICTION, SegmentColumns.DURATION]
    if annotations.empty:
        print("Warning: Annotations DataFrame is empty. Returning empty segments DataFrame.")
        return pd.DataFrame(columns=output_columns)
    if PredictionColumns.TIMESTAMP not in annotations.columns:
        raise ValueError(f"Annotations must contain '{PredictionColumns.TIMESTAMP}'.")
    if PredictionColumns.FRAME_INDEX not in annotations.columns:
        raise ValueError(f"Annotations must contain '{PredictionColumns.FRAME_INDEX}'.")

    is_gesture = annotations[PredictionColumns.PREDICTION] == Labels.GESTURE
    is_move = annotations[PredictionColumns.PREDICTION] == Labels.MOVE
    is_any_gesture = is_gesture | is_move
    if not is_any_gesture.any():
        print("Warning: No gesture or move labels found in annotations. Returning empty segments DataFrame.")
        return pd.DataFrame(columns=output_columns)

    if segments_policy == "combine":
        changes = np.diff(is_any_gesture.astype(int), prepend=0)
        start_idxs = np.where(changes == 1)[0]
        # Falling edges point to the first inactive row.
        end_idxs = np.where(changes == -1)[0] - 1
        if len(start_idxs) > len(end_idxs):
            end_idxs = np.append(end_idxs, len(annotations) - 1)
    else:
        active = is_any_gesture.to_numpy()
        labels = annotations[PredictionColumns.PREDICTION].to_numpy()
        label_changes = labels[1:] != labels[:-1] # compare consecutive labels
        # Start indices are where we have an active label and either the previous label was inactive or the label changed.
        start_idxs = np.flatnonzero(active & np.r_[True, label_changes])
        # End indices are where we have an active label and either the next label is inactive or the label changes.
        end_idxs = np.flatnonzero(active & np.r_[label_changes, True])

    initial_segments = []
    for start_idx, end_idx in zip(start_idxs, end_idxs):
        segment_labels = annotations.iloc[start_idx:end_idx + 1][PredictionColumns.PREDICTION]
        current_label = segment_labels.mode()[0] if segments_policy == "combine" else segment_labels.iloc[0]
        initial_segments.append({
            SegmentColumns.START_TIME: annotations.iloc[start_idx][PredictionColumns.TIMESTAMP],
            SegmentColumns.START_FRAME_IDX: annotations[PredictionColumns.FRAME_INDEX].iloc[start_idx],
            SegmentColumns.END_TIME: annotations.iloc[end_idx][PredictionColumns.TIMESTAMP],
            SegmentColumns.END_FRAME_IDX: annotations[PredictionColumns.FRAME_INDEX].iloc[end_idx],
            SegmentColumns.PREDICTION: current_label,
        })

    if not initial_segments:
        print("Warning: No valid segments found after initial segmentation. Returning empty segments DataFrame.")
        return pd.DataFrame(columns=output_columns)

    merged_segments = []
    current_segment = initial_segments[0]
    for next_segment in initial_segments[1:]:
        time_gap = next_segment[SegmentColumns.START_TIME] - current_segment[SegmentColumns.END_TIME]
        same_label = current_segment[SegmentColumns.PREDICTION] == next_segment[SegmentColumns.PREDICTION]
        if time_gap <= min_gap_s and same_label:
            current_segment[SegmentColumns.END_TIME] = next_segment[SegmentColumns.END_TIME]
            current_segment[SegmentColumns.END_FRAME_IDX] = next_segment[SegmentColumns.END_FRAME_IDX]
        else:
            if current_segment[SegmentColumns.END_TIME] - current_segment[SegmentColumns.START_TIME] >= min_length_s:
                merged_segments.append(current_segment)
            current_segment = next_segment
    # last segment check
    if current_segment[SegmentColumns.END_TIME] - current_segment[SegmentColumns.START_TIME] >= min_length_s:
        merged_segments.append(current_segment)

    return pd.DataFrame([
        {
            SegmentColumns.START_TIME: segment[SegmentColumns.START_TIME],
            SegmentColumns.START_FRAME_IDX: segment[SegmentColumns.START_FRAME_IDX],
            SegmentColumns.END_TIME: segment[SegmentColumns.END_TIME],
            SegmentColumns.END_FRAME_IDX: segment[SegmentColumns.END_FRAME_IDX],
            SegmentColumns.SEGMENT_IDX: index,
            SegmentColumns.PREDICTION: segment[SegmentColumns.PREDICTION],
            SegmentColumns.DURATION: segment[SegmentColumns.END_TIME] - segment[SegmentColumns.START_TIME],
        }
        for index, segment in enumerate(merged_segments, start=1)
    ], columns=output_columns)

def cut_by_segments_batch(
    segments_folder: str,
    output_folder: str,
    videos_folder: str,
    features_folder: str,
    suffix_to_remove: Optional[str] = None # example: "_labeled"
):
    '''
    Batch: Cuts videos and features into segments based on the matching segments files.

    Args:
        videos_folder: Path to the input videos folder. Recursive glob will find all .mp4 videos
        segments_folder: Path to the folder with CSV files containing segment information. Search on segments_folder/video_name/video_name_segments.csv
        features_folder: Path to the folder with NPY files containing feature information. Search on features_folder/video_name/video_name_features.npy
        output_folder: Path to the folder where segmented videos will be saved. Structure will be output_folder/video_name/file.mp4 or file.npy
    Optional Args:
        suffix_to_remove: Optional suffix to remove from video names when matching with segments and features files. To support both original and labeled videos cutting.
    '''
    videos_folder = Path(videos_folder)
    segments_folder = Path(segments_folder)
    features_folder = Path(features_folder)
    output_folder = Path(output_folder)

    if not videos_folder.exists():
        raise ValueError(f"Videos folder does not exist: {videos_folder}")
    if not segments_folder.exists():
        raise ValueError(f"Segments folder does not exist: {segments_folder}")
    if not features_folder.exists():
        raise ValueError(f"Features folder does not exist: {features_folder}")
    
    output_folder.mkdir(parents=True, exist_ok=True)

    video_paths = list(videos_folder.rglob("*.mp4"))
    for video_path in video_paths:
        video_name = video_path.stem
        if suffix_to_remove and video_name.endswith(suffix_to_remove):
            video_name = video_name.replace(suffix_to_remove, "")

        segment_csv_path = segments_folder / video_name / f"{video_name}_segments.csv"
        features_npy_path = features_folder / video_name / f"{video_name}_features.npy"
        video_output_folder = output_folder / video_name
        video_output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Processing video: {video_path},\nsegments: {segment_csv_path},\nfeatures: {features_npy_path}")
        try:
            cut_by_segments(
                video_path=str(video_path),
                output_folder=str(video_output_folder),
                segments_csv_path=str(segment_csv_path),
                features_npy_path=str(features_npy_path),
                video_name=video_name
            )
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            continue

def cut_by_segments(
    segments_csv_path: str,
    output_folder: str,
    video_name: str,
    video_path: Optional[str] = None,
    features_npy_path: Optional[str] = None,
):
    """
    Cuts a single video and features file into segments based on the matching the segment file.
    
    Args:
        segments_csv_path: Path to the CSV file containing segment information
        video_name: Name of the video file
        output_folder: Path to the folder where segmented videos will be saved. Structure will be output_folder/file.mp4 or file.npy
    
    Optional Args:
        video_path: Path to the input video file
        features_npy_path: Path to the NPY file containing feature information
    """
    segments_csv_path = Path(segments_csv_path)
    output_folder = Path(output_folder)
    video_path = Path(video_path) if video_path is not None else None
    features_npy_path = Path(features_npy_path) if features_npy_path is not None else None

    if not segments_csv_path.exists():
        raise ValueError(f"Segments CSV file does not exist: {segments_csv_path}")

    # We need at least one of video_path or features_npy_path to exist to proceed
    has_video = video_path is not None and video_path.is_file()
    has_features = features_npy_path is not None and features_npy_path.is_file()
    if not has_video and not has_features:
        raise ValueError(f"Both video file and features NPY file do not exist: {video_path}, {features_npy_path}")
    
    output_folder.mkdir(parents=True, exist_ok=True)
    segments_df = pd.read_csv(segments_csv_path)
    if segments_df.empty:
        raise ValueError(f"No segments found in {segments_csv_path}")

    if has_video:
        print(f"Cutting video: {video_path} into segments based on {segments_csv_path}")
        _cut_video(video_path=str(video_path), segments_df=segments_df, output_folder=output_folder, video_name=video_name)

    if has_features:
        print(f"Cutting features: {features_npy_path} into segments based on {segments_csv_path}")
        _cut_features(segments_df=segments_df, features_npy_path=features_npy_path, output_folder=output_folder, video_name=video_name)

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
            segment_filename = f"{video_name}_segment_{segment_idx}_{prediction}_{start_time:.2f}_{end_time:.2f}.mp4"
            segment_path = output_folder / segment_filename
            temp_path = segment_path.with_name(f".{segment_path.stem}.{uuid4().hex}.tmp.mp4")

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
                temp_path.replace(segment_path)
            except Exception as e:
                print(f"Error creating video segment {segment_filename}: {e}")
            finally:
                try:
                    if segment_clip is not None:
                        segment_clip.close()
                finally:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except OSError as cleanup_error:
                        print(f"Could not remove partial video {temp_path}: {cleanup_error}")
                
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
    finally:
        if video_obj is not None:
            video_obj.close()
        
def _cut_features(segments_df: pd.DataFrame, features_npy_path: Path, output_folder: Path, video_name: str):
    features = np.load(features_npy_path)
    for _, segment in segments_df.iterrows():
        temp_path = None
        try:
            segment_idx =segment[SegmentColumns.SEGMENT_IDX] 
            start_time = segment[SegmentColumns.START_TIME]
            end_time = segment[SegmentColumns.END_TIME]
            prediction = segment[SegmentColumns.PREDICTION]
            start_frame = segment[SegmentColumns.START_FRAME_IDX]
            end_frame = segment[SegmentColumns.END_FRAME_IDX]

            # extract features segments
            features_filename = f"{video_name}_segment_{segment_idx}_{prediction}_{start_time:.2f}_{end_time:.2f}_features.npy"
            features_path = output_folder / features_filename

            if 0 <= start_frame <= end_frame < len(features):
                segment_features = features[start_frame:end_frame + 1]
                temp_path = features_path.with_name(f".{features_path.stem}.{uuid4().hex}.tmp.npy")
                np.save(temp_path, segment_features)
                temp_path.replace(features_path)
                print(f"Created features segment: {features_filename}")
            else:
                print(f"Warning: Frame indices {start_frame}:{end_frame} out of bounds for features array of length {len(features)}")
        except Exception as e:
            print(f"Error creating features segment for {video_name}: {e}")
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    print(f"Could not remove partial features {temp_path}: {cleanup_error}")
