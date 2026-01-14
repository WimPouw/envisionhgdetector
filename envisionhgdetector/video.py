# envisionhgdetector/video.py
"""
Video processing utilities for gesture detection.
Handles video labeling, segmentation, and file operations.
"""

import os
import glob
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm


class VideoProcessingError(Exception):
    """Exception raised for errors during video processing."""
    pass


@contextmanager
def video_capture(source: Union[str, int]) -> Generator[cv2.VideoCapture, None, None]:
    """
    Context manager for cv2.VideoCapture that ensures proper resource cleanup.

    Args:
        source: Video file path or camera index

    Yields:
        cv2.VideoCapture object

    Raises:
        VideoProcessingError: If video cannot be opened

    Example:
        with video_capture("video.mp4") as cap:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
    """
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video source: {source}")
        yield cap
    finally:
        cap.release()


def get_video_info(video_path: str) -> Dict[str, Union[int, float]]:
    """
    Get video metadata without keeping the file open.

    Args:
        video_path: Path to video file

    Returns:
        Dict with 'fps', 'frame_count', 'width', 'height', 'duration'
    """
    with video_capture(video_path) as cap:
        return {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }


def validate_video_path(video_path: str, must_exist: bool = True) -> str:
    """
    Validate a video file path.

    Args:
        video_path: Path to validate
        must_exist: If True, verify the file exists

    Returns:
        Validated path string

    Raises:
        VideoProcessingError: If validation fails
    """
    if not isinstance(video_path, str):
        raise VideoProcessingError(
            f"video_path must be a string, got {type(video_path).__name__}"
        )

    if not video_path.strip():
        raise VideoProcessingError("video_path cannot be empty")

    if must_exist and not os.path.exists(video_path):
        raise VideoProcessingError(f"Video file not found: {video_path}")

    return video_path


def validate_output_path(output_path: str) -> str:
    """
    Validate an output file path and ensure parent directory exists.

    Args:
        output_path: Path to validate

    Returns:
        Validated path string

    Raises:
        VideoProcessingError: If validation fails
    """
    if not isinstance(output_path, str):
        raise VideoProcessingError(
            f"output_path must be a string, got {type(output_path).__name__}"
        )

    if not output_path.strip():
        raise VideoProcessingError("output_path cannot be empty")

    # Ensure parent directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except OSError as e:
            raise VideoProcessingError(
                f"Cannot create output directory {parent_dir}: {e}"
            )

    return output_path


def validate_segments_dataframe(segments: pd.DataFrame) -> pd.DataFrame:
    """
    Validate segments DataFrame has required columns.

    Args:
        segments: DataFrame to validate

    Returns:
        Validated DataFrame

    Raises:
        VideoProcessingError: If validation fails
    """
    if not isinstance(segments, pd.DataFrame):
        raise VideoProcessingError(
            f"segments must be a pandas DataFrame, got {type(segments).__name__}"
        )

    if segments.empty:
        return segments  # Empty is valid

    required_cols = ['start_time', 'end_time', 'label']
    missing_cols = [col for col in required_cols if col not in segments.columns]

    if missing_cols:
        raise VideoProcessingError(
            f"segments DataFrame missing required columns: {missing_cols}. "
            f"Found columns: {list(segments.columns)}"
        )

    return segments


def validate_positive_number(value: Union[int, float], name: str, allow_zero: bool = False) -> float:
    """
    Validate that a value is a positive number.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        allow_zero: If True, zero is allowed

    Returns:
        Validated float value

    Raises:
        VideoProcessingError: If validation fails
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise VideoProcessingError(
            f"{name} must be a number, got {type(value).__name__}"
        )

    if allow_zero:
        if value < 0:
            raise VideoProcessingError(f"{name} must be >= 0, got {value}")
    else:
        if value <= 0:
            raise VideoProcessingError(f"{name} must be > 0, got {value}")

    return value


def label_video(
    video_path: str,
    segments: pd.DataFrame,
    output_path: str,
    predictions_df: Optional[pd.DataFrame] = None,
    valid_timestamps: Optional[List[float]] = None,
    motion_threshold: Optional[float] = None,
    gesture_threshold: Optional[float] = None,
    window_duration: float = 10.0,
    target_fps: float = 25.0
) -> None:
    """
    Label a video with predicted gestures based on segments.
    Creates output at target_fps regardless of input fps.

    Args:
        video_path: Path to input video file
        segments: DataFrame with segment information (start_time, end_time, label)
        output_path: Path to save labeled video
        predictions_df: Optional DataFrame with frame-by-frame predictions for overlay
        valid_timestamps: Optional list of valid timestamps
        motion_threshold: Motion threshold for display (optional, 0.0-1.0)
        gesture_threshold: Gesture threshold for display (optional, 0.0-1.0)
        window_duration: Duration of confidence graph window in seconds (> 0)
        target_fps: Output video frame rate (> 0)

    Raises:
        VideoProcessingError: If input validation fails or video cannot be processed
    """
    # Validate inputs
    video_path = validate_video_path(video_path, must_exist=True)
    output_path = validate_output_path(output_path)
    segments = validate_segments_dataframe(segments)
    window_duration = validate_positive_number(window_duration, 'window_duration')
    target_fps = validate_positive_number(target_fps, 'target_fps')

    if motion_threshold is not None:
        motion_threshold = validate_positive_number(motion_threshold, 'motion_threshold', allow_zero=True)
        if motion_threshold > 1.0:
            raise VideoProcessingError(f"motion_threshold must be <= 1.0, got {motion_threshold}")

    if gesture_threshold is not None:
        gesture_threshold = validate_positive_number(gesture_threshold, 'gesture_threshold', allow_zero=True)
        if gesture_threshold > 1.0:
            raise VideoProcessingError(f"gesture_threshold must be <= 1.0, got {gesture_threshold}")

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise VideoProcessingError(f"Cannot open video file: {video_path}")
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / input_fps

    # Create VideoWriter object at target FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    # Color mapping for labels
    color_map = {
        'NoGesture': (50, 50, 50),      # Dark gray
        'Gesture': (0, 204, 204),        # Vibrant teal
        'Move': (255, 94, 98)            # Soft coral red
    }

    # Fixed y-axis parameters for absolute scale
    y_min = 0.0
    y_max = 1.0

    # Determine graph dimensions
    graph_width = int(width * 0.3)
    graph_height = int(height * 0.2)
    graph_margin = 10

    # Check if we have predictions
    has_predictions = predictions_df is not None and not predictions_df.empty

    if has_predictions:
        # Ensure time column exists
        if 'time' not in predictions_df.columns:
            has_predictions = False
            print("Warning: predictions_df doesn't have a 'time' column")

    if has_predictions:
        # Get confidence data
        times = predictions_df['time'].values
        predictions_start_time = times.min() if len(times) > 0 else None
        gesture_conf = predictions_df['Gesture_confidence'].values if 'Gesture_confidence' in predictions_df.columns else None
        move_conf = predictions_df['Move_confidence'].values if 'Move_confidence' in predictions_df.columns else None
        motion_conf = predictions_df['has_motion'].values if 'has_motion' in predictions_df.columns else None

    # Prepare segment lookup
    def get_label_at_time(time: float) -> str:
        if segments.empty:
            return 'NoGesture'

        matching_segments = segments[
            (segments['start_time'] <= time) &
            (segments['end_time'] >= time)
        ]
        return matching_segments['label'].iloc[0] if len(matching_segments) > 0 else 'NoGesture'

    # Calculate total output frames at target FPS
    output_frames = int(video_duration * target_fps)

    progress_bar = tqdm(total=output_frames, desc="Labeling video", unit="frames")

    # Read frames sequentially (much faster than seeking)
    # Track which input frames map to output frames
    frame_ratio = input_fps / target_fps
    current_input_frame = 0
    last_frame = None

    for output_frame_idx in range(output_frames):
        output_time = output_frame_idx / target_fps
        target_input_frame = int(output_time * input_fps)

        # Read frames sequentially until we reach the target
        while current_input_frame <= target_input_frame:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame
            current_input_frame += 1

        if last_frame is None:
            break

        # Work with a copy for annotation (avoid modifying cached frame)
        frame = last_frame.copy() if frame_ratio > 1 else last_frame

        # Get the label at current time
        try:
            current_label = get_label_at_time(output_time)
        except Exception as e:
            print(f"Error getting label at time {output_time}: {str(e)}")
            current_label = 'NoGesture'

        # Add text label to frame
        cv2.putText(
            frame,
            current_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_map.get(current_label, (255, 255, 255)),
            2
        )

        # Add moving window confidence graph if predictions are available
        if has_predictions and predictions_start_time is not None and output_time >= predictions_start_time:
            _draw_confidence_graph(
                frame, predictions_df, output_time, times,
                gesture_conf, move_conf, motion_conf,
                motion_threshold, gesture_threshold,
                graph_width, graph_height, graph_margin,
                width, window_duration, y_min, y_max
            )

        out.write(frame)
        progress_bar.update(1)

    progress_bar.close()
    cap.release()
    out.release()

    print(f"Video labeled at {target_fps}fps saved to {output_path}")


def _draw_confidence_graph(
    frame: np.ndarray,
    predictions_df: pd.DataFrame,
    output_time: float,
    times: np.ndarray,
    gesture_conf: Optional[np.ndarray],
    move_conf: Optional[np.ndarray],
    motion_conf: Optional[np.ndarray],
    motion_threshold: Optional[float],
    gesture_threshold: Optional[float],
    graph_width: int,
    graph_height: int,
    graph_margin: int,
    frame_width: int,
    window_duration: float,
    y_min: float,
    y_max: float
) -> None:
    """Draw confidence graph overlay on video frame."""
    graph_pos_x = frame_width - graph_width - graph_margin
    graph_pos_y = graph_margin

    # Draw background with semi-transparency
    overlay = frame.copy()
    cv2.rectangle(overlay,
                 (graph_pos_x - 35, graph_pos_y - 5),
                 (graph_pos_x + graph_width + 5, graph_pos_y + graph_height + 25),
                 (0, 0, 0),
                 -1)
    frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Calculate window bounds
    min_time = min(times) if len(times) > 0 else 0
    max_time = max(times) if len(times) > 0 else output_time + window_duration

    # For beginning of video
    if output_time < min_time + (window_duration * 0.2):
        window_start = min_time
        window_end = min(max_time, min_time + window_duration)
    # For end of video
    elif output_time > max_time - (window_duration * 0.2):
        window_end = max_time
        window_start = max(min_time, max_time - window_duration)
    # For middle of video (standard sliding window)
    else:
        window_start = max(min_time, output_time - (window_duration * 0.8))
        window_end = min(max_time, window_start + window_duration)

    # Add a safeguard
    if window_end <= window_start:
        window_start = max(0, output_time - (window_duration * 0.5))
        window_end = window_start + window_duration

    # Add title with timestamp info
    cv2.putText(
        frame,
        f"Confidence: {window_start:.1f}s - {window_end:.1f}s",
        (graph_pos_x, graph_pos_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

    # Draw axes
    cv2.line(frame,
            (graph_pos_x, graph_pos_y + graph_height),
            (graph_pos_x + graph_width, graph_pos_y + graph_height),
            (255, 255, 255), 1)  # X-axis
    cv2.line(frame,
            (graph_pos_x, graph_pos_y),
            (graph_pos_x, graph_pos_y + graph_height),
            (255, 255, 255), 1)  # Y-axis

    # Add Y-axis ticks and grid lines
    tick_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    for tick in tick_positions:
        tick_y = graph_pos_y + graph_height - int(tick * graph_height)
        cv2.line(frame,
                (graph_pos_x - 3, tick_y),
                (graph_pos_x, tick_y),
                (180, 180, 180), 1)
        cv2.putText(frame, f"{tick:.1f}",
                  (graph_pos_x - 25, tick_y + 4),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.line(frame,
                (graph_pos_x, tick_y),
                (graph_pos_x + graph_width, tick_y),
                (50, 50, 50), 1, cv2.LINE_AA)

    # Draw threshold lines
    if motion_threshold is not None:
        motion_y = graph_pos_y + graph_height - int(motion_threshold * graph_height)
        for x in range(graph_pos_x, graph_pos_x + graph_width, 8):
            cv2.line(frame, (x, motion_y), (x+4, motion_y), (200, 200, 200), 1)
        cv2.putText(frame, f"M:{motion_threshold:.1f}",
                  (graph_pos_x + graph_width + 2, motion_y + 4),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    if gesture_threshold is not None:
        gesture_y = graph_pos_y + graph_height - int(gesture_threshold * graph_height)
        for x in range(graph_pos_x, graph_pos_x + graph_width, 8):
            cv2.line(frame, (x, gesture_y), (x+4, gesture_y), (128, 150, 150), 1)
        cv2.putText(frame, f"G:{gesture_threshold:.1f}",
                  (graph_pos_x + graph_width + 2, gesture_y + 4),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 150, 150), 1)

    # Find indices within the time window
    mask = (times >= window_start) & (times <= window_end)
    if np.any(mask):
        window_times = times[mask]

        # Plot confidence lines
        if gesture_conf is not None:
            _plot_confidence_line(frame, window_times, gesture_conf[mask],
                                 window_start, window_duration, graph_pos_x,
                                 graph_pos_y, graph_width, graph_height,
                                 y_min, y_max, (0, 204, 204))

        if move_conf is not None:
            _plot_confidence_line(frame, window_times, move_conf[mask],
                                 window_start, window_duration, graph_pos_x,
                                 graph_pos_y, graph_width, graph_height,
                                 y_min, y_max, (255, 94, 98))

        if motion_conf is not None:
            _plot_confidence_line(frame, window_times, motion_conf[mask],
                                 window_start, window_duration, graph_pos_x,
                                 graph_pos_y, graph_width, graph_height,
                                 y_min, y_max, (200, 200, 200))

    # Add current time indicator
    x_current = graph_pos_x + int(((output_time - window_start) / window_duration) * graph_width)
    if graph_pos_x <= x_current <= graph_pos_x + graph_width:
        cv2.line(frame,
                (x_current, graph_pos_y),
                (x_current, graph_pos_y + graph_height),
                (255, 255, 100), 2)

    # Add legend
    legend_y = graph_pos_y + graph_height + 15
    cv2.putText(frame, "G", (graph_pos_x + 5, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 204, 204), 1)
    cv2.putText(frame, "M", (graph_pos_x + 25, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 94, 98), 1)
    cv2.putText(frame, "Motion", (graph_pos_x + 45, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def _plot_confidence_line(
    frame: np.ndarray,
    window_times: np.ndarray,
    conf_values: np.ndarray,
    window_start: float,
    window_duration: float,
    graph_pos_x: int,
    graph_pos_y: int,
    graph_width: int,
    graph_height: int,
    y_min: float,
    y_max: float,
    color: tuple
) -> None:
    """Plot a single confidence line on the graph."""
    prev_point = None

    for t, conf in zip(window_times, conf_values):
        x = graph_pos_x + int(((t - window_start) / window_duration) * graph_width)
        conf_clamped = max(min(conf, y_max), y_min)
        y = graph_pos_y + graph_height - int((conf_clamped - y_min) / (y_max - y_min) * graph_height)

        if prev_point:
            cv2.line(frame, prev_point, (x, y), color, 1, cv2.LINE_AA)
        prev_point = (x, y)


def cut_video_by_segments(
    output_folder: str,
    segments_pattern: str = "*_segments.csv",
    labeled_video_prefix: str = "labeled_",
    output_subfolder: str = "gesture_segments"
) -> Dict[str, List[str]]:
    """
    Extract video segments and corresponding features from labeled videos.

    Args:
        output_folder: Path to folder containing segments.csv files and labeled videos
        segments_pattern: Pattern to match segment CSV files
        labeled_video_prefix: Prefix of labeled video files
        output_subfolder: Name of subfolder to store segmented videos

    Returns:
        Dictionary mapping original video names to lists of generated segment paths
    """
    # Create subfolder for segments if it doesn't exist
    segments_folder = os.path.join(output_folder, output_subfolder)
    os.makedirs(segments_folder, exist_ok=True)

    # Get all segment CSV files
    segment_files = glob.glob(os.path.join(output_folder, segments_pattern))
    results = {}

    for segment_file in segment_files:
        try:
            # Get original video name from segments file name
            base_name = os.path.basename(segment_file).replace('_segments.csv', '')
            labeled_video = os.path.join(output_folder, f"{labeled_video_prefix}{base_name}")
            features_path = os.path.join(output_folder, f"{base_name}_features.npy")

            # Check if labeled video and features exist
            if not os.path.exists(labeled_video):
                print(f"Warning: Labeled video not found for {base_name}")
                continue
            if not os.path.exists(features_path):
                print(f"Warning: Features file not found for {base_name}")
                continue

            # Read segments file
            segments_df = pd.read_csv(segment_file)

            if segments_df.empty:
                print(f"No segments found in {segment_file}")
                continue

            # Create subfolder for this video's segments
            video_segments_folder = os.path.join(segments_folder, base_name)
            os.makedirs(video_segments_folder, exist_ok=True)

            # Load video and get fps
            video = VideoFileClip(labeled_video)
            fps = video.fps

            # Load features
            features = np.load(features_path)

            segment_paths = []

            # Process each segment
            for idx, segment in segments_df.iterrows():
                start_time = segment['start_time']
                end_time = segment['end_time']
                label = segment['label']

                # Calculate frame indices
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)

                # Create segment filenames
                segment_filename = f"{base_name}_segment_{idx+1}_{label}_{start_time:.2f}_{end_time:.2f}.mp4"
                features_filename = f"{base_name}_segment_{idx+1}_{label}_{start_time:.2f}_{end_time:.2f}_features.npy"

                segment_path = os.path.join(video_segments_folder, segment_filename)
                segment_features_path = os.path.join(video_segments_folder, features_filename)

                # Extract and save video segment
                try:
                    # Cut video
                    segment_clip = video.subclipped(start_time, end_time)
                    segment_clip.write_videofile(
                        segment_path,
                        codec='libx264',
                        audio=False
                    )
                    segment_clip.close()

                    # Cut and save features
                    if start_frame < len(features) and end_frame <= len(features):
                        segment_features = features[start_frame:end_frame]
                        np.save(segment_features_path, segment_features)
                        print(f"Created segment and features: {segment_filename}")
                    else:
                        print(f"Warning: Frame indices {start_frame}:{end_frame} out of bounds "
                              f"for features array of length {len(features)}")

                    segment_paths.append(segment_path)

                except Exception as e:
                    print(f"Error creating segment {segment_filename}: {str(e)}")
                    continue

            # Clean up
            video.close()

            results[base_name] = segment_paths
            print(f"Completed processing segments for {base_name}")

        except Exception as e:
            print(f"Error processing {segment_file}: {str(e)}")
            continue

    return results


def find_all_videos(folder: str, pattern: str = "*.mp4") -> List[str]:
    """
    Recursively find all video files in a folder and its subfolders.

    Args:
        folder: Root folder to search
        pattern: File pattern to match (default: "*.mp4")

    Returns:
        List of full paths to video files

    Raises:
        VideoProcessingError: If folder path is invalid
    """
    if not isinstance(folder, str):
        raise VideoProcessingError(
            f"folder must be a string, got {type(folder).__name__}"
        )

    if not folder.strip():
        raise VideoProcessingError("folder path cannot be empty")

    if not os.path.exists(folder):
        raise VideoProcessingError(f"Folder not found: {folder}")

    if not os.path.isdir(folder):
        raise VideoProcessingError(f"Path is not a directory: {folder}")

    videos = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.mp4'):
                videos.append(os.path.join(root, file))
    return videos


def create_sliding_windows(
    features: List[List[float]],
    seq_length: int,
    stride: int = 1,
    input_fps: Optional[float] = None,
    target_fps: float = 25.0
) -> np.ndarray:
    """
    Create sliding windows from feature sequence.

    Args:
        features: List of feature vectors
        seq_length: Length of each window (must be > 0)
        stride: Step size between windows (must be > 0, default: 1)
        input_fps: Original video FPS (if provided, will adjust stride)
        target_fps: Target FPS for analysis

    Returns:
        NumPy array of windowed features with shape (num_windows, seq_length, num_features)

    Raises:
        VideoProcessingError: If input validation fails
    """
    # Validate inputs
    if not isinstance(seq_length, int) or seq_length <= 0:
        raise VideoProcessingError(
            f"seq_length must be a positive integer, got {seq_length}"
        )

    if not isinstance(stride, int) or stride <= 0:
        raise VideoProcessingError(
            f"stride must be a positive integer, got {stride}"
        )

    if features is None:
        raise VideoProcessingError("features cannot be None")

    # Handle empty features
    if len(features) == 0:
        return np.array([])

    if len(features) < seq_length:
        return np.array([])

    windows = []
    for i in range(0, len(features) - seq_length + 1, stride):
        window = features[i:i + seq_length]
        if len(window) == seq_length:
            windows.append(window)

    return np.array(windows)
