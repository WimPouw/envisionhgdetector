"""Render predicted labels on video frames."""
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from envisionhgdetector.state import Labels, PredictionColumns, SegmentColumns, Thresholds, color_map

def label_video_batch(
    videos_input_folder: str,
    videos_output_folder: str,
    plot_confidence: Optional[bool] = True, # predictions_df is required
    target_fps: Optional[float] = None,
    motion_threshold: Optional[float] = None,
    gesture_threshold: Optional[float] = None,
    window_duration: Optional[float] = None,
    video_extension: Optional[str] = ".mp4"
):
    videos_input_folder = Path(videos_input_folder)
    videos_output_folder = Path(videos_output_folder)

    if not videos_input_folder.exists():
        print(f"Input folder not found: {videos_input_folder}")
        return
    if not videos_output_folder.exists():
        print(f"Output folder not found: {videos_output_folder}")
        return

    for video_file in videos_input_folder.glob(f"*{video_extension}"):
        video_name = video_file.stem
        print(f"Labeling video: {video_name}")

        video_output_folder = videos_output_folder / video_name
        if not video_output_folder.exists():
            print(f"Output folder for video not found: {video_output_folder}. Skipping video.")
            continue

        label_video(
            str(video_file),
            str(video_output_folder),
            plot_confidence=plot_confidence,
            target_fps=target_fps,
            motion_threshold=motion_threshold,
            gesture_threshold=gesture_threshold,
            window_duration=window_duration
        )

def label_video(
    video_path: str,
    video_output_folder: str,
    plot_confidence: Optional[bool] = True,
    target_fps: Optional[float] = None,
    motion_threshold: Optional[float] = None,
    gesture_threshold: Optional[float] = None,
    window_duration: Optional[float] = None
):
    video_path = Path(video_path)
    video_output_folder = Path(video_output_folder)
    video_name = video_path.stem

    segments_save_path = video_output_folder / f"{video_name}_segments.csv"
    predictions_save_path = video_output_folder / f"{video_name}_predictions.csv"
    labeled_video_path = video_output_folder / f"{video_name}_labeled.mp4"

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return
    if not video_output_folder.exists():
        print(f"Output folder not found: {video_output_folder}")
        return
    if not segments_save_path.exists():
        print(f"Segments file not found: {segments_save_path}")
        return

    predictions_df = None
    if plot_confidence:
        if not predictions_save_path.exists():
            print(f"Predictions file not found: {predictions_save_path}. Skipping confidence plots.")
        else:
            predictions_df = pd.read_csv(predictions_save_path)

    default_thresholds = Thresholds()
    if motion_threshold is None:
        motion_threshold = default_thresholds.motion_threshold
    if gesture_threshold is None:
        gesture_threshold = default_thresholds.gesture_threshold
    if window_duration is None:
        window_duration = 10.0  # Default window duration in seconds

    segments = pd.read_csv(segments_save_path)
    try:
        _render_labeled_video(
            video_path=str(video_path),
            output_path=str(labeled_video_path),
            segments=segments,
            predictions_df=predictions_df,
            motion_threshold=motion_threshold,
            gesture_threshold=gesture_threshold,
            window_duration=window_duration,
            target_fps=target_fps
        )
    except Exception as e:
        print(f"Error labeling video {video_name}: {str(e)}")
        if labeled_video_path.exists():
            labeled_video_path.unlink()  # Remove partially created video
    
    return

def _render_labeled_video(
    video_path: str,
    output_path: str,
    segments: pd.DataFrame,
    predictions_df: pd.DataFrame,
    motion_threshold: float,
    gesture_threshold: float,
    window_duration: float,
    target_fps: float,
) -> None:
    """
    Label a video with predicted gestures based on segments.
    Uses the input frame rate when target_fps is None.

    Segments has short detections removed, and nearby intervals merged. This is what is saved in elan format as well.
    """
    required_segments = {SegmentColumns.START_TIME, SegmentColumns.END_TIME, SegmentColumns.PREDICTION}
    missing = required_segments - set(segments.columns)
    if missing:
        raise ValueError(f"Segments are missing columns: {sorted(missing)}")
    segment_times = segments[[SegmentColumns.START_TIME, SegmentColumns.END_TIME]].to_numpy(dtype=float)
    if not np.isfinite(segment_times).all() or (segment_times < 0).any():
        raise ValueError("Segment times must be finite and nonnegative.")
    if (segment_times[:, 1] < segment_times[:, 0]).any():
        raise ValueError("Segment end times must not precede start times.")
    if segments[SegmentColumns.PREDICTION].isna().any():
        raise ValueError("Segment predictions must not be missing.")
    if not np.isfinite(window_duration) or window_duration <= 0:
        raise ValueError("window_duration must be finite and positive.")
    if target_fps is not None and (not np.isfinite(target_fps) or target_fps <= 0):
        raise ValueError("target_fps must be finite and positive.")
    for name, threshold in (("motion_threshold", motion_threshold), ("gesture_threshold", gesture_threshold)):
        if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError(f"{name} must be finite and between 0 and 1.")

    has_predictions = predictions_df is not None and not predictions_df.empty
    if has_predictions:
        confidence_columns = [PredictionColumns.GESTURE_CONFIDENCE, PredictionColumns.MOVE_CONFIDENCE, PredictionColumns.MOTION_CONFIDENCE]
        missing = {PredictionColumns.TIMESTAMP, *confidence_columns} - set(predictions_df.columns)
        if missing:
            raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
        times = predictions_df[PredictionColumns.TIMESTAMP].to_numpy(dtype=float)
        if not np.isfinite(times).all() or (times < 0).any() or (np.diff(times) < 0).any():
            raise ValueError("Prediction timestamps must be finite, nonnegative, and sorted.")
        confidence = predictions_df[confidence_columns].to_numpy(dtype=float)
        # NaN confidence values are allowed and skipped when drawing curves.
        if np.isinf(confidence).any():
            raise ValueError("Confidence values must not be infinite.")
        gesture_conf, move_conf, motion_conf = confidence.T
        min_time, max_time = times.min(), times.max()

    cap = out = progress_bar = None
    # Open video
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(input_fps) or input_fps <= 0:
            raise ValueError("Input video FPS must be finite and positive.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or total_frames <= 0:
            raise ValueError("Input video must have positive dimensions and frame count.")
        video_duration = total_frames / input_fps

        output_fps = input_fps if target_fps is None else target_fps
        output_frames = total_frames if target_fps is None else int(video_duration * output_fps)
        
        # Create VideoWriter object at the selected FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Could not open video writer: {output_path}")
        graph_width = max(1, int(width * 0.3))
        graph_height = max(1, int(height * 0.2))
        graph_layout = (width - graph_width - 10, 10, graph_width, graph_height)
            
        progress_bar = tqdm(total=output_frames, desc="Labeling video", unit="frames")
        for output_frame_idx in range(output_frames):
            # Calculate which input frame to read
            output_time = output_frame_idx / output_fps
            input_frame_idx = int(output_time * input_fps)
            
            # Ensure we don't exceed video bounds
            if input_frame_idx >= total_frames:
                break
                
            # Seek to the correct frame
            if target_fps is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, input_frame_idx)
            # otherwise read sequentially, which is faster and more efficient
            
            ret, frame = cap.read()
            if not ret:
                break

            # Add text label to frame
            current_label = _get_label_at_time(output_time, segments)
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
            if has_predictions:
                frame = _draw_confidence_graph(
                    frame, graph_layout, min_time, max_time,
                    times, output_time, window_duration,
                    motion_conf, gesture_conf, move_conf,
                    motion_threshold, gesture_threshold
                )

            out.write(frame)
            progress_bar.update(1)
    except Exception as e:
        print(f"Error during video labeling: {str(e)}")
        raise
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
    
    print(f"Video labeled at {output_fps}fps saved to {output_path}")

def _get_label_at_time(time: float, segments) -> str:
    if segments.empty:
        return Labels.NOGESTURE
        
    matching_segments = segments[
        (segments[SegmentColumns.START_TIME] <= time) & 
        (segments[SegmentColumns.END_TIME] >= time)
    ]
    try:
        if not matching_segments.empty:
            return matching_segments[SegmentColumns.PREDICTION].iloc[0]
        else:
            return Labels.NOGESTURE
    except Exception as e:
        print(f"Error determining label at time {time}: {str(e)}")
        return Labels.NOGESTURE
    
def _get_confidence_window(output_time, min_time, max_time, window_duration):
    """Return the moving graph window, keeping the original edge behavior."""
    if output_time < min_time + window_duration * 0.2:
        window_start = min_time
        window_end = min(max_time, min_time + window_duration)
    elif output_time > max_time - window_duration * 0.2:
        window_end = max_time
        window_start = max(min_time, max_time - window_duration)
    else:
        window_start = max(min_time, output_time - window_duration * 0.8)
        window_end = min(max_time, window_start + window_duration)
    if window_end <= window_start:
        window_start = max(0, output_time - window_duration * 0.5)
        window_end = window_start + window_duration
    return window_start, window_end


def _draw_confidence_graph(frame, graph_layout, min_time, max_time, times, output_time, window_duration, motion_conf, gesture_conf, move_conf, motion_threshold, gesture_threshold):
    # Fixed y-axis parameters for absolute scale
    y_min = 0.0
    y_max = 1.0
    graph_pos_x, graph_pos_y, graph_width, graph_height = graph_layout
    
    # Draw background with semi-transparency
    overlay = frame.copy()
    cv2.rectangle(overlay, 
    (graph_pos_x - 35, graph_pos_y - 5), 
    (graph_pos_x + graph_width + 5, graph_pos_y + graph_height + 25), 
    (0, 0, 0), 
    -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    window_start, window_end = _get_confidence_window(
        output_time, min_time, max_time, window_duration
    )
    
    # Add title with timestamp info
    cv2.putText(
        frame,
        f"Confidence: {window_start:.1f}s - {window_end:.1f}s",
        (graph_pos_x + 10, graph_pos_y + 10),
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
        frame = _draw_threshold_line(frame, graph_pos_x, graph_pos_y, graph_width, graph_height, motion_threshold, color=(200, 200, 200), label="M")
    
    if gesture_threshold is not None:
        frame = _draw_threshold_line(frame, graph_pos_x, graph_pos_y, graph_width, graph_height, gesture_threshold, color=(128, 150, 150), label="G")
    
    # Find indices within the time window
    mask = (times >= window_start) & (times <= window_end)
    if np.any(mask):
        window_times = times[mask]
        # Plot confidence lines
        if gesture_conf is not None:
            frame = _draw_confidence_line(frame, gesture_conf, window_times, window_start, window_duration, graph_pos_x, graph_pos_y, graph_width, graph_height, y_min, y_max, mask, color=(0, 204, 204))
        
        if move_conf is not None:
            frame = _draw_confidence_line(frame, move_conf, window_times, window_start, window_duration, graph_pos_x, graph_pos_y, graph_width, graph_height, y_min, y_max, mask, color=(255, 94, 98))
        
        if motion_conf is not None:
            frame = _draw_confidence_line(frame, motion_conf, window_times, window_start, window_duration, graph_pos_x, graph_pos_y, graph_width, graph_height, y_min, y_max, mask, color=(200, 200, 200))
    
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

    return frame

def _draw_threshold_line(frame, graph_pos_x, graph_pos_y, graph_width, graph_height, threshold, color, label):
    
    y = graph_pos_y + graph_height - int(threshold * graph_height)
    for x in range(graph_pos_x, graph_pos_x + graph_width, 8):
        cv2.line(frame, (x, y), (x+4, y), color, 1)
    cv2.putText(frame, f"{label}:{threshold:.1f}", 
                (graph_pos_x + graph_width + 4, y + 4), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return frame
    
def _draw_confidence_line(frame, confidence, window_times, window_start, window_duration, graph_pos_x, graph_pos_y, graph_width, graph_height, y_min, y_max, mask, color):
    window_motion = confidence[mask]
    prev_point = None
    
    for i, (t, conf) in enumerate(zip(window_times, window_motion)):
        if conf is None or np.isnan(conf):
            continue # TODO why is conf nan - lightgbm
        x = graph_pos_x + int(((t - window_start) / window_duration) * graph_width)
        conf_clamped = max(min(conf, y_max), y_min)
        y = graph_pos_y + graph_height - int((conf_clamped - y_min) / (y_max - y_min) * graph_height)
        
        if prev_point:
            cv2.line(frame, prev_point, (x, y), color, 1, cv2.LINE_AA)
        prev_point = (x, y)
    return frame
