"""
Dual-panel video labeling for Combined CNN + LightGBM model.
Shows both models' confidence timeseries and segmented labels side-by-side.
"""

import cv2
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm
from envisionhgdetector.state import Labels, ModelNames, PredictionColumns, SegmentColumns
from .helpers import get_label_at_time, get_confidence_window, validate_prediction_times


def draw_confidence_graph(
    frame: np.ndarray,
    times: np.ndarray,
    confidences: dict,  # {'line_name': (values, color), ...}
    current_time: float,
    threshold_lines: dict,  # {'name': (value, color), ...}
    window_duration: float,
    graph_x: int,
    graph_y: int,
    graph_width: int,
    graph_height: int,
    title: str = ""
) -> None:
    """
    Draw a confidence graph on the frame.
    
    Args:
        frame: Video frame to draw on
        times: Array of timestamps
        confidences: Dict of {name: (values_array, color_bgr)}
        current_time: Current playback time
        threshold_lines: Dict of {name: (threshold_value, color_bgr)}
        window_duration: Width of time window in seconds
        graph_x, graph_y: Top-left position of graph
        graph_width, graph_height: Dimensions of graph
        title: Title to display above graph
    """
    if len(times) < 2:
        return
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                  (graph_x - 5, graph_y - 20), 
                  (graph_x + graph_width + 5, graph_y + graph_height + 5), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw title
    if title:
        cv2.putText(frame, title, (graph_x, graph_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Calculate window bounds
    window_start, window_end = get_confidence_window(
        current_time, times.min(), times.max(), window_duration,
        history_fraction=0.5,
    )
    
    # Get data in window
    window_mask = (times >= window_start) & (times <= window_end)
    window_times = times[window_mask]
    
    if len(window_times) == 0:
        return
    
    # Draw threshold lines (dashed)
    for name, (thresh_val, color) in threshold_lines.items():
        y_pos = int(graph_y + graph_height - (thresh_val * graph_height))
        # Draw dashed line
        dash_length = 5
        for x in range(graph_x, graph_x + graph_width, dash_length * 2):
            x_end = min(x + dash_length, graph_x + graph_width)
            cv2.line(frame, (x, y_pos), (x_end, y_pos), color, 1)
    
    # Draw confidence lines
    for line_name, (values, color) in confidences.items():
        if values is None:
            continue
        window_values = values[window_mask]
        
        if len(window_values) < 2:
            continue
        
        # Convert to pixel coordinates
        previous_point = None
        for t, v in zip(window_times, window_values):
            if not (np.isfinite(t) and np.isfinite(v)):
                previous_point = None
                continue
            x = int(graph_x + ((t - window_start) / (window_end - window_start)) * graph_width)
            y = int(graph_y + graph_height - (v * graph_height))
            point = (x, y)
            if previous_point is not None:
                cv2.line(frame, previous_point, point, color, 1, cv2.LINE_AA)
            previous_point = point
    
    # Draw current time indicator (yellow vertical line)
    if window_start <= current_time <= window_end:
        x_current = int(graph_x + ((current_time - window_start) / (window_end - window_start)) * graph_width)
        cv2.line(frame, (x_current, graph_y), (x_current, graph_y + graph_height), 
                 (0, 255, 255), 2)
    
    # Draw border
    cv2.rectangle(frame, (graph_x, graph_y), 
                  (graph_x + graph_width, graph_y + graph_height), 
                  (100, 100, 100), 1)


def label_video_combined(
    video_path: str,
    video_output_folder: str,
    cnn_motion_threshold: float = 0.5,
    cnn_gesture_threshold: float = 0.5,
    lgbm_threshold: float = 0.5,
    window_duration: float = 10.0,
    target_fps: float = 25.0,
) -> None:
    """
    Create a labeled video with dual-panel display for CNN and LightGBM comparison.
    
    Shows:
    - Left side: CNN label, LightGBM label, agreement indicator
    - Top-right: CNN confidence graph (Gesture, Move, Motion lines)
    - Bottom-right: LightGBM confidence graph (Gesture line)
    - Both graphs show thresholds and current time indicator
    
    Loads predictions and per-model segments saved by combined detection.
    Thresholds only control graph lines; saved segments determine labels.
    
    Args:
        video_path: Path to input video
        video_output_folder: Folder containing the saved combined output CSVs.
            Writes <video_name>_comparison.mp4 to this folder.
        cnn_motion_threshold: Motion threshold line for the CNN graph
        cnn_gesture_threshold: Gesture threshold line for the CNN graph
        lgbm_threshold: Confidence threshold line for the LightGBM graph
        window_duration: Width of confidence graph window in seconds
        target_fps: Output video frame rate
    """
    video_path = Path(video_path)
    video_output_folder = Path(video_output_folder)
    video_name = video_path.stem
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be finite and positive.")
    if not np.isfinite(window_duration) or window_duration <= 0:
        raise ValueError("window_duration must be finite and positive.")
    for threshold in (cnn_motion_threshold, cnn_gesture_threshold, lgbm_threshold):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Graph thresholds must be finite and between 0 and 1.")
    predictions_df = pd.read_csv(video_output_folder / f"{video_name}_predictions.csv")
    cnn_segments = pd.read_csv(video_output_folder / f"{video_name}_{ModelNames.CNN_B}_segments.csv")
    lgbm_segments = pd.read_csv(video_output_folder / f"{video_name}_{ModelNames.LIGHTGBM}_segments.csv")
    required_segments = {SegmentColumns.START_TIME, SegmentColumns.END_TIME, SegmentColumns.PREDICTION}
    for name, segments in ((ModelNames.CNN_B, cnn_segments), (ModelNames.LIGHTGBM, lgbm_segments)):
        missing = required_segments - set(segments.columns)
        if missing:
            raise ValueError(f"{name} segments are missing columns: {sorted(missing)}")
        bounds = segments[[SegmentColumns.START_TIME, SegmentColumns.END_TIME]].to_numpy(dtype=float)
        if not np.isfinite(bounds).all() or (bounds < 0).any() or (bounds[:, 1] < bounds[:, 0]).any():
            raise ValueError(f"{name} segments have invalid time boundaries.")
        if not segments[SegmentColumns.PREDICTION].isin([Labels.GESTURE, Labels.MOVE]).all():
            raise ValueError(f"{name} segments contain invalid predictions.")
    if PredictionColumns.TIMESTAMP not in predictions_df.columns:
        raise ValueError("Predictions must contain timestamps.")
    times = validate_prediction_times(predictions_df[PredictionColumns.TIMESTAMP])
    output_path = str(video_output_folder / f"{video_name}_comparison.mp4")
    cap = out = progress_bar = None
    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(input_fps) or input_fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0:
            raise ValueError("Input video must have positive FPS, dimensions, and frame count.")
        video_duration = total_frames / input_fps

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Could not open video writer: {output_path}")

        # Colors (BGR)
        COLOR_GESTURE = (200, 200, 50)     # Teal/Cyan for CNN Gesture
        COLOR_MOVE = (100, 100, 255)       # Coral/Orange for CNN Move
        COLOR_MOTION = (150, 150, 150)     # Gray for motion line
        COLOR_NOGESTURE = (100, 100, 100)  # Dark gray
        COLOR_LGBM_GESTURE = (50, 200, 50) # Green for LightGBM
        COLOR_AGREE = (0, 255, 0)          # Green
        COLOR_DIFFER = (0, 165, 255)       # Orange

        # Check available columns
        cnn_motion_column = f"{ModelNames.CNN_B}_{PredictionColumns.MOTION_CONFIDENCE}"
        cnn_gesture_column = f"{ModelNames.CNN_B}_{PredictionColumns.GESTURE_CONFIDENCE}"
        cnn_move_column = f"{ModelNames.CNN_B}_{PredictionColumns.MOVE_CONFIDENCE}"
        lgbm_gesture_column = f"{ModelNames.LIGHTGBM}_{PredictionColumns.GESTURE_CONFIDENCE}"
        cnn_prediction_column = f"{ModelNames.CNN_B}_{PredictionColumns.PREDICTION}"
        lgbm_prediction_column = f"{ModelNames.LIGHTGBM}_{PredictionColumns.PREDICTION}"
        has_cnn = all(col in predictions_df.columns for col in [
            cnn_motion_column,
            cnn_gesture_column,
            cnn_move_column,
            cnn_prediction_column,
        ])
        has_lgbm = all(col in predictions_df.columns for col in [lgbm_gesture_column, lgbm_prediction_column])

        if not has_cnn or not has_lgbm:
            raise ValueError("Comparison predictions must contain both models' labels and confidence columns.")

        # Get time array
        for column in (cnn_prediction_column, lgbm_prediction_column):
            if not predictions_df[column].isin([Labels.GESTURE, Labels.MOVE, Labels.NOGESTURE]).all():
                raise ValueError(f"Invalid or missing predictions in {column}.")

        # Get confidence arrays for plotting
        gesture_conf = predictions_df[cnn_gesture_column].to_numpy(dtype=float)
        move_conf = predictions_df[cnn_move_column].to_numpy(dtype=float)
        motion_conf = predictions_df[cnn_motion_column].to_numpy(dtype=float)
        lgbm_conf = predictions_df[lgbm_gesture_column].to_numpy(dtype=float)
        for values in (gesture_conf, move_conf, motion_conf, lgbm_conf):
            if np.isinf(values).any():
                raise ValueError("Confidence values must not be infinite.")
        plot_cnn = any(np.isfinite(values).sum() >= 2 for values in (gesture_conf, move_conf, motion_conf))
        plot_lgbm = np.isfinite(lgbm_conf).sum() >= 2
        if not plot_cnn:
            warnings.warn("Skipping CNN confidence graph: fewer than two usable samples.", UserWarning, stacklevel=2)
        if not plot_lgbm:
            warnings.warn("Skipping LightGBM confidence graph: fewer than two usable samples.", UserWarning, stacklevel=2)

        # Graph dimensions
        graph_width = max(1, int(width * 0.28))
        graph_height = max(1, int(height * 0.15))
        graph_margin = 10
        graph_x = width - graph_width - graph_margin

        # Calculate output frames
        output_frames = int(video_duration * target_fps)

        print(f"Generating dual-panel labeled video...")
        progress_bar = tqdm(total=output_frames, desc="Labeling video", unit="frames")

        for output_frame_idx in range(output_frames):
            output_time = output_frame_idx / target_fps
            input_frame_idx = int(output_time * input_fps)

            if input_frame_idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, input_frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # Get labels from segments (post-processed)
            cnn_label = get_label_at_time(output_time, cnn_segments)
            lgbm_label = get_label_at_time(output_time, lgbm_segments)

            # Determine colors for labels
            if cnn_label == Labels.GESTURE:
                cnn_color = COLOR_GESTURE
            elif cnn_label == Labels.MOVE:
                cnn_color = COLOR_MOVE
            else:
                cnn_color = COLOR_NOGESTURE

            lgbm_color = COLOR_LGBM_GESTURE if lgbm_label == Labels.GESTURE else COLOR_NOGESTURE

            # Check agreement (both detecting gesture/move or both not)
            cnn_is_gesture = cnn_label in [Labels.GESTURE, Labels.MOVE]
            lgbm_is_gesture = lgbm_label == Labels.GESTURE
            agree = cnn_is_gesture == lgbm_is_gesture

            # Draw labels on left side
            y_offset = 30
            cv2.putText(frame, f"CNN: {cnn_label}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cnn_color, 2)

            y_offset += 30
            cv2.putText(frame, f"LGBM: {lgbm_label}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, lgbm_color, 2)

            y_offset += 30
            agree_text = "[AGREE]" if agree else "[DIFFER]"
            agree_color = COLOR_AGREE if agree else COLOR_DIFFER
            if not (has_cnn and has_lgbm):
                agree_text = "[N/A]"
                agree_color = COLOR_NOGESTURE
            cv2.putText(frame, agree_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, agree_color, 2)

            # Draw CNN confidence graph (top-right)
            if plot_cnn:
                cnn_graph_y = graph_margin

                cnn_confidences = {
                    'Gesture': (gesture_conf, COLOR_GESTURE),
                    'Move': (move_conf, COLOR_MOVE),
                    'Motion': (motion_conf, COLOR_MOTION),
                }
                cnn_thresholds = {
                    'motion': (cnn_motion_threshold, (0, 100, 255)),  # Orange-red dashed
                    'gesture': (cnn_gesture_threshold, (100, 100, 255)),  # Light red dashed
                }

                draw_confidence_graph(
                    frame, times, cnn_confidences, output_time,
                    cnn_thresholds, window_duration,
                    graph_x, cnn_graph_y, graph_width, graph_height,
                    title="CNN (G=teal, M=coral, Motion=gray)"
                )

            # Draw LightGBM confidence graph (below CNN)
            if plot_lgbm:
                lgbm_graph_y = graph_margin + graph_height + 35

                lgbm_confidences = {
                    'Gesture': (lgbm_conf, COLOR_LGBM_GESTURE),
                }
                lgbm_thresholds = {
                    'threshold': (lgbm_threshold, (0, 100, 255)),
                }

                draw_confidence_graph(
                    frame, times, lgbm_confidences, output_time,
                    lgbm_thresholds, window_duration,
                    graph_x, lgbm_graph_y, graph_width, graph_height,
                    title="LightGBM (Gesture=green)"
                )

            # Draw timestamp
            cv2.putText(frame, f"Time: {output_time:.2f}s", (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out.write(frame)
            progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()

    print(f"Saved dual-panel labeled video to: {output_path}")


# For testing
if __name__ == "__main__":
    print("label_video_combined module loaded")
    print("Usage:")
    print("  from envisionhgdetector.rendering import label_video_combined")
    print("  label_video_combined(video_path, video_output_folder, ...)")
