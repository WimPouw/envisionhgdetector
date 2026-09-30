"""Render predicted labels on video frames."""

from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..state import Labels, PredictionColumns, SegmentColumns, color_map

def label_video(
    video_path: str,
    segments: pd.DataFrame,
    output_path: str,
    predictions_df: Optional[pd.DataFrame] = None,
    valid_timestamps: Optional[List[float]] = None,
    motion_threshold: float = None,
    gesture_threshold: float = None,
    window_duration: float = 10.0,
    target_fps: Optional[float] = None
) -> None:
    """
    Label a video with predicted gestures based on segments.
    Uses the input frame rate when target_fps is None.
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / input_fps

    output_fps = input_fps if target_fps is None else target_fps
    
    # Create VideoWriter object at the selected FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
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
        # Ensure timestamp column exists
        if PredictionColumns.TIMESTAMP not in predictions_df.columns:
            has_predictions = False
            print(f"Warning: predictions_df doesn't have a '{PredictionColumns.TIMESTAMP}' column")
            
    if has_predictions:
        # Get confidence data
        times = predictions_df[PredictionColumns.TIMESTAMP].values
        predictions_start_time = times.min() if len(times) > 0 else None
        gesture_conf = predictions_df[PredictionColumns.GESTURE_CONFIDENCE].values
        move_conf = predictions_df[PredictionColumns.MOVE_CONFIDENCE].values
        motion_conf = predictions_df[PredictionColumns.MOTION_CONFIDENCE].values
        
    # Prepare segment lookup
    def get_label_at_time(time: float) -> str:
        if segments.empty:
            return Labels.NOGESTURE
            
        matching_segments = segments[
            (segments[SegmentColumns.START_TIME] <= time) & 
            (segments[SegmentColumns.END_TIME] >= time)
        ]
        # TODO shoudlnt this be prediction? why is it label
        if SegmentColumns.LABEL not in matching_segments.columns:
            return matching_segments[SegmentColumns.PREDICTION].iloc[0] if len(matching_segments) > 0 else Labels.NOGESTURE
        else:
            return matching_segments[SegmentColumns.LABEL].iloc[0] if len(matching_segments) > 0 else Labels.NOGESTURE
    
    # Calculate total output frames at the selected FPS
    output_frames = total_frames if target_fps is None else int(video_duration * output_fps)
    
    progress_bar = tqdm(total=output_frames, desc="Labeling video", unit="frames")

    # Process frames at the selected rate
    for output_frame_idx in range(output_frames):
        # Calculate which input frame to read
        output_time = output_frame_idx / output_fps
        input_frame_idx = int(output_time * input_fps)
        
        # Ensure we don't exceed video bounds
        if input_frame_idx >= total_frames:
            break
            
        # Seek to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, input_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break

        # Get the label at current time
        try:
            current_label = get_label_at_time(output_time)
        except Exception as e:
            print(f"Error getting label at time {output_time}: {str(e)}")
            current_label = Labels.NOGESTURE
        
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
            # Create a blank sub-image for the graph with black semi-transparent background
            graph_pos_x = width - graph_width - graph_margin
            graph_pos_y = graph_margin
            
            # Draw background with semi-transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (graph_pos_x - 35, graph_pos_y - 5), 
                         (graph_pos_x + graph_width + 5, graph_pos_y + graph_height + 25), 
                         (0, 0, 0), 
                         -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
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
                    window_gesture = gesture_conf[mask]
                    prev_point = None
                    
                    for i, (t, conf) in enumerate(zip(window_times, window_gesture)):
                        if conf is None or np.isnan(conf):
                            continue # TODO why is conf nan - lightgbm - could it be due to 5 frame window? - yes, it is due to the 5 frame window in lightgbm, which can produce NaN values at the edges of the data
                        x = graph_pos_x + int(((t - window_start) / window_duration) * graph_width)
                        conf_clamped = max(min(conf, y_max), y_min)
                        y = graph_pos_y + graph_height - int((conf_clamped - y_min) / (y_max - y_min) * graph_height)
                        
                        if prev_point:
                            cv2.line(frame, prev_point, (x, y), (0, 204, 204), 1, cv2.LINE_AA)
                        prev_point = (x, y)
                
                if move_conf is not None:
                    window_move = move_conf[mask]
                    prev_point = None
                    
                    for i, (t, conf) in enumerate(zip(window_times, window_move)):
                        if conf is None or np.isnan(conf):
                            continue # TODO why is conf nan - lightgbm
                        x = graph_pos_x + int(((t - window_start) / window_duration) * graph_width)
                        conf_clamped = max(min(conf, y_max), y_min)
                        y = graph_pos_y + graph_height - int((conf_clamped - y_min) / (y_max - y_min) * graph_height)
                        
                        if prev_point:
                            cv2.line(frame, prev_point, (x, y), (255, 94, 98), 1, cv2.LINE_AA)
                        prev_point = (x, y)
                
                if motion_conf is not None:
                    window_motion = motion_conf[mask]
                    prev_point = None
                    
                    for i, (t, conf) in enumerate(zip(window_times, window_motion)):
                        if conf is None or np.isnan(conf):
                            continue # TODO why is conf nan - lightgbm
                        x = graph_pos_x + int(((t - window_start) / window_duration) * graph_width)
                        conf_clamped = max(min(conf, y_max), y_min)
                        y = graph_pos_y + graph_height - int((conf_clamped - y_min) / (y_max - y_min) * graph_height)
                        
                        if prev_point:
                            cv2.line(frame, prev_point, (x, y), (200, 200, 200), 1, cv2.LINE_AA)
                        prev_point = (x, y)
            
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
        
        out.write(frame)
        progress_bar.update(1)

    progress_bar.close()
    cap.release()
    out.release()
    
    print(f"Video labeled at {output_fps}fps saved to {output_path}")

