# envisionhgdetector/label_video_combined.py
"""
Dual-panel video labeling for Combined CNN + LightGBM model.
Shows both models' confidence timeseries and segmented labels side-by-side.
"""

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from envisionhgdetector.state import Labels, MoveMode, PredictionColumns, SegmentColumns
from envisionhgdetector.utils import create_segments_from_labels


def get_label_at_time(segments_df: pd.DataFrame, time: float) -> str:
    """Get the label at a specific time from segments."""
    if segments_df.empty:
        return Labels.NOGESTURE
    
    matching = segments_df[
        (segments_df[SegmentColumns.START_TIME] <= time) &
        (segments_df[SegmentColumns.END_TIME] >= time)
    ]
    
    if len(matching) > 0:
        return matching[SegmentColumns.PREDICTION].iloc[0]
    return Labels.NOGESTURE


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
    min_time = times.min() if len(times) > 0 else 0
    max_time = times.max() if len(times) > 0 else current_time + window_duration
    
    # Determine window position
    if current_time < min_time + (window_duration * 0.2):
        window_start = min_time
        window_end = min(max_time, min_time + window_duration)
    elif current_time > max_time - (window_duration * 0.2):
        window_start = max(min_time, max_time - window_duration)
        window_end = max_time
    else:
        window_start = current_time - (window_duration * 0.5)
        window_end = current_time + (window_duration * 0.5)
    
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
        points = []
        for t, v in zip(window_times, window_values):
            x = int(graph_x + ((t - window_start) / (window_end - window_start)) * graph_width)
            y = int(graph_y + graph_height - (v * graph_height))
            points.append((x, y))
        
        # Draw line
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i+1], color, 1, cv2.LINE_AA)
    
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
    predictions_df: pd.DataFrame,
    output_path: str,
    cnn_motion_threshold: float = 0.5,
    cnn_gesture_threshold: float = 0.5,
    lgbm_threshold: float = 0.5,
    min_gap_s: float = 0.1,
    min_length_s: float = 0.1,
    window_duration: float = 10.0,
    target_fps: float = 25.0,
    move_mode: MoveMode | str = MoveMode.SEPARATE,
) -> None:
    """
    Create a labeled video with dual-panel display for CNN and LightGBM comparison.
    
    Shows:
    - Left side: CNN label, LightGBM label, agreement indicator
    - Top-right: CNN confidence graph (Gesture, Move, Motion lines)
    - Bottom-right: LightGBM confidence graph (Gesture line)
    - Both graphs show thresholds and current time indicator
    
    Labels are computed from SEGMENTS (with min_gap_s and min_length_s applied),
    not raw frame-by-frame predictions.
    
    Args:
        video_path: Path to input video
        predictions_df: DataFrame with both CNN and LightGBM predictions
        output_path: Path for output video
        cnn_motion_threshold: Motion threshold for CNN
        cnn_gesture_threshold: Gesture threshold for CNN
        lgbm_threshold: Confidence threshold for LightGBM
        min_gap_s: Minimum gap between segments (for post-processing)
        min_length_s: Minimum segment length (for post-processing)
        window_duration: Width of confidence graph window in seconds
        target_fps: Output video frame rate
        move_mode: Whether Move is separate, treated as Gesture, or ignored.
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / input_fps
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    # Colors (BGR)
    COLOR_GESTURE = (200, 200, 50)     # Teal/Cyan for CNN Gesture
    COLOR_MOVE = (100, 100, 255)       # Coral/Orange for CNN Move
    COLOR_MOTION = (150, 150, 150)     # Gray for motion line
    COLOR_NOGESTURE = (100, 100, 100)  # Dark gray
    COLOR_LGBM_GESTURE = (50, 200, 50) # Green for LightGBM
    COLOR_AGREE = (0, 255, 0)          # Green
    COLOR_DIFFER = (0, 165, 255)       # Orange
    
    # Check available columns
    has_cnn = all(col in predictions_df.columns for col in [
        'cnn_motion_confidence',
        'cnn_gesture_confidence',
    ])
    has_lgbm = 'lgbm_gesture_confidence' in predictions_df.columns
    
    if not has_cnn and not has_lgbm:
        print("Warning: No CNN or LightGBM predictions found in DataFrame")
        cap.release()
        out.release()
        return
    
    # Get time array
    times = predictions_df[PredictionColumns.TIMESTAMP].to_numpy()
    
    # === Create CNN segments ===
    print(f"Creating CNN segments (motion_thresh={cnn_motion_threshold}, gesture_thresh={cnn_gesture_threshold})...")
    segment_columns = [
        SegmentColumns.START_TIME,
        SegmentColumns.END_TIME,
        SegmentColumns.PREDICTION,
        SegmentColumns.PREDICTION_ID,
        SegmentColumns.DURATION,
    ]
    cnn_segments = pd.DataFrame(columns=segment_columns)
    
    if has_cnn:
        # Compute CNN labels per frame
        cnn_labels = []
        for _, row in predictions_df.iterrows():
            has_motion = row.get('cnn_motion_confidence', 0)
            gesture_conf = row.get('cnn_gesture_confidence', 0)
            move_conf = row.get('cnn_move_confidence', 0)
            
            if has_motion < cnn_motion_threshold:
                cnn_labels.append(Labels.NOGESTURE)
            elif gesture_conf >= cnn_gesture_threshold:
                cnn_labels.append(Labels.GESTURE)
            elif move_conf > gesture_conf:
                cnn_labels.append(Labels.MOVE)
            else:
                # When gesture_conf < threshold but move_conf is not higher, 
                # classify based on which is higher
                cnn_labels.append(Labels.MOVE)
        
        cnn_segments = create_segments_from_labels(
            times, cnn_labels, min_gap_s, min_length_s, move_mode
        )
        print(f"  CNN segments: {len(cnn_segments)}")
    
    # === Create LightGBM segments ===
    print(f"Creating LightGBM segments (threshold={lgbm_threshold})...")
    lgbm_segments = pd.DataFrame(columns=segment_columns)
    
    if has_lgbm:
        # Compute LightGBM labels per frame
        lgbm_labels = []
        for _, row in predictions_df.iterrows():
            lgbm_prob = row.get('lgbm_gesture_confidence', 0)
            if lgbm_prob >= lgbm_threshold:
                lgbm_labels.append(Labels.GESTURE)
            else:
                lgbm_labels.append(Labels.NOGESTURE)
        
        lgbm_segments = create_segments_from_labels(
            times, lgbm_labels, min_gap_s, min_length_s, move_mode
        )
        print(f"  LightGBM segments: {len(lgbm_segments)}")
    
    # Get confidence arrays for plotting
    gesture_conf = predictions_df['cnn_gesture_confidence'].values if has_cnn else None
    move_conf = predictions_df['cnn_move_confidence'].values if has_cnn else None
    motion_conf = predictions_df['cnn_motion_confidence'].values if has_cnn else None
    lgbm_conf = predictions_df['lgbm_gesture_confidence'].values if has_lgbm else None
    
    # Graph dimensions
    graph_width = int(width * 0.28)
    graph_height = int(height * 0.15)
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
        cnn_label = get_label_at_time(cnn_segments, output_time) if has_cnn else "N/A"
        lgbm_label = get_label_at_time(lgbm_segments, output_time) if has_lgbm else "N/A"
        
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
        cv2.putText(frame, agree_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, agree_color, 2)
        
        # Draw CNN confidence graph (top-right)
        if has_cnn and len(times) > 0:
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
        if has_lgbm and len(times) > 0:
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
    
    progress_bar.close()
    cap.release()
    out.release()
    
    print(f"Saved dual-panel labeled video to: {output_path}")


# For testing
if __name__ == "__main__":
    print("label_video_combined module loaded")
    print("Usage:")
    print("  from envisionhgdetector.label_video_combined import label_video_combined")
    print("  label_video_combined(video_path, predictions_df, output_path, ...)")