"""Retrack gesture clips and save world landmarks."""
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from envisionhgdetector.mediapipe_processing import PoseProcessor, drawing_utils, pose
from envisionhgdetector.state import DIRS

def retrack_gesture_videos(
    input_folder: str,
    output_folder: str,
    video_pattern: str = ".mp4"
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Retrack gesture videos using MediaPipe world landmarks and save visualization.
    Now also tracks and saves visibility scores separately.
    
    Args:
        input_folder: Folder containing input videos
        output_folder: Folder to save tracked data
        video_pattern: Pattern to match video files
        
    Returns:
        Dictionary mapping video names to tuples of (landmarks, visibility scores)
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    tracked_folder = output_folder / DIRS.TRACKED_VIDEOS
    output_folder.mkdir(parents=True, exist_ok=True)
    tracked_folder.mkdir(parents=True, exist_ok=True)

    tracked_data = {}
    
    # Find all videos recursively
    video_paths = [
        str(path)
        for path in Path(input_folder).rglob("*")
        if path.is_file() and path.name.endswith(video_pattern)
    ]
    
    # Process each video
    for video_path in video_paths:
        video_name = Path(video_path).stem
        print(f"Processing {video_name}")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        out_path = tracked_folder / f"{video_name}_tracked.mp4"
        out = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
        
        # Store world landmarks, visibility, and frame indices
        world_landmarks = []
        visibility_scores = []
        frame_indices = []
        
        with PoseProcessor(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True
        ) as processor:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = processor.process_frame(frame)
                
                if results.pose_world_landmarks:
                    # Extract world landmarks
                    frame_landmarks = [coord for landmark in results.pose_world_landmarks.landmark 
                                    for coord in (landmark.x, landmark.y, landmark.z)]
                    
                    # Extract visibility scores separately
                    frame_visibility = [landmark.visibility for landmark in results.pose_world_landmarks.landmark]
                    
                    world_landmarks.append(frame_landmarks)
                    visibility_scores.append(frame_visibility)
                    frame_indices.append(frame_idx)
                    
                    # Draw pose on frame
                    annotated_frame = frame.copy()
                    drawing_utils.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        pose.POSE_CONNECTIONS
                    )
                else:
                    # For frames without landmarks, just write the original frame
                    annotated_frame = frame
                    
                out.write(annotated_frame)
                frame_idx += 1
                
            cap.release()
            out.release()
        
        if world_landmarks:
            # Convert landmarks to numpy array
            landmarks_array = np.array(world_landmarks)
            visibility_array = np.array(visibility_scores)
            frame_indices = np.array(frame_indices)
            
            # Reshape landmarks to (frames, num_keypoints, 3)
            num_landmarks = landmarks_array.shape[1] // 3
            landmarks_array = landmarks_array.reshape(-1, num_landmarks, 3)
            
            # Create full arrays with all frames
            full_landmarks = np.zeros((total_frames, num_landmarks, 3))
            full_visibility = np.zeros((total_frames, num_landmarks))
            
            # Fill detected frames
            full_landmarks[frame_indices] = landmarks_array
            full_visibility[frame_indices] = visibility_array
            
            # Fill missing frames with nearest neighbor
            missing_indices = np.setdiff1d(np.arange(total_frames), frame_indices)
            
            if len(missing_indices) > 0:
                print(f"Filling {len(missing_indices)} missing frames with nearest neighbor values")
                
                for missing_idx in missing_indices:
                    # Find nearest detected frame
                    nearest_idx = frame_indices[np.abs(frame_indices - missing_idx).argmin()]
                    full_landmarks[missing_idx] = full_landmarks[nearest_idx]
                    full_visibility[missing_idx] = full_visibility[nearest_idx]
            
            # Apply smoothing (Gaussian filter) to landmarks
            smoothed = np.zeros_like(full_landmarks)
            for i in range(full_landmarks.shape[1]):  # Iterate over keypoints
                smoothed[:, i] = gaussian_filter1d(
                    full_landmarks[:, i], 
                    sigma=1  # Adjust sigma as needed
                )
            
            # Save smoothed landmarks
            landmarks_save_path = output_folder / f"{video_name}_world_landmarks.npy"
            np.save(landmarks_save_path, smoothed)
            
            # Save visibility scores
            visibility_save_path = output_folder / f"{video_name}_visibility.npy"
            np.save(visibility_save_path, full_visibility)
            
            tracked_data[video_name] = (smoothed, full_visibility)
    
    return tracked_data
