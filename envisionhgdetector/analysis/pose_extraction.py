"""Extract pose from gesture clips and save world and image landmarks."""
import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

from envisionhgdetector.utils import return_path
from envisionhgdetector.rendering.helpers import fix_codex
from envisionhgdetector.mediapipe_processing import PoseProcessor, drawing_utils, pose


def extract_pose_from_videos_batch(videos_folder: str | None, data_folder: str, video_pattern: str):
    '''
    Extracts pose from all videos in the videos_folder (matching the video_pattern) and saves the results in the data_folder.
    Note: video has detected landmarks while landmarks file has smoothed landmarks and visibility scores.
    Args:
        videos_folder: Path to the input videos folder. If None then labeled videos in data_folder is used.
        data_folder: Path where tracked data will be saved.
        video_pattern: Pattern to match video files (ex: .mp4).
    '''
    data_folder = Path(data_folder)
    if not data_folder.exists():
        raise ValueError(f"data folder does not exist: {data_folder}")
    
    remove_suffix = None
    if videos_folder is None:
        # use labeled videos in data_folder
        remove_suffix = "_labeled"
        video_paths = [
            path
            for path in data_folder.rglob(f"*{remove_suffix}.mp4")
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

    # Process each video
    for video_path in video_paths:
        try:
            video_name = video_path.stem
            if remove_suffix and video_name.endswith(remove_suffix):
                video_name = video_name[:-len(remove_suffix)]
            print(f"Processing {video_name}")
            extract_pose_from_video(
                video_name=video_name,
                video_path=str(video_path),
                data_folder=str(data_folder)
            )
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            continue

def extract_pose_from_video(video_name: str, video_path: str, data_folder: str):
    """
    Extracts pose from a single video and saves the results in the data_folder.
    Note: video has detected landmarks while landmarks file has smoothed landmarks and visibility scores.
    
    Args:
        video_name: Name of the video
        video_path: Path to the input video
        data_folder: Folder to save tracked data
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    out_path = return_path("retracked_video", data_folder, video_name)
    if out_path is None:
        raise ValueError(f"Could not determine output path for video: {video_name}")
    
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    if not out.isOpened():
        raise ValueError(f"Could not open video writer for output file: {out_path}")
    
    # Store world landmarks, visibility, and frame indices
    image_landmarks = []
    image_visibility_scores = []
    image_frame_indices = []
    world_landmarks = []
    world_visibility_scores = []
    world_frame_indices = []
    
    try:
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
                    frame_landmarks = [
                        coord
                        for landmark in results.pose_world_landmarks.landmark 
                        for coord in (landmark.x, landmark.y, landmark.z)
                    ]
                    
                    # Extract visibility scores separately
                    frame_visibility = [landmark.visibility for landmark in results.pose_world_landmarks.landmark]
                    
                    world_landmarks.append(frame_landmarks)
                    world_visibility_scores.append(frame_visibility)
                    world_frame_indices.append(frame_idx)

                if results.pose_landmarks:
                    # Extract image landmarks
                    frame_landmarks = [
                        coord 
                        for landmark in results.pose_landmarks.landmark
                        for coord in (landmark.x, landmark.y)
                    ]
                    # Extract image visibility scores
                    frame_visibility = [landmark.visibility for landmark in results.pose_landmarks.landmark]
                    image_landmarks.append(frame_landmarks)
                    image_visibility_scores.append(frame_visibility)
                    image_frame_indices.append(frame_idx)

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
    except Exception as e:
        print(f"Error during pose extraction for video {video_name}: {e}")
    finally:
        if cap.isOpened():
            cap.release()
        if out.isOpened():
            out.release()
        try:
            out_path = Path(out_path)
            fix_codex(out_path, out_path)  # Fix potential codec issues
        except Exception as e:
            print(f"Warning: Could not fix codec for {out_path}: {e}")

    if world_landmarks:
        smoothed_world, world_visibility = _smooth_results(world_landmarks, world_visibility_scores, world_frame_indices, total_frames, n_coords=3)
        landmarks_save_path = return_path("world_landmarks", data_folder, video_name)
        if landmarks_save_path is None:
            raise ValueError(f"Could not determine output path for world landmarks: {video_name}")
        np.save(landmarks_save_path, smoothed_world)
        visibility_save_path = return_path("world_visibility_scores", data_folder, video_name)
        if visibility_save_path is None:
            raise ValueError(f"Could not determine output path for visibility scores: {video_name}")
        np.save(visibility_save_path, world_visibility)
    if image_landmarks:
        smoothed_image, image_visibility = _smooth_results(image_landmarks, image_visibility_scores, image_frame_indices, total_frames, n_coords=2)
        image_landmarks_save_path = return_path("image_landmarks", data_folder, video_name)
        if image_landmarks_save_path is None:
            raise ValueError(f"Could not determine output path for image landmarks: {video_name}")
        np.save(image_landmarks_save_path, smoothed_image)
        
        visibility_save_path = return_path("image_visibility_scores", data_folder, video_name)
        if visibility_save_path is None:
            raise ValueError(f"Could not determine output path for visibility scores: {video_name}")
        np.save(visibility_save_path, image_visibility)
    return

def _smooth_results(landmarks, visibility_scores, frame_indices, total_frames, n_coords):
    # Convert landmarks to numpy array
    landmarks_array = np.array(landmarks)
    visibility_array = np.array(visibility_scores)
    frame_indices = np.array(frame_indices)
    
    # Reshape landmarks to (frames, num_keypoints, n_coords)
    num_landmarks = landmarks_array.shape[1] // n_coords
    landmarks_array = landmarks_array.reshape(-1, num_landmarks, n_coords)
    
    # Create full arrays with all frames
    full_landmarks = np.zeros((total_frames, num_landmarks, n_coords))
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
    smoothed = gaussian_filter1d(full_landmarks, sigma=1, axis=0)

    return smoothed, full_visibility
