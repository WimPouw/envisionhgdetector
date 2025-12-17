# envisionhgdetector/tracking.py
"""
Pose tracking utilities for gesture detection.
Handles MediaPipe pose estimation and landmark extraction.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .video import find_all_videos


def extract_upper_limb_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract and format upper limb features from world landmarks.

    Args:
        landmarks: Array of world landmarks in format [N, num_points, 3]
                  where 3 represents (x, y, z)

    Returns:
        Array of upper limb features containing coordinates for shoulders,
        elbows, wrists, and mean-centered fingers.

    Raises:
        ValueError: If landmarks shape is invalid
    """
    if landmarks.ndim != 3 or landmarks.shape[2] != 3:
        raise ValueError(
            f"Landmarks must be a 3D array with shape [N, num_points, 3], "
            f"got shape: {landmarks.shape}"
        )

    # Keypoint indices for upper body joints (MediaPipe pose landmarks)
    ordered_keypoints = [
        ('left_shoulder', 11),
        ('left_elbow', 13),
        ('left_wrist', 15),
        ('right_shoulder', 12),
        ('right_elbow', 14),
        ('right_wrist', 16)
    ]

    # Finger indices for mean centering
    left_finger_indices = [17, 19, 21]   # pinky, index, thumb
    right_finger_indices = [18, 20, 22]  # pinky, index, thumb

    all_features = []

    # Extract main keypoint features
    for key, index in ordered_keypoints:
        feature = landmarks[:, index]
        if not (np.any(np.isnan(feature)) or feature.size == 0):
            all_features.append(feature.reshape(-1, 3))

    # Process fingers with mean centering
    left_fingers = _process_hand_fingers(landmarks, left_finger_indices)
    right_fingers = _process_hand_fingers(landmarks, right_finger_indices)

    if left_fingers is not None:
        all_features.append(left_fingers)
    if right_fingers is not None:
        all_features.append(right_fingers)

    features = np.concatenate(all_features, axis=1)
    return features


def _process_hand_fingers(
    landmarks: np.ndarray,
    finger_indices: List[int]
) -> Optional[np.ndarray]:
    """
    Process finger landmarks for one hand with mean centering.

    Args:
        landmarks: Full landmarks array
        finger_indices: List of indices for finger landmarks

    Returns:
        Mean-centered finger features or None if no valid data
    """
    fingers = []
    for idx in finger_indices:
        feature = landmarks[:, idx]
        if not (np.any(np.isnan(feature)) or feature.size == 0):
            fingers.append(feature.reshape(-1, 3))

    if fingers:
        fingers = np.concatenate(fingers, axis=1)
        fingers_mean = np.mean(fingers, axis=0)
        return fingers - fingers_mean
    return None


def remove_nans(features: np.ndarray) -> np.ndarray:
    """
    Remove NaN values from the feature matrix by replacing them with zeros.

    Args:
        features: 2D numpy array (gesture features)

    Returns:
        Cleaned features with NaN values replaced by 0.0
    """
    return np.nan_to_num(features, nan=0.0)


def retrack_gesture_videos(
    input_folder: str,
    output_folder: str,
    video_pattern: str = "*.mp4"
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Retrack gesture videos using MediaPipe world landmarks and save visualization.

    This function processes videos to extract 3D pose landmarks using MediaPipe's
    world coordinate system, which provides metric-scale coordinates.

    Args:
        input_folder: Folder containing input videos
        output_folder: Folder to save tracked data and visualization videos
        video_pattern: Pattern to match video files

    Returns:
        Dictionary mapping video names to tuples of (landmarks, visibility_scores)
    """
    os.makedirs(output_folder, exist_ok=True)
    tracked_folder = os.path.join(output_folder, "tracked_videos")
    os.makedirs(tracked_folder, exist_ok=True)

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    tracked_data = {}

    # Find all videos recursively
    video_paths = find_all_videos(input_folder, video_pattern)

    # Process each video
    for video_path in video_paths:
        video_name = Path(video_path).stem
        print(f"Processing {video_name}")

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video writer
        out_path = os.path.join(tracked_folder, f"{video_name}_tracked.mp4")
        out = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )

        # Storage for landmarks and visibility
        world_landmarks = []
        visibility_scores = []
        frame_indices = []

        with mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True
        ) as pose:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_world_landmarks:
                    # Extract world landmarks
                    frame_landmarks = [
                        coord
                        for landmark in results.pose_world_landmarks.landmark
                        for coord in (landmark.x, landmark.y, landmark.z)
                    ]

                    # Extract visibility scores
                    frame_visibility = [
                        landmark.visibility
                        for landmark in results.pose_world_landmarks.landmark
                    ]

                    world_landmarks.append(frame_landmarks)
                    visibility_scores.append(frame_visibility)
                    frame_indices.append(frame_idx)

                    # Draw pose on frame
                    annotated_frame = frame.copy()
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                else:
                    annotated_frame = frame

                out.write(annotated_frame)
                frame_idx += 1

            cap.release()
            out.release()

        if world_landmarks:
            # Convert to numpy arrays
            landmarks_array = np.array(world_landmarks)
            visibility_array = np.array(visibility_scores)
            frame_indices = np.array(frame_indices)

            # Reshape landmarks to (frames, num_keypoints, 3)
            num_landmarks = landmarks_array.shape[1] // 3
            landmarks_array = landmarks_array.reshape(-1, num_landmarks, 3)

            # Create full arrays for all frames
            full_landmarks = np.zeros((total_frames, num_landmarks, 3))
            full_visibility = np.zeros((total_frames, num_landmarks))

            # Fill detected frames
            full_landmarks[frame_indices] = landmarks_array
            full_visibility[frame_indices] = visibility_array

            # Fill missing frames with nearest neighbor
            missing_indices = np.setdiff1d(np.arange(total_frames), frame_indices)

            if len(missing_indices) > 0:
                print(f"Filling {len(missing_indices)} missing frames with nearest neighbor")

                for missing_idx in missing_indices:
                    nearest_idx = frame_indices[np.abs(frame_indices - missing_idx).argmin()]
                    full_landmarks[missing_idx] = full_landmarks[nearest_idx]
                    full_visibility[missing_idx] = full_visibility[nearest_idx]

            # Apply Gaussian smoothing to landmarks
            smoothed = np.zeros_like(full_landmarks)
            for i in range(full_landmarks.shape[1]):
                smoothed[:, i] = gaussian_filter1d(full_landmarks[:, i], sigma=1)

            # Save smoothed landmarks
            landmarks_save_path = os.path.join(
                output_folder, f"{video_name}_world_landmarks.npy"
            )
            np.save(landmarks_save_path, smoothed)

            # Save visibility scores
            visibility_save_path = os.path.join(
                output_folder, f"{video_name}_visibility.npy"
            )
            np.save(visibility_save_path, full_visibility)

            tracked_data[video_name] = (smoothed, full_visibility)

    return tracked_data
