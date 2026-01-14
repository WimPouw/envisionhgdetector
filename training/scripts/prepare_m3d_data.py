"""
Data preparation script for M3D TED dataset.

This script converts M3D TED talk videos with ELAN annotations into
the NPZ training format required by the LightGBM model.

Usage:
    python prepare_m3d_data.py --data_dir ../data/m3d --output ../data/m3d_landmarks.npz

The script:
1. Parses ELAN annotation files (.eaf) to get gesture timing
2. Extracts MediaPipe pose landmarks from videos
3. Labels each frame as "Gesture" or "NoGesture" based on annotations
4. Saves landmarks grouped by label to NPZ format
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# Constants
NUM_LANDMARKS = 23
LANDMARKS_PER_FRAME = NUM_LANDMARKS * 3  # x, y, z per landmark

# Add parent package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import cv2
    import mediapipe as mp
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Install with: pip install opencv-python mediapipe")
    sys.exit(1)


def parse_elan_file(eaf_path: str) -> List[Tuple[float, float, str]]:
    """
    Parse ELAN annotation file to extract gesture segments.

    Args:
        eaf_path: Path to .eaf file

    Returns:
        List of (start_time_ms, end_time_ms, label) tuples
    """
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    # Build time slot lookup: ts_id -> time_value (ms)
    time_slots = {}
    for ts in root.findall('.//TIME_SLOT'):
        ts_id = ts.get('TIME_SLOT_ID')
        ts_value = ts.get('TIME_VALUE')
        if ts_id and ts_value:
            time_slots[ts_id] = int(ts_value)

    segments = []

    # Look for gesture annotation tiers
    # M3D uses "Manual_GUnit" for gesture units
    gesture_tiers = ['Manual_GUnit', 'GUnit', 'Gesture', 'gesture', 'Manual_Gesture']

    for tier in root.findall('.//TIER'):
        tier_id = tier.get('TIER_ID', '')

        # Check if this is a gesture tier
        is_gesture_tier = any(gt.lower() in tier_id.lower() for gt in gesture_tiers)

        if is_gesture_tier:
            for annotation in tier.findall('.//ALIGNABLE_ANNOTATION'):
                ts1 = annotation.get('TIME_SLOT_REF1')
                ts2 = annotation.get('TIME_SLOT_REF2')

                # Note: We use 'Gesture' as the label regardless of annotation text
                # since we only need binary classification (Gesture vs NoGesture)
                if ts1 in time_slots and ts2 in time_slots:
                    start_ms = time_slots[ts1]
                    end_ms = time_slots[ts2]
                    segments.append((start_ms, end_ms, 'Gesture'))

    # Sort by start time
    segments.sort(key=lambda x: x[0])

    return segments


def extract_landmarks_from_video(
    video_path: str,
    progress_bar: bool = True
) -> Tuple[np.ndarray, float, int]:
    """
    Extract MediaPipe pose landmarks from video.

    Args:
        video_path: Path to video file
        progress_bar: Show progress bar

    Returns:
        Tuple of (landmarks array, fps, total_frames)
        landmarks shape: (num_frames, LANDMARKS_PER_FRAME) where LANDMARKS_PER_FRAME = 23 landmarks * 3 coords
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # NUM_LANDMARKS pose landmarks (indices 0-22 from MediaPipe Pose)
    # Each has (x, y, z) -> LANDMARKS_PER_FRAME values per frame
    landmarks_list = []

    iterator = range(total_frames)
    if progress_bar:
        iterator = tqdm(iterator, desc=f"Processing {Path(video_path).stem}")

    try:
        for _ in iterator:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = pose.process(rgb_frame)

            if results.pose_world_landmarks:
                # Extract world landmarks (real-world 3D coordinates in meters)
                # This matches the inference code in model_lightgbm.py
                frame_landmarks = []
                for i in range(NUM_LANDMARKS):
                    lm = results.pose_world_landmarks.landmark[i]
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_list.append(frame_landmarks)
            else:
                # No pose detected - use zeros
                landmarks_list.append([0.0] * LANDMARKS_PER_FRAME)
    finally:
        cap.release()
        pose.close()

    return np.array(landmarks_list, dtype=np.float32), fps, total_frames


def label_frames(
    total_frames: int,
    fps: float,
    gesture_segments: List[Tuple[float, float, str]]
) -> List[str]:
    """
    Assign labels to each frame based on gesture segments.

    Args:
        total_frames: Total number of frames in video
        fps: Frames per second
        gesture_segments: List of (start_ms, end_ms, label) tuples

    Returns:
        List of labels for each frame
    """
    labels = ['NoGesture'] * total_frames

    for start_ms, end_ms, label in gesture_segments:
        # Convert ms to frame indices
        start_frame = int((start_ms / 1000.0) * fps)
        end_frame = int((end_ms / 1000.0) * fps)

        # Clamp to valid range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames))

        for i in range(start_frame, end_frame):
            labels[i] = label

    return labels


def find_video_for_eaf(eaf_path: str, data_dir: str) -> Optional[str]:
    """
    Find the corresponding video file for an ELAN annotation file.

    Looks in Media subdirectory and common locations.
    """
    eaf_stem = Path(eaf_path).stem
    eaf_dir = Path(eaf_path).parent

    # Common video extensions
    extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    # Search locations
    search_dirs = [
        eaf_dir / 'Media',           # M3D structure: Media subfolder
        eaf_dir,                      # Same directory
        Path(data_dir),              # Data directory root
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in extensions:
            video_path = search_dir / f"{eaf_stem}{ext}"
            if video_path.exists():
                return str(video_path)

    return None


def process_dataset(
    data_dir: str,
    output_path: str,
    max_videos: Optional[int] = None,
    min_gesture_ratio: float = 0.05
) -> Dict[str, np.ndarray]:
    """
    Process all videos in the dataset.

    Args:
        data_dir: Root directory containing .eaf files and videos
        output_path: Output NPZ file path
        max_videos: Maximum number of videos to process (for testing)
        min_gesture_ratio: Minimum ratio of gesture frames (skip videos below this)

    Returns:
        Dictionary of label -> landmarks arrays
    """
    # Find all ELAN files
    eaf_files = list(Path(data_dir).rglob('*.eaf'))

    if not eaf_files:
        print(f"No .eaf files found in {data_dir}")
        return {}

    print(f"Found {len(eaf_files)} ELAN annotation files")

    # Collect landmarks by label
    landmarks_by_label = defaultdict(list)

    processed = 0
    skipped = 0

    for eaf_path in eaf_files:
        if max_videos and processed >= max_videos:
            break

        print(f"\n{'='*60}")
        print(f"Processing: {eaf_path.name}")

        # Find corresponding video
        video_path = find_video_for_eaf(str(eaf_path), data_dir)

        if not video_path:
            print(f"  Warning: No video found for {eaf_path.name}, skipping")
            skipped += 1
            continue

        print(f"  Video: {Path(video_path).name}")

        # Parse annotations
        try:
            segments = parse_elan_file(str(eaf_path))
            print(f"  Found {len(segments)} gesture segments")
        except Exception as e:
            print(f"  Error parsing ELAN file: {e}")
            skipped += 1
            continue

        if not segments:
            print(f"  Warning: No gesture annotations found, skipping")
            skipped += 1
            continue

        # Extract landmarks
        try:
            landmarks, fps, total_frames = extract_landmarks_from_video(video_path)
            print(f"  Extracted {len(landmarks)} frames at {fps:.1f} fps")
        except Exception as e:
            print(f"  Error extracting landmarks: {e}")
            skipped += 1
            continue

        # Label frames
        labels = label_frames(len(landmarks), fps, segments)

        # Count labels
        label_counts = Counter(labels)
        gesture_count = label_counts['Gesture']
        no_gesture_count = label_counts['NoGesture']
        gesture_ratio = gesture_count / len(labels) if labels else 0

        print(f"  Labels: Gesture={gesture_count}, NoGesture={no_gesture_count} ({gesture_ratio:.1%} gesture)")

        if gesture_ratio < min_gesture_ratio:
            print(f"  Warning: Gesture ratio too low ({gesture_ratio:.1%} < {min_gesture_ratio:.1%}), skipping")
            skipped += 1
            continue

        # Group landmarks by label
        for lm, label in zip(landmarks, labels):
            landmarks_by_label[label].append(lm)

        processed += 1

    print(f"\n{'='*60}")
    print(f"Processed {processed} videos, skipped {skipped}")

    # Convert to arrays
    result = {}
    for label, lm_list in landmarks_by_label.items():
        arr = np.array(lm_list, dtype=np.float32)
        result[label] = arr
        print(f"  {label}: {len(arr)} frames")

    # Save to NPZ
    if result:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.savez(output_path, **result)
        print(f"\nSaved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Prepare M3D TED dataset for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all videos in M3D dataset
    python prepare_m3d_data.py --data_dir ../data/m3d --output ../data/m3d_landmarks.npz

    # Process only first 2 videos (for testing)
    python prepare_m3d_data.py --data_dir ../data/m3d --output ../data/test.npz --max_videos 2
        """
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data/m3d',
        help='Directory containing M3D dataset (ELAN files + videos)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../data/m3d_landmarks.npz',
        help='Output NPZ file path'
    )

    parser.add_argument(
        '--max_videos',
        type=int,
        default=None,
        help='Maximum number of videos to process (for testing)'
    )

    parser.add_argument(
        '--min_gesture_ratio',
        type=float,
        default=0.05,
        help='Minimum ratio of gesture frames to include video (default: 0.05)'
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_path = (script_dir / args.output).resolve()

    print(f"M3D Data Preparation Script")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Output file: {output_path}")

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    result = process_dataset(
        str(data_dir),
        str(output_path),
        max_videos=args.max_videos,
        min_gesture_ratio=args.min_gesture_ratio
    )

    if result:
        print(f"\nSuccess! Training data saved to: {output_path}")
        print(f"\nNext step: Train the model:")
        print(f"  cd training/TrainingcodeGBM")
        print(f"  python EnvisionRealTimeTrain.py --npz_path {output_path} --output_path ../../envisionhgdetector/model/lightgbm_gesture_model_v2.pkl")
    else:
        print("\nNo data was processed. Check the data directory structure.")


if __name__ == '__main__':
    main()
