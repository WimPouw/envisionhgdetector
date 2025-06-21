from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import os
import multiprocessing
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from enum import Enum, auto
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic

@dataclass
class Config:
    gesture_labels: Tuple[str, ...] = ("Gesture", "Move") #your gesture labels here
    undefined_gesture_label: str = "Undefined"
    stationary_label: str = "NoGesture"
    npz_filename: str = "./training/bodylandmarks7wlonly.npz" # this is where the training data is stored (mediapipe holistic body from the training videos)
    # seq_length: int = 25 # this sets the window size for the classifier, so 25 frames input (so the model needs 25 frames to make a prediction) (THIS IS NOT USED HERE, SO COMMENTING OUT)
    num_features: int = 69 # number of features that we originally take in (shown below, be careful this number should match you Feature set)
    #weights_filename: str = f"./training/saga_gesturenogesture_trained_model_weightsv1.h5" # this contains the CNN model weights

# Define body landmark names for reference
UPPER_BODY_LANDMARK_NAMES = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 
    'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 
    'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
    'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB'
]
UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

class VideoSegment(Enum):
    """Video segment selection for processing."""
    BEGINNING = auto()
    LAST = auto()

def augment_videos_with_mirrors(videofiles: List[str]) -> List[str]:
    """
    Create mirrored versions of all videos for data augmentation.
    
    Args:
        videofiles: List of video paths
        
    Returns:
        Updated list including the original and mirrored videos
    """
    # Filter out videos that are already mirrored
    original_videos = [x for x in videofiles if '_mirror' not in x]
    
    # Process each video to create mirrored version
    for video in tqdm(original_videos, desc="Creating mirrored videos"):
        video_dir = os.path.dirname(video)
        video_name = os.path.basename(video)
        mirror_path = os.path.join(video_dir, f"{os.path.splitext(video_name)[0]}_mirror.mp4")
        
        # Create mirror version if it doesn't exist
        if not os.path.exists(mirror_path):
            command = [
                'ffmpeg', 
                '-i', video, 
                '-vf', 'hflip', 
                '-c:v', 'libx264', 
                '-preset', 'fast', 
                '-crf', '22', 
                '-an',  # Disable audio
                mirror_path,
                '-y'   # Overwrite existing files
            ]
            
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"Error creating mirror for {video}: {e}")
                continue
    
    # Refresh video list to include mirrored versions
    all_videos = []
    for root, _, files in os.walk(os.path.dirname(videofiles[0])):
        for file in files:
            if file.endswith('.mp4'):
                all_videos.append(os.path.join(root, file))
    
    return all_videos

def extract_world_landmarks(results) -> Optional[List[float]]:
    """
    Extract upper body world landmarks from MediaPipe Holistic results.
    
    Args:
        results: MediaPipe Holistic processing results
        
    Returns:
        List of upper body landmark coordinates in world space or None if not detected
    """
    if not results.pose_world_landmarks:
        return None
        
    features = []
    # Only extract upper body landmarks using the defined indices
    for idx in UPPER_BODY_INDICES:
        if idx < len(results.pose_world_landmarks.landmark):
            landmark = results.pose_world_landmarks.landmark[idx]
            features.extend([landmark.x, landmark.y, landmark.z])
    
    return features

def process_video(
    video_path: str,
    max_num_frames: Optional[int] = None,
    video_segment: VideoSegment = VideoSegment.BEGINNING,
    end_padding: bool = True,
    drop_consecutive_duplicates: bool = False,
    target_fps: int = 25,
    detection_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
    min_detection_percentage: float = 0.01
) -> Tuple[List[List[float]], List[float]]:
    """
    Process a single video to extract world landmarks.
    
    Args:
        video_path: Path to the video file
        max_num_frames: Maximum number of frames to process
        video_segment: Which segment of the video to process
        end_padding: Whether to pad the end if fewer frames than max_num_frames
        drop_consecutive_duplicates: Whether to drop consecutive duplicate frames
        target_fps: Target frame rate for processing
        detection_confidence: MediaPipe detection confidence threshold
        tracking_confidence: MediaPipe tracking confidence threshold
        
    Returns:
        Tuple of (landmarks, frame_timestamps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return [], []
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(original_fps / target_fps)))
    
    # Initialize collections
    landmarks = []
    frame_timestamps = []
    prev_features = None
    frame_index = 0
    valid_frame_count = 0
    
    # Initialize detection tracking
    attempted_frames = 0
    detected_frames = 0
    
    # Process with MediaPipe
    # Process with MediaPipe
    with mp_holistic.Holistic(
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
        static_image_mode=False
    ) as holistic:
        with tqdm(total=min(max_num_frames or float('inf'), total_frames // frame_interval), 
                  desc=f"Processing {os.path.basename(video_path)}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process at target FPS
                if frame_index % frame_interval == 0:
                    # Check if we've reached max frames (for BEGINNING mode)
                    if max_num_frames and video_segment == VideoSegment.BEGINNING and valid_frame_count >= max_num_frames:
                        break
                    
                    # Count this as an attempted frame
                    attempted_frames += 1
                    
                    # Get current timestamp
                    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    # Process frame with MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb_frame)
                    
                    # Extract landmarks
                    features = extract_world_landmarks(results)
                    
                    # Only add valid landmarks
                    if features:
                        # Count this as a successful detection
                        detected_frames += 1
                        
                        # Skip duplicates if requested
                        if drop_consecutive_duplicates and prev_features and np.array_equal(
                                np.round(features, decimals=2),
                                np.round(prev_features, decimals=2)
                        ):
                            frame_index += 1
                            continue
                        
                        landmarks.append(features)
                        frame_timestamps.append(current_time)
                        prev_features = features
                        valid_frame_count += 1
                        pbar.update(1)
                    else:
                        pbar.update(1)  # Update progress bar even when no landmarks are detected
                
                frame_index += 1
    
    # Clean up
    cap.release()
    
    # Calculate detection percentage
    detection_percentage = detected_frames / attempted_frames if attempted_frames > 0 else 0
    
    # Log the detection rates
    print(f"{os.path.basename(video_path)}: Detected person in {detected_frames}/{attempted_frames} frames ({detection_percentage:.2%})")
    
    # Skip videos with poor detection rates
    if detection_percentage < min_detection_percentage:
        print(f"Skipping {video_path}: Detection rate {detection_percentage:.2%} below threshold {min_detection_percentage:.2%}")
        return [], []
    
    # Handle empty results
    if not landmarks:
        print(f"No valid landmarks detected in {video_path}")
        return [], []
    
    # Handle LAST segment selection
    if max_num_frames and video_segment == VideoSegment.LAST:
        landmarks = landmarks[-max_num_frames:]
        frame_timestamps = frame_timestamps[-max_num_frames:]
    
    # Handle padding
    if max_num_frames and end_padding and len(landmarks) < max_num_frames:
        last = landmarks[-1]
        padding_count = max_num_frames - len(landmarks)
        
        # Pad landmarks
        landmarks.extend([last] * padding_count)
        
        # Pad timestamps with incrementing values
        if frame_timestamps:
            last_time = frame_timestamps[-1]
            frame_interval = 1.0 / target_fps
            for i in range(padding_count):
                frame_timestamps.append(last_time + (i+1) * frame_interval)
    
    return landmarks, frame_timestamps

def process_video_wrapper(args_dict):
    """Wrapper for process_video to use with multiprocessing."""
    return process_video(**args_dict)

def collect_landmarks_parallel(
    videofiles: List[str],
    labels: Tuple[str, ...],
    npz_path: str,
    max_num_frames: int = 800,
    num_workers: Optional[int] = None
) -> None:
    """
    Collect landmarks from videos in parallel and save to npz file with dataset tracking.
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 3) # Leave some cores free for system tasks
    
    print(f"Using {num_workers} parallel workers")
    
    # Process videos for each label
    landmark_dict = {}
    dataset_tracking = {}  # This will store which dataset each landmark came from
    
    # First, identify all unique datasets present in the videos
    datasets = set()
    for video in videofiles:
        basename = os.path.basename(video)
        # Extract dataset prefix (e.g., "ECOLANG")
        if "_" in basename:
            dataset = basename.split("_")[0]
            datasets.add(dataset)
    
    print(f"Found {len(datasets)} unique datasets: {', '.join(datasets)}")
    
    for label in labels:
        print(f"Processing videos with label: {label}")
        # Filter videos based on label
        if label == 'NoGesture':
            glob_list = [x for x in videofiles if any(term in os.path.basename(x).lower() 
                        for term in ['nogesture'])]
        elif label == 'Gesture':
            glob_list = [x for x in videofiles if not any(term in os.path.basename(x).lower() 
                        for term in ['nogesture', 'move', 'objman'])]
        elif label == 'Move':
            glob_list = [x for x in videofiles if 'move' in os.path.basename(x).lower()]
            
        # Group videos by dataset
        dataset_videos = {dataset: [] for dataset in datasets}
        for video in glob_list:
            basename = os.path.basename(video)
            if "_" in basename:
                dataset = basename.split("_")[0]
                dataset_videos[dataset].append(video)
        
        for dataset, videos in dataset_videos.items():
            print(f"  Dataset {dataset}: {len(videos)} videos for label '{label}'")
        
        # Initialize empty list for this label
        landmark_dict[label] = []
        dataset_tracking[label] = []
        
        if not glob_list:
            print(f"WARNING: No videos found for label '{label}'")
            continue
        
        videos = glob_list
        print(f"Processing {len(videos)} videos for label '{label}'...")
        
        # Prepare arguments for parallel processing
        process_args = [
            {
                'video_path': video,
                'max_num_frames': max_num_frames,
                'video_segment': VideoSegment.BEGINNING,
                'end_padding': True,
                'drop_consecutive_duplicates': False, # were going to set this to False because we have a no gesture label with little movement
                'target_fps': 25,
                'min_detection_percentage': 0.4 # some reasonable threshold for detection
            }
            for video in videos
        ]
        
        # Process videos in parallel
        all_landmarks = []
        all_dataset_info = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_video_wrapper, args) for args in process_args]
            
            for future, video_path in zip(as_completed(futures), videos):
                try:
                    landmarks, _ = future.result()
                    if landmarks:
                        # Extract dataset from filename
                        basename = os.path.basename(video_path)
                        if "_" in basename:
                            dataset = basename.split("_")[0]
                        else:
                            dataset = "unknown"
                        
                        all_landmarks.extend(landmarks)
                        # Record which dataset each frame came from
                        all_dataset_info.extend([dataset] * len(landmarks))
                except Exception as e:
                    print(f"Error in worker: {e}")
        
        landmark_dict[label] = all_landmarks
        dataset_tracking[label] = all_dataset_info
        print(f"Collected {len(all_landmarks)} landmark frames for label '{label}'")
    
    # Get feature names automatically
    feature_names = []
    for landmark_name in UPPER_BODY_LANDMARK_NAMES:  # Changed from BODY_LANDMARK_NAMES
        for dim in ['X', 'Y', 'Z']:
            feature_names.append(f"{landmark_name}_{dim}")
    
    # Save the results with dataset tracking information
    print(f"Saving landmarks to {npz_path}")
    np.savez_compressed(
        npz_path, 
        feature_names=feature_names, 
        dataset_tracking=dataset_tracking,
        **landmark_dict
    )
    print(f"Landmarks saved successfully to {npz_path}")

def find_training_videos(training_dir='../trainingsetcleaned/'):
    """Find all training videos."""
    # Find training videos
    training_videos = []
    for root, _, files in os.walk(training_dir):
        for file in files:
            if file.endswith('.mp4'):
                training_videos.append(os.path.join(root, file))
    return training_videos

if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    # Find videos
    print("Finding training and validation videos...")
    training_videos = find_training_videos()
    print(f"Found {len(training_videos)} training videos")
    # Augment videos with mirrored versions
    #print("Augmenting training data with mirrored videos...")
    #augmented_training_videos = augment_videos_with_mirrors(training_videos)
    #print(f"Total training videos after augmentation: {len(augmented_training_videos)}")
    
    # Collect landmarks in parallel
    print("Collecting world landmarks from videos...")
    #print("NOTE: Only using videos with people detected in at least 75% of frames")
    collect_landmarks_parallel(
        videofiles=training_videos,
        labels=(config.stationary_label,) + config.gesture_labels,
        npz_path=config.npz_filename,  # Use the configured path
        max_num_frames=800
    )
    print(f"World landmarks collection complete! Data saved to {config.npz_filename}")