# ============================================================================
# STRUCTURED FEATURE EXTRACTION v3.x (Data Creation Only)
# ============================================================================
# Creates a structured dataset:
# World:    92 features (23 landmarks × 4: x, y, z, visibility)
#
# Three categories: NoGesture, Gesture, Move
# Full metadata for speaker-independent splits
# Timestamps preserved per video
#
# For plotting, use: plot_dataset_statistics.py

from dataclasses import dataclass
from typing import  List, Optional, Dict, Any
import json
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from pathlib import Path

mp_holistic = mp.solutions.holistic

"""
- u can import  this function in main.py for easier pipeline

"""


"""
Note:
originally we set non-existing idx to features.extend([0.0, 0.0, 0.0, 0.0]) 
but we will make it n/a. 0 is misleading
"""



def load_config(output_directory: Path) -> Dict[str, Any]:
    npz_config = dict()
    output_directory.mkdir(parents=True, exist_ok=True)
    npz_config['output_directory'] = output_directory
    npz_config['npz_output_file'] = npz_config['output_directory'] / 'landmarks_world_92.npz'
    npz_config['metadata_file'] = npz_config['output_directory'] / 'dataset_metadata.json'

    npz_config['num_world_features'] = 92  # 23 landmarks × 4 (x, y, z, visibility)

    npz_config['category_labels'] = ('NoGesture', 'Gesture', 'Move')

    npz_config['upperbody_landmark_names'] = [
        'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 
        'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 
        'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
        'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB'
    ]
    npz_config['num_upperbody_landmarks'] = len(npz_config['upperbody_landmark_names'])
    return npz_config

@dataclass
class VideoMetadata:
    """Metadata extracted from video filename"""
    video_name: str
    corpus: str
    speaker: str
    clip_id: str
    category: str
    subtype: str
    original_fps: int
    width: int
    height: int
    is_mirror: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'video_name': self.video_name,
            'corpus': self.corpus,
            'speaker': self.speaker,
            'clip_id': self.clip_id,
            'category': self.category,
            'subtype': self.subtype,
            'original_fps': self.original_fps,
            'width': self.width,
            'height': self.height,
            'is_mirror': self.is_mirror
        }

def extract_npz(npz_path: Path, max_idx: int) -> Dict[str, Any]: # check what original code did and how to adapt it to this function
    data = np.load(npz_path)
    video_metadata = VideoMetadata(
        video_name=data['video_name'].item(),
        corpus=data['corpus'].item(),
        speaker=data['speaker'].item(),
        clip_id=data['clip_id'].item(),
        category=data['category'].item(),
        subtype=data['subtype'].item(),
        original_fps=int(data['fps'].item()),
        width=int(data['width'].item()),
        height=int(data['height'].item()),
    )
    # we are only interested in upper body, so convert the rest to None
    world_landmarks = data['world_body_landmarks']
    for frame in world_landmarks:
        for keypoints in frame:
            for idx in keypoints:
                if idx >= max_idx:
                    keypoints[idx] = [np.nan, np.nan, np.nan, np.nan]  # x,y,z,visibility | convert to NaN for remaining landmarks

    return {
        'video_metadata': video_metadata.to_dict(),
        'world_landmarks': data['world_body_landmarks'],
        }
    
def extract_features_parllel(
    npz_config: Dict[str, Any],
    num_workers: int = None
) -> Dict[str, Any]:
    """Collect features from all videos and save structured datasets."""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)

    input_dir = npz_config['input_directory']
    npz_files = list(input_dir.glob("*.npz"))

    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(lambda x: extract_npz(x, npz_config['num_upperbody_landmarks']), npz_files), 
                          total=len(npz_files), desc="Extracting features from npz"):
            if result is not None:
                results.append(result)
    
    # Organize by training label
    data_by_label = defaultdict(list)
    all_metadata = []
    
    for r in results:
        video_metadata = r['video_metadata']
        all_metadata.append(video_metadata)
        label = video_metadata['category']
        data_by_label[label].append(r)
    
    # World landmarks
    world_dataset = {}
    for label in npz_config['category_labels']:
        videos_data = data_by_label.get(label, [])
        world_dataset[f'{label}_landmarks'] = np.array([v['world_landmarks'] for v in videos_data], dtype=object)
        world_dataset[f'{label}_metadata'] = [v['metadata'] for v in videos_data]
        world_dataset[f'{label}_n_videos'] = len(videos_data)
        world_dataset[f'{label}_n_frames'] = sum(len(v['world_landmarks']) for v in videos_data)
    
    world_feature_names = []
    for lm_name in npz_config['upperbody_landmark_names']:
        for dim in ['X', 'Y', 'Z', 'visibility']:
            world_feature_names.append(f"{lm_name}_{dim}")
    world_dataset['feature_names'] = world_feature_names
    
    world_path = npz_config['npz_output_file']
    np.savez_compressed(world_path, **world_dataset)

    print(f"\n✓ Saved: {world_path}")
    print(f"  Features: {npz_config['num_world_features']}")
    for label in npz_config['category_labels']:
        n_vid = world_dataset[f'{label}_n_videos']
        n_fr = world_dataset[f'{label}_n_frames']
        print(f"  {label}: {n_vid} videos, {n_fr} frames")
    
    # Save metadata JSON
    metadata_path = npz_config['metadata_file']
    with open(metadata_path, 'w') as f:
        json.dump({
            'all_videos': all_metadata,
            'by_label': {k: [v['metadata'] for v in vlist] for k, vlist in data_by_label.items()},
            'config': {
                'num_world_features': npz_config['num_world_features'],
                'target_fps': npz_config['target_fps'],
                'category_labels': npz_config['category_labels']
            }
        }, f, indent=2)
    print(f"\n✓ Metadata: {metadata_path}")
    
    return {
        'results': results,
        'metadata': all_metadata,
        'by_label': data_by_label
    }

if __name__ == "__main__":
    output_directory = Path('ENTER')
    input_directory = Path('maskbench')
    npz_config = load_config(output_directory)
    extract_features_parllel(npz_config)