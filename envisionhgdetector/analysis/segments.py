"""Cut videos into gesture segments."""

import os
import glob
from typing import Dict, List

import numpy as np
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip

def cut_video_by_segments(
    segments_folder: str,
    videos_folder: str = None,
    video_prefix: str = "_labelled",
    segments_pattern: str = "*_segments.csv",
    output_subfolder: str = "gesture_segments"
) -> Dict[str, List[str]]:
    """
    Extracts video segments and corresponding features based on segments.csv files.
    
    Args:
        segments_folder: Path to the folder containing segments.csv and features.npy files
        videos_folder: Path to the folder containing videos; defaults to segments_folder
        video_prefix: Suffix added before the video extension, or empty for original videos
        segments_pattern: Pattern to match segment CSV files
        output_subfolder: Name of subfolder to store segmented videos
        
    Returns:
        Dictionary mapping original video names to lists of generated segment paths
    """
    # Create subfolder for segments if it doesn't exist
    videos_folder = videos_folder or segments_folder
    output_segments_folder = os.path.join(segments_folder, output_subfolder)
    os.makedirs(output_segments_folder, exist_ok=True)
    
    # Get all segment CSV files
    segment_files = glob.glob(os.path.join(segments_folder, segments_pattern))
    results = {}
    
    for segment_file in segment_files:
        try:
            # Get original video name from segments file name
            base_name = os.path.basename(segment_file).replace('_segments.csv', '')
            video_stem, video_extension = os.path.splitext(base_name)
            video_extension = video_extension or ".mp4"
            input_video = os.path.join(videos_folder, f"{video_stem}{video_prefix}{video_extension}")
            features_path = os.path.join(segments_folder, f"{base_name}_features.npy")
            
            # Check if video and features exist
            if not os.path.exists(input_video):
                print(f"Warning: Video not found for {base_name}: {input_video}")
                continue
            if not os.path.exists(features_path):
                print(f"Warning: Features file not found for {base_name}")
                continue
                
            # Read segments file
            segments_df = pd.read_csv(segment_file)
            
            if segments_df.empty:
                print(f"No segments found in {segment_file}")
                continue
            
            # Create subfolder for this video's segments
            video_segments_folder = os.path.join(output_segments_folder, base_name)
            os.makedirs(video_segments_folder, exist_ok=True)
            
            # Load video and get fps
            video = VideoFileClip(input_video)
            fps = video.fps
            
            # Load features
            features = np.load(features_path)
            
            segment_paths = []
            
            # Process each segment
            for idx, segment in segments_df.iterrows():
                start_time = segment['start_time']
                end_time = segment['end_time']
                label = segment['label']
                
                # Calculate frame indices
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                
                # Create segment filenames
                segment_filename = f"{base_name}_segment_{idx+1}_{label}_{start_time:.2f}_{end_time:.2f}.mp4"
                features_filename = f"{base_name}_segment_{idx+1}_{label}_{start_time:.2f}_{end_time:.2f}_features.npy"
                
                segment_path = os.path.join(video_segments_folder, segment_filename)
                features_path = os.path.join(video_segments_folder, features_filename)
                
                # Extract and save video segment
                try:
                    # Cut video
                    segment_clip = video.subclipped(start_time, end_time)
                    segment_clip.write_videofile(
                        segment_path,
                        codec='libx264',
                        audio=False
                    )
                    segment_clip.close()
                    
                    # Cut and save features
                    if start_frame < len(features) and end_frame <= len(features):
                        segment_features = features[start_frame:end_frame]
                        np.save(features_path, segment_features)
                        print(f"Created segment and features: {segment_filename}")
                    else:
                        print(f"Warning: Frame indices {start_frame}:{end_frame} out of bounds for features array of length {len(features)}")
                    
                    segment_paths.append(segment_path)
                    
                except Exception as e:
                    print(f"Error creating segment {segment_filename}: {str(e)}")
                    continue
            
            # Clean up
            video.close()
            
            results[base_name] = segment_paths
            print(f"Completed processing segments for {base_name}")
            
        except Exception as e:
            print(f"Error processing {segment_file}: {str(e)}")
            continue
    
    return results

