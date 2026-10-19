# from parent folder: python -m tests.test_postprocessing
from pathlib import Path
from envisionhgdetector import PostProcessing

def main():
    data_folder = Path("path/to/data_folder")  # Replace with the desired output folder path
    if not data_folder.exists():
        raise FileNotFoundError(f"Video file {data_folder} does not exist.")
    
    post_processor = PostProcessing(data_folder=data_folder)
    # Cut labeled full videos, excluding anything inside a segments folder.
    post_processor.cut_by_segments_batch(video_scope="full")
    # Retrack the generated clips inside segments folders.
    post_processor.extract_pose_from_videos_batch(video_scope="segments")
    # To retrack labeled full videos instead, use video_scope="full".
    # Compare retracked segments using each segment video's frame rate.
    dtw_matrix, gesture_names, kinematic_features = post_processor.compute_gesture_kinematics_dtw(
        max_gap=3,
    )
    

if __name__ == "__main__":
    main()