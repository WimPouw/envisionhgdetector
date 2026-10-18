from pathlib import Path
from envisionhgdetector import PostProcessing

def main():
    data_folder = Path("path/to/data_folder")  # Replace with the desired output folder path
    if not data_folder.exists():
        raise FileNotFoundError(f"Video file {data_folder} does not exist.")
    
    post_processor = PostProcessing(data_folder=data_folder)
    post_processor.cut_by_segments_batch()
    post_processor.extract_pose_from_videos_batch()

if __name__ == "__main__":
    main()