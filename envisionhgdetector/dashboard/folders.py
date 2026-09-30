"""Prepare analysis files for the dashboard."""

import os

def setup_dashboard_folders(data_folder: str, assets_folder: str) -> None:
    """
    Set up necessary folders for the dashboard.
    
    Args:
        data_folder: Path to analysis data folder
        assets_folder: Path to Dash assets folder
    """
    # Create assets folder if it doesn't exist
    os.makedirs(assets_folder, exist_ok=True)
    
    # Adjust path to tracked videos to point to the retracked directory
    retracked_folder = os.path.join(os.path.dirname(data_folder), "retracked", "tracked_videos")
    if not os.path.exists(retracked_folder):
        raise FileNotFoundError(f"Tracked videos folder not found at {retracked_folder}")
        
    # Copy videos if they don't exist in assets
    for video in os.listdir(retracked_folder):
        if video.endswith("_tracked.mp4"):
            source = os.path.join(retracked_folder, video)
            dest = os.path.join(assets_folder, video)
            if not os.path.exists(dest):
                import shutil
                print(f"Copying {video} to assets folder...")
                shutil.copy2(source, dest)
                
    # Correct path to visualization data from the analysis folder
    viz_path = os.path.join(data_folder, "gesture_visualization.csv")
    if not os.path.exists(viz_path):
        raise FileNotFoundError(f"Visualization data not found at {viz_path}")
        
    print(f"Dashboard folders set up successfully:")
    print(f"- Assets folder: {assets_folder}")
    print(f"- Data folder: {data_folder}")
    print(f"- {len(os.listdir(assets_folder))} videos in assets")


