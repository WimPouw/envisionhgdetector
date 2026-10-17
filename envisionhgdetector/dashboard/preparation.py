"""Dashboard preparation exposed by gesture detectors."""

import os
import shutil
from typing import Optional

from .folders import setup_dashboard_folders
from .css import css_content


def prepare_gesture_dashboard(self, data_folder: str, assets_folder: Optional[str] = None) -> None:
    """Prepare dashboard (works with both models).
        assets contains? - analyze_dtw_kinematics
    """
    try:
        if assets_folder is None:
            assets_folder = os.path.join(os.path.dirname(data_folder), "assets")

        # Set up folders and copy necessary files
        setup_dashboard_folders(data_folder, assets_folder)
        
        # Get the output directory (parent of analysis folder)
        output_dir = os.path.dirname(data_folder)
        
        # Copy the app.py to the output directory - TODO why?
        dashboard_script_path = os.path.join(os.path.dirname(__file__), "app.py")
        destination_script_path = os.path.join(output_dir, "app.py")
        shutil.copy(dashboard_script_path, destination_script_path)
        
        print(f"Dashboard prepared for {self.model_type.upper()} results")
        print(f"App dashboard copied to: {destination_script_path}")
        
        # Create the CSS file in the assets folder
        
        css_file_path = os.path.join(assets_folder, "styles.css")
        with open(css_file_path, "w") as css_file:
            css_file.write(css_content.strip())
        
        print(f"CSS file created at: {css_file_path}")
        print("Run 'python app.py' to start the dashboard")
        
    except Exception as e:
        print(f"Error preparing dashboard: {str(e)}")
        raise
