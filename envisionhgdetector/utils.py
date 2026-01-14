# envisionhgdetector/utils.py
"""
Utilities module for gesture detection.

This module re-exports functions from specialized submodules for backward compatibility.
New code should import directly from the specific modules:
- segmentation: create_segments, get_prediction_at_threshold
- elan: create_elan_file
- video: label_video, cut_video_by_segments, find_all_videos, create_sliding_windows
- tracking: extract_upper_limb_features, remove_nans, retrack_gesture_videos
- kinematics: compute_kinematic_features, calc_mcneillian_space, calc_vert_height, etc.
- dtw_analysis: compute_gesture_kinematics_dtw, create_gesture_visualization
- dashboard_utils: create_dashboard, setup_dashboard_folders
"""

# Re-export from segmentation module
from .segmentation import (
    create_segments,
    get_prediction_at_threshold,
)

# Re-export from elan module
from .elan import create_elan_file

# Re-export from video module
from .video import (
    label_video,
    cut_video_by_segments,
    find_all_videos,
    create_sliding_windows,
)

# Re-export from tracking module
from .tracking import (
    extract_upper_limb_features,
    remove_nans,
    retrack_gesture_videos,
)

# Re-export from kinematics module
from .kinematics import (
    JOINT_MAP as joint_map,  # Alias for backward compatibility
    ArmKinematics,
    KinematicFeatures,
    calculate_derivatives,
    find_submovements,
    compute_limb_kinematics,
    define_mcneillian_grid,
    get_mcneillian_mode,
    calc_mcneillian_space,
    calc_volume_size,
    calc_vert_height,
    find_movepauses,
    calculate_distance,
    calc_holds,
    compute_kinematic_features,
)

# Re-export from dtw_analysis module
from .dtw_analysis import (
    compute_gesture_kinematics_dtw,
    create_gesture_visualization,
)

# Re-export from dashboard_utils module
from .dashboard_utils import (
    setup_dashboard_folders,
    create_dashboard,
    get_dashboard_css,
)


# Define __all__ for explicit public API
__all__ = [
    # Segmentation
    'create_segments',
    'get_prediction_at_threshold',
    # ELAN
    'create_elan_file',
    # Video
    'label_video',
    'cut_video_by_segments',
    'find_all_videos',
    'create_sliding_windows',
    # Tracking
    'extract_upper_limb_features',
    'remove_nans',
    'retrack_gesture_videos',
    # Kinematics
    'joint_map',
    'ArmKinematics',
    'KinematicFeatures',
    'calculate_derivatives',
    'find_submovements',
    'compute_limb_kinematics',
    'define_mcneillian_grid',
    'get_mcneillian_mode',
    'calc_mcneillian_space',
    'calc_volume_size',
    'calc_vert_height',
    'find_movepauses',
    'calculate_distance',
    'calc_holds',
    'compute_kinematic_features',
    # DTW Analysis
    'compute_gesture_kinematics_dtw',
    'create_gesture_visualization',
    # Dashboard
    'setup_dashboard_folders',
    'create_dashboard',
    'get_dashboard_css',
]
