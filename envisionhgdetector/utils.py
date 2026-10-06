# Standard library imports
import os
import glob
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from scipy.spatial.distance import euclidean
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .state import Labels, MoveMode, PredictionColumns, Row, SegmentColumns
from .state import *
from .analysis.video_files import find_all_videos
from .analysis.retracking import retrack_gesture_videos
from .analysis.visualization import create_gesture_visualization
from .analysis.features import process_hand_fingers, extract_upper_limb_features
from .analysis.kinematics import ArmKinematics, calculate_derivatives, compute_limb_kinematics, find_submovements, find_movepauses, calculate_distance, calc_holds
from .analysis.spatial import define_mcneillian_grid, get_mcneillian_mode, calc_mcneillian_space, calc_volume_size, calc_vert_height
from .analysis.gesture_kinematics import joint_map, KinematicFeatures, compute_kinematic_features
from .analysis.dtw import compute_gesture_kinematics_dtw
from .analysis.segments import cut_video_by_segments
from .rendering.label_video import label_video
from .dashboard.folders import setup_dashboard_folders
from .dashboard.gesture_space import create_dashboard
from .inference.video import get_video_fps
from .inference.validation import valid_float
from .inference.frames import expand_predictions_to_frames
from .inference.thresholds import get_label_from_prediction, get_prediction_at_threshold
from .inference.segments import create_segments, create_segments_from_labels
from .inference.elan import create_elan_file

