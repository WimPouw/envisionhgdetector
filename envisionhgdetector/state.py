from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass
from typing import Literal, get_args

class DIRS:
    ANALYSIS = "analysis"
    RETRACKED = "retracked"
    TRACKED_VIDEOS = "tracked_videos" # retracked/tracked_videos
    GESTURE_SEGMENTS = "gesture_segments"
    ASSETS = "assets"
    VIDEOS_RERENDERED = "videos_rerendered" # assets/videos_rerendered

class Labels:
    GESTURE = "Gesture"
    NOGESTURE = "NoGesture"
    MOVE = "Move"

# Color mapping for labels
color_map = {
    Labels.NOGESTURE: (50, 50, 50),      # Dark gray
    Labels.GESTURE: (0, 204, 204),        # Vibrant teal
    Labels.MOVE: (255, 94, 98)            # Soft coral red
}

LABELS_LITERAL = Literal[Labels.GESTURE, Labels.NOGESTURE, Labels.MOVE]
LABELS = get_args(LABELS_LITERAL)

class PredictionColumns:
    """Canonical prediction DataFrame column names."""

    FRAME_INDEX = "frame_index"
    PREDICTION = "prediction"
    CONFIDENCE = "confidence"
    MOTION_CONFIDENCE = "motion_confidence"
    GESTURE_CONFIDENCE = "gesture_confidence"
    NO_GESTURE_CONFIDENCE = "no_gesture_confidence"
    MOVE_CONFIDENCE = "move_confidence"
    TIMESTAMP = "timestamp"
    FRAME = "frame"
    WALL_CLOCK_TIME = "wall_clock_time"
    RAW_GESTURE = "raw_gesture"
    THRESHOLD = "threshold"
    PREDICTION_AVAILABLE = "prediction_available"
    SOURCE_FRAME_INDEX = "frame_idx"

# TODO - how do i join these 2
# maybe init and get names function or smth
@dataclass
class Row:
    frame_index: int
    prediction: LABELS_LITERAL
    confidence: float
    motion_confidence: float
    gesture_confidence: float
    no_gesture_confidence: float
    move_confidence: float
    timestamp: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            PredictionColumns.FRAME_INDEX: self.frame_index,
            PredictionColumns.PREDICTION: self.prediction,
            PredictionColumns.CONFIDENCE: self.confidence,
            PredictionColumns.MOTION_CONFIDENCE: self.motion_confidence,
            PredictionColumns.GESTURE_CONFIDENCE: self.gesture_confidence,
            PredictionColumns.NO_GESTURE_CONFIDENCE: self.no_gesture_confidence,
            PredictionColumns.MOVE_CONFIDENCE: self.move_confidence,
            PredictionColumns.TIMESTAMP: self.timestamp
        }

class SegmentColumns:
    """Canonical segment DataFrame column names."""

    START_TIME = "start_time"
    START_FRAME_IDX = "start_frame_idx"
    END_TIME = "end_time"
    END_FRAME_IDX = "end_frame_idx"
    PREDICTION = "prediction"
    SEGMENT_IDX = "segment_idx"
    DURATION = "duration"
    LABEL_ID = "labelid"



class ModelNames:
    """Canonical model identifiers."""

    CNN = "cnn"
    CNN_B = "cnn_b"
    LIGHTGBM = "lightgbm"
VALID_MODEL_NAMES_LITERAL = Literal[ModelNames.CNN, ModelNames.CNN_B, ModelNames.LIGHTGBM]
VALID_MODEL_NAMES = get_args(VALID_MODEL_NAMES_LITERAL)



class StatsKeys:
    """Canonical statistics dictionary keys."""

    MODEL_TYPE = "model_type"
    AVERAGE_MOTION = "average_motion"
    AVERAGE_GESTURE = "average_gesture"
    AVERAGE_MOVE = "average_move"
    CNN_WEIGHT = "cnn_weight"
    LIGHTGBM_WEIGHT = "lgbm_weight"



@dataclass
class Segment:
    start_time: float
    end_time: float
    prediction: LABELS_LITERAL
    segment_idx: int
    duration: float


class Thresholds:
    def __init__(self, 
        motion_threshold: Optional[float] = None,
        gesture_threshold: Optional[float] = None,
        min_gap_s: Optional[float] = None,
        min_length_s: Optional[float] = None,
        gesture_class_bias: Optional[float] = None,
        cnn_motion_threshold: Optional[float] = None,
        cnn_gesture_threshold: Optional[float] = None,
        lgbm_threshold: Optional[float] = None,
        mediapipe_min_detection_confidence: Optional[float] = None,
        mediapipe_min_tracking_confidence: Optional[float] = None
    ):
        self.motion_threshold =  motion_threshold or 0.5
        self.gesture_threshold = gesture_threshold or 0.5
        self.min_gap_s = min_gap_s or 0.3
        self.min_length_s = min_length_s or 0.5
        self.gesture_class_bias = gesture_class_bias or 0.0

        # combined model specific thresholds
        self.cnn_motion_threshold = cnn_motion_threshold or 0.5
        self.cnn_gesture_threshold = cnn_gesture_threshold or 0.5
        self.lgbm_threshold = lgbm_threshold or 0.5

        # lightgbm specific thresholds
        self.mediapipe_min_detection_confidence = mediapipe_min_detection_confidence or 0.5
        self.mediapipe_min_tracking_confidence = mediapipe_min_tracking_confidence or 0.5


class CNN_B_Config:
    def __init__(self, config: dict, weights_path: Path, thresholds: Thresholds):
        self.weights_path = weights_path
        self.thresholds = thresholds

        data_dict = config.get('data', {})
        model_dict = config.get('model', {})

        self.seq_length = data_dict.get('seq_length')
        self.num_features = data_dict.get('num_features')
        self.dataset_name = data_dict.get('dataset_name') # "world", "extended", "basic"
        self.target_fps = data_dict.get('target_fps')

        self.conv_filters = model_dict.get('conv_filters')
        self.conv_kernel_size = model_dict.get('conv_kernel_size')
        self.pool_size = model_dict.get('pool_size')
        self.dense_units = model_dict.get('dense_units')
        self.dropout_rate = model_dict.get('dropout_rate')
        self.l2_weight = model_dict.get('l2_weight')
        self.preprocessing = model_dict.get('preprocessing')
        self.noise_stddev = model_dict.get('noise_stddev')
        self.spatial_dropout_rate = model_dict.get('spatial_dropout_rate')

        # defaults
        self.jitter_sigma = 0.001
        self.scale_range = (0.995, 1.005)
        self.drop_prob = 0.01

class CNN_Config(CNN_B_Config):
    def __init__(self, config: dict, weights_path: Path, thresholds: Thresholds):
        super().__init__(config, weights_path, thresholds)        

        # TODO - hardcoded for now - since last run was cnn_b only
        self.gesture_labels = Tuple[str, str] = ("Gesture", "Move")  # Motion classes (excluding NoGesture)
        self.all_labels: Tuple[str, str, str] = ("NoGesture", "Gesture", "Move")

class LIGHTGBM_Config:
    def __init__(self, config: dict, weights_path: Path, thresholds: Thresholds):
        self.weights_path = weights_path
        self.thresholds = thresholds

        data_dict = config.get('data', {})

        self.window_size = data_dict.get('window_size')
        self.n_features = data_dict.get('num_features')
        self.gesture_labels = data_dict.get('class_labels')

        # Mediapipe
        self.min_detection_confidence = thresholds.mediapipe_min_detection_confidence
        self.min_tracking_confidence = thresholds.mediapipe_min_tracking_confidence

        # Defaults for LightGBM model parameters
        # Upper body landmark indices (23 landmarks, matching training)
        self.UPPER_BODY_INDICES = list(range(23))

        # Key joint indices for feature extraction
        self.KEY_JOINT_INDICES = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
        self.LEFT_WRIST_IDX = 15
        self.RIGHT_WRIST_IDX = 16

        # Visibility landmark indices
        self.VISIBILITY_LANDMARKS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        self.UPPER_BODY_VIS = np.array([11, 12, 13, 14, 15, 16])