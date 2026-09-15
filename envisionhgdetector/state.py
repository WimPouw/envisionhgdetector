from pathlib import Path
from typing import Optional
import numpy as np
from dataclasses import dataclass
from typing import Literal

@dataclass
class Row:
    frame_index: int
    prediction: Literal["Gesture", "NoGesture", "Move"]
    confidence: float
    motion_confidence: float
    gesture_confidence: float
    no_gesture_confidence: float
    move_confidence: float
    timestamp: Optional[float] = None

class Thresholds:
    def __init__(self, 
        motion_threshold: Optional[float] = None,
        gesture_threshold: Optional[float] = None,
        min_gap_s: Optional[float] = None,
        min_length_s: Optional[float] = None,
        gesture_class_bias: Optional[float] = None,
        cnn_motion_threshold: Optional[float] = None,
        cnn_gesture_threshold: Optional[float] = None,
        lgbm_threshold: Optional[float] = None
    ):
        self.motion_threshold =  motion_threshold or 0.7
        self.gesture_threshold = gesture_threshold or 0.7
        self.min_gap_s = min_gap_s or 0.3
        self.min_length_s = min_length_s or 0.5
        self.gesture_class_bias = gesture_class_bias or 0.0
        self.cnn_motion_threshold = cnn_motion_threshold or 0.7
        self.cnn_gesture_threshold = cnn_gesture_threshold or 0.7
        self.lgbm_threshold = lgbm_threshold or 0.5


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


class LIGHTGBM_Config:
    def __init__(self, config: dict, weights_path: Path, thresholds: Thresholds):
        self.weights_path = weights_path
        self.thresholds = thresholds

        data_dict = config.get('data', {})
        model_dict = config.get('model', {})

        self.window_size = data_dict.get('window_size')
        self.n_features = data_dict.get('num_features')
        self.gesture_labels = data_dict.get('class_labels')

        # Mediapipe
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5

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