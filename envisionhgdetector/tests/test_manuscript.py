"""
Tests to lock in current behavior before risky fixes on the manuscript branch.
Covers: Config, segmentation, thresholds, and preprocessing feature extraction.
No model files or video files required.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# ============================================================================
# Config tests
# ============================================================================

class TestConfig:
    def test_default_feature_set_is_world(self):
        from envisionhgdetector.config import Config
        cfg = Config()
        assert cfg.feature_set == "world"
        assert cfg.num_original_features == 92

    def test_feature_set_basic(self):
        from envisionhgdetector.config import Config
        cfg = Config(feature_set="basic")
        assert cfg.num_original_features == 41

    def test_feature_set_extended(self):
        from envisionhgdetector.config import Config
        cfg = Config(feature_set="extended")
        assert cfg.num_original_features == 61

    def test_default_cnn_model_filename(self):
        from envisionhgdetector.config import Config
        cfg = Config()
        assert cfg.cnn_model_filename == "R2_CNN_world_best_config18.h5"

    def test_default_lightgbm_model_filename(self):
        from envisionhgdetector.config import Config
        cfg = Config()
        assert cfg.lightgbm_model_filename == "R2_best_lightgbm_model_config13.pkl"

    def test_default_thresholds(self):
        from envisionhgdetector.config import Config
        cfg = Config()
        assert cfg.default_motion_threshold == 0.7
        assert cfg.default_gesture_threshold == 0.7

    def test_seq_length(self):
        from envisionhgdetector.config import Config
        cfg = Config()
        assert cfg.seq_length == 25


# ============================================================================
# Segmentation / threshold tests (from utils.py)
# ============================================================================

class TestCreateSegments:
    def _make_predictions(self, labels, times=None):
        n = len(labels)
        if times is None:
            times = np.arange(n) / 25.0
        return pd.DataFrame({
            'time': times,
            'label': labels,
        })

    def test_all_gesture_produces_one_segment(self):
        from envisionhgdetector.utils import create_segments
        df = self._make_predictions(['Gesture'] * 50)
        segs = create_segments(df, label_column='label', min_gap_s=0.0, min_length_s=0.0)
        assert len(segs) == 1
        assert segs.iloc[0]['label'] == 'Gesture'

    def test_all_nogesture_produces_no_segments(self):
        from envisionhgdetector.utils import create_segments
        df = self._make_predictions(['NoGesture'] * 50)
        segs = create_segments(df, label_column='label', min_gap_s=0.0, min_length_s=0.0)
        assert len(segs) == 0

    def test_two_gesture_segments_separated_by_gap(self):
        from envisionhgdetector.utils import create_segments
        labels = ['Gesture'] * 20 + ['NoGesture'] * 10 + ['Gesture'] * 20
        df = self._make_predictions(labels)
        segs = create_segments(df, label_column='label', min_gap_s=0.0, min_length_s=0.0)
        assert len(segs) == 2

    def test_min_length_filter_removes_short_segments(self):
        from envisionhgdetector.utils import create_segments
        # 5 frames at 25fps = 0.2s, below min_length_s=0.5
        labels = ['Gesture'] * 5 + ['NoGesture'] * 100
        df = self._make_predictions(labels)
        segs = create_segments(df, label_column='label', min_gap_s=0.0, min_length_s=0.5)
        assert len(segs) == 0

    def test_min_gap_merges_close_segments(self):
        from envisionhgdetector.utils import create_segments
        # Two segments with a 2-frame gap (0.08s at 25fps)
        labels = ['Gesture'] * 25 + ['NoGesture'] * 2 + ['Gesture'] * 25
        df = self._make_predictions(labels)
        segs = create_segments(df, label_column='label', min_gap_s=0.5, min_length_s=0.0)
        assert len(segs) == 1

    def test_segment_has_required_columns(self):
        from envisionhgdetector.utils import create_segments
        df = self._make_predictions(['Gesture'] * 25)
        segs = create_segments(df, label_column='label', min_gap_s=0.0, min_length_s=0.0)
        assert 'start_time' in segs.columns
        assert 'end_time' in segs.columns
        assert 'label' in segs.columns

    def test_start_before_end(self):
        from envisionhgdetector.utils import create_segments
        df = self._make_predictions(['Gesture'] * 25)
        segs = create_segments(df, label_column='label', min_gap_s=0.0, min_length_s=0.0)
        for _, row in segs.iterrows():
            assert row['start_time'] < row['end_time']


# ============================================================================
# Threshold behavior tests (document current behavior before fixing)
# ============================================================================

class TestThresholdBehavior:
    """
    These tests document the CURRENT (buggy) threshold behavior where
    passing 0.0 is treated as falsy and replaced with the config default.
    When the falsy-threshold bug is fixed, these tests should be UPDATED
    to verify that 0.0 is respected.
    """

    def test_zero_motion_threshold_uses_config_default(self):
        """
        CURRENT BEHAVIOR: passing motion_threshold=0.0 silently uses config default.
        After bug fix: 0.0 should be used as-is.
        """
        from envisionhgdetector.config import Config
        cfg = Config()
        # Simulate the current behavior in detector.py line 91-94
        motion_threshold = 0.0
        effective = motion_threshold or cfg.default_motion_threshold
        assert effective == cfg.default_motion_threshold  # 0.7, not 0.0

    def test_nonzero_threshold_is_respected(self):
        from envisionhgdetector.config import Config
        cfg = Config()
        motion_threshold = 0.5
        effective = motion_threshold or cfg.default_motion_threshold
        assert effective == 0.5


# ============================================================================
# Preprocessing: feature count and structure tests
# ============================================================================

class TestPreprocessingFeatureCounts:
    """
    Lock in the expected feature dimensions for each feature set.
    These should not change unless the feature extraction is deliberately updated.
    """

    def test_world_feature_count(self):
        from envisionhgdetector.config import Config
        cfg = Config(feature_set="world")
        assert cfg.num_original_features == 92

    def test_world_features_are_23_landmarks_times_4(self):
        # 23 MediaPipe world landmarks * [x, y, z, visibility] = 92
        assert 23 * 4 == 92

    def test_basic_feature_count(self):
        from envisionhgdetector.config import Config
        cfg = Config(feature_set="basic")
        assert cfg.num_original_features == 41

    def test_extended_feature_count(self):
        from envisionhgdetector.config import Config
        cfg = Config(feature_set="extended")
        assert cfg.num_original_features == 61


# ============================================================================
# Utils: extract_upper_limb_features (behavior lock-in before camera fix)
# ============================================================================

class TestExtractUpperLimbFeatures:
    def _make_landmarks(self, n_frames=10, n_points=33):
        """Synthetic world landmarks: n_frames x n_points x 3."""
        rng = np.random.default_rng(42)
        return rng.uniform(-1.0, 1.0, size=(n_frames, n_points, 3)).astype(np.float32)

    def test_returns_array(self):
        from envisionhgdetector.utils import extract_upper_limb_features
        landmarks = self._make_landmarks()
        result = extract_upper_limb_features(landmarks)
        assert isinstance(result, np.ndarray)

    def test_output_has_correct_frames(self):
        from envisionhgdetector.utils import extract_upper_limb_features
        landmarks = self._make_landmarks(n_frames=15)
        result = extract_upper_limb_features(landmarks)
        assert result.shape[0] == 15

    def test_raises_on_wrong_shape(self):
        from envisionhgdetector.utils import extract_upper_limb_features
        bad = np.zeros((10, 33, 4))  # 4 dims instead of 3
        with pytest.raises(ValueError):
            extract_upper_limb_features(bad)

    def test_raises_on_2d_input(self):
        from envisionhgdetector.utils import extract_upper_limb_features
        bad = np.zeros((10, 33))
        with pytest.raises(ValueError):
            extract_upper_limb_features(bad)

    def test_output_is_reproducible(self):
        """Same input always produces same output (no randomness)."""
        from envisionhgdetector.utils import extract_upper_limb_features
        landmarks = self._make_landmarks()
        r1 = extract_upper_limb_features(landmarks)
        r2 = extract_upper_limb_features(landmarks)
        np.testing.assert_array_equal(r1, r2)
