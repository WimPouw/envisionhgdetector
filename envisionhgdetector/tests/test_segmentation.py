# tests/test_segmentation.py
"""Unit tests for segmentation module."""

import pytest
import numpy as np
import pandas as pd

from envisionhgdetector.segmentation import (
    create_segments,
    get_prediction_at_threshold,
)


class TestCreateSegments:
    """Tests for create_segments function."""

    def test_empty_input(self):
        """Test with no gesture frames."""
        df = pd.DataFrame({
            'time': [0.0, 0.1, 0.2, 0.3],
            'label': ['NoGesture', 'NoGesture', 'NoGesture', 'NoGesture']
        })
        result = create_segments(df, 'label')
        assert result.empty
        assert list(result.columns) == ['start_time', 'end_time', 'labelid', 'label', 'duration']

    def test_single_gesture_segment(self):
        """Test with a single continuous gesture."""
        df = pd.DataFrame({
            'time': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'label': ['NoGesture', 'Gesture', 'Gesture', 'Gesture', 'Gesture',
                     'Gesture', 'Gesture', 'NoGesture', 'NoGesture', 'NoGesture']
        })
        result = create_segments(df, 'label', min_length_s=0.3)

        assert len(result) == 1
        assert result.iloc[0]['label'] == 'Gesture'
        assert result.iloc[0]['start_time'] == 0.1
        # End time is the timestamp where state changes (index 7 = 0.7)
        assert result.iloc[0]['end_time'] == 0.7

    def test_merge_close_segments(self):
        """Test that close segments are merged."""
        df = pd.DataFrame({
            'time': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'label': ['NoGesture', 'Gesture', 'Gesture', 'NoGesture', 'Gesture',
                     'Gesture', 'NoGesture', 'NoGesture', 'NoGesture', 'NoGesture', 'NoGesture']
        })
        # With min_gap_s=0.2, the two gesture segments should be merged
        result = create_segments(df, 'label', min_gap_s=0.2, min_length_s=0.1)

        assert len(result) == 1
        assert result.iloc[0]['start_time'] == 0.1
        # Merged segment ends at index 6 = 0.6
        assert result.iloc[0]['end_time'] == 0.6

    def test_filter_short_segments(self):
        """Test that segments shorter than min_length_s are filtered."""
        df = pd.DataFrame({
            'time': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'label': ['NoGesture', 'Gesture', 'NoGesture', 'NoGesture', 'NoGesture', 'NoGesture']
        })
        # Segment is only 0.1s, should be filtered with min_length_s=0.5
        result = create_segments(df, 'label', min_length_s=0.5)
        assert result.empty

    def test_move_label(self):
        """Test with Move label."""
        df = pd.DataFrame({
            'time': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            'label': ['NoGesture', 'Move', 'Move', 'Move', 'Move', 'Move', 'NoGesture', 'NoGesture']
        })
        result = create_segments(df, 'label', min_length_s=0.3)

        assert len(result) == 1
        assert result.iloc[0]['label'] == 'Move'


class TestGetPredictionAtThreshold:
    """Tests for get_prediction_at_threshold function."""

    def test_no_motion(self):
        """Test when NoGesture confidence is high."""
        row = pd.Series({
            'NoGesture_confidence': 0.9,
            'Gesture_confidence': 0.05,
            'Move_confidence': 0.05
        })
        result = get_prediction_at_threshold(row, motion_threshold=0.6, gesture_threshold=0.6)
        assert result == 'NoGesture'

    def test_gesture_detected(self):
        """Test when gesture confidence is high."""
        row = pd.Series({
            'NoGesture_confidence': 0.2,
            'Gesture_confidence': 0.7,
            'Move_confidence': 0.1
        })
        result = get_prediction_at_threshold(row, motion_threshold=0.6, gesture_threshold=0.6)
        assert result == 'Gesture'

    def test_move_detected(self):
        """Test when move confidence is high."""
        row = pd.Series({
            'NoGesture_confidence': 0.2,
            'Gesture_confidence': 0.1,
            'Move_confidence': 0.7
        })
        result = get_prediction_at_threshold(row, motion_threshold=0.6, gesture_threshold=0.6)
        assert result == 'Move'

    def test_gesture_vs_move_gesture_wins(self):
        """Test that higher confidence wins between gesture and move."""
        row = pd.Series({
            'NoGesture_confidence': 0.1,
            'Gesture_confidence': 0.75,
            'Move_confidence': 0.65
        })
        result = get_prediction_at_threshold(row, motion_threshold=0.6, gesture_threshold=0.6)
        assert result == 'Gesture'

    def test_below_gesture_threshold(self):
        """Test when motion detected but gesture below threshold."""
        row = pd.Series({
            'NoGesture_confidence': 0.3,
            'Gesture_confidence': 0.4,  # Below 0.6 threshold
            'Move_confidence': 0.3
        })
        result = get_prediction_at_threshold(row, motion_threshold=0.6, gesture_threshold=0.6)
        assert result == 'NoGesture'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
