# tests/test_dtw_analysis.py
"""Unit tests for DTW analysis module."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from envisionhgdetector.dtw_analysis import (
    create_gesture_visualization,
    DTWAnalysisError,
)


class TestCreateGestureVisualization:
    """Tests for create_gesture_visualization function."""

    def test_creates_visualization_file(self):
        """Test that visualization CSV file is created."""
        # Create a simple DTW matrix (needs at least 4 gestures for proper UMAP)
        dtw_matrix = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.5, 2.5],
            [2.0, 1.5, 0.0, 1.0],
            [3.0, 2.5, 1.0, 0.0]
        ])
        gesture_names = ['gesture_1', 'gesture_2', 'gesture_3', 'gesture_4']

        with tempfile.TemporaryDirectory() as tmpdir:
            create_gesture_visualization(dtw_matrix, gesture_names, tmpdir)

            output_path = os.path.join(tmpdir, "gesture_visualization.csv")
            assert os.path.exists(output_path)

    def test_output_has_correct_columns(self):
        """Test that output CSV has expected columns."""
        dtw_matrix = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.5, 2.5],
            [2.0, 1.5, 0.0, 1.0],
            [3.0, 2.5, 1.0, 0.0]
        ])
        gesture_names = ['gesture_1', 'gesture_2', 'gesture_3', 'gesture_4']

        with tempfile.TemporaryDirectory() as tmpdir:
            create_gesture_visualization(dtw_matrix, gesture_names, tmpdir)

            output_path = os.path.join(tmpdir, "gesture_visualization.csv")
            df = pd.read_csv(output_path)

            assert 'x' in df.columns
            assert 'y' in df.columns
            assert 'gesture' in df.columns

    def test_output_has_correct_rows(self):
        """Test that output has one row per gesture."""
        dtw_matrix = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.5, 2.5],
            [2.0, 1.5, 0.0, 1.0],
            [3.0, 2.5, 1.0, 0.0]
        ])
        gesture_names = ['g1', 'g2', 'g3', 'g4']

        with tempfile.TemporaryDirectory() as tmpdir:
            create_gesture_visualization(dtw_matrix, gesture_names, tmpdir)

            output_path = os.path.join(tmpdir, "gesture_visualization.csv")
            df = pd.read_csv(output_path)

            assert len(df) == 4
            assert set(df['gesture'].tolist()) == set(gesture_names)

    def test_handles_nan_values(self):
        """Test that NaN values in DTW matrix are handled."""
        dtw_matrix = np.array([
            [0.0, 1.0, 2.0, np.nan],
            [1.0, 0.0, 1.5, 2.0],
            [2.0, 1.5, 0.0, 1.0],
            [np.nan, 2.0, 1.0, 0.0]
        ])
        gesture_names = ['g1', 'g2', 'g3', 'g4']

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise an error
            create_gesture_visualization(dtw_matrix, gesture_names, tmpdir)

            output_path = os.path.join(tmpdir, "gesture_visualization.csv")
            assert os.path.exists(output_path)

    def test_too_few_gestures_raises_error(self):
        """Test that less than 3 gestures raises an error (UMAP requires n_neighbors >= 2)."""
        dtw_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        gesture_names = ['g1', 'g2']

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(DTWAnalysisError):
                create_gesture_visualization(
                    dtw_matrix, gesture_names, tmpdir, n_neighbors=15
                )


class TestKinematicFeaturesDataFrame:
    """Tests for _kinematic_features_to_dataframe helper."""

    def test_dataframe_columns(self):
        """Test that all expected columns are present."""
        from envisionhgdetector.dtw_analysis import _kinematic_features_to_dataframe
        from envisionhgdetector.kinematics import KinematicFeatures

        # Create a mock KinematicFeatures object
        features = KinematicFeatures(
            gesture_id='test_gesture',
            video_id='test_video',
            active_hand='L',
            space_use=5,
            mcneillian_max=3.0,
            mcneillian_mode=2,
            volume=100.0,
            max_height=2.5,
            duration=1.5,
            hold_count=2,
            hold_time=0.5,
            hold_avg_duration=0.25,
            hand_submovements=3,
            hand_submovement_peaks=[0.5, 0.8, 0.6],
            hand_mean_submovement_amplitude=0.63,
            elbow_submovements=2,
            elbow_mean_submovement_amplitude=0.4,
            hand_peak_speed=1.5,
            hand_mean_speed=0.8,
            hand_peak_acceleration=2.0,
            hand_peak_deceleration=-1.5,
            hand_peak_jerk=3.0,
            elbow_peak_speed=1.0,
            elbow_mean_speed=0.5,
            elbow_peak_acceleration=1.5,
            elbow_peak_deceleration=-1.0,
            elbow_peak_jerk=2.0
        )

        df = _kinematic_features_to_dataframe([features])

        expected_columns = [
            'gesture_id', 'video_id', 'active_hand',
            'space_use', 'mcneillian_max', 'mcneillian_mode',
            'volume', 'max_height', 'duration',
            'hold_count', 'hold_time', 'hold_avg_duration',
            'hand_submovements', 'hand_submovement_peak_max',
            'hand_submovement_peak_mean', 'hand_mean_submovement_amplitude',
            'elbow_submovements', 'elbow_mean_submovement_amplitude',
            'hand_peak_speed', 'hand_mean_speed',
            'hand_peak_acceleration', 'hand_peak_deceleration',
            'hand_peak_jerk', 'elbow_peak_speed', 'elbow_mean_speed',
            'elbow_peak_acceleration', 'elbow_peak_deceleration',
            'elbow_peak_jerk'
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_peak_calculations(self):
        """Test that peak max and mean are calculated correctly."""
        from envisionhgdetector.dtw_analysis import _kinematic_features_to_dataframe
        from envisionhgdetector.kinematics import KinematicFeatures

        features = KinematicFeatures(
            gesture_id='test', video_id='test', active_hand='R',
            space_use=1, mcneillian_max=1, mcneillian_mode=1,
            volume=1, max_height=1, duration=1,
            hold_count=0, hold_time=0, hold_avg_duration=0,
            hand_submovements=3,
            hand_submovement_peaks=[1.0, 2.0, 3.0],  # max=3, mean=2
            hand_mean_submovement_amplitude=2.0,
            elbow_submovements=1,
            elbow_mean_submovement_amplitude=1.0,
            hand_peak_speed=1, hand_mean_speed=1,
            hand_peak_acceleration=1, hand_peak_deceleration=1,
            hand_peak_jerk=1, elbow_peak_speed=1, elbow_mean_speed=1,
            elbow_peak_acceleration=1, elbow_peak_deceleration=1,
            elbow_peak_jerk=1
        )

        df = _kinematic_features_to_dataframe([features])

        assert df.iloc[0]['hand_submovement_peak_max'] == 3.0
        assert df.iloc[0]['hand_submovement_peak_mean'] == 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
