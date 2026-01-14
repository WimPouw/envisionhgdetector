# tests/conftest.py
"""Pytest configuration and shared fixtures."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_landmarks():
    """Generate sample pose landmarks for testing."""
    # 50 frames, 33 keypoints, 3 coordinates (x, y, z)
    np.random.seed(42)
    return np.random.randn(50, 33, 3)


@pytest.fixture
def sample_segments_df():
    """Generate sample segments DataFrame for testing."""
    return pd.DataFrame({
        'start_time': [1.0, 5.0, 10.0],
        'end_time': [3.0, 8.0, 15.0],
        'labelid': [1, 2, 3],
        'label': ['Gesture', 'Move', 'Gesture'],
        'duration': [2.0, 3.0, 5.0]
    })


@pytest.fixture
def sample_predictions_df():
    """Generate sample predictions DataFrame for testing."""
    times = np.arange(0, 10, 0.1)
    n = len(times)

    return pd.DataFrame({
        'time': times,
        'has_motion': np.random.rand(n),
        'NoGesture_confidence': np.random.rand(n),
        'Gesture_confidence': np.random.rand(n),
        'Move_confidence': np.random.rand(n),
        'label': np.random.choice(['Gesture', 'Move', 'NoGesture'], n)
    })


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_dtw_matrix():
    """Generate sample DTW distance matrix for testing."""
    n = 5
    matrix = np.random.rand(n, n) * 10
    # Make symmetric with zeros on diagonal
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    return matrix


@pytest.fixture
def sample_gesture_names():
    """Generate sample gesture names for testing."""
    return ['gesture_a', 'gesture_b', 'gesture_c', 'gesture_d', 'gesture_e']


@pytest.fixture
def sample_kinematic_features():
    """Generate sample KinematicFeatures object for testing."""
    from envisionhgdetector.kinematics import KinematicFeatures

    return KinematicFeatures(
        gesture_id='test_gesture_001',
        video_id='test_video',
        active_hand='L',
        space_use=5,
        mcneillian_max=3.0,
        mcneillian_mode=2,
        volume=150.0,
        max_height=2.5,
        duration=1.8,
        hold_count=2,
        hold_time=0.5,
        hold_avg_duration=0.25,
        hand_submovements=4,
        hand_submovement_peaks=[0.5, 0.8, 0.6, 0.9],
        hand_mean_submovement_amplitude=0.7,
        elbow_submovements=2,
        elbow_mean_submovement_amplitude=0.4,
        hand_peak_speed=1.8,
        hand_mean_speed=0.9,
        hand_peak_acceleration=2.5,
        hand_peak_deceleration=-2.0,
        hand_peak_jerk=4.0,
        elbow_peak_speed=1.2,
        elbow_mean_speed=0.6,
        elbow_peak_acceleration=1.8,
        elbow_peak_deceleration=-1.5,
        elbow_peak_jerk=2.5
    )
