# tests/test_error_handling.py
"""Tests for error handling and input validation across all modules."""

import tempfile
import os

import numpy as np
import pandas as pd
import pytest


class TestSegmentationErrors:
    """Tests for segmentation module error handling."""

    def test_invalid_annotations_type(self):
        """Test error when annotations is not a DataFrame."""
        from envisionhgdetector.segmentation import create_segments, SegmentationError

        with pytest.raises(SegmentationError, match="must be a pandas DataFrame"):
            create_segments("not a dataframe", "label")

        with pytest.raises(SegmentationError, match="must be a pandas DataFrame"):
            create_segments([1, 2, 3], "label")

    def test_missing_time_column(self):
        """Test error when time column is missing."""
        from envisionhgdetector.segmentation import create_segments, SegmentationError

        df = pd.DataFrame({'label': ['Gesture', 'NoGesture']})
        with pytest.raises(SegmentationError, match="must contain 'time' column"):
            create_segments(df, 'label')

    def test_missing_label_column(self):
        """Test error when label column is missing."""
        from envisionhgdetector.segmentation import create_segments, SegmentationError

        df = pd.DataFrame({'time': [0.0, 0.1]})
        with pytest.raises(SegmentationError, match="must contain 'gesture' column"):
            create_segments(df, 'gesture')

    def test_invalid_min_gap(self):
        """Test error when min_gap_s is invalid."""
        from envisionhgdetector.segmentation import create_segments, SegmentationError

        df = pd.DataFrame({'time': [0.0], 'label': ['Gesture']})

        with pytest.raises(SegmentationError, match="must be >= 0"):
            create_segments(df, 'label', min_gap_s=-1.0)

        with pytest.raises(SegmentationError, match="must be a number"):
            create_segments(df, 'label', min_gap_s="invalid")

    def test_invalid_threshold_values(self):
        """Test error when threshold values are out of range."""
        from envisionhgdetector.segmentation import get_prediction_at_threshold, SegmentationError

        row = pd.Series({
            'NoGesture_confidence': 0.5,
            'Gesture_confidence': 0.3,
            'Move_confidence': 0.2
        })

        with pytest.raises(SegmentationError, match="must be <= 1.0"):
            get_prediction_at_threshold(row, motion_threshold=1.5)

        with pytest.raises(SegmentationError, match="must be >= 0"):
            get_prediction_at_threshold(row, gesture_threshold=-0.5)

    def test_missing_confidence_columns(self):
        """Test error when confidence columns are missing."""
        from envisionhgdetector.segmentation import get_prediction_at_threshold, SegmentationError

        row = pd.Series({'NoGesture_confidence': 0.5})  # Missing Gesture and Move

        with pytest.raises(SegmentationError, match="missing required confidence columns"):
            get_prediction_at_threshold(row)


class TestVideoErrors:
    """Tests for video module error handling."""

    def test_invalid_video_path_type(self):
        """Test error when video_path is not a string."""
        from envisionhgdetector.video import validate_video_path, VideoProcessingError

        with pytest.raises(VideoProcessingError, match="must be a string"):
            validate_video_path(123)

        with pytest.raises(VideoProcessingError, match="must be a string"):
            validate_video_path(None)

    def test_empty_video_path(self):
        """Test error when video_path is empty."""
        from envisionhgdetector.video import validate_video_path, VideoProcessingError

        with pytest.raises(VideoProcessingError, match="cannot be empty"):
            validate_video_path("")

        with pytest.raises(VideoProcessingError, match="cannot be empty"):
            validate_video_path("   ")

    def test_video_file_not_found(self):
        """Test error when video file doesn't exist."""
        from envisionhgdetector.video import validate_video_path, VideoProcessingError

        with pytest.raises(VideoProcessingError, match="not found"):
            validate_video_path("/nonexistent/path/video.mp4")

    def test_invalid_sliding_window_params(self):
        """Test error when sliding window params are invalid."""
        from envisionhgdetector.video import create_sliding_windows, VideoProcessingError

        features = [[1, 2], [3, 4], [5, 6]]

        with pytest.raises(VideoProcessingError, match="must be a positive integer"):
            create_sliding_windows(features, seq_length=0)

        with pytest.raises(VideoProcessingError, match="must be a positive integer"):
            create_sliding_windows(features, seq_length=-1)

        with pytest.raises(VideoProcessingError, match="must be a positive integer"):
            create_sliding_windows(features, seq_length=2, stride=0)

        with pytest.raises(VideoProcessingError, match="cannot be None"):
            create_sliding_windows(None, seq_length=2)

    def test_invalid_folder_for_find_videos(self):
        """Test error when folder is invalid."""
        from envisionhgdetector.video import find_all_videos, VideoProcessingError

        with pytest.raises(VideoProcessingError, match="must be a string"):
            find_all_videos(123)

        with pytest.raises(VideoProcessingError, match="not found"):
            find_all_videos("/nonexistent/folder")


class TestKinematicsErrors:
    """Tests for kinematics module error handling."""

    def test_invalid_positions_type(self):
        """Test error when positions is not a numpy array."""
        from envisionhgdetector.kinematics import calculate_derivatives, KinematicsError

        with pytest.raises(KinematicsError, match="must be a numpy array"):
            calculate_derivatives([[1, 2, 3]], fps=25.0)

    def test_empty_positions(self):
        """Test error when positions array is empty."""
        from envisionhgdetector.kinematics import calculate_derivatives, KinematicsError

        with pytest.raises(KinematicsError, match="cannot be empty"):
            calculate_derivatives(np.array([]), fps=25.0)

    def test_invalid_positions_shape(self):
        """Test error when positions has wrong shape."""
        from envisionhgdetector.kinematics import calculate_derivatives, KinematicsError

        # 1D array
        with pytest.raises(KinematicsError, match="must have shape"):
            calculate_derivatives(np.array([1, 2, 3]), fps=25.0)

        # Wrong last dimension
        positions = np.random.randn(10, 4)  # Should be (N, 3)
        with pytest.raises(KinematicsError, match="must have shape"):
            calculate_derivatives(positions, fps=25.0)

    def test_invalid_fps(self):
        """Test error when fps is invalid."""
        from envisionhgdetector.kinematics import calculate_derivatives, KinematicsError

        positions = np.random.randn(10, 3)

        with pytest.raises(KinematicsError, match="must be positive"):
            calculate_derivatives(positions, fps=0)

        with pytest.raises(KinematicsError, match="must be positive"):
            calculate_derivatives(positions, fps=-25)

        with pytest.raises(KinematicsError, match="must be a number"):
            calculate_derivatives(positions, fps="invalid")

    def test_insufficient_frames(self):
        """Test error when not enough frames for derivatives."""
        from envisionhgdetector.kinematics import calculate_derivatives, KinematicsError

        positions = np.random.randn(1, 3)  # Only 1 frame

        with pytest.raises(KinematicsError, match="at least 2 frames"):
            calculate_derivatives(positions, fps=25.0)


class TestTrackingErrors:
    """Tests for tracking module error handling."""

    def test_invalid_landmarks_type(self):
        """Test error when landmarks is not a numpy array."""
        from envisionhgdetector.tracking import extract_upper_limb_features, TrackingError

        with pytest.raises(TrackingError, match="must be a numpy array"):
            extract_upper_limb_features([[1, 2, 3]])

    def test_empty_landmarks(self):
        """Test error when landmarks is empty."""
        from envisionhgdetector.tracking import extract_upper_limb_features, TrackingError

        with pytest.raises(TrackingError, match="cannot be empty"):
            extract_upper_limb_features(np.array([]))

    def test_invalid_landmarks_dimensions(self):
        """Test error when landmarks has wrong dimensions."""
        from envisionhgdetector.tracking import extract_upper_limb_features, TrackingError

        # 2D array instead of 3D
        landmarks_2d = np.random.randn(10, 99)
        with pytest.raises(TrackingError, match="must be a 3D array"):
            extract_upper_limb_features(landmarks_2d)

        # Wrong last dimension
        landmarks_wrong = np.random.randn(10, 33, 4)
        with pytest.raises(TrackingError, match="must be 3"):
            extract_upper_limb_features(landmarks_wrong)

    def test_insufficient_keypoints(self):
        """Test error when not enough keypoints."""
        from envisionhgdetector.tracking import extract_upper_limb_features, TrackingError

        landmarks = np.random.randn(10, 10, 3)  # Only 10 keypoints, need 23

        with pytest.raises(TrackingError, match="at least 23 keypoints"):
            extract_upper_limb_features(landmarks)

    def test_remove_nans_invalid_type(self):
        """Test error when features is not numpy array."""
        from envisionhgdetector.tracking import remove_nans, TrackingError

        with pytest.raises(TrackingError, match="must be a numpy array"):
            remove_nans([[1, 2], [3, 4]])


class TestDTWAnalysisErrors:
    """Tests for DTW analysis module error handling."""

    def test_invalid_dtw_matrix_type(self):
        """Test error when dtw_matrix is not a numpy array."""
        from envisionhgdetector.dtw_analysis import create_gesture_visualization, DTWAnalysisError

        with pytest.raises(DTWAnalysisError, match="must be a numpy array"):
            create_gesture_visualization([[1, 2], [2, 1]], ['g1', 'g2'], '/tmp')

    def test_empty_dtw_matrix(self):
        """Test error when dtw_matrix is empty."""
        from envisionhgdetector.dtw_analysis import create_gesture_visualization, DTWAnalysisError

        with pytest.raises(DTWAnalysisError, match="cannot be empty"):
            create_gesture_visualization(np.array([]), [], '/tmp')

    def test_non_square_dtw_matrix(self):
        """Test error when dtw_matrix is not square."""
        from envisionhgdetector.dtw_analysis import create_gesture_visualization, DTWAnalysisError

        matrix = np.random.randn(3, 4)  # Not square

        with pytest.raises(DTWAnalysisError, match="must be square"):
            create_gesture_visualization(matrix, ['g1', 'g2', 'g3'], '/tmp')

    def test_mismatched_gesture_names(self):
        """Test error when gesture_names doesn't match matrix size."""
        from envisionhgdetector.dtw_analysis import create_gesture_visualization, DTWAnalysisError

        matrix = np.zeros((3, 3))

        with pytest.raises(DTWAnalysisError, match="must match"):
            create_gesture_visualization(matrix, ['g1', 'g2'], '/tmp')  # Only 2 names

    def test_too_few_gestures(self):
        """Test error when less than 3 gestures (UMAP requires n_neighbors >= 2)."""
        from envisionhgdetector.dtw_analysis import create_gesture_visualization, DTWAnalysisError

        matrix = np.array([[0.0]])

        with pytest.raises(DTWAnalysisError, match="at least 3 gestures"):
            create_gesture_visualization(matrix, ['g1'], '/tmp')

    def test_invalid_n_neighbors(self):
        """Test error when n_neighbors is invalid."""
        from envisionhgdetector.dtw_analysis import create_gesture_visualization, DTWAnalysisError

        matrix = np.zeros((3, 3))

        with pytest.raises(DTWAnalysisError, match="n_neighbors must be an integer >= 2"):
            create_gesture_visualization(matrix, ['g1', 'g2', 'g3'], '/tmp', n_neighbors=0)

        with pytest.raises(DTWAnalysisError, match="n_neighbors must be an integer >= 2"):
            create_gesture_visualization(matrix, ['g1', 'g2', 'g3'], '/tmp', n_neighbors=-1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
