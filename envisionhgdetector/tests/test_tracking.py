# tests/test_tracking.py
"""Unit tests for tracking module (pose tracking and feature extraction)."""

import numpy as np
import pytest

from envisionhgdetector.tracking import (
    extract_upper_limb_features,
    remove_nans,
    TrackingError,
)


class TestExtractUpperLimbFeatures:
    """Tests for extract_upper_limb_features function."""

    def test_basic_extraction(self):
        """Test basic feature extraction from landmarks."""
        # Create mock landmarks: 10 frames, 33 keypoints, 3 coordinates
        landmarks = np.random.randn(10, 33, 3)

        features = extract_upper_limb_features(landmarks)

        # Should return 2D array with correct number of frames
        assert features.ndim == 2
        assert features.shape[0] == 10

    def test_invalid_shape_raises(self):
        """Test that invalid landmark shapes raise TrackingError."""
        # 2D array instead of 3D
        landmarks_2d = np.random.randn(10, 99)
        with pytest.raises(TrackingError):
            extract_upper_limb_features(landmarks_2d)

        # Wrong last dimension (should be 3 for x,y,z)
        landmarks_wrong = np.random.randn(10, 33, 4)
        with pytest.raises(TrackingError):
            extract_upper_limb_features(landmarks_wrong)

    def test_output_shape_consistency(self):
        """Test that output shape is consistent across different inputs."""
        # Different number of frames should produce same feature dimension
        landmarks_10 = np.random.randn(10, 33, 3)
        landmarks_50 = np.random.randn(50, 33, 3)

        features_10 = extract_upper_limb_features(landmarks_10)
        features_50 = extract_upper_limb_features(landmarks_50)

        # Feature dimension should be the same
        assert features_10.shape[1] == features_50.shape[1]

    def test_extracts_upper_body_joints(self):
        """Test that upper body joints are extracted."""
        # Create landmarks with known values
        landmarks = np.zeros((5, 33, 3))

        # Set specific values for upper body joints
        # Left shoulder (index 11)
        landmarks[:, 11, :] = [1, 2, 3]
        # Right shoulder (index 12)
        landmarks[:, 12, :] = [4, 5, 6]
        # Left wrist (index 15)
        landmarks[:, 15, :] = [7, 8, 9]

        features = extract_upper_limb_features(landmarks)

        # Features should contain these values (flattened)
        # Note: exact positions depend on ordering in the function
        assert features.shape[0] == 5
        assert features.shape[1] > 0


class TestRemoveNans:
    """Tests for remove_nans function."""

    def test_replaces_nans_with_zeros(self):
        """Test that NaN values are replaced with zeros."""
        features = np.array([
            [1.0, np.nan, 3.0],
            [np.nan, 5.0, np.nan]
        ])

        result = remove_nans(features)

        expected = np.array([
            [1.0, 0.0, 3.0],
            [0.0, 5.0, 0.0]
        ])

        np.testing.assert_array_equal(result, expected)

    def test_preserves_non_nan_values(self):
        """Test that non-NaN values are preserved."""
        features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = remove_nans(features)

        np.testing.assert_array_equal(result, features)

    def test_handles_all_nan(self):
        """Test handling of array with all NaN values."""
        features = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        result = remove_nans(features)

        expected = np.zeros((2, 2))
        np.testing.assert_array_equal(result, expected)

    def test_handles_empty_array(self):
        """Test handling of empty array."""
        features = np.array([])
        result = remove_nans(features)

        assert len(result) == 0

    def test_preserves_shape(self):
        """Test that output shape matches input shape."""
        features = np.random.randn(10, 5)
        features[3, 2] = np.nan

        result = remove_nans(features)

        assert result.shape == features.shape


class TestProcessHandFingers:
    """Tests for _process_hand_fingers helper function."""

    def test_mean_centering(self):
        """Test that finger features are mean-centered."""
        from envisionhgdetector.tracking import _process_hand_fingers

        # Create landmarks with known finger positions
        landmarks = np.zeros((5, 33, 3))

        # Set finger positions for left hand (indices 17, 19, 21)
        landmarks[:, 17, :] = [1, 1, 1]  # pinky
        landmarks[:, 19, :] = [2, 2, 2]  # index
        landmarks[:, 21, :] = [3, 3, 3]  # thumb

        result = _process_hand_fingers(landmarks, [17, 19, 21])

        # Mean should be [2, 2, 2], so after centering:
        # pinky: [-1, -1, -1], index: [0, 0, 0], thumb: [1, 1, 1]
        assert result is not None
        # Check that result is approximately mean-centered
        mean_per_frame = np.mean(result.reshape(5, 3, 3), axis=1)
        np.testing.assert_array_almost_equal(mean_per_frame, 0, decimal=10)

    def test_returns_none_for_nan_data(self):
        """Test that None is returned when finger data contains NaN."""
        from envisionhgdetector.tracking import _process_hand_fingers

        landmarks = np.full((5, 33, 3), np.nan)

        result = _process_hand_fingers(landmarks, [17, 19, 21])

        assert result is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
