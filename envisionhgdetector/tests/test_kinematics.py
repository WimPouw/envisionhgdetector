# tests/test_kinematics.py
"""Unit tests for kinematics module."""

import pytest
import numpy as np

from envisionhgdetector.kinematics import (
    calculate_derivatives,
    find_submovements,
    compute_limb_kinematics,
    ArmKinematics,
    KinematicsError,
)


class TestCalculateDerivatives:
    """Tests for calculate_derivatives function."""

    def test_basic_derivatives(self):
        """Test basic derivative calculation."""
        # Linear motion: positions increase linearly
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ])
        fps = 10.0

        velocity, acceleration, jerk = calculate_derivatives(positions, fps)

        # Velocity should be approximately constant (10 units/s in x)
        assert velocity.shape == positions.shape
        assert acceleration.shape == positions.shape
        assert jerk.shape == positions.shape

    def test_empty_positions_raises(self):
        """Test that empty positions raises KinematicsError."""
        with pytest.raises(KinematicsError):
            calculate_derivatives(np.array([]), 25.0)

    def test_invalid_fps_raises(self):
        """Test that non-positive fps raises KinematicsError."""
        positions = np.array([[0, 0, 0], [1, 1, 1]])
        with pytest.raises(KinematicsError):
            calculate_derivatives(positions, 0.0)
        with pytest.raises(KinematicsError):
            calculate_derivatives(positions, -10.0)


class TestFindSubmovements:
    """Tests for find_submovements function."""

    def test_single_peak(self):
        """Test finding a single peak in speed profile."""
        # Create a simple speed profile with one peak
        speed = np.array([0, 1, 2, 3, 2, 1, 0])
        peaks, heights = find_submovements(speed, fps=25.0)

        assert len(peaks) >= 1
        assert len(heights) >= 1

    def test_multiple_peaks(self):
        """Test finding multiple peaks in speed profile."""
        # Create speed profile with two distinct peaks
        speed = np.concatenate([
            np.linspace(0, 5, 20),
            np.linspace(5, 0, 20),
            np.linspace(0, 4, 20),
            np.linspace(4, 0, 20)
        ])
        peaks, heights = find_submovements(speed, fps=25.0)

        assert len(peaks) >= 1  # Should find at least one peak

    def test_short_sequence(self):
        """Test handling of very short sequences."""
        speed = np.array([1, 2])
        peaks, heights = find_submovements(speed, fps=25.0)

        # Should return something valid even for short sequences
        assert len(peaks) >= 1
        assert len(heights) >= 1

    def test_empty_sequence(self):
        """Test handling of empty sequence."""
        speed = np.array([])
        peaks, heights = find_submovements(speed, fps=25.0)

        assert len(peaks) == 1
        assert peaks[0] == 0


class TestComputeLimbKinematics:
    """Tests for compute_limb_kinematics function."""

    def test_returns_arm_kinematics(self):
        """Test that function returns ArmKinematics namedtuple."""
        positions = np.random.randn(50, 3)  # 50 frames, 3D positions
        result = compute_limb_kinematics(positions, fps=25.0)

        assert isinstance(result, ArmKinematics)
        assert hasattr(result, 'velocity')
        assert hasattr(result, 'acceleration')
        assert hasattr(result, 'jerk')
        assert hasattr(result, 'speed')
        assert hasattr(result, 'peaks')
        assert hasattr(result, 'peak_heights')

    def test_output_shapes(self):
        """Test that output arrays have correct shapes."""
        num_frames = 100
        positions = np.random.randn(num_frames, 3)
        result = compute_limb_kinematics(positions, fps=25.0)

        assert result.velocity.shape == (num_frames, 3)
        assert result.acceleration.shape == (num_frames, 3)
        assert result.jerk.shape == (num_frames, 3)
        assert result.speed.shape == (num_frames,)

    def test_speed_is_positive(self):
        """Test that speed values are non-negative."""
        positions = np.random.randn(50, 3)
        result = compute_limb_kinematics(positions, fps=25.0)

        assert np.all(result.speed >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
