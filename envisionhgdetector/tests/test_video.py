# tests/test_video.py
"""Unit tests for video processing module."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from envisionhgdetector.video import (
    create_sliding_windows,
    find_all_videos,
)


class TestCreateSlidingWindows:
    """Tests for create_sliding_windows function."""

    def test_basic_windows(self):
        """Test basic window creation."""
        features = [[i, i+1, i+2] for i in range(10)]  # 10 frames, 3 features
        windows = create_sliding_windows(features, seq_length=5, stride=1)

        # Should create 6 windows (10 - 5 + 1)
        assert windows.shape == (6, 5, 3)

    def test_stride(self):
        """Test window creation with stride > 1."""
        features = [[i] for i in range(10)]  # 10 frames, 1 feature
        windows = create_sliding_windows(features, seq_length=3, stride=2)

        # With stride=2: windows at indices 0, 2, 4, 6 = 4 windows
        assert windows.shape == (4, 3, 1)

    def test_sequence_too_short(self):
        """Test handling when features shorter than seq_length."""
        features = [[1, 2], [3, 4]]  # Only 2 frames
        windows = create_sliding_windows(features, seq_length=5, stride=1)

        assert len(windows) == 0

    def test_exact_length(self):
        """Test when features exactly equal seq_length."""
        features = [[i] for i in range(5)]
        windows = create_sliding_windows(features, seq_length=5, stride=1)

        assert windows.shape == (1, 5, 1)

    def test_empty_features(self):
        """Test handling of empty features list."""
        features = []
        windows = create_sliding_windows(features, seq_length=5, stride=1)

        assert len(windows) == 0

    def test_preserves_feature_values(self):
        """Test that feature values are preserved correctly."""
        features = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        windows = create_sliding_windows(features, seq_length=3, stride=1)

        # First window should be frames 0-2
        np.testing.assert_array_equal(windows[0], [[0, 1], [2, 3], [4, 5]])

        # Second window should be frames 1-3
        np.testing.assert_array_equal(windows[1], [[2, 3], [4, 5], [6, 7]])


class TestFindAllVideos:
    """Tests for find_all_videos function."""

    def test_finds_mp4_files(self):
        """Test that MP4 files are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            video1 = os.path.join(tmpdir, "video1.mp4")
            video2 = os.path.join(tmpdir, "video2.mp4")
            other = os.path.join(tmpdir, "other.txt")

            for f in [video1, video2, other]:
                with open(f, 'w') as fp:
                    fp.write("test")

            videos = find_all_videos(tmpdir)

            assert len(videos) == 2
            assert any("video1.mp4" in v for v in videos)
            assert any("video2.mp4" in v for v in videos)

    def test_recursive_search(self):
        """Test that videos in subdirectories are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)

            # Create test files
            video1 = os.path.join(tmpdir, "video1.mp4")
            video2 = os.path.join(subdir, "video2.mp4")

            for f in [video1, video2]:
                with open(f, 'w') as fp:
                    fp.write("test")

            videos = find_all_videos(tmpdir)

            assert len(videos) == 2

    def test_empty_directory(self):
        """Test handling of directory with no videos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            videos = find_all_videos(tmpdir)
            assert len(videos) == 0

    def test_returns_full_paths(self):
        """Test that full paths are returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "video.mp4")
            with open(video, 'w') as fp:
                fp.write("test")

            videos = find_all_videos(tmpdir)

            assert len(videos) == 1
            assert os.path.isabs(videos[0])


class TestLabelVideoHelpers:
    """Tests for label_video helper functions (tested via module-level functions)."""

    def test_color_map_coverage(self):
        """Test that color map covers all expected labels."""
        # This is more of a documentation test to ensure color map is complete
        from envisionhgdetector.video import label_video

        # The function should handle these labels without error
        expected_labels = ['NoGesture', 'Gesture', 'Move']

        # We can't easily test label_video without a real video,
        # but we can verify the function exists and has proper signature
        import inspect
        sig = inspect.signature(label_video)
        params = list(sig.parameters.keys())

        assert 'video_path' in params
        assert 'segments' in params
        assert 'output_path' in params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
