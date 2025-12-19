# tests/test_config.py
"""Unit tests for config module."""

import pytest

from envisionhgdetector.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_default_values(self):
        """Test that default configuration values are set."""
        config = Config()

        assert config.seq_length == 25
        assert config.default_motion_threshold == 0.7
        assert config.default_gesture_threshold == 0.7
        assert config.default_min_gap_s == 0.5
        assert config.default_min_length_s == 0.5

    def test_gesture_labels(self):
        """Test that gesture labels are properly defined."""
        config = Config()

        assert 'Gesture' in config.gesture_labels
        assert 'Move' in config.gesture_labels
        # NoGesture is stored in stationary_label, not gesture_labels
        assert config.stationary_label == 'NoGesture'

    def test_is_model_available(self):
        """Test model availability validation."""
        config = Config()

        # Should return True or False without raising
        cnn_available = config.is_model_available('cnn')
        lgbm_available = config.is_model_available('lightgbm')

        assert isinstance(cnn_available, bool)
        assert isinstance(lgbm_available, bool)

    def test_invalid_model_type(self):
        """Test validation with invalid model type."""
        config = Config()

        # Invalid model type should raise ValueError
        with pytest.raises(ValueError):
            config.is_model_available('invalid_model')

    def test_str_representation(self):
        """Test string representation of config."""
        config = Config()

        str_repr = str(config)
        assert 'Config' in str_repr or 'seq_length' in str_repr.lower()


class TestConfigPaths:
    """Tests for Config path handling."""

    def test_weights_path_attribute(self):
        """Test that weights_path attribute exists."""
        config = Config()

        # Should have weights_path attribute (may be None if not found)
        assert hasattr(config, 'weights_path')

    def test_lightgbm_model_path_attribute(self):
        """Test that lightgbm_weights_path attribute exists."""
        config = Config()

        assert hasattr(config, 'lightgbm_weights_path')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
