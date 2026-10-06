"""Shared helpers for single-model and comparison video rendering."""

import numpy as np

from envisionhgdetector.state import Labels, SegmentColumns


def get_label_at_time(time, segments):
    """Return the saved segment label, or NoGesture outside active segments."""
    matching = segments[
        (segments[SegmentColumns.START_TIME] <= time)
        & (segments[SegmentColumns.END_TIME] >= time)
    ]
    return matching[SegmentColumns.PREDICTION].iloc[0] if not matching.empty else Labels.NOGESTURE


def get_confidence_window(output_time, min_time, max_time, window_duration, history_fraction=0.8):
    """Calculate a positive graph span within the observed timestamp range."""
    if not np.isfinite(window_duration) or window_duration <= 0:
        raise ValueError("window_duration must be finite and positive.")
    if not np.isfinite(min_time) or not np.isfinite(max_time) or max_time <= min_time:
        raise ValueError("A confidence graph requires distinct timestamp bounds.")
    if output_time < min_time + window_duration * 0.2:
        return min_time, min(max_time, min_time + window_duration)
    elif output_time > max_time - window_duration * 0.2:
        return max(min_time, max_time - window_duration), max_time
    else:
        start = max(min_time, output_time - window_duration * history_fraction)
        return start, min(max_time, start + window_duration)


def validate_prediction_times(values):
    """Return numeric timestamps; reject missing, duplicate, or reversed times."""
    times = np.asarray(values, dtype=float)
    if times.ndim != 1:
        raise ValueError("Prediction timestamps must be one-dimensional.")
    if not np.isfinite(times).all() or (times < 0).any() or (np.diff(times) <= 0).any():
        raise ValueError("Prediction timestamps must be finite, nonnegative, and strictly increasing.")
    return times
