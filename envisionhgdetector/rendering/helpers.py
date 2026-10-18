"""Shared helpers for single-model and comparison video rendering."""

import cv2
import shutil
import imageio
import tempfile
import numpy as np
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip

from envisionhgdetector.state import Labels, SegmentColumns

def create_gif_from_video(video_path: str, output_path: str, start: float, end: float | None, fps: float):
    """Save a GIF of [start, end) in seconds, sampled at the provided FPS.

    Set end=None to use the remainder of the video. Only sampled frames
    from the selected interval are retained in memory.
    """
    if not np.isfinite(start) or start < 0:
        raise ValueError("GIF start must be a finite, non-negative time in seconds.")
    if end is not None and (not np.isfinite(end) or end <= start):
        raise ValueError("GIF end must be finite and greater than start.")
    if not np.isfinite(fps) or not 0 < fps <= 100:
        raise ValueError("GIF FPS must be greater than zero and at most 100.")

    if Path(video_path).resolve() == Path(output_path).resolve():
        raise ValueError("GIF output must differ from the input video path.")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(source_fps) or source_fps <= 0 or frame_count <= 0:
            raise ValueError("Video has invalid FPS or frame count.")
        if source_fps < fps:
            print(f"Warning: GIF FPS ({fps}) is greater than video FPS ({source_fps}). GIF will be sampled at video FPS.")
            fps = source_fps

        video_duration = frame_count / source_fps
        stop = video_duration if end is None else min(end, video_duration)
        if start >= stop:
            raise ValueError("GIF start is beyond the end of the video.")

        frame_index = int(start * source_fps)
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise ValueError("Could not seek to the GIF start time.")
        frames_rgb = []
        durations_ms = []
        sample_index = 0
        while True:
            sample_time = start + sample_index / fps
            if sample_time >= stop:
                break
            target_index = int(sample_time * source_fps)
            while frame_index <= target_index:
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("Video ended before the requested GIF interval.")
                frame_index += 1
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Pillow uses milliseconds; GIF delays have 10 ms precision.
            delay = min(1 / fps, stop - sample_time) # for final frame, we may have a shorter delay than 1/fps
            durations_ms.append(max(10, round(delay * 100) * 10))
            sample_index += 1

        imageio.mimsave(
            output_path, frames_rgb, format="pillow",
            duration=durations_ms, loop=0,
        )
    except Exception as e:
        raise ValueError(f"Failed to create GIF: {e}")
    finally:
        cap.release()

def fix_codex(video_path: Path, output_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

    try:
        print(f"Converting: {video_path} to {output_path}")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_video_path = Path(temp_dir) / video_path.name
            video = VideoFileClip(str(video_path))
            video.write_videofile(str(temp_video_path), codec='libx264', audio=False, preset='ultrafast', fps=video.fps)
            video.close()
            # Move the temporary file to the desired output path
            shutil.copy2(temp_video_path, output_path)
    except Exception as e:
        print(f"Error converting {video_path} to {output_path}: {e}")

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
