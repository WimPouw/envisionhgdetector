import cv2
import tempfile
import imageio
import numpy as np
from typing import Optional, Literal
from pathlib import Path

from envisionhgdetector.analysis.pose_extraction import extract_pose_from_video
from envisionhgdetector.utils import return_path
from .helpers import fix_codex

# https://github.com/maskbench/maskbench/blob/envision_gesture_challenge/src/keypoint_pairs.py
MEDIAPIPE_KEYPOINT_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 19), (15, 17), (17, 19), (15, 21),
    (12, 14), (14, 16), (16, 20), (16, 18), (18, 20), (11, 23), (12, 24), (16, 22),
    (23, 25), (24, 26), (25, 27), (26, 28), (23, 24), (28, 30), (28, 32), (30, 32),
    (27, 29), (27, 31), (29, 31)
]

def render_pose_video(
    video_path: str,
    output_path: str, # full MP4 output path
    logo_path: Optional[str] = None, # optionally add a logo to the output video
    logo_position: Optional[Literal["top-left", "top-right", "bottom-left", "bottom-right"]] = "top-right", # position of the logo
    only_skeleton: bool = False, # only skeleton result or side by side of original and skeleton
    make_gif: bool = False, # also save a GIF preview
    gif_start: float = 0,
    gif_end: Optional[float] = 5,
    gif_fps: float = 10,
):
    '''
    Renders a full MP4 with pose landmarks and optionally saves a GIF preview.
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the full MP4 video.
        logo_path (Optional[str]): Path to the logo image file. If None, no logo will be added.
        logo_position (Optional[Literal["top-left", "top-right", "bottom-left", "bottom-right"]]): Position of the logo in the output video. Default is "top-right".
        only_skeleton (bool): If True, only the skeleton will be rendered. If False, the output will be a side-by-side comparison of the original video and the skeleton. Default is False.    
        make_gif (bool): Also save a GIF alongside the MP4. Default is False.
        gif_start (float): GIF start time in seconds. Default is 0.
        gif_end (Optional[float]): Exclusive GIF end time in seconds. None uses the video end. Default is 5.
        gif_fps (float): GIF sampling rate, between 1 and 100. Default is 10.
    '''
    video_path = Path(video_path)
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("output_path must have an .mp4 extension.")
    if output_path.is_file():
        print(f"Warning: Output file already exists, replacing: {output_path}")
    if output_path.resolve() == video_path.resolve():
        raise ValueError("Output path must differ from the input video path.")

    video_name = video_path.stem
    cap = None
    video_writer = None
    try:
        landmarks = get_landmarks(str(video_path), video_name)
        if landmarks is None:
            raise ValueError("No landmarks were extracted from the video.")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_width = width if only_skeleton else width * 2
        fps = cap.get(cv2.CAP_PROP_FPS)

        video_writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, height))
        if not video_writer.isOpened():
            raise ValueError(f"Could not open video writer for output file: {output_path}")

        logo = get_logo(logo_path, output_width, height, logo_position, margin=10, scale=0.3)
        add_landmarks(landmarks, cap, video_writer, width, height, only_skeleton, logo)

        cap.release()
        video_writer.release()

        fix_codex(output_path, output_path)  # Fix potential codec issues for MP4 files

        if make_gif:
            gif_path = str(output_path.with_suffix(".gif"))
            try:
                create_gif_from_video(str(output_path), gif_path, gif_start, gif_end, gif_fps)
            except Exception as e:
                print(f"Error occurred while creating GIF: {e}")
    except Exception as e:
        raise ValueError(f"Error occurred while rendering pose video: {e}")
    finally:
        if cap is not None:
            cap.release()
        if video_writer is not None:
            video_writer.release()

    return

def add_landmarks(landmarks: list, cap: cv2.VideoCapture, video_writer: cv2.VideoWriter, width: int, height: int, only_skeleton: bool, logo: Optional[tuple]):
    line_color = (0, 255, 255)  # Yellow color for lines
    circle_color = (255, 255, 0)  # Cyan color for circles
    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_copy = np.zeros_like(frame)  # Create a blank copy of the original frame
            frame_landmarks = landmarks[frame_idx] # we assume we have an entry (empty list or landmarks) for each frame in the video
            for landmark in frame_landmarks: # add all landmarks for this frame
                if landmark is None or len(landmark) < 2:
                    continue  # Skip if the landmark is None or does not have enough coordinates
                # get pixel coordinates of the landmark
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                cv2.circle(frame_copy, (x, y), radius=5, color=circle_color, thickness=-1)

            for pair in MEDIAPIPE_KEYPOINT_PAIRS: # draw lines between the landmarks
                start_idx, end_idx = pair
                if start_idx < len(frame_landmarks) and end_idx < len(frame_landmarks):
                    start_landmark = frame_landmarks[start_idx]
                    end_landmark = frame_landmarks[end_idx]
                    if start_landmark is not None and end_landmark is not None:
                        start_x = int(start_landmark[0] * width)
                        start_y = int(start_landmark[1] * height)
                        end_x = int(end_landmark[0] * width)
                        end_y = int(end_landmark[1] * height)
                        cv2.line(frame_copy, (start_x, start_y), (end_x, end_y), color=line_color, thickness=2)  # Draw a line between the landmarks
        
            if not only_skeleton:
                frame_copy = np.hstack((frame, frame_copy))  # Side by side of original and skeleton

            frame = add_logo(frame_copy, logo)
            video_writer.write(frame)
            frame_idx += 1
    except Exception as e:
        raise ValueError(f"Error occurred while adding landmarks: {e}")

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

def get_landmarks(video_path: str, video_name: str):
    landmarks = None
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract pose from the video and save the results in the temporary directory
            extract_pose_from_video(video_name=video_name, video_path=video_path, data_folder=temp_dir)
            image_landmarks_save_path = return_path("image_landmarks", temp_dir, video_name)
            if not Path(image_landmarks_save_path).exists():
                raise ValueError(f"Image landmarks file does not exist: {image_landmarks_save_path}")
            landmarks = np.load(image_landmarks_save_path, allow_pickle=False)  # Load the image landmarks to ensure they are valid
        except Exception as e:
            print(f"Error occurred while extracting landmarks: {e}")
            return None

    if landmarks is None or landmarks.size == 0:
        print("Warning: No landmarks were extracted from the video.")
        return None

    return landmarks

def get_logo(logo_path: str, frame_width: int, frame_height: int, logo_position: str, margin: int, scale: float = 1.0):
    if scale <= 0 or scale > 1 or not np.isfinite(scale):
        raise ValueError("Scale must be a positive value between 0 and 1.")
    
    if logo_path is None:
        return None

    if not Path(logo_path).is_file():
        print(f"Warning: Logo file does not exist: {logo_path}")
        return None

    logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
    if logo is None:
        print(f"Warning: Could not read logo image from {logo_path}.")
        return None

    if logo.dtype != np.uint8:
        print("Warning: Logo must be an 8-bit image.")
        return None

    if logo.ndim == 2: # grayscale image
        logo = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGR)

    logo_height, logo_width = logo.shape[:2]
    logo = cv2.resize(logo, dsize=(
            max(1, round(logo_width * scale)),
            max(1, round(logo_height * scale)),
        ), interpolation=cv2.INTER_AREA)
    logo_height, logo_width = logo.shape[:2]

    positions = {
        "top-left": (margin, margin),
        "top-right": (frame_width - logo_width - margin, margin),
        "bottom-left": (margin, frame_height - logo_height - margin),
        "bottom-right": (
            frame_width - logo_width - margin,
            frame_height - logo_height - margin,
        ),
    }

    if logo_position not in positions:
        print(f"Warning: Invalid logo position '{logo_position}'.")
        return None

    x, y = positions[logo_position]
    x1, y1 = max(0, x), max(0, y)
    x2 = min(frame_width, x + logo_width)
    y2 = min(frame_height, y + logo_height)

    if x1 >= x2 or y1 >= y2:
        print("Warning: Logo is outside the frame.")
        return None

    logo = logo[y1 - y:y2 - y, x1 - x:x2 - x]
    colors = logo[:, :, :3]

    alpha = (
        logo[:, :, 3:4].astype(np.float32) / 255.0
        if logo.shape[2] == 4
        else None
    )

    return colors, alpha, (x1, y1, x2, y2)

def add_logo(frame, logo_data):
    if logo_data is None:
        return frame

    colors, alpha, (x1, y1, x2, y2) = logo_data
    region = frame[y1:y2, x1:x2]

    if alpha is None:
        region[:] = colors
    else:
        region[:] = (
            colors * alpha + region * (1 - alpha)
        ).astype(np.uint8)

    return frame