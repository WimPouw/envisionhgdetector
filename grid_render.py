import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from envisionhgdetector import GestureDetector, PostProcessing, label_video
from envisionhgdetector.utils import return_path
from envisionhgdetector.rendering.helpers import create_gif_from_video

def render_grid_pipeline(
    video_path: str,
    output_folder: str,
    make_gif: bool = False,
    detect_and_label: bool = True,
    retrack: bool = True,
    logo_path: Optional[str] = None,
    gif_start: float = 0,
    gif_end: Optional[float] = 5,
    gif_fps: float = 10,
    logo_scale: float = 0.75,
):
    """Run selected processing stages, save a grid MP4, and optionally a GIF.

    Disabled stages are omitted from the grid, even if their outputs already
    exist. Return a dictionary containing the generated output Paths.
    logo_scale (0 < scale <= 1) controls logo size relative to its grid cell.
    """
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    if not np.isfinite(logo_scale) or not 0 < logo_scale <= 1:
        raise ValueError("logo_scale must be finite, greater than zero, and at most 1.")
    if not video_path.is_file():
        raise ValueError(f"Input video does not exist: {video_path}")
    if logo_path is not None and not Path(logo_path).is_file():
        raise ValueError(f"Logo file does not exist: {logo_path}")
    if make_gif:
        if not np.isfinite(gif_start) or gif_start < 0:
            raise ValueError("gif_start must be finite and non-negative.")
        if gif_end is not None and (not np.isfinite(gif_end) or gif_end <= gif_start):
            raise ValueError("gif_end must be finite and greater than gif_start.")
        if not np.isfinite(gif_fps) or not 0 < gif_fps <= 100:
            raise ValueError("gif_fps must be greater than zero and at most 100.")

    video_name = video_path.stem
    video_output_folder = output_folder / video_name
    grid_path = video_output_folder / f"{video_name}_grid.mp4"
    gif_path = grid_path.with_suffix(".gif") if make_gif else None
    for path in (grid_path, gif_path):
        if path is not None and path.exists():
            raise ValueError(f"Output already exists: {path}")
        if path is not None and path.resolve() == video_path.resolve():
            raise ValueError("Output must differ from the input video.")
    video_output_folder.mkdir(parents=True, exist_ok=True)

    labeled_path = None
    retracked_path = None
    if detect_and_label:
        detector = GestureDetector(model_type="lightgbm")
        result = detector.process_video(str(video_path), str(output_folder))
        if result.get("error"):
            raise RuntimeError(f"Detection failed: {result.get('error')}")
        label_video(
            video_path=str(video_path),
            video_output_folder=str(video_output_folder),
        )
        labeled_path = Path(return_path("labeled_video", str(output_folder), video_name))
        if not labeled_path.is_file():
            raise RuntimeError(f"Labeling did not create the expected video: {labeled_path}")

    if retrack:
        post_processor = PostProcessing(data_folder=str(output_folder))
        post_processor.extract_pose_from_video(str(video_path), video_name=video_name, blank_frame=True)
        retracked_path = Path(return_path("retracked_video", str(output_folder), video_name))
        if not retracked_path.is_file():
            raise RuntimeError(f"Retracking did not create the expected video: {retracked_path}")

    grid_creator(
        output_path=str(grid_path), video_path=str(video_path),
        labeled_video_path=str(labeled_path) if labeled_path is not None else None,
        retracked_video_path=str(retracked_path) if retracked_path is not None else None,
        logo_path=logo_path, logo_scale=logo_scale,
    )
    if make_gif:
        create_gif_from_video(
            str(grid_path), str(gif_path), start=gif_start, end=gif_end, fps=gif_fps,
        )
    return {
        "grid_video": grid_path, "gif": gif_path,
        "labeled_video": labeled_path, "retracked_video": retracked_path,
    }

def grid_creator(
    output_path: str,
    video_path: Optional[str] = None,
    retracked_video_path: Optional[str] = None,
    labeled_video_path: Optional[str] = None,
    logo_path: Optional[str] = None,
    logo_scale: float = 1.0,
):
    """Stream synchronized videos and an optional static logo into a silent MP4.

    Cells fill in original, logo, labeled, retracked order. Return the output Path.
    Videos must have matching frame counts and FPS (within 0.001).
    logo_scale is a fraction of the maximum logo size that fits its cell,
    with 0 < logo_scale <= 1. The logo stays centered.
    """
    output_path = Path(output_path)
    if not np.isfinite(logo_scale) or not 0 < logo_scale <= 1:
        raise ValueError("logo_scale must be finite, greater than zero, and at most 1.")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Grid output must have an .mp4 extension.")
    if output_path.exists():
        raise ValueError(f"Output already exists: {output_path}")

    inputs = [
        ("video", video_path), ("logo", logo_path),
        ("labeled", labeled_video_path), ("retracked", retracked_video_path),
    ]
    inputs = [(kind, Path(path)) for kind, path in inputs if path is not None]
    if not any(kind != "logo" for kind, _ in inputs):
        raise ValueError("At least one video is required.")
    for _, path in inputs:
        if output_path.resolve() == path.resolve():
            raise ValueError("Output must differ from every input path.")
        if not path.is_file():
            raise ValueError(f"Input file does not exist: {path}")
    if not output_path.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {output_path.parent}")

    captures = {}
    writer = None
    try:
        metadata = []
        logo = None
        for kind, path in inputs:
            if kind == "logo":
                logo = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if logo is None or logo.dtype != np.uint8:
                    raise ValueError(f"Could not read an 8-bit logo image: {path}")
                if logo.ndim == 2:
                    logo = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGR)
                if logo.shape[2] not in (3, 4):
                    raise ValueError("Logo must be grayscale, BGR, or BGRA.")
                continue
            cap = cv2.VideoCapture(str(path))
            captures[kind] = cap
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {path}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError(f"Invalid video dimensions, FPS, or frame count: {path}")
            metadata.append((kind, width, height, fps, count))

        output_fps, frame_count = metadata[0][3:]
        for kind, _, _, fps, count in metadata[1:]:
            if abs(fps - output_fps) > 0.001 or count != frame_count:
                raise ValueError(f"""Video '{kind}' must match the other videos' FPS and frame count.\n
                Expected: {output_fps} FPS, {frame_count} frames. \n
                Found: {fps} FPS, {count} frames.""")
        # Even canvas dimensions work with the MP4 encoder.
        canvas_width = max(item[1] for item in metadata)
        canvas_height = max(item[2] for item in metadata)
        canvas_width += canvas_width % 2
        canvas_height += canvas_height % 2
        columns = 1 if len(inputs) == 1 else 2
        rows = 1 if len(inputs) <= 2 else 2

        cells = []
        for index, (kind, _) in enumerate(inputs):
            column, row = index % columns, index // columns
            x1 = column * canvas_width // columns
            x2 = (column + 1) * canvas_width // columns
            y1 = row * canvas_height // rows
            y2 = (row + 1) * canvas_height // rows
            if kind == "logo":
                source_height, source_width = logo.shape[:2]
            else:
                info = next(item for item in metadata if item[0] == kind)
                source_width, source_height = info[1:3]
            scale = min((x2 - x1) / source_width, (y2 - y1) / source_height)
            if kind == "logo":
                scale *= logo_scale
            fitted_width = max(1, min(x2 - x1, round(source_width * scale)))
            fitted_height = max(1, min(y2 - y1, round(source_height * scale)))
            x = x1 + (x2 - x1 - fitted_width) // 2
            y = y1 + (y2 - y1 - fitted_height) // 2
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            static_image = None
            if kind == "logo":
                static_image = cv2.resize(logo, (fitted_width, fitted_height), interpolation=interpolation)
                if static_image.shape[2] == 4:
                    alpha = static_image[:, :, 3:4].astype(np.float32) / 255
                    static_image = (static_image[:, :, :3] * alpha + (1 - alpha) * 255).astype(np.uint8) # white background
                    # static_image = (static_image[:, :, :3] * alpha).astype(np.uint8) # black background
            cells.append((kind, x, y, fitted_width, fitted_height, interpolation, static_image))

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps, (canvas_width, canvas_height),
        )
        if not writer.isOpened():
            raise ValueError(f"Could not open grid video writer: {output_path}")
        for frame_index in range(frame_count):
            canvas = np.full((canvas_height, canvas_width, 3), fill_value=255, dtype=np.uint8) # black: full(255), white: zeros
            for kind, x, y, width, height, interpolation, static_image in cells:
                if kind == "logo":
                    image = static_image
                else:
                    ret, frame = captures[kind].read()
                    if not ret:
                        raise ValueError(f"Video '{kind}' ended unexpectedly at frame {frame_index}.")
                    image = cv2.resize(frame, (width, height), interpolation=interpolation)
                canvas[y:y + height, x:x + width] = image
            writer.write(canvas)
        return output_path
    finally:
        for cap in captures.values():
            cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    render_grid_pipeline(
        video_path="path/to/input/video.mp4",
        output_folder="path/to/output/folder",
        make_gif=True,
        detect_and_label=True,
        retrack=True,
        logo_path="path/to/logo.png",
        gif_start=0,
        gif_end=5,
        gif_fps=10,
    )
