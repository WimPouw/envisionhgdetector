"""Read input video metadata."""

import cv2

def get_video_fps(video_path: str) -> int:
	"""Get video FPS."""
	cap = cv2.VideoCapture(video_path)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	cap.release()
	return fps

