import logging
import subprocess
import json
import os

from dataclasses import dataclass
from pathlib import Path

@dataclass()
class ClipInfo:
    id: int
    label: str
    start: float
    end: float
    output_path: Path
    type: str = None

@dataclass
class GapInfo:
    start: float
    end: float

def get_video_dimensions(video_path: Path):
    """Get video dimensions using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        import json
        data = json.loads(result.stdout)
        
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                return int(stream['width']), int(stream['height'])
        
        return None, None
    except Exception as e:
        print(f"Error getting video dimensions: {e}")
        return None, None

def split_video_horizontally(input_path: Path, output_left: Path, output_right: Path):
    """
    Split video horizontally into left and right halves using ffmpeg.
    """
    try:
        # Get video dimensions
        width, height = get_video_dimensions(input_path)
        if width is None or height is None:
            raise Exception("Could not determine video dimensions")
        
        half_width = width // 2
        
        # Split left half (Front view)
        cmd_left = [
            'ffmpeg', '-i', input_path,
            '-vf', f'crop={half_width}:{height}:0:0',
            '-c:a', 'copy',  # Copy audio stream
            '-y',  # Overwrite output files
            output_left
        ]
        
        # Split right half (Side view)
        cmd_right = [
            'ffmpeg', '-i', input_path,
            '-vf', f'crop={half_width}:{height}:{half_width}:0',
            '-c:a', 'copy',  # Copy audio stream
            '-y',  # Overwrite output files
            output_right
        ]
        
        print(f"  Splitting left half to: {os.path.basename(output_left)}")
        subprocess.run(cmd_left, check=True, capture_output=True)
        
        print(f"  Splitting right half to: {os.path.basename(output_right)}")
        subprocess.run(cmd_right, check=True, capture_output=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  Error splitting video: {e}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def return_file_output_path(output_dir: Path, corpus_name: str, video_name: str, unique_id: str, label: str, label_type: str) -> Path:
    """Construct output file path for a clip"""
    sanitized_corpus_name = sanitize_filename_component(corpus_name)
    sanitized_video_name = sanitize_filename_component(video_name)
    sanitized_label = sanitize_filename_component(label)
    sanitized_label_type = sanitize_filename_component(label_type)

    filename = f"{sanitized_corpus_name}_{sanitized_video_name}_{unique_id}_{sanitized_label}_{sanitized_label_type}.mp4"
    return output_dir / filename

def sanitize_filename_component(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(value)).strip("_") or "unknown"

def extract_clip_with_padding(input_file_path: Path, output_file_path: Path, start_time: float, end_time: float, video_duration: float, padding=1.0):
    """Extract clip using ffmpeg with padding before and after"""
    try:
        # Add padding and ensure we don't go out of bounds
        padded_start = max(0, start_time - padding)
        padded_end = min(video_duration, end_time + padding)
        clip_duration = padded_end - padded_start
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(padded_start),
            '-i', input_file_path,
            '-t', str(clip_duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            output_file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Checks if video is > 1KB to ensure it's not an empty file
        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 1024:
            return
        else:
            print(f"Failed to create valid output file: {output_file_path}")
            print(f"FFmpeg output: {result.stderr}")
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            logging.error(f"Failed to extract clip: {output_file_path} (Start: {padded_start:.2f}s, End: {padded_end:.2f}s). FFmpeg output: {result.stderr}")
            return
            
    except Exception as e:
        print(f"Error extracting clip: {e}")
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        logging.error(f"Exception while extracting clip: {output_file_path} (Start: {padded_start:.2f}s, End: {padded_end:.2f}s). Exception: {e}")
        return

def get_video_info(video_path: Path):
    """Get video information using ffprobe"""
    try:
        # Get duration and fps using ffprobe
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-select_streams', 'v:0', 
            '-show_entries', 'stream=duration,r_frame_rate', 
            '-of', 'json', 
            video_path.name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        
        # Extract duration and fps
        duration = float(info['streams'][0]['duration'])
        
        # r_frame_rate is usually in the format "num/den"
        fps_str = info['streams'][0]['r_frame_rate']
        if '/' in fps_str:
            num, den = map(float, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
            
        return {'duration': duration, 'fps': fps}
    
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {'duration': 0, 'fps': 0}

def check_ffmpeg():
    """Check if ffmpeg is available for video processing."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
