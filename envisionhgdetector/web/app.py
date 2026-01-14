# envisionhgdetector/web/app.py
"""
Flask web application for EnvisionHG gesture detection.
Provides REST API endpoints for the frontend UI.
"""

import os
import json
import uuid
import tempfile
import time
import random
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Try to import gesture detection modules (may fail if TensorFlow not installed)
DEMO_MODE = False
try:
    import cv2
    from ..detector import GestureDetector
    from ..elan import create_elan_file
    from ..video import label_video
except ImportError as e:
    print(f"Warning: Running in DEMO MODE - {e}")
    print("Install TensorFlow and other dependencies for full functionality.")
    DEMO_MODE = True
    GestureDetector = None
    create_elan_file = None

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='envisionhg_')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# In-memory job storage (for production, use Redis or database)
jobs: Dict[str, Dict[str, Any]] = {}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/info')
def get_info():
    """Get server info including demo mode status."""
    return jsonify({
        'demo_mode': DEMO_MODE,
        'version': '2.0.0',
        'allowed_extensions': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_CONTENT_LENGTH // (1024 * 1024)
    })


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """
    Upload a video file for processing.

    Returns:
        JSON with job_id and file info
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8]

    # Save file
    filename = secure_filename(file.filename)
    job_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(job_folder, exist_ok=True)

    filepath = os.path.join(job_folder, filename)
    file.save(filepath)

    # Initialize job
    jobs[job_id] = {
        'id': job_id,
        'filename': filename,
        'filepath': filepath,
        'status': 'uploaded',
        'progress': 0,
        'step': 'uploaded',
        'results': None,
        'error': None,
        'created_at': datetime.now().isoformat()
    }

    return jsonify({
        'job_id': job_id,
        'filename': filename,
        'size': os.path.getsize(filepath),
        'status': 'uploaded'
    })


@app.route('/api/process/<job_id>', methods=['POST'])
def process_video(job_id: str):
    """
    Start processing a video.

    Expected JSON body:
    {
        "model": "cnn" or "lightgbm",
        "motion_threshold": 0.3,
        "gesture_threshold": 0.5,
        "min_duration": 15
    }
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]

    if job['status'] == 'processing':
        return jsonify({'error': 'Job already processing'}), 400

    # Get parameters
    params = request.get_json() or {}
    model_type = params.get('model', 'cnn')
    motion_threshold = float(params.get('motion_threshold', 0.3))
    gesture_threshold = float(params.get('gesture_threshold', 0.5))
    min_duration = int(params.get('min_duration', 15))

    # Update job status
    job['status'] = 'processing'
    job['progress'] = 0
    job['step'] = 'initializing'
    job['params'] = params

    # Process in background (for production, use Celery or similar)
    try:
        _process_video_sync(job_id, model_type, motion_threshold, gesture_threshold, min_duration)
    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)
        return jsonify({'error': str(e)}), 500

    return jsonify({'status': 'processing', 'job_id': job_id})


def _process_video_demo(job_id: str, model_type: str, motion_threshold: float,
                        gesture_threshold: float, min_duration: int):
    """
    Demo mode processing - generates fake results for UI testing.
    """
    job = jobs[job_id]

    def update_progress(step: str, progress: int):
        job['step'] = step
        job['progress'] = progress

    # Simulate processing steps
    steps = [
        ('loading', 10, 0.5),
        ('configuring', 25, 0.3),
        ('processing', 50, 1.5),
        ('processing', 75, 1.0),
        ('finalizing', 90, 0.5),
    ]

    for step, progress, delay in steps:
        update_progress(step, progress)
        time.sleep(delay)

    # Generate demo results
    num_segments = random.randint(4, 10)
    segments = []
    current_time = 0.5

    gesture_types = ['Gesture', 'Move']

    for i in range(num_segments):
        gesture = random.choice(gesture_types)
        duration = random.uniform(0.8, 2.5)
        start_time = current_time
        end_time = start_time + duration

        segments.append({
            'id': i + 1,
            'gesture': gesture,
            'startTime': round(start_time, 3),
            'endTime': round(end_time, 3),
            'startFrame': int(start_time * 30),
            'endFrame': int(end_time * 30),
            'confidence': round(random.uniform(0.75, 0.98), 2),
            'duration': round(duration, 3)
        })

        current_time = end_time + random.uniform(0.5, 2.0)

    job['results'] = {
        'segments': segments,
        'total_segments': len(segments),
        'video_duration': current_time,
        'fps': 30,
        'labeled_video': None,
        'predictions_csv': None,
        'segments_csv': None,
        'elan_file': None,
        'demo_mode': True
    }

    update_progress('complete', 100)
    job['status'] = 'complete'


def _process_video_sync(job_id: str, model_type: str, motion_threshold: float,
                        gesture_threshold: float, min_duration: int):
    """
    Synchronous video processing (for async, wrap with Celery task).
    """
    # Use demo mode if dependencies not available
    if DEMO_MODE:
        return _process_video_demo(job_id, model_type, motion_threshold, gesture_threshold, min_duration)

    job = jobs[job_id]

    def update_progress(step: str, progress: int):
        job['step'] = step
        job['progress'] = progress

    try:
        # Step 1: Initialize detector with all parameters
        update_progress('loading', 10)
        detector = GestureDetector(
            model_type=model_type,
            motion_threshold=motion_threshold,
            gesture_threshold=gesture_threshold,
            min_length_s=min_duration / 30.0  # Convert frames to seconds at 30fps
        )

        # Step 2: Get video FPS
        update_progress('configuring', 20)
        cap = cv2.VideoCapture(job['filepath'])
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count / fps if fps > 0 else 0
        cap.release()

        # Step 3: Process video
        update_progress('processing', 30)
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id, 'output')
        os.makedirs(output_folder, exist_ok=True)

        predictions_df, stats, segments_df, features, timestamps = detector.predict_video(job['filepath'])

        update_progress('labeling', 60)

        # Step 4: Generate labeled video
        # Ensure output has .mp4 extension
        base_name = Path(job['filename']).stem
        labeled_video_path = os.path.join(output_folder, f"labeled_{base_name}.mp4")
        label_video(job['filepath'], segments_df, labeled_video_path, predictions_df)

        # Step 5: Generate ELAN file
        update_progress('exporting', 80)
        elan_path = os.path.join(output_folder, f"{Path(job['filename']).stem}.eaf")
        create_elan_file(job['filepath'], segments_df, elan_path, fps)

        # Save predictions CSV
        predictions_csv_path = os.path.join(output_folder, f"predictions_{Path(job['filename']).stem}.csv")
        predictions_df.to_csv(predictions_csv_path, index=False)

        # Save segments CSV
        segments_csv_path = os.path.join(output_folder, f"segments_{Path(job['filename']).stem}.csv")
        segments_df.to_csv(segments_csv_path, index=False)

        update_progress('finalizing', 90)

        # Extract results for JSON response
        segments = []
        for idx, row in segments_df.iterrows():
            segments.append({
                'id': idx + 1,
                'gesture': row.get('label', 'Unknown'),
                'startTime': float(row.get('start_time', 0)),
                'endTime': float(row.get('end_time', 0)),
                'startFrame': int(row.get('start_time', 0) * fps),
                'endFrame': int(row.get('end_time', 0) * fps),
                'confidence': 0.85,  # Segments don't have confidence, use default
                'duration': float(row.get('duration', row.get('end_time', 0) - row.get('start_time', 0)))
            })

        job['results'] = {
            'segments': segments,
            'total_segments': len(segments),
            'video_duration': video_duration,
            'fps': int(fps),
            'labeled_video': labeled_video_path,
            'predictions_csv': predictions_csv_path,
            'segments_csv': segments_csv_path,
            'elan_file': elan_path
        }

        update_progress('complete', 100)
        job['status'] = 'complete'

    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)
        raise


@app.route('/api/status/<job_id>')
def get_status(job_id: str):
    """Get processing status for a job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'step': job['step'],
        'error': job.get('error')
    })


@app.route('/api/results/<job_id>')
def get_results(job_id: str):
    """Get processing results for a completed job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]

    if job['status'] != 'complete':
        return jsonify({'error': 'Job not complete', 'status': job['status']}), 400

    return jsonify(job['results'])


@app.route('/api/export/<job_id>/<format>')
def export_results(job_id: str, format: str):
    """
    Export results in various formats.

    Formats: csv, json, elan
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]

    if job['status'] != 'complete':
        return jsonify({'error': 'Job not complete'}), 400

    results = job['results']

    if format == 'csv':
        csv_path = results.get('segments_csv')
        if csv_path and os.path.exists(csv_path):
            return send_file(csv_path, as_attachment=True, download_name='segments.csv')

        # Generate CSV from results
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['segment_id', 'gesture_type', 'start_frame', 'end_frame',
                        'start_time', 'end_time', 'confidence', 'duration_ms'])
        for seg in results['segments']:
            writer.writerow([
                seg['id'], seg['gesture'], seg['startFrame'], seg['endFrame'],
                f"{seg['startTime']:.3f}", f"{seg['endTime']:.3f}",
                seg['confidence'], int(seg['duration'] * 1000)
            ])
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='segments.csv'
        )

    elif format == 'json':
        return jsonify({
            'metadata': {
                'file': job['filename'],
                'model': job.get('params', {}).get('model', 'cnn'),
                'date': datetime.now().isoformat(),
                'total_segments': results['total_segments'],
                'video_duration': results['video_duration']
            },
            'segments': results['segments']
        })

    elif format == 'elan':
        elan_path = results.get('elan_file')
        if elan_path and os.path.exists(elan_path):
            return send_file(elan_path, as_attachment=True, download_name='gestures.eaf')
        return jsonify({'error': 'ELAN file not available'}), 404

    elif format == 'video':
        video_path = results.get('labeled_video')
        if video_path and os.path.exists(video_path):
            return send_file(video_path, as_attachment=True,
                           download_name=f'labeled_{job["filename"]}')
        return jsonify({'error': 'Labeled video not available'}), 404

    return jsonify({'error': f'Unknown format: {format}'}), 400


@app.route('/api/video/<job_id>')
def get_video(job_id: str):
    """Stream the original uploaded video."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    return send_file(job['filepath'], mimetype='video/mp4')


@app.route('/api/labeled-video/<job_id>')
def get_labeled_video(job_id: str):
    """Stream the labeled output video."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]

    if job['status'] != 'complete':
        return jsonify({'error': 'Job not complete'}), 400

    video_path = job['results'].get('labeled_video')
    if video_path and os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')

    return jsonify({'error': 'Labeled video not available'}), 404


def run_server(host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
    """Run the Flask development server."""
    print(f"\n{'='*50}")
    print("EnvisionHG Web Interface")
    print(f"{'='*50}")
    if DEMO_MODE:
        print("MODE: DEMO (TensorFlow not installed)")
        print("      Results will be simulated for UI testing")
    else:
        print("MODE: FULL (All dependencies available)")
    print(f"Server running at: http://{host}:{port}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"{'='*50}\n")

    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    run_server(debug=True)
