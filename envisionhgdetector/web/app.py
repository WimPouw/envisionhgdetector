# envisionhgdetector/web/app.py
"""
Flask web application for EnvisionHG gesture detection.
Provides REST API endpoints for the frontend UI.
"""

import os
import json
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import gesture detection modules
from ..detector import GestureDetector
from ..elan import create_elan_file

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


def _process_video_sync(job_id: str, model_type: str, motion_threshold: float,
                        gesture_threshold: float, min_duration: int):
    """
    Synchronous video processing (for async, wrap with Celery task).
    """
    job = jobs[job_id]

    def update_progress(step: str, progress: int):
        job['step'] = step
        job['progress'] = progress

    try:
        # Step 1: Initialize detector
        update_progress('loading', 10)
        detector = GestureDetector(model_type=model_type)

        # Step 2: Configure parameters
        update_progress('configuring', 20)
        detector.set_params(
            motion_threshold=motion_threshold,
            gesture_threshold=gesture_threshold,
            min_length_s=min_duration / 30.0  # Convert frames to seconds at 30fps
        )

        # Step 3: Process video
        update_progress('processing', 30)
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id, 'output')
        os.makedirs(output_folder, exist_ok=True)

        results = detector.process_video(
            video_path=job['filepath'],
            output_folder=output_folder,
            create_labeled_video=True,
            create_elan_file=True
        )

        update_progress('finalizing', 90)

        # Extract results
        segments = []
        if results and 'segments' in results:
            segments_df = results['segments']
            for idx, row in segments_df.iterrows():
                segments.append({
                    'id': idx + 1,
                    'gesture': row.get('label', 'Unknown'),
                    'startTime': float(row.get('start_time', 0)),
                    'endTime': float(row.get('end_time', 0)),
                    'startFrame': int(row.get('start_time', 0) * 30),
                    'endFrame': int(row.get('end_time', 0) * 30),
                    'confidence': float(row.get('confidence', 0.85)),
                    'duration': float(row.get('duration', row.get('end_time', 0) - row.get('start_time', 0)))
                })

        job['results'] = {
            'segments': segments,
            'total_segments': len(segments),
            'video_duration': results.get('duration', 0),
            'fps': results.get('fps', 30),
            'labeled_video': results.get('labeled_video_path'),
            'predictions_csv': results.get('predictions_path'),
            'segments_csv': results.get('segments_path'),
            'elan_file': results.get('elan_path')
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
    print(f"Server running at: http://{host}:{port}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"{'='*50}\n")

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
