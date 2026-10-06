import cv2
import os
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from envisionhgdetector import GestureDetector
from envisionhgdetector.state import Labels, ModelNames, PredictionColumns, SegmentColumns
from envisionhgdetector.mediapipe_processing import HolisticProcessor
from envisionhgdetector.utils import create_elan_file, create_segments

class RealtimeGestureDetector:
    """
    Real-time gesture detection class (LightGBM: Binary Gesture v NoGesture Detection).
    Provides webcam processing and real-time inference capabilities with post-processing.
    """
    def __init__(
        self,
        confidence_threshold: float,
        min_gap_s: float,
        min_length_s: float,
    ):
        """Initialize real-time detector with LightGBM model and refinement parameters."""        
        # Force LightGBM model
        self.model = GestureDetector(model_type=ModelNames.LIGHTGBM).model # run with default LightGBM model
        self.confidence_threshold = confidence_threshold
        self.min_gap_s = min_gap_s
        self.min_length_s = min_length_s
        
        print(f"Initialized real-time LightGBM detector")
        print(f"Confidence threshold: {self.confidence_threshold:.2f} (fixed)")
        print(f"Min gap between gestures: {self.min_gap_s:.2f}s (fixed)")
        print(f"Min gesture length: {self.min_length_s:.2f}s (fixed)")
        print(f"Fingers Included: {'True' if self.model.includes_fingers else 'False'}")
        
    def process_webcam(
        self,
        duration: Optional[float] = None,
        camera_index: int = 0,
        show_display: bool = True,
        save_video: bool = True,
        create_gesture_segments: bool = True,
        output_folder: Optional[str] = None,
        output_fps: float = 20.0,
        verbose: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process webcam feed in real-time with post-processing.
        
        Args:
            duration: Maximum duration in seconds (None = unlimited)
            camera_index: Camera device index
            show_display: Whether to show real-time display
            save_video: Whether to save annotated video
            create_gesture_segments: Whether to create gesture/move segments and their exports
            verbose: Whether to print predictions for every frame
            
        Returns:
            Tuple of (raw_results_df, segments_df)
        """
        if not np.isfinite(output_fps) or output_fps <= 0:
            raise ValueError("output_fps must be finite and positive.")
        if duration is not None and (not np.isfinite(duration) or duration <= 0):
            raise ValueError("duration must be finite and positive, or None.")
        print(f"Starting real-time webcam processing...")
        if duration:
            print(f"Duration: {duration} seconds")
        else:
            print("Duration: Unlimited (press 'q' to quit)")
        
        # Create output folder structure
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if output_folder is None: # use default in current working directory
            output_folder = Path.cwd() / "output_realtime"

        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        session_folder = output_folder / f"session_{timestamp}"
        session_folder.mkdir(parents=True, exist_ok=True)        
        print(f"Session output folder: {session_folder}")
        
        cap = None
        writer = None
        video_path = None
        
        frame_results = []
        frame_count = 0
        start_time = time.time()
        
        print("\nControls:")
        print("  - Q: Quit session")
        print("  - SPACE: Show current status")
        print()
        
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise ValueError(f"Could not open camera {camera_index}")
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if width <= 0 or height <= 0:
                raise ValueError("Camera dimensions must be positive.")
            
            print(f"Camera: {width}x{height} at {fps:.1f}fps")
            if save_video:
                video_path = str(session_folder / "webcam_session.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(video_path, fourcc, output_fps, (width, height))
                if not writer.isOpened():
                    raise ValueError(f"Could not open video writer: {video_path}")
                print(f"Saving video to: {video_path} at {output_fps} FPS")

            self.model.reset_buffer()
            with HolisticProcessor(
                model_complexity=1,
                static_image_mode=False,
                enable_segmentation=False,
                smooth_landmarks=True,
                min_detection_confidence=self.model.config.min_detection_confidence,
                min_tracking_confidence=self.model.config.min_tracking_confidence,
            ) as processor:
                while True:
                    current_time = time.time() - start_time
                    if duration is not None and current_time >= duration:
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame from camera; ending session and saving recorded results.")
                        break
                
                    # Extract features and predict
                    features = self.model.extract_features_from_frame(frame, processor=processor)
                
                    prediction = Labels.NOGESTURE
                    org_prediction = Labels.NOGESTURE # original prediction for reference
                    confidence = 0.0
                
                    if features is not None:
                        pred_probs = self.model.predict(features.reshape(1, -1))[0]
                        predicted_class = np.argmax(pred_probs)
                        confidence = pred_probs[predicted_class]
                    
                        org_prediction = self.model.label_encoder.inverse_transform([predicted_class])[0]
                        prediction = org_prediction
                        if confidence < self.confidence_threshold:
                            prediction = Labels.NOGESTURE
                
                    # Calculate frame-based timestamp that matches video output
                    # This ensures ELAN timestamps align with video frames
                    # Use frame_count/fps for video sync, wall clock for user display
                    video_timestamp = frame_count / output_fps if save_video else current_time
                
                    # Store results with both timestamps
                    frame_result = {
                        PredictionColumns.FRAME_INDEX: frame_count,
                        PredictionColumns.TIMESTAMP: video_timestamp,  # Video-aligned timestamp for ELAN
                        PredictionColumns.PREDICTION: prediction,
                        'wall_clock_time': current_time,  # Real time for user feedback
                        'confidence': confidence,
                        'threshold': self.confidence_threshold,
                        'org_prediction': org_prediction
                    }
                
                    # Display on frame
                    if show_display or save_video:
                        display_frame = cv2.flip(frame, 1)  # Mirror effect
                    
                        # Add text overlay (use wall clock time for display)
                        color = (0, 255, 0) if prediction == Labels.GESTURE else (128, 128, 128)
                        cv2.putText(display_frame, f"Prediction: {prediction}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.putText(display_frame, f"Confidence: {confidence:.2f}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(display_frame, f"Time: {current_time:.1f}s",
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"Frame: {frame_count}",
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                        # Save frame if requested
                        if writer is not None:
                            writer.write(display_frame)

                    frame_results.append(frame_result)
                    frame_count += 1
                    if show_display:
                        cv2.imshow('Real-time Gesture Detection', display_frame)
                    
                        # Handle keyboard input (simplified)
                        key = cv2.waitKey(1) & 0xFF
                    
                        if key == ord('q') or key == ord('Q'):
                            print("Quit requested")
                            break
                        elif key == ord(' '):  # Status
                            print(f"Current: {prediction} ({confidence:.3f})")
                            print(f"Parameters: threshold={self.confidence_threshold:.2f}, gap={self.min_gap_s:.1f}s, minlen={self.min_length_s:.1f}s")
                
                    # Periodic status updates
                    if frame_count % 1500 == 0:
                        runtime_mins = current_time / 60.0
                        gesture_frames = len([r for r in frame_results if r[PredictionColumns.PREDICTION] == Labels.GESTURE])
                        gesture_percentage = (gesture_frames / len(frame_results)) * 100 if frame_results else 0
                        print(f"Status: {runtime_mins:.1f}m runtime, {frame_count} frames, {gesture_percentage:.1f}% gestures")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as exc:
            if not frame_results:
                raise
            print(f"Session stopped due to an error: {exc}. Saving {len(frame_results)} recorded frames.")
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()
        
        # Convert to DataFrame
        raw_df = pd.DataFrame(frame_results)
        
        if raw_df.empty:
            print("No data recorded")
            return pd.DataFrame(), pd.DataFrame()
        
        # Save raw results
        raw_csv_path = session_folder / "raw_frame_results.csv"
        raw_df.to_csv(raw_csv_path, index=False)
        print(f"Raw results saved to: {raw_csv_path}")
        
        # Debug timing information
        if save_video and not raw_df.empty:
            print(f"Timing alignment info:")
            print(f"   Video duration: {raw_df[PredictionColumns.TIMESTAMP].max():.1f}s (based on {output_fps} FPS)")
            print(f"   Wall clock duration: {raw_df['wall_clock_time'].max():.1f}s")
            print(f"   Frame count: {len(raw_df)} frames")
            print(f"   Expected video duration: {len(raw_df) / output_fps:.1f}s")
        
        # Create segments if requested
        segments_df = pd.DataFrame()
        if create_gesture_segments:
            try:
                # Apply segmentation
                segments_df = create_segments(
                    raw_df,
                    min_gap_s=self.min_gap_s,
                    min_length_s=self.min_length_s,
                    segments_policy="separate",
                )
                
                if not segments_df.empty:
                    # Save processed segments
                    segments_csv_path = session_folder / "gesture_segments.csv"
                    segments_df.to_csv(segments_csv_path, index=False)
                    print(f"Segments saved to: {segments_csv_path}")
                    
                    # Create ELAN file - only if video was saved
                    if save_video and video_path and os.path.exists(video_path):
                        try:
                            print("Creating ELAN file...")
                            elan_path = session_folder / "gesture_segments.eaf"
                            
                                
                            create_elan_file(
                                video_path=video_path,
                                segments_df=segments_df,
                                output_path=elan_path,
                            )
                            print(f"ELAN file saved to: {elan_path}")
                        except Exception as e:
                            print(f"Error creating ELAN file: {str(e)}")
                            traceback.print_exc()
                    else:
                        print("Skipping ELAN creation (video not saved or not found)")
                    
                    # Print summary
                    total_segments = len(segments_df)
                    total_gesture_time = segments_df[SegmentColumns.DURATION].sum()
                    avg_segment_length = segments_df[SegmentColumns.DURATION].mean()
                    
                    print(f"\nPost-processing Summary:")
                    print(f"   Gesture segments created: {total_segments}")
                    print(f"   Total gesture time: {total_gesture_time:.1f}s")
                    print(f"   Average segment duration: {avg_segment_length:.1f}s")
                    if 'wall_clock_time' in raw_df.columns:
                        total_time = raw_df['wall_clock_time'].max()
                        print(f"   Gestures per minute: {total_segments / (total_time / 60):.1f}")
                else:
                    print("\nNo gesture segments found after post-processing")
                    print("Suggestions:")
                    print(f"- Try reducing min_length (current: {self.min_length_s:.2f}s)")
                    print(f"- Try increasing min_gap (current: {self.min_gap_s:.2f}s)")
                    print("- Check if gestures are being detected consistently in the video")
                    
            except Exception as e:
                print(f"Error during post-processing: {str(e)}")
                traceback.print_exc()
        
        # Save session summary
        self._save_session_summary(session_folder, raw_df, segments_df)
        
        total_time = time.time() - start_time
        print(f"\nReal-time session completed:")
        print(f"   Processed {frame_count} frames in {total_time:.1f}s")
        print(f"   Average FPS: {frame_count/total_time:.1f}")
        print(f"   Session folder: {session_folder}")
        
        return raw_df, segments_df
    
    def _save_session_summary(self, session_folder: Path, raw_df: pd.DataFrame, segments_df: pd.DataFrame):
        """Save a summary of the session parameters and results as CSV."""        
        # Create flattened summary data for CSV
        summary_data = {
            # Session info
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_frames': len(raw_df),
            'duration_seconds': raw_df[PredictionColumns.TIMESTAMP].max() if not raw_df.empty else 0,
            'wall_clock_duration_seconds': raw_df['wall_clock_time'].max() if not raw_df.empty and 'wall_clock_time' in raw_df.columns else 0,
            'average_fps': len(raw_df) / raw_df[PredictionColumns.TIMESTAMP].max() if not raw_df.empty and raw_df[PredictionColumns.TIMESTAMP].max() > 0 else 0,
            
            # Parameters
            'confidence_threshold': self.confidence_threshold,
            'min_gap_s': self.min_gap_s,
            'min_length_s': self.min_length_s,
            'model_type': 'LightGBM',
            'advanced_features': self.model.includes_fingers,
            
            # Results
            'org_gesture_percentage': (len(raw_df[raw_df['org_prediction'] == Labels.GESTURE]) / len(raw_df) * 100) if not raw_df.empty else 0,
            'thresholded_gesture_percentage': (len(raw_df[raw_df[PredictionColumns.PREDICTION] == Labels.GESTURE]) / len(raw_df) * 100) if not raw_df.empty else 0,
            'processed_segments': len(segments_df) if not segments_df.empty else 0,
            'total_gesture_time': segments_df[SegmentColumns.DURATION].sum() if not segments_df.empty else 0,
            'average_segment_duration': segments_df[SegmentColumns.DURATION].mean() if not segments_df.empty else 0,
            'gestures_per_minute': (len(segments_df) / (raw_df['wall_clock_time'].max() / 60)) if not raw_df.empty and 'wall_clock_time' in raw_df.columns and raw_df['wall_clock_time'].max() > 0 else 0
        }
        
        # Convert to DataFrame with single row
        summary_df = pd.DataFrame([summary_data])
        
        # Save as CSV
        summary_path = session_folder / "session_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Session summary saved to: {summary_path}")
        
        # Also save a more detailed version with individual segment information if segments exist
        if not segments_df.empty:
            detailed_summary = []
            for idx, segment in segments_df.iterrows():
                detailed_row = summary_data.copy()  # Include all session info
                detailed_row.update({
                    SegmentColumns.SEGMENT_IDX: segment[SegmentColumns.SEGMENT_IDX],
                    SegmentColumns.PREDICTION: segment[SegmentColumns.PREDICTION],
                    SegmentColumns.START_TIME: segment[SegmentColumns.START_TIME],
                    SegmentColumns.START_FRAME_IDX: segment[SegmentColumns.START_FRAME_IDX],
                    SegmentColumns.END_TIME: segment[SegmentColumns.END_TIME],
                    SegmentColumns.END_FRAME_IDX: segment[SegmentColumns.END_FRAME_IDX],
                    SegmentColumns.DURATION: segment[SegmentColumns.DURATION],
                })
                detailed_summary.append(detailed_row)
            
            detailed_df = pd.DataFrame(detailed_summary)
            detailed_path = session_folder / "session_summary_detailed.csv"
            detailed_df.to_csv(detailed_path, index=False)
            
            print(f"Detailed session summary saved to: {detailed_path}")
    
    def set_refinement_parameters(self, min_gap_s: float = None, min_length_s: float = None):
        """Update post-processing refinement parameters."""
        if min_gap_s is not None:
            self.min_gap_s = max(0.1, min(2.0, min_gap_s))
            print(f"Min gap updated to: {self.min_gap_s}s")
        
        if min_length_s is not None:
            self.min_length_s = max(0.1, min(3.0, min_length_s))
            print(f"Min length updated to: {self.min_length_s}s")
    
    def load_and_analyze_session(self, session_folder: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and analyze a previous session."""
        raw_csv = session_folder / "raw_frame_results.csv"
        segments_csv = session_folder / "gesture_segments.csv"
        summary_csv = session_folder / "session_summary.csv"
        
        raw_df = pd.DataFrame()
        segments_df = pd.DataFrame()
        
        if raw_csv.exists():
            raw_df = pd.read_csv(raw_csv)
            print(f"Loaded raw results: {len(raw_df)} frames")
        
        if segments_csv.exists():
            segments_df = pd.read_csv(segments_csv)
            print(f"Loaded segments: {len(segments_df)} segments")
        
        if summary_csv.exists():
            summary = pd.read_csv(summary_csv).iloc[0]
            print(f"Session summary:")
            print(f"   Duration: {summary['duration_seconds']:.1f}s")
            print(f"   Parameters: threshold={summary['confidence_threshold']:.2f}")
            print(f"   Results: {int(summary['processed_segments'])} segments")
        
        return raw_df, segments_df
