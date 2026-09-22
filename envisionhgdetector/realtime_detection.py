class RealtimeGestureDetector:
    """
    Real-time gesture detection class (LightGBM only).
    Provides webcam processing and real-time inference capabilities with post-processing.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.2,
        min_gap_s: float = 0.3,          # Fixed - no real-time adjustment
        min_length_s: float = 0.5,       # Fixed - no real-time adjustment
        config: Optional[Config] = None
    ):
        """Initialize real-time detector with LightGBM model and refinement parameters."""
        self.config = config or Config()
        
        # Force LightGBM model
        self.model = LightGBMGestureModel(self.config)
        
        # Validate and store parameters with proper defaults
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self.min_gap_s = max(0.0, float(min_gap_s))
        self.min_length_s = max(0.0, float(min_length_s))
        
        # Set confidence threshold on model if supported
        if hasattr(self.model, 'set_confidence_threshold'):
            self.model.set_confidence_threshold(self.confidence_threshold)
        
        print(f"Initialized real-time LightGBM detector")
        print(f"Confidence threshold: {self.confidence_threshold:.2f} (fixed)")
        print(f"Min gap between gestures: {self.min_gap_s:.2f}s (fixed)")
        print(f"Min gesture length: {self.min_length_s:.2f}s (fixed)")
        print(f"Advanced features: {'ENABLED' if self.model.includes_fingers else 'DISABLED'}")
        
        # Debug: verify parameters are set
        assert hasattr(self, 'confidence_threshold'), "confidence_threshold not set"
        assert hasattr(self, 'min_gap_s'), "min_gap_s not set"
        assert hasattr(self, 'min_length_s'), "min_length_s not set"
        print(f"Parameter validation passed.")
        
    def process_webcam(
        self,
        duration: Optional[float] = None,
        camera_index: int = 0,
        show_display: bool = True,
        save_video: bool = True,
        apply_post_processing: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process webcam feed in real-time with post-processing.
        
        Args:
            duration: Maximum duration in seconds (None = unlimited)
            camera_index: Camera device index
            show_display: Whether to show real-time display
            save_video: Whether to save annotated video
            apply_post_processing: Whether to apply segment refinement
            
        Returns:
            Tuple of (raw_results_df, segments_df)
        """
        print(f"Starting real-time webcam processing...")
        if duration:
            print(f"Duration: {duration} seconds")
        else:
            print("Duration: Unlimited (press 'q' to quit)")
        
        # Create output folder structure
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_base = "output_realtime"
        session_folder = os.path.join(output_base, f"session_{timestamp}")
        os.makedirs(session_folder, exist_ok=True)
        
        print(f"Output folder: {session_folder}")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera: {width}x{height} at {fps:.1f}fps")
        
        # Setup video writer if requested
        writer = None
        video_path = None
        output_fps = 20.0  # Define output FPS as a variable
        if save_video:
            video_path = os.path.join(session_folder, f"webcam_session.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, output_fps, (width, height))
            print(f"Saving video to: {video_path} at {output_fps} FPS")
        
        # Reset model state
        self.model.key_joints_buffer.clear()
        if hasattr(self.model, 'left_fingers_buffer'):
            self.model.left_fingers_buffer.clear()
            self.model.right_fingers_buffer.clear()
        
        frame_results = []
        frame_count = 0
        start_time = time.time()
        
        print("\nControls:")
        print("  - Q: Quit session")
        print("  - SPACE: Show current status")
        print()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    continue
                
                current_time = time.time() - start_time
                
                # Check duration limit
                if duration and current_time > duration:
                    break
                
                # Extract features and predict
                features = self.model.extract_features_from_frame(frame)
                
                gesture_name = "NoGesture"
                confidence = 0.0
                
                if features is not None:
                    pred_probs = self.model.predict(features.reshape(1, -1))[0]
                    predicted_class = np.argmax(pred_probs)
                    confidence = pred_probs[predicted_class]
                    
                    raw_gesture_name = self.model.label_encoder.inverse_transform([predicted_class])[0]
                    gesture_name = self.model.standardize_gesture_name(raw_gesture_name)
                    
                    # Apply confidence threshold (fixed)
                    if gesture_name != "NoGesture" and confidence < self.confidence_threshold:
                        gesture_name = "NoGesture"
                        confidence = 0.0
                
                # Calculate frame-based timestamp that matches video output
                # This ensures ELAN timestamps align with video frames
                # Use frame_count/fps for video sync, wall clock for user display
                video_timestamp = frame_count / output_fps if save_video else current_time
                
                # Store results with both timestamps
                frame_results.append({
                    'frame': frame_count,
                    'timestamp': video_timestamp,  # Video-aligned timestamp for ELAN
                    'wall_clock_time': current_time,  # Real time for user feedback
                    Labels.GESTURE: gesture_name,
                    'confidence': confidence,
                    'threshold': self.confidence_threshold,
                    'raw_gesture': gesture_name
                })
                
                # Display on frame
                if show_display:
                    display_frame = cv2.flip(frame, 1)  # Mirror effect
                    
                    # Add text overlay (use wall clock time for display)
                    color = (0, 255, 0) if gesture_name != "NoGesture" else (128, 128, 128)
                    cv2.putText(display_frame, f"Gesture: {gesture_name}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(display_frame, f"Confidence: {confidence:.2f}", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(display_frame, f"Time: {current_time:.1f}s", 
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Frame: {frame_count}", 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save frame if requested
                    if writer:
                        writer.write(display_frame)
                    
                    cv2.imshow('Real-time Gesture Detection', display_frame)
                    
                    # Handle keyboard input (simplified)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q') or key == ord('Q'):
                        print("Quit requested")
                        break
                    elif key == ord(' '):  # Status
                        print(f"Current: {gesture_name} ({confidence:.3f})")
                        print(f"Parameters: threshold={self.confidence_threshold:.2f}, gap={self.min_gap_s:.1f}s, minlen={self.min_length_s:.1f}s")
                
                frame_count += 1
                
                # Periodic status updates
                if frame_count % 1500 == 0:
                    runtime_mins = current_time / 60.0
                    gesture_frames = len([r for r in frame_results if r[Labels.GESTURE] != Labels.NOGESTURE])
                    gesture_percentage = (gesture_frames / len(frame_results)) * 100 if frame_results else 0
                    print(f"Status: {runtime_mins:.1f}m runtime, {frame_count} frames, {gesture_percentage:.1f}% gestures")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()
        
        # Convert to DataFrame
        raw_df = pd.DataFrame(frame_results)
        
        if raw_df.empty:
            print("No data recorded")
            return pd.DataFrame(), pd.DataFrame()
        
        # Save raw results
        raw_csv_path = os.path.join(session_folder, "raw_frame_results.csv")
        raw_df.to_csv(raw_csv_path, index=False)
        print(f"Raw results saved to: {raw_csv_path}")
        
        # Debug timing information
        if save_video and not raw_df.empty:
            print(f"Timing alignment info:")
            print(f"   Video duration: {raw_df['timestamp'].max():.1f}s (based on {output_fps} FPS)")
            print(f"   Wall clock duration: {raw_df['wall_clock_time'].max():.1f}s")
            print(f"   Frame count: {len(raw_df)} frames")
            print(f"   Expected video duration: {len(raw_df) / output_fps:.1f}s")
        
        # Apply post-processing if requested
        segments_df = pd.DataFrame()
        if apply_post_processing:
            try:
                print("Applying post-processing segmentation...")
                
                # Create processed dataframe 
                processed_df = raw_df.copy()
                processed_df['time'] = processed_df['timestamp']  # Required for segmentation
                processed_df['original_gesture'] = processed_df[Labels.GESTURE]  # Keep original gesture names
                
                # Count gesture vs non-gesture frames
                gesture_frames = processed_df[processed_df[Labels.GESTURE].apply(
                    lambda x: x not in [Labels.NOGESTURE, Labels.NOGESTURE]
                )].shape[0]
                total_frames = len(processed_df)
                
                print(f"Frame analysis:")
                print(f"  Total frames: {total_frames}")
                print(f"  Gesture frames: {gesture_frames} ({gesture_frames/total_frames*100:.1f}%)")
                print(f"  Post-processing parameters: min_gap={self.min_gap_s:.2f}s, min_length={self.min_length_s:.2f}s")
                
                # Apply segmentation
                segments_df = self._create_gesture_segments(processed_df)
                
                if not segments_df.empty:
                    # Save processed segments
                    segments_csv_path = os.path.join(session_folder, "gesture_segments.csv")
                    segments_df.to_csv(segments_csv_path, index=False)
                    print(f"Segments saved to: {segments_csv_path}")
                    
                    # Create ELAN file - only if video was saved
                    if save_video and video_path and os.path.exists(video_path):
                        try:
                            print("Creating ELAN file...")
                            elan_path = os.path.join(session_folder, "gesture_segments.eaf")
                            
                            try:
                                from .utils import create_elan_file
                            except ImportError:
                                from utils import create_elan_file
                                
                            create_elan_file(
                                video_path=video_path,
                                segments_df=segments_df,
                                output_path=elan_path,
                                fps=output_fps,
                                include_ground_truth=False
                            )
                            print(f"ELAN file saved to: {elan_path}")
                        except Exception as e:
                            print(f"Error creating ELAN file: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print("Skipping ELAN creation (video not saved or not found)")
                    
                    # Print summary
                    total_segments = len(segments_df)
                    total_gesture_time = segments_df['duration'].sum()
                    avg_segment_length = segments_df['duration'].mean()
                    
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
                import traceback
                traceback.print_exc()
        
        # Save session summary
        self._save_session_summary(session_folder, raw_df, segments_df)
        
        total_time = time.time() - start_time
        print(f"\nReal-time session completed:")
        print(f"   Processed {frame_count} frames in {total_time:.1f}s")
        print(f"   Average FPS: {frame_count/total_time:.1f}")
        print(f"   Session folder: {session_folder}")
        
        if not raw_df.empty:
            gesture_frames = len(raw_df[raw_df[Labels.GESTURE] != Labels.NOGESTURE])
            print(f"   Raw gestures detected: {gesture_frames} frames ({gesture_frames/len(raw_df)*100:.1f}%)")
        
        return raw_df, segments_df

    def _create_gesture_segments(self, processed_df):
        """
        Create gesture segments from captured frame data using standard segmentation logic.
        This runs AFTER capture is complete, not during real-time processing.
        """
        import pandas as pd
        import numpy as np
        
        # Create binary gesture indicator (anything not NoGesture is a gesture)
        is_gesture = processed_df['original_gesture'].apply(
            lambda x: x not in [Labels.NOGESTURE, Labels.NOGESTURE]
        ).astype(int)
        
        # Find state changes
        changes = np.diff(is_gesture, prepend=0)
        start_indices = np.where(changes == 1)[0]  # Start of gesture periods
        end_indices = np.where(changes == -1)[0]   # End of gesture periods
        
        # Handle case where recording ends during a gesture
        if len(start_indices) > len(end_indices):
            end_indices = np.append(end_indices, len(processed_df) - 1)
        
        print(f"Found {len(start_indices)} potential gesture periods before filtering")
        
        # Create segments with gap merging and minimum length filtering
        segments = []
        segment_id = 1
        
        for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
            start_time = processed_df.iloc[start_idx]['time']
            end_time = processed_df.iloc[end_idx]['time']
            duration = end_time - start_time
            
            # Apply minimum length filter
            if duration >= self.min_length_s:
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'prediction': Labels.GESTURE,
                    'prediction_id': segment_id,
                    'duration': duration
                })
                segment_id += 1
        
        # Apply gap merging if we have multiple segments
        if len(segments) > 1:
            merged_segments = []
            current_segment = segments[0]
            
            for next_segment in segments[1:]:
                gap = next_segment['start_time'] - current_segment['end_time']
                
                # If gap is smaller than min_gap_s, merge segments
                if gap <= self.min_gap_s:
                    current_segment['end_time'] = next_segment['end_time']
                    current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
                else:
                    merged_segments.append(current_segment)
                    current_segment = next_segment
            
            # Add the last segment
            merged_segments.append(current_segment)
            segments = merged_segments
            
            print(f"After gap merging (gap<={self.min_gap_s:.2f}s): {len(segments)} segments")
        
        # Convert to DataFrame
        if segments:
            segments_df = pd.DataFrame(segments)
            print(f"Final gesture segments: {len(segments_df)}")
            
            # Print details
            for idx, seg in segments_df.iterrows():
                print(f"  Segment {idx+1}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s ({seg['duration']:.2f}s)")
                
            return segments_df
        else:
            print("No gesture segments found after applying filters")
            return pd.DataFrame(columns=['start_time', 'end_time', 'prediction_id', 'prediction', 'duration'])
    
    def _save_session_summary(self, session_folder: str, raw_df: pd.DataFrame, segments_df: pd.DataFrame):
        """Save a summary of the session parameters and results as CSV."""
        import pandas as pd
        
        # Create flattened summary data for CSV
        summary_data = {
            # Session info
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_frames': len(raw_df),
            'duration_seconds': raw_df['timestamp'].max() if not raw_df.empty else 0,
            'wall_clock_duration_seconds': raw_df['wall_clock_time'].max() if not raw_df.empty and 'wall_clock_time' in raw_df.columns else 0,
            'average_fps': len(raw_df) / raw_df['timestamp'].max() if not raw_df.empty and raw_df['timestamp'].max() > 0 else 0,
            
            # Parameters
            'confidence_threshold': self.confidence_threshold,
            'min_gap_s': self.min_gap_s,
            'min_length_s': self.min_length_s,
            'model_type': 'LightGBM',
            'advanced_features': self.model.includes_fingers,
            
            # Results
            'raw_gesture_frames': len(raw_df[raw_df[Labels.GESTURE] != Labels.NOGESTURE]) if not raw_df.empty else 0,
            'raw_gesture_percentage': (len(raw_df[raw_df[Labels.GESTURE] != Labels.NOGESTURE]) / len(raw_df) * 100) if not raw_df.empty else 0,
            'processed_segments': len(segments_df) if not segments_df.empty else 0,
            'total_gesture_time': segments_df['duration'].sum() if not segments_df.empty else 0,
            'average_segment_duration': segments_df['duration'].mean() if not segments_df.empty else 0,
            'gestures_per_minute': (len(segments_df) / (raw_df['wall_clock_time'].max() / 60)) if not raw_df.empty and 'wall_clock_time' in raw_df.columns and raw_df['wall_clock_time'].max() > 0 else 0
        }
        
        # Convert to DataFrame with single row
        summary_df = pd.DataFrame([summary_data])
        
        # Save as CSV
        summary_path = os.path.join(session_folder, "session_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Session summary saved to: {summary_path}")
        
        # Also save a more detailed version with individual segment information if segments exist
        if not segments_df.empty:
            detailed_summary = []
            for idx, segment in segments_df.iterrows():
                detailed_row = summary_data.copy()  # Include all session info
                detailed_row.update({
                    'segment_id': segment['prediction_id'],
                    'segment_prediction': segment['prediction'],
                    'segment_start_time': segment['start_time'],
                    'segment_end_time': segment['end_time'],
                    'segment_duration': segment['duration']
                })
                detailed_summary.append(detailed_row)
            
            detailed_df = pd.DataFrame(detailed_summary)
            detailed_path = os.path.join(session_folder, "session_summary_detailed.csv")
            detailed_df.to_csv(detailed_path, index=False)
            
            print(f"Detailed session summary saved to: {detailed_path}")
    
    def load_and_analyze_session(self, session_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and analyze a previous session."""
        raw_csv = os.path.join(session_folder, "raw_frame_results.csv")
        segments_csv = os.path.join(session_folder, "gesture_segments.csv")
        summary_json = os.path.join(session_folder, "session_summary.csv")
        
        raw_df = pd.DataFrame()
        segments_df = pd.DataFrame()
        
        if os.path.exists(raw_csv):
            raw_df = pd.read_csv(raw_csv)
            print(f"Loaded raw results: {len(raw_df)} frames")
        
        if os.path.exists(segments_csv):
            segments_df = pd.read_csv(segments_csv)
            print(f"Loaded segments: {len(segments_df)} segments")
        
        if os.path.exists(summary_json):
            import json
            with open(summary_json, 'r') as f:
                summary = json.load(f)
            print(f"Session summary:")
            print(f"   Duration: {summary['session_info']['duration_seconds']:.1f}s")
            print(f"   Parameters: threshold={summary['parameters']['confidence_threshold']:.2f}")
            print(f"   Results: {summary['results']['processed_segments']} segments")
        
        return raw_df, segments_df
    
    def set_refinement_parameters(self, min_gap_s: float = None, min_length_s: float = None):
        """Update post-processing refinement parameters."""
        if min_gap_s is not None:
            self.min_gap_s = max(0.1, min(2.0, min_gap_s))
            print(f"Min gap updated to: {self.min_gap_s}s")
        
        if min_length_s is not None:
            self.min_length_s = max(0.1, min(3.0, min_length_s))
            print(f"Min length updated to: {self.min_length_s}s")
    
    def load_and_analyze_session(self, session_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and analyze a previous session."""
        raw_csv = os.path.join(session_folder, "raw_frame_results.csv")
        segments_csv = os.path.join(session_folder, "gesture_segments.csv")
        summary_json = os.path.join(session_folder, "session_summary.csv")
        
        raw_df = pd.DataFrame()
        segments_df = pd.DataFrame()
        
        if os.path.exists(raw_csv):
            raw_df = pd.read_csv(raw_csv)
            print(f"Loaded raw results: {len(raw_df)} frames")
        
        if os.path.exists(segments_csv):
            segments_df = pd.read_csv(segments_csv)
            print(f"Loaded segments: {len(segments_df)} segments")
        
        if os.path.exists(summary_json):
            import json
            with open(summary_json, 'r') as f:
                summary = json.load(f)
            print(f"Session summary:")
            print(f"   Duration: {summary['session_info']['duration_seconds']:.1f}s")
            print(f"   Parameters: threshold={summary['parameters']['confidence_threshold']:.2f}")
            print(f"   Results: {summary['results']['processed_segments']} segments")
        
        return raw_df, segments_df