# Enhanced webcam demo script with parameter input
from envisionhgdetector import RealtimeGestureDetector
import pandas as pd
import os
from datetime import datetime

def get_parameters():
    """Get detection and post-processing parameters from user."""
    print("Detector Configuration")
    print("-" * 30)
    print("Configure detection and post-processing parameters:")
    print()
    
    # Get confidence threshold (applied during detection)
    while True:
        try:
            conf_input = input("Confidence threshold (0.0-1.0, default 0.2): ").strip()
            confidence_threshold = float(conf_input) if conf_input else 0.2
            if 0.0 <= confidence_threshold <= 1.0:
                break
            else:
                print("Please enter a value between 0.0 and 1.0")
        except ValueError:
            print("Please enter a valid number")
    
    # Get min gap (applied post-hoc)
    while True:
        try:
            gap_input = input("Min gap between gestures in seconds (default 0.2): ").strip()
            min_gap_s = float(gap_input) if gap_input else 0.2
            if min_gap_s >= 0.0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Get min length (applied post-hoc)
    while True:
        try:
            length_input = input("Min gesture length in seconds (default 0.3): ").strip()  # Changed default from 0.2 to 0.3
            min_length_s = float(length_input) if length_input else 0.3
            if min_length_s >= 0.0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nConfiguration:")
    print(f"  Confidence threshold: {confidence_threshold:.2f} (applied during detection)")
    print(f"  Min gap: {min_gap_s:.2f}s (applied post-hoc)")
    print(f"  Min length: {min_length_s:.2f}s (applied post-hoc)")
    
    return confidence_threshold, min_gap_s, min_length_s

def run_webcam_demo():
    """
    Enhanced webcam demo with user-configurable parameters.
    """
    print("LightGBM Real-time Gesture Detection Demo")
    print("=" * 50)
    
    # Get parameters from user
    confidence_threshold, min_gap_s, min_length_s = get_parameters()
    
    # Initialize detector with user parameters
    try:
        detector = RealtimeGestureDetector(
            confidence_threshold=confidence_threshold,
            min_gap_s=min_gap_s,
            min_length_s=min_length_s
        )
        print("\nLightGBM detector initialized successfully!")
        
        # Verify parameters were set correctly
        print(f"Detector parameters:")
        print(f"  Confidence threshold: {detector.confidence_threshold:.2f}")
        print(f"  Min gap: {detector.min_gap_s:.2f}s")
        print(f"  Min length: {detector.min_length_s:.2f}s")
        
        print(f"Model features: {detector.model.expected_features}")
        print(f"Gesture labels: {detector.model.gesture_labels}")
        print(f"Advanced features: {'ENABLED' if detector.model.includes_fingers else 'DISABLED'}")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return None
    
    print("\nStarting webcam demo...")
    print("Controls:")
    print("  - Q: Quit session")
    print("  - SPACE: Show current status")
    print("\nPosition yourself in front of the camera and start gesturing!")
    print("   (The system will start detecting after a few frames to build a buffer)\n")
    
    try:
        # Run webcam processing with post-processing
        raw_results, segments = detector.process_webcam(
            duration=None,               # Unlimited duration (use Q to quit)
            camera_index=0,              # Default camera
            show_display=True,           # Show real-time window
            save_video=True,             # Save annotated video
            apply_post_processing=True   # Apply segment refinement
        )
        
        print(f"\nSession complete! Processed {len(raw_results)} frames")
        
        if not raw_results.empty:
            # Analyze raw results
            gesture_frames = len(raw_results[raw_results['gesture'] != 'NoGesture'])
            gesture_percentage = (gesture_frames/len(raw_results)*100) if len(raw_results) > 0 else 0
            unique_gestures = raw_results[raw_results['gesture'] != 'NoGesture']['gesture'].unique()
            avg_confidence = raw_results['confidence'].mean()
            
            print(f"Raw gesture frames: {gesture_frames} ({gesture_percentage:.1f}%)")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Unique gestures detected: {list(unique_gestures)}")
            
            # Analyze processed segments
            if not segments.empty:
                total_segments = len(segments)
                total_gesture_time = segments['duration'].sum()
                avg_segment_duration = segments['duration'].mean()
                
                print(f"\nProcessed Segments (after applying gap={min_gap_s:.2f}s, minlen={min_length_s:.2f}s):")
                print(f"Total segments: {total_segments}")
                print(f"Total gesture time: {total_gesture_time:.1f}s")
                print(f"Average segment duration: {avg_segment_duration:.1f}s")
                
                if 'wall_clock_time' in raw_results.columns:
                    total_time = raw_results['wall_clock_time'].max()
                    print(f"Gestures per minute: {total_segments / (total_time / 60):.1f}")
                
                # Show segment details
                print(f"\nSegment Details:")
                for idx, seg in segments.iterrows():
                    print(f"  {idx+1}: {seg['label']} ({seg['start_time']:.1f}s - {seg['end_time']:.1f}s, {seg['duration']:.1f}s)")
            else:
                print(f"\nNo gesture segments found after post-processing")
                print(f"(Applied filters: gap={min_gap_s:.2f}s, minlen={min_length_s:.2f}s)")
                print("Suggestions:")
                print("- Try using smaller values for gap and minimum length")
                print("- Check if gestures are being detected consistently")
                print("- Verify confidence threshold isn't too high")
                
            # Show output files
            print(f"\nFiles saved in output_realtime/ folder:")
            print(f"- raw_frame_results.csv (frame-by-frame predictions)")
            print(f"- gesture_segments.csv (refined gesture segments)")
            print(f"- gesture_segments.eaf (ELAN annotation file)")
            print(f"- webcam_session.mp4 (annotated video)")
            print(f"- session_summary.json (session parameters & statistics)")
        else:
            print("No data recorded (session may have been too short)")
        
        return raw_results, segments
        
    except KeyboardInterrupt:
        print("\nSession interrupted by user")
        return None, None
    except Exception as e:
        print(f"\nError during webcam processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_default_demo():
    """Run demo with default parameters (no user input)."""
    print("LightGBM Real-time Gesture Detection Demo (Default Parameters)")
    print("=" * 60)
    
    try:
        detector = RealtimeGestureDetector(
            confidence_threshold=0.2,
            min_gap_s=0.2,
            min_length_s=0.3
        )
        print("Using default parameters:")
        print(f"  Confidence: {detector.confidence_threshold:.2f}")
        print(f"  Gap: {detector.min_gap_s:.2f}s") 
        print(f"  Min length: {detector.min_length_s:.2f}s")
        print("LightGBM detector initialized successfully!")
        
        print("\nStarting webcam demo...")
        print("Controls: Q=quit, SPACE=status")
        print("Position yourself in front of the camera and start gesturing!\n")
        
        raw_results, segments = detector.process_webcam()
        
        if not raw_results.empty:
            gesture_frames = len(raw_results[raw_results['gesture'] != 'NoGesture'])
            gesture_percentage = (gesture_frames/len(raw_results)*100)
            print(f"\nSession complete: {gesture_frames} gesture frames ({gesture_percentage:.1f}%)")
            if not segments.empty:
                print(f"Processed segments: {len(segments)}")
        
        return raw_results, segments
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def quick_test():
    """Quick test without webcam to verify installation."""
    print("Quick Installation Test")
    print("-" * 30)
    try:
        detector = RealtimeGestureDetector(confidence_threshold=0.2)
        print("RealtimeGestureDetector imported successfully")
        print("LightGBM model loaded")
        print("MediaPipe initialized")
        print("Installation test passed! Ready for webcam demo.")
        return True
    except Exception as e:
        print(f"Installation test failed: {e}")
        return False

def analyze_previous_session():
    """Analyze a previous session from output_realtime folder."""
    import glob
    
    # Find most recent session
    session_folders = glob.glob("output_realtime/session_*")
    if not session_folders:
        print("No previous sessions found in output_realtime/")
        return
    
    latest_session = max(session_folders, key=lambda x: os.path.getctime(x))
    print(f"Loading latest session: {latest_session}")
    
    try:
        detector = RealtimeGestureDetector(confidence_threshold=0.2)
        raw_df, segments_df = detector.load_and_analyze_session(latest_session)
        
        if not raw_df.empty:
            print("\nSession Analysis:")
            gesture_frames = len(raw_df[raw_df['gesture'] != 'NoGesture'])
            print(f"Total frames: {len(raw_df)}")
            print(f"Gesture frames: {gesture_frames} ({gesture_frames/len(raw_df)*100:.1f}%)")
            print(f"Session duration: {raw_df['timestamp'].max():.1f}s")
            
            if not segments_df.empty:
                print(f"Processed segments: {len(segments_df)}")
                print(f"Total gesture time: {segments_df['duration'].sum():.1f}s")
                print(f"Average segment duration: {segments_df['duration'].mean():.1f}s")
                
                # Show files available
                session_files = os.listdir(latest_session)
                print(f"Available files: {session_files}")
        
        return raw_df, segments_df
        
    except Exception as e:
        print(f"Error analyzing session: {e}")
        return None, None

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_test()
        elif sys.argv[1] == "analyze":
            analyze_previous_session()
        elif sys.argv[1] == "default":
            if quick_test():
                print("\n" + "="*50)
                input("Press Enter to start default webcam demo (or Ctrl+C to cancel)...")
                run_default_demo()
        else:
            print("Usage:")
            print("  python webcam_demo.py         # Run webcam demo with parameter input")
            print("  python webcam_demo.py test    # Test installation")
            print("  python webcam_demo.py default # Run with default parameters (no input)")
            print("  python webcam_demo.py analyze # Analyze previous session")
    else:
        # Run the full webcam demo with parameter input
        if quick_test():
            print("\n" + "="*50)
            input("Press Enter to configure and start webcam demo (or Ctrl+C to cancel)...")
            raw_results, segments = run_webcam_demo()
        else:
            print("\nPlease fix installation issues before running webcam demo")