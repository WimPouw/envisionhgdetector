# envisionhgdetector/model_combined.py
"""
Combined CNN + LightGBM gesture detection model.
Single MediaPipe pass feeds both models for efficient inference.

Both models run independently and output their results separately.
User can compare predictions from both models.

CNN: 3-class (NoGesture, Gesture, Move) - 25 frame window, 92 world landmarks
LightGBM: 2-class (NoGesture, Gesture) - 5 frame window, 100 engineered features

Output includes:
- CNN predictions: class, confidence, has_motion, gesture_prob, move_prob
- LightGBM predictions: class, confidence, gesture_prob, nogesture_prob
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from typing import Optional, List, Dict, Any, Tuple, Literal
from collections import deque
from dataclasses import dataclass
import tensorflow as tf

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ============================================================================
# COMBINED MODEL CLASS
# ============================================================================

class CombinedGestureModel:
    """
    Combined CNN + LightGBM gesture detection.
    Single MediaPipe pass feeds both models.
    """
    
    def __init__(self, config: Optional[CombinedConfig] = None):
        """
        Initialize combined model.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or CombinedConfig()
        
        # Initialize MediaPipe (single instance for both models)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark buffer (stores raw 92-feature world landmarks)
        # Sized for CNN (25 frames), LightGBM uses last 5
        self.landmarks_buffer = deque(maxlen=self.config.cnn_seq_length)
        
        # Load models
        self.cnn_model = None
        self.lgbm_model = None
        self.lgbm_scaler = None
        self.lgbm_label_encoder = None
        
        self._load_cnn_model()
        self._load_lgbm_model()
        
        print(f"\n✓ Combined model initialized")
        print(f"  CNN thresholds: motion={self.config.cnn_motion_threshold}, gesture={self.config.cnn_gesture_threshold}")
        print(f"  LightGBM threshold: {self.config.lgbm_threshold}")
        print(f"  CNN ready: {self.cnn_model is not None}")
        print(f"  LightGBM ready: {self.lgbm_model is not None}")
 
    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a single frame and get predictions from both models.
        
        Args:
            frame: BGR video frame
            
        Returns:
            Dictionary with separate predictions from CNN and LightGBM, or None if not ready
        """
        # Extract landmarks (single MediaPipe call)
        landmarks = self.extract_world_landmarks(frame)
        
        if landmarks is None:
            return None
        
        # Add to buffer
        self.landmarks_buffer.append(landmarks)
        
        # Check if we have enough frames for CNN
        if len(self.landmarks_buffer) < self.config.cnn_seq_length:
            return None
        
        results = {
            'cnn': None,
            'lightgbm': None
        }
        
        # Get full sequence for CNN (25 frames)
        cnn_sequence = np.array(list(self.landmarks_buffer))
        
        # Get last 5 frames for LightGBM
        lgbm_sequence = cnn_sequence[-self.config.lgbm_window_size:]
        
        # ===== CNN Prediction =====
        if self.cnn_model is not None:
            cnn_input = cnn_sequence.reshape(1, self.config.cnn_seq_length, self.config.cnn_num_features)
            cnn_pred = self.cnn_model.predict(cnn_input, verbose=0)[0]
            
            # Parse CNN output: [has_motion, gesture_prob, move_prob]
            cnn_has_motion = float(cnn_pred[0])
            cnn_gesture_prob = float(cnn_pred[1])
            cnn_move_prob = float(cnn_pred[2])
            
            # Determine class using thresholds
            if cnn_has_motion < self.config.cnn_motion_threshold:
                cnn_class = "NoGesture"
                cnn_confidence = 1 - cnn_has_motion
            elif cnn_gesture_prob > cnn_move_prob:
                cnn_class = "Gesture"
                cnn_confidence = cnn_has_motion * cnn_gesture_prob
            else:
                cnn_class = "Move"
                cnn_confidence = cnn_has_motion * cnn_move_prob
            
            results['cnn'] = {
                'class': cnn_class,
                'confidence': cnn_confidence,
                'has_motion': cnn_has_motion,
                'gesture_prob': cnn_gesture_prob,
                'move_prob': cnn_move_prob,
            }
        
        # ===== LightGBM Prediction =====
        if self.lgbm_model is not None:
            lgbm_features = extract_lgbm_features(lgbm_sequence)
            lgbm_features_scaled = self.lgbm_scaler.transform(lgbm_features.reshape(1, -1))
            
            # Handle both Booster and Classifier objects
            if hasattr(self.lgbm_model, 'predict_proba'):
                # LGBMClassifier - returns probabilities directly
                lgbm_probs = self.lgbm_model.predict_proba(lgbm_features_scaled)[0]
            else:
                # Raw Booster trained with multiclass objective
                # predict() already returns probabilities with shape (n_samples, n_classes)
                raw_output = self.lgbm_model.predict(lgbm_features_scaled)
                raw_output = np.array(raw_output)
                
                # For multiclass, output is already probabilities (n_samples, n_classes)
                if raw_output.ndim == 2:
                    lgbm_probs = raw_output[0]  # Get first sample
                elif raw_output.ndim == 1 and len(raw_output) == 2:
                    # Already probabilities for 2 classes
                    lgbm_probs = raw_output
                else:
                    # Single value - binary, apply sigmoid
                    raw_score = float(raw_output.flatten()[0])
                    gesture_prob = 1.0 / (1.0 + np.exp(-raw_score))
                    lgbm_probs = np.array([1 - gesture_prob, gesture_prob])
            
            # LightGBM: LabelEncoder sorts alphabetically!
            # So class order is: [Gesture_prob, NoGesture_prob] (alphabetical)
            lgbm_gesture_prob = float(lgbm_probs[0])      # Class 0 = Gesture
            lgbm_nogesture_prob = float(lgbm_probs[1])    # Class 1 = NoGesture
            
            # Determine class using threshold
            if lgbm_gesture_prob >= self.config.lgbm_threshold:
                lgbm_class = "Gesture"
                lgbm_confidence = lgbm_gesture_prob
            else:
                lgbm_class = "NoGesture"
                lgbm_confidence = lgbm_nogesture_prob
            
            results['lightgbm'] = {
                'class': lgbm_class,
                'confidence': lgbm_confidence,
                'nogesture_prob': lgbm_nogesture_prob,
                'gesture_prob': lgbm_gesture_prob,
            }
        
        return results
    
    def predict_video(
        self,
        video_path: str,
        target_fps: int = 25,
        return_all: bool = False
    ) -> Dict[str, Any]:
        """
        Process entire video and return predictions from both models.
        
        Args:
            video_path: Path to video file
            target_fps: Target FPS for processing
            return_all: If True, return predictions for all frames
            
        Returns:
            Dictionary with frame-by-frame predictions from both models
        """
        from tqdm import tqdm
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, int(original_fps / target_fps))
        
        self.reset_buffer()
        
        all_predictions = []
        frame_idx = 0
        processed_idx = 0
        
        # Progress bar
        pbar = tqdm(total=total_frames, desc="Processing frames", unit="frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                result = self.process_frame(frame)
                
                if result is not None:
                    result['frame_idx'] = frame_idx
                    result['time_s'] = frame_idx / original_fps
                    all_predictions.append(result)
                
                processed_idx += 1
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Build summary
        summary = {
            'video_path': video_path,
            'total_frames': frame_idx,
            'processed_frames': len(all_predictions),
            'original_fps': original_fps,
            'target_fps': target_fps,
            'predictions': all_predictions if return_all else None,
            'cnn_available': self.cnn_model is not None,
            'lgbm_available': self.lgbm_model is not None,
        }
        
        return summary
    
    def reset_buffer(self):
        """Clear the landmark buffer."""
        self.landmarks_buffer.clear()
    
    def set_thresholds(
        self,
        cnn_motion_threshold: Optional[float] = None,
        cnn_gesture_threshold: Optional[float] = None,
        lgbm_threshold: Optional[float] = None
    ):
        """Update thresholds for both models."""
        if cnn_motion_threshold is not None:
            self.config.cnn_motion_threshold = cnn_motion_threshold
        if cnn_gesture_threshold is not None:
            self.config.cnn_gesture_threshold = cnn_gesture_threshold
        if lgbm_threshold is not None:
            self.config.lgbm_threshold = lgbm_threshold
        print(f"Thresholds updated: CNN motion={self.config.cnn_motion_threshold}, "
              f"CNN gesture={self.config.cnn_gesture_threshold}, "
              f"LightGBM={self.config.lgbm_threshold}")
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'holistic') and self.holistic:
            self.holistic.close()

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
# NOT USED ANYWHERE?
def load_combined_model(
    cnn_weights: Optional[str] = None,
    lgbm_weights: Optional[str] = None,
    cnn_motion_threshold: float = 0.5,
    cnn_gesture_threshold: float = 0.5,
    lgbm_threshold: float = 0.5
) -> CombinedGestureModel:
    """
    Load combined model with custom weights and thresholds.
    
    Args:
        cnn_weights: Path to CNN weights (.h5)
        lgbm_weights: Path to LightGBM model (.pkl)
        cnn_motion_threshold: Motion threshold for CNN
        cnn_gesture_threshold: Gesture threshold for CNN
        lgbm_threshold: Confidence threshold for LightGBM
        
    Returns:
        CombinedGestureModel instance
    """
    config = CombinedConfig(
        cnn_weights_path=cnn_weights,
        lgbm_weights_path=lgbm_weights,
        cnn_motion_threshold=cnn_motion_threshold,
        cnn_gesture_threshold=cnn_gesture_threshold,
        lgbm_threshold=lgbm_threshold
    )
    return CombinedGestureModel(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Combined CNN + LightGBM Gesture Detection")
    print("=" * 50)
    print("Both models run independently - compare results yourself!")
    
    # Load model
    model = CombinedGestureModel()
    
    # Process video
    # results = model.predict_video("test_video.mp4")
    # print(f"Processed {results['processed_frames']} frames")
    
    # Or process frame by frame
    # cap = cv2.VideoCapture(0)  # Webcam
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     
    #     result = model.process_frame(frame)
    #     if result:
    #         print(f"CNN: {result['cnn']['class']} ({result['cnn']['confidence']:.2f})")
    #         print(f"LightGBM: {result['lightgbm']['class']} ({result['lightgbm']['confidence']:.2f})")