# EnvisionRealTimeTrain.py

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import time
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter


# EXACT SAME as inference script
KEY_JOINT_INDICES = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
KEY_JOINT_NAMES = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST']

# Finger landmark indices from MediaPipe pose
FINGER_INDICES = [17, 18, 19, 20, 21, 22]  # left_pinky, right_pinky, left_index, right_index, left_thumb, right_thumb
FINGER_NAMES = ['LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB']

# Wrist indices for centering
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16


class MaximumDataLightGBMTrainerWithFingers:
    """
    Enhanced LightGBM trainer that maximizes data usage while matching inference.
    Now includes finger landmarks (pinky, index, thumb) centered around wrists for hand shape information.
    FIXED for 23-landmark data structure.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize trainer to match inference.
        
        Args:
            window_size: Same as inference (5 frames)
        """
        self.window_size = window_size
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.gesture_labels = []
        
    def analyze_available_data(self, npz_path: str) -> Dict[str, Any]:
        """
        Analyze what data is available before training.
        
        Returns:
            Dictionary with data analysis
        """
        print(f"🔍 Analyzing available data in {npz_path}")
        
        data = np.load(npz_path, allow_pickle=True)
        
        analysis = {
            'total_frames': 0,
            'gesture_counts': {},
            'potential_sequences': {},
            'total_potential_sequences': 0,
            'landmark_info': {
                'landmarks_per_frame': 23,
                'features_per_frame': 69,
                'finger_landmarks_available': True
            }
        }
        
        # Verify data structure with first gesture
        first_gesture_analyzed = False
        
        for key in data.keys():
            if key in ['feature_names', 'dataset_tracking']:
                continue
                
            gesture_data = data[key]
            frame_count = len(gesture_data)
            
            # Verify landmark structure (only once)
            if not first_gesture_analyzed and len(gesture_data) > 0:
                sample_frame = gesture_data[0]
                total_values = len(sample_frame)
                
                print(f"\n✅ CONFIRMED DATA STRUCTURE:")
                print(f"Features per frame: {total_values}")
                print(f"Landmarks per frame: {total_values // 3}")
                print(f"Finger landmarks at indices 17-22: AVAILABLE")
                
                # Quick check of finger landmarks
                frame_3d = sample_frame.reshape(23, 3)
                print(f"Sample finger values:")
                print(f"  Left pinky (17): {frame_3d[17]}")
                print(f"  Right thumb (22): {frame_3d[22]}")
                
                first_gesture_analyzed = True
            
            # Calculate potential sequences with different strides
            potential_seqs_stride1 = max(0, frame_count - self.window_size + 1)
            potential_seqs_stride2 = max(0, (frame_count - self.window_size) // 2 + 1)
            
            analysis['gesture_counts'][key] = frame_count
            analysis['potential_sequences'][key] = {
                'stride_1': potential_seqs_stride1,
                'stride_2': potential_seqs_stride2
            }
            analysis['total_frames'] += frame_count
            analysis['total_potential_sequences'] += potential_seqs_stride1
        
        print(f"\n📊 DATA ANALYSIS RESULTS:")
        print(f"Total frames across all gestures: {analysis['total_frames']:,}")
        print(f"Number of gesture types: {len(analysis['gesture_counts'])}")
        print(f"Potential sequences (stride=1): {analysis['total_potential_sequences']:,}")
        print(f"Potential sequences (stride=2): {sum(seq['stride_2'] for seq in analysis['potential_sequences'].values()):,}")
        
        print(f"\n📋 Per-gesture breakdown:")
        for gesture, count in analysis['gesture_counts'].items():
            stride1_seqs = analysis['potential_sequences'][gesture]['stride_1']
            stride2_seqs = analysis['potential_sequences'][gesture]['stride_2']
            print(f"  {gesture}: {count:,} frames → {stride1_seqs:,} sequences (stride=1), {stride2_seqs:,} (stride=2)")
        
        return analysis
    
    def extract_fingers_from_pose(self, full_landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract finger landmarks from pose data, centered around wrists.
        Data structure: 23 landmarks (69 features) with fingers at indices 17-22.
        
        Args:
            full_landmarks: Full pose landmarks for a sequence of frames (shape: [..., 69])
            
        Returns:
            (left_fingers_relative, right_fingers_relative) each as [pinky_x, pinky_y, pinky_z, index_x, index_y, index_z, thumb_x, thumb_y, thumb_z]
            Relative to respective wrist positions
        """
        if len(full_landmarks.shape) == 1:
            # Single frame: reshape from 69 features to (23, 3)
            landmarks_3d = full_landmarks.reshape(23, 3)
        else:
            # Multiple frames - process each frame
            results = []
            for frame_landmarks in full_landmarks:
                left_rel, right_rel = self.extract_fingers_from_pose(frame_landmarks)
                results.append((left_rel, right_rel))
            
            # Stack results
            left_fingers_all = np.array([r[0] for r in results])
            right_fingers_all = np.array([r[1] for r in results])
            return left_fingers_all, right_fingers_all
        
        # Single frame processing - we know we have exactly 23 landmarks
        left_fingers_relative = np.zeros(9, dtype=np.float32)   # [pinky_x, pinky_y, pinky_z, index_x, index_y, index_z, thumb_x, thumb_y, thumb_z]
        right_fingers_relative = np.zeros(9, dtype=np.float32)
        
        # Get wrist positions (indices 15 and 16 in 23-landmark structure)
        left_wrist = landmarks_3d[LEFT_WRIST_IDX].astype(np.float32)   # Index 15
        right_wrist = landmarks_3d[RIGHT_WRIST_IDX].astype(np.float32)  # Index 16
        
        # Extract left fingers relative to left wrist
        # Left pinky (17), left index (19), left thumb (21)
        if np.any(left_wrist):
            left_pinky = landmarks_3d[17].astype(np.float32) - left_wrist
            left_fingers_relative[0:3] = left_pinky
            
            left_index = landmarks_3d[19].astype(np.float32) - left_wrist
            left_fingers_relative[3:6] = left_index
                
            left_thumb = landmarks_3d[21].astype(np.float32) - left_wrist
            left_fingers_relative[6:9] = left_thumb
        
        # Extract right fingers relative to right wrist
        # Right pinky (18), right index (20), right thumb (22)
        if np.any(right_wrist):
            right_pinky = landmarks_3d[18].astype(np.float32) - right_wrist
            right_fingers_relative[0:3] = right_pinky
            
            right_index = landmarks_3d[20].astype(np.float32) - right_wrist
            right_fingers_relative[3:6] = right_index
                
            right_thumb = landmarks_3d[22].astype(np.float32) - right_wrist
            right_fingers_relative[6:9] = right_thumb
        
        return left_fingers_relative, right_fingers_relative
    
    def load_npz_data_maximum(self, npz_path: str, max_sequences_per_label: int = None, 
                             stride: int = 1, augment_data: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data with maximum utilization, now including finger landmarks.
        
        Args:
            npz_path: Path to NPZ training data
            max_sequences_per_label: Max sequences per gesture (None = unlimited)
            stride: Frame stride (1 = use all possible sequences)
            augment_data: Apply data augmentation
            
        Returns:
            (X, y) arrays with maximum data including finger features
        """
        print(f"🚀 Loading data with MAXIMUM utilization (including finger landmarks):")
        print(f"  • Stride: {stride} (lower = more data)")
        print(f"  • Max sequences per gesture: {max_sequences_per_label or 'UNLIMITED'}")
        print(f"  • Data augmentation: {augment_data}")
        print(f"  • 🤏 Finger landmarks: ENABLED (pinky, index, thumb relative to wrists)")
        
        data = np.load(npz_path, allow_pickle=True)
        
        all_sequences = []
        all_labels = []
        
        for key in data.keys():
            if key in ['feature_names', 'dataset_tracking']:
                continue
                
            gesture_data = data[key]
            print(f"\nProcessing '{key}': {len(gesture_data)} frames")
            
            # Create sequences with specified stride
            sequences = self._create_sequences_with_stride(gesture_data, stride)
            original_count = len(sequences)
            
            # Apply data augmentation if requested
            if augment_data and len(sequences) > 0:
                augmented = self._augment_sequences(sequences)
                sequences.extend(augmented)
                print(f"  Added {len(augmented)} augmented sequences")
            
            # Limit only if specified
            if max_sequences_per_label and len(sequences) > max_sequences_per_label:
                print(f"  Limiting to {max_sequences_per_label:,} sequences (from {len(sequences):,})")
                indices = np.random.choice(len(sequences), max_sequences_per_label, replace=False)
                sequences = [sequences[i] for i in indices]
            
            all_sequences.extend(sequences)
            all_labels.extend([key] * len(sequences))
            
            print(f"  Final sequences for '{key}': {len(sequences):,}")
        
        self.gesture_labels = list(set(all_labels))
        
        print(f"\n🎯 FINAL DATA SUMMARY:")
        print(f"Total sequences: {len(all_sequences):,}")
        print(f"Gesture types: {len(self.gesture_labels)}")
        
        # Show data distribution
        label_counts = Counter(all_labels)
        print(f"\n📊 Data distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / len(all_labels)) * 100
            print(f"  {label}: {count:,} sequences ({percentage:.1f}%)")
        
        return np.array(all_sequences), np.array(all_labels)
    
    def _create_sequences_with_stride(self, frames: np.ndarray, stride: int = 1) -> List[np.ndarray]:
        """Create sequences with configurable stride for maximum data usage."""
        sequences = []
        
        for i in range(0, len(frames) - self.window_size + 1, stride):
            sequence = frames[i:i + self.window_size]
            sequences.append(sequence)
        
        return sequences
    
    def _augment_sequences(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply data augmentation to increase training data variety.
        
        Returns:
            List of augmented sequences
        """
        if len(sequences) == 0:
            return []
        
        augmented = []
        
        # Augment up to 30% of sequences to avoid overwhelming original data
        num_to_augment = min(len(sequences) // 3, 5000)  # Cap at 5000 per gesture
        
        selected_sequences = np.random.choice(len(sequences), num_to_augment, replace=False)
        
        for idx in selected_sequences:
            seq = sequences[idx]
            
            # 1. Add small Gaussian noise (simulate sensor noise)
            noise_factor = 0.008  # Small noise to avoid changing gesture meaning
            noisy_seq = seq + np.random.normal(0, noise_factor, seq.shape)
            augmented.append(noisy_seq.astype(np.float32))
            
            # 2. Small scale variation (simulate distance changes)
            scale_factor = np.random.uniform(0.97, 1.03)  # Very small scale changes
            scaled_seq = seq * scale_factor
            augmented.append(scaled_seq.astype(np.float32))
            
            # 3. Temporal jitter (slight timing variations)
            if len(seq) >= 4:  # Only if we have enough frames
                # Randomly drop one frame and duplicate another
                drop_idx = np.random.randint(1, len(seq) - 1)  # Don't drop first/last
                jittered_seq = np.delete(seq, drop_idx, axis=0)
                # Duplicate a random remaining frame
                dup_idx = np.random.randint(0, len(jittered_seq))
                jittered_seq = np.insert(jittered_seq, dup_idx, jittered_seq[dup_idx], axis=0)
                augmented.append(jittered_seq.astype(np.float32))
        
        return augmented
    
    def check_data_balance(self, y_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze data balance and provide recommendations.
        
        Returns:
            Analysis results and recommendations
        """
        label_counts = Counter(y_labels)
        total_samples = len(y_labels)
        
        analysis = {
            'total_samples': total_samples,
            'label_counts': dict(label_counts),
            'num_classes': len(label_counts),
            'min_samples': min(label_counts.values()),
            'max_samples': max(label_counts.values()),
            'imbalance_ratio': max(label_counts.values()) / min(label_counts.values()),
            'mean_samples': total_samples / len(label_counts)
        }
        
        print(f"\n⚖️  DATA BALANCE ANALYSIS:")
        print(f"Total samples: {total_samples:,}")
        print(f"Number of classes: {analysis['num_classes']}")
        print(f"Samples range: {analysis['min_samples']:,} - {analysis['max_samples']:,}")
        print(f"Mean samples per class: {analysis['mean_samples']:.0f}")
        print(f"Imbalance ratio: {analysis['imbalance_ratio']:.2f}")
        
        # Provide recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if analysis['imbalance_ratio'] > 5:
            print(f"  ⚠️  HIGH IMBALANCE: Consider class weights or SMOTE")
        elif analysis['imbalance_ratio'] > 2:
            print(f"  ⚠️  MODERATE IMBALANCE: Monitor per-class performance")
        else:
            print(f"  ✅ GOOD BALANCE: Data is well distributed")
        
        if analysis['min_samples'] < 500:
            print(f"  ⚠️  LOW SAMPLE COUNT: Consider more augmentation for sparse classes")
        
        if total_samples > 100000:
            print(f"  🚀 LARGE DATASET: Consider gradient-based sampling or distributed training")
        
        return analysis
    
    def get_optimized_params(self, num_samples: int, num_classes: int, imbalance_ratio: float) -> Dict[str, Any]:
        """
        Get parameters optimized for dataset characteristics.
        
        Args:
            num_samples: Total training samples
            num_classes: Number of gesture classes
            imbalance_ratio: Class imbalance ratio
            
        Returns:
            Optimized LightGBM parameters
        """
        base_params = {
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'num_threads': -1,
            'force_col_wise': True,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
        }
        
        # Adjust parameters based on dataset size
        if num_samples < 5000:
            # Small dataset - prevent overfitting
            params = {
                **base_params,
                'num_leaves': 31,
                'max_depth': 5,
                'learning_rate': 0.1,
                'min_data_in_leaf': 10,
                'feature_fraction': 0.9,
                'num_boost_round': 200
            }
        elif num_samples < 20000:
            # Medium dataset
            params = {
                **base_params,
                'num_leaves': 63,
                'max_depth': 7,
                'learning_rate': 0.08,
                'min_data_in_leaf': 30,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'num_boost_round': 400
            }
        elif num_samples < 100000:
            # Large dataset
            params = {
                **base_params,
                'num_leaves': 127,
                'max_depth': 9,
                'learning_rate': 0.06,
                'min_data_in_leaf': 50,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'num_boost_round': 600
            }
        else:
            # Very large dataset
            params = {
                **base_params,
                'num_leaves': 255,
                'max_depth': 11,
                'learning_rate': 0.04,
                'min_data_in_leaf': 100,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'num_boost_round': 1000
            }
        
        # Adjust for imbalanced data
        if imbalance_ratio > 3:
            params['is_unbalance'] = True
            print(f"  🎯 Enabled imbalanced data handling")
        
        print(f"📈 Using parameters optimized for {num_samples:,} samples")
        return params
    
    def extract_key_landmarks_from_full(self, full_landmarks: np.ndarray) -> np.ndarray:
        """Extract same 6 key joints as inference from full landmarks."""
        # FIXED: Your data has exactly 23 landmarks (69 features ÷ 3 = 23)
        landmarks_3d = full_landmarks.reshape(-1, 23, 3)
        key_landmarks = landmarks_3d[:, KEY_JOINT_INDICES, :]
        key_landmarks_flat = key_landmarks.reshape(key_landmarks.shape[0], -1)
        return key_landmarks_flat.astype(np.float32)
    
    def extract_enhanced_features_with_fingers(self, key_joints_sequence: np.ndarray, 
                                             left_fingers_sequence: np.ndarray, 
                                             right_fingers_sequence: np.ndarray) -> np.ndarray:
        """
        ENHANCED feature extraction including finger landmarks.
        This is the NEW version that includes finger shape information.
        """
        if len(key_joints_sequence) == 0:
            return np.zeros(80, dtype=np.float32)  # Increased to 80 for finger features
        
        features = []
        
        # Original pose features (same as before)
        # Current pose (18 values: 6 joints * 3 coords)
        current_pose = key_joints_sequence[-1]
        features.extend(current_pose)
        
        if len(key_joints_sequence) > 1:
            # Simple velocity (18 values)
            velocity = key_joints_sequence[-1] - key_joints_sequence[-2]
            features.extend(velocity)
            
            # Wrist speeds only (2 values)
            left_wrist_speed = np.linalg.norm(velocity[12:15])
            right_wrist_speed = np.linalg.norm(velocity[15:18])
            features.extend([left_wrist_speed, right_wrist_speed])
        else:
            features.extend([0.0] * 20)
        
        # Simple range over window for pose
        if len(key_joints_sequence) >= 3:
            # Range for wrists only (6 values)
            wrist_data = key_joints_sequence[:, 12:18]
            wrist_ranges = np.ptp(wrist_data, axis=0)
            features.extend(wrist_ranges)
        else:
            features.extend([0.0] * 6)
        
        # NEW: Finger features (30 additional values)
        
        # Current finger positions (18 values: 2 hands * 9 coords each)
        current_left_fingers = left_fingers_sequence[-1] if len(left_fingers_sequence) > 0 else np.zeros(9)
        current_right_fingers = right_fingers_sequence[-1] if len(right_fingers_sequence) > 0 else np.zeros(9)
        features.extend(current_left_fingers)
        features.extend(current_right_fingers)
        
        # Finger shape features for each hand (6 values total)
        # Left hand distances
        left_pinky_thumb_dist = 0.0
        left_index_thumb_dist = 0.0
        left_pinky_index_dist = 0.0
        
        if len(current_left_fingers) >= 9 and np.any(current_left_fingers):
            left_pinky_pos = current_left_fingers[0:3]
            left_index_pos = current_left_fingers[3:6]
            left_thumb_pos = current_left_fingers[6:9]
            
            left_pinky_thumb_dist = np.linalg.norm(left_pinky_pos - left_thumb_pos)
            left_index_thumb_dist = np.linalg.norm(left_index_pos - left_thumb_pos)
            left_pinky_index_dist = np.linalg.norm(left_pinky_pos - left_index_pos)
        
        # Right hand distances
        right_pinky_thumb_dist = 0.0
        right_index_thumb_dist = 0.0
        right_pinky_index_dist = 0.0
        
        if len(current_right_fingers) >= 9 and np.any(current_right_fingers):
            right_pinky_pos = current_right_fingers[0:3]
            right_index_pos = current_right_fingers[3:6]
            right_thumb_pos = current_right_fingers[6:9]
            
            right_pinky_thumb_dist = np.linalg.norm(right_pinky_pos - right_thumb_pos)
            right_index_thumb_dist = np.linalg.norm(right_index_pos - right_thumb_pos)
            right_pinky_index_dist = np.linalg.norm(right_pinky_pos - right_index_pos)
        
        features.extend([
            left_pinky_thumb_dist, left_index_thumb_dist, left_pinky_index_dist,
            right_pinky_thumb_dist, right_index_thumb_dist, right_pinky_index_dist
        ])
        
        # Total so far: 18 + 18 + 2 + 6 + 9 + 9 + 6 = 68 features
        
        # Pad to consistent size (80)
        while len(features) < 80:
            features.append(0.0)
        
        return np.array(features[:80], dtype=np.float32)
    
    def prepare_training_data_with_fingers(self, X_sequences: np.ndarray, y_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert sequences to features INCLUDING finger landmarks."""
        print("🔄 Extracting features with FINGER LANDMARKS...")
        
        X_features = []
        y_encoded = []
        
        batch_size = 10000
        total_sequences = len(X_sequences)
        
        # Track finger detection stats
        finger_detection_stats = {
            'left_fingers_detected': 0,
            'right_fingers_detected': 0,
            'total_sequences': 0
        }
        
        for i in range(0, total_sequences, batch_size):
            end_idx = min(i + batch_size, total_sequences)
            print(f"Processing batch {i//batch_size + 1}: sequences {i:,} - {end_idx:,}")
            
            for j in range(i, end_idx):
                sequence, label = X_sequences[j], y_labels[j]
                
                # Extract key landmarks
                key_joints_sequence = self.extract_key_landmarks_from_full(sequence)
                
                # Extract finger landmarks for each frame in sequence
                left_fingers_sequence, right_fingers_sequence = self.extract_fingers_from_pose(sequence)
                
                # Track finger detection
                finger_detection_stats['total_sequences'] += 1
                if np.any(left_fingers_sequence):
                    finger_detection_stats['left_fingers_detected'] += 1
                if np.any(right_fingers_sequence):
                    finger_detection_stats['right_fingers_detected'] += 1
                
                # Print sample data for first few sequences to debug
                if j < 3 and i == 0:
                    print(f"  Debug sample {j}: Left fingers shape: {left_fingers_sequence.shape}, any values: {np.any(left_fingers_sequence)}")
                    print(f"  Debug sample {j}: Right fingers shape: {right_fingers_sequence.shape}, any values: {np.any(right_fingers_sequence)}")
                    if np.any(left_fingers_sequence):
                        print(f"  Debug sample {j}: Left fingers sample: {left_fingers_sequence[-1][:6]}")  # Show first 6 values
                
                # Extract enhanced features including fingers
                features = self.extract_enhanced_features_with_fingers(
                    key_joints_sequence, left_fingers_sequence, right_fingers_sequence
                )
                
                X_features.append(features)
                y_encoded.append(label)
        
        X_features = np.array(X_features)
        
        # Print finger detection statistics
        print(f"\n🤏 FINGER DETECTION STATISTICS:")
        print(f"Total sequences processed: {finger_detection_stats['total_sequences']:,}")
        print(f"Left fingers detected: {finger_detection_stats['left_fingers_detected']:,} ({finger_detection_stats['left_fingers_detected']/finger_detection_stats['total_sequences']*100:.1f}%)")
        print(f"Right fingers detected: {finger_detection_stats['right_fingers_detected']:,} ({finger_detection_stats['right_fingers_detected']/finger_detection_stats['total_sequences']*100:.1f}%)")
        
        print(f"✅ Extracted {X_features.shape[1]} features (including finger landmarks) from {X_features.shape[0]:,} sequences")
        return X_features, np.array(y_encoded)
    
    def train_with_maximum_data(self, X_features: np.ndarray, y_labels: np.ndarray, 
                               use_cv: bool = True, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train using all available data with proper validation.
        
        Args:
            X_features: Feature matrix (now including finger features)
            y_labels: Labels
            use_cv: Use cross-validation for robust evaluation
            cv_folds: Number of CV folds
            
        Returns:
            Training results dictionary
        """
        print(f"🚀 Training with {len(X_features):,} samples using maximum data approach (WITH FINGERS)...")
        
        # Analyze data balance
        balance_analysis = self.check_data_balance(y_labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_labels)
        
        # Get optimized parameters
        params = self.get_optimized_params(
            len(X_features), 
            len(self.gesture_labels), 
            balance_analysis['imbalance_ratio']
        )
        
        start_time = time.time()
        
        if use_cv and len(X_features) > 10000:  # Use CV for larger datasets
            print(f"🔄 Using {cv_folds}-fold stratified cross-validation...")
            
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            fold_times = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_features, y_encoded)):
                fold_start = time.time()
                print(f"\n📋 Training fold {fold + 1}/{cv_folds}...")
                
                X_train_fold = X_features[train_idx]
                X_val_fold = X_features[val_idx]
                y_train_fold = y_encoded[train_idx]
                y_val_fold = y_encoded[val_idx]
                
                # Scale features
                scaler_fold = StandardScaler()
                X_train_scaled = scaler_fold.fit_transform(X_train_fold)
                X_val_scaled = scaler_fold.transform(X_val_fold)
                
                # Train fold model
                train_data = lgb.Dataset(X_train_scaled, label=y_train_fold)
                val_data = lgb.Dataset(X_val_scaled, label=y_val_fold, reference=train_data)
                
                fold_model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=params['num_boost_round'],
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                )
                
                # Evaluate fold
                val_pred = fold_model.predict(X_val_scaled)
                val_pred_classes = np.argmax(val_pred, axis=1)
                fold_accuracy = accuracy_score(y_val_fold, val_pred_classes)
                
                cv_scores.append(fold_accuracy)
                fold_times.append(time.time() - fold_start)
                
                print(f"  Fold {fold + 1} accuracy: {fold_accuracy:.4f} (time: {fold_times[-1]:.1f}s)")
                
                # Keep best model
                if fold == 0 or fold_accuracy == max(cv_scores):
                    self.model = fold_model
                    self.scaler = scaler_fold
            
            # Final training on all data with best parameters
            print(f"\n🎯 Training final model on all data...")
            X_all_scaled = self.scaler.fit_transform(X_features)
            train_data_all = lgb.Dataset(X_all_scaled, label=y_encoded)
            
            self.model = lgb.train(
                params,
                train_data_all,
                num_boost_round=int(params['num_boost_round'] * 0.8)  # Slightly fewer rounds
            )
            
            training_time = time.time() - start_time
            
            # Final evaluation
            final_pred = self.model.predict(X_all_scaled)
            final_pred_classes = np.argmax(final_pred, axis=1)
            final_accuracy = accuracy_score(y_encoded, final_pred_classes)
            
            results = {
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'final_accuracy': final_accuracy,
                'training_time': training_time,
                'num_features': X_features.shape[1],
                'num_trees': self.model.num_trees(),
                'total_samples': len(X_features),
                'balance_analysis': balance_analysis,
                'training_method': 'cross_validation',
                'includes_fingers': True
            }
            
            print(f"\n🎯 CROSS-VALIDATION RESULTS:")
            print(f"CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"Final Model Accuracy: {final_accuracy:.4f}")
            
        else:
            # Standard train/test split for smaller datasets
            print(f"🔄 Using train/test split...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y_encoded, test_size=0.15, 
                random_state=42, stratify=y_encoded
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=params['num_boost_round'],
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
            )
            
            training_time = time.time() - start_time
            
            # Evaluate
            test_pred = self.model.predict(X_test_scaled)
            test_pred_classes = np.argmax(test_pred, axis=1)
            test_accuracy = accuracy_score(y_test, test_pred_classes)
            
            results = {
                'test_accuracy': test_accuracy,
                'training_time': training_time,
                'num_features': X_features.shape[1],
                'num_trees': self.model.num_trees(),
                'total_samples': len(X_features),
                'balance_analysis': balance_analysis,
                'training_method': 'train_test_split',
                'includes_fingers': True
            }
            
            print(f"\n🎯 TRAINING RESULTS:")
            print(f"Test Accuracy: {test_accuracy:.4f}")
        
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Model Trees: {self.model.num_trees()}")
        print(f"Total Samples Used: {len(X_features):,}")
        print(f"Features (including fingers): {X_features.shape[1]}")
        
        return results
    
    def save_model(self, filepath: str):
        """Save model in EXACT format expected by inference."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'gesture_labels': self.gesture_labels,
            'window_size': self.window_size,
            'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'feature_names': [f'enhanced_feature_{i}' for i in range(80)],  # Updated for 80 features
            'overlap': 0,
            'includes_fingers': True,  # Flag to indicate finger features are included
            'feature_count': 80  # Explicit feature count
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to: {filepath}")
        print(f"   🔗 Compatible with enhanced inference script!")
        print(f"   🤏 Includes finger landmarks (pinky, index, thumb relative to wrists)")


def main():
    """Enhanced main training pipeline for maximum data usage with fingers."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LightGBM with MAXIMUM data usage and FINGER LANDMARKS')
    parser.add_argument('--npz_path', type=str, default='./training/bodylandmarks7wlonly.npz',
                       help='Path to NPZ training data')
    parser.add_argument('--max_sequences', type=int, default=None,
                       help='Maximum sequences per gesture (None = unlimited)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Frame stride (1=all frames, 2=every other, etc.)')
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation')
    parser.add_argument('--output_model', type=str, default='gesture_model_with_fingers.pkl',
                       help='Output model filename')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze data without training')
    
    args = parser.parse_args()
    
    print("🚀 MAXIMUM DATA USAGE TRAINING WITH FINGER LANDMARKS")
    print("=" * 70)
    print(f"Training data: {args.npz_path}")
    print(f"Frame stride: {args.stride} ({'ALL sequences' if args.stride == 1 else 'reduced sequences'})")
    print(f"Max sequences per gesture: {args.max_sequences or 'UNLIMITED'}")
    print(f"Data augmentation: {'ON' if args.augment else 'OFF'}")
    print(f"Output model: {args.output_model}")
    print(f"🤏 NEW: Finger landmarks (pinky, index, thumb) ENABLED from 23-landmark pose data")
    print(f"📊 Expected data structure: 23 landmarks, 69 features per frame")
    
    # Initialize trainer
    trainer = MaximumDataLightGBMTrainerWithFingers(window_size=5)
    
    try:
        # First, analyze available data
        analysis = trainer.analyze_available_data(args.npz_path)
        
        if args.analyze_only:
            print(f"\n📊 ANALYSIS COMPLETE - No training performed")
            return
        
        # Load data with maximum usage
        X_sequences, y_labels = trainer.load_npz_data_maximum(
            args.npz_path, 
            max_sequences_per_label=args.max_sequences,
            stride=args.stride,
            augment_data=args.augment
        )
        
        # Prepare features with fingers
        X_features, y_prepared = trainer.prepare_training_data_with_fingers(X_sequences, y_labels)
        
        # Train with maximum data
        results = trainer.train_with_maximum_data(X_features, y_prepared, cv_folds=args.cv_folds)
        
        # Save model
        trainer.save_model(args.output_model)
        
        print(f"\n🎉 MAXIMUM DATA TRAINING WITH FINGERS COMPLETED!")
        print(f"📁 Model saved to: {args.output_model}")
        if 'cv_mean' in results:
            print(f"🎯 CV Accuracy: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        else:
            print(f"🎯 Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"📊 Total samples used: {results['total_samples']:,}")
        print(f"🤏 Features (with fingers): {results['num_features']}")
        print(f"⚡ Training time: {results['training_time']:.1f} seconds")
        
        # Show data efficiency gained
        theoretical_max = analysis['total_potential_sequences']
        actual_used = results['total_samples']
        if args.augment:
            # Account for augmentation
            base_sequences = actual_used // 1.3  # Rough estimate
            efficiency = (base_sequences / theoretical_max) * 100
        else:
            efficiency = (actual_used / theoretical_max) * 100
        
        print(f"\n📈 DATA EFFICIENCY:")
        print(f"Used {actual_used:,} out of {theoretical_max:,} possible sequences")
        print(f"Data efficiency: {efficiency:.1f}%")
        
        print(f"\n🔥 Use this enhanced model with inference:")
        print(f"   python EnvisionRealTimeInference_WithFingers.py --model {args.output_model}")
        
    except FileNotFoundError:
        print(f"❌ Training data not found: {args.npz_path}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()