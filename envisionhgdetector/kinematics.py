# envisionhgdetector/kinematics.py
"""
Kinematic analysis utilities for gesture detection.
Computes movement features including velocity, acceleration, jerk,
McNeillian space usage, and submovement detection.
"""

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d


class KinematicsError(Exception):
    """Exception raised for errors during kinematic analysis."""
    pass


def validate_landmarks_array(landmarks: np.ndarray, name: str = "landmarks") -> np.ndarray:
    """
    Validate landmarks array has correct shape and type.

    Args:
        landmarks: Array to validate
        name: Parameter name for error messages

    Returns:
        Validated numpy array

    Raises:
        KinematicsError: If validation fails
    """
    if not isinstance(landmarks, np.ndarray):
        raise KinematicsError(
            f"{name} must be a numpy array, got {type(landmarks).__name__}"
        )

    if landmarks.size == 0:
        raise KinematicsError(f"{name} array cannot be empty")

    return landmarks


def validate_fps(fps: Union[int, float], name: str = "fps") -> float:
    """
    Validate frames per second value.

    Args:
        fps: FPS value to validate
        name: Parameter name for error messages

    Returns:
        Validated float value

    Raises:
        KinematicsError: If validation fails
    """
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        raise KinematicsError(
            f"{name} must be a number, got {type(fps).__name__}"
        )

    if fps <= 0:
        raise KinematicsError(f"{name} must be positive, got {fps}")

    if fps > 1000:
        # Sanity check - unusually high FPS
        import warnings
        warnings.warn(f"Unusually high {name} value: {fps}. Verify this is correct.")

    return fps


# Mapping from joint names to MediaPipe pose landmark indices
JOINT_MAP = {
    'L_Hand': 15,      # Left wrist
    'R_Hand': 16,      # Right wrist
    'LElb': 13,        # Left elbow
    'RElb': 14,        # Right elbow
    'LShoulder': 11,   # Left shoulder
    'RShoulder': 12,   # Right shoulder
    'Neck': 23,        # Neck (approximated as top of spine)
    'MidHip': 24,      # Mid hip
    'LEye': 2,         # Left eye
    'REye': 5,         # Right eye
    'Nose': 0,         # Nose
    'LHip': 23,        # Left hip
    'RHip': 24         # Right hip
}


class ArmKinematics(NamedTuple):
    """Container for arm kinematic measurements."""
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    speed: np.ndarray
    peaks: np.ndarray
    peak_heights: np.ndarray


@dataclass
class KinematicFeatures:
    """Data class to store comprehensive kinematic features for a gesture."""
    gesture_id: str
    video_id: str

    # Which hand was more active in this specific gesture
    active_hand: str  # 'L' or 'R'

    # Spatial features
    space_use: int
    mcneillian_max: float
    mcneillian_mode: int
    volume: float
    max_height: float

    # Temporal features
    duration: float
    hold_count: int
    hold_time: float
    hold_avg_duration: float

    # Submovement features
    hand_submovements: int
    hand_submovement_peaks: List[float]
    hand_mean_submovement_amplitude: float

    elbow_submovements: int
    elbow_mean_submovement_amplitude: float

    # Dynamic features
    hand_peak_speed: float
    hand_mean_speed: float
    hand_peak_acceleration: float
    hand_peak_deceleration: float
    hand_peak_jerk: float

    elbow_peak_speed: float
    elbow_mean_speed: float
    elbow_peak_acceleration: float
    elbow_peak_deceleration: float
    elbow_peak_jerk: float


def calculate_derivatives(
    positions: np.ndarray,
    fps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate velocity, acceleration and jerk from position data.

    Args:
        positions: Array of 3D positions over time with shape (N, 3)
        fps: Frames per second

    Returns:
        Tuple of (velocity, acceleration, jerk) arrays

    Raises:
        KinematicsError: If positions is empty or fps is non-positive
    """
    positions = validate_landmarks_array(positions, "positions")
    fps = validate_fps(fps, "fps")

    # Validate positions shape
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise KinematicsError(
            f"positions must have shape (N, 3), got {positions.shape}"
        )

    if len(positions) < 2:
        raise KinematicsError(
            f"positions must have at least 2 frames for derivative calculation, "
            f"got {len(positions)}"
        )

    dt = 1 / fps

    # Smooth positions
    positions = gaussian_filter1d(positions, sigma=2, axis=0)

    # Calculate velocity (first derivative)
    velocity = np.gradient(positions, dt, axis=0)

    # Calculate acceleration (second derivative)
    acceleration = np.gradient(velocity, dt, axis=0)

    # Calculate jerk (third derivative)
    jerk = np.gradient(acceleration, dt, axis=0)

    return velocity, acceleration, jerk


def find_submovements(
    speed_profile: np.ndarray,
    fps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find submovements in a speed profile using peak detection.

    Args:
        speed_profile: Array of speeds over time
        fps: Frames per second

    Returns:
        Tuple of (peak indices, peak heights)
    """
    # Handle very short sequences
    if len(speed_profile) < 3:
        if len(speed_profile) > 0:
            max_idx = np.argmax(speed_profile)
            return np.array([max_idx]), np.array([speed_profile[max_idx]])
        else:
            return np.array([0]), np.array([0])

    # Apply Savitzky-Golay smoothing with proper parameter handling
    if len(speed_profile) >= 15:
        smoothed = signal.savgol_filter(speed_profile, 15, 5)
    else:
        window = len(speed_profile)
        if window % 2 == 0:
            window = window - 1
        if window < 3:
            window = 3

        polyorder = min(5, window - 1)
        polyorder = max(1, polyorder)

        if window < 5 or polyorder < 1:
            if len(speed_profile) >= 3:
                smoothed = np.convolve(speed_profile, np.ones(3)/3, mode='same')
            else:
                smoothed = speed_profile.copy()
        else:
            try:
                smoothed = signal.savgol_filter(speed_profile, window, polyorder)
            except ValueError:
                smoothed = np.convolve(speed_profile, np.ones(3)/3, mode='same')

    # Find peaks with prominence and distance constraints
    peaks, properties = signal.find_peaks(
        smoothed,
        distance=max(1, int(5 * fps / 25)),
        height=0,
        prominence=max(0.01, np.std(smoothed) * 0.1)
    )

    peak_heights = smoothed[peaks] if len(peaks) > 0 else np.array([0])

    # If no peaks found, use the maximum value as a peak
    if len(peaks) == 0:
        max_idx = np.argmax(smoothed)
        peaks = np.array([max_idx])
        peak_heights = np.array([smoothed[max_idx]])

    return peaks, peak_heights


def compute_limb_kinematics(positions: np.ndarray, fps: float) -> ArmKinematics:
    """
    Compute comprehensive kinematics for a limb segment.

    Args:
        positions: Array of 3D positions over time
        fps: Frames per second

    Returns:
        ArmKinematics object containing computed measures
    """
    velocity, acceleration, jerk = calculate_derivatives(positions, fps)
    speed = np.linalg.norm(velocity, axis=1)
    peaks, peak_heights = find_submovements(speed, fps)

    if len(peaks) == 0:
        peaks = np.array([0])
        peak_heights = np.array([0])

    return ArmKinematics(
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        speed=speed,
        peaks=peaks,
        peak_heights=peak_heights
    )


def define_mcneillian_grid(df: pd.DataFrame, frame: int) -> Tuple:
    """
    Define the McNeillian gesture space grid based on body proportions.

    Args:
        df: DataFrame with pose keypoints
        frame: Frame index

    Returns:
        Tuple of grid boundaries (cc_xmin, cc_xmax, cc_ymin, cc_ymax,
                                  c_xmin, c_xmax, c_ymin, c_ymax,
                                  p_xmin, p_xmax, p_ymin, p_ymax)
    """
    bodycent = df['Neck'][frame][1] - (df['Neck'][frame][1] - df['MidHip'][frame][1]) / 2
    face_width = (df['LEye'][frame][0] - df['REye'][frame][0]) * 2
    body_width = df['LShoulder'][frame][0] - df['RShoulder'][frame][0]

    # Center-center boundaries
    cc_xmin = df['RShoulder'][frame][0]
    cc_xmax = df['LShoulder'][frame][0]
    cc_len = cc_xmax - cc_xmin
    cc_ymin = bodycent - cc_len / 2
    cc_ymax = bodycent + cc_len / 2

    # Center boundaries
    c_xmin = df['RShoulder'][frame][0] - body_width / 2
    c_xmax = df['LShoulder'][frame][0] + body_width / 2
    c_len = c_xmax - c_xmin
    c_ymin = bodycent - c_len / 2
    c_ymax = bodycent + c_len / 2

    # Periphery boundaries
    p_ymax = df['LEye'][frame][1] + (df['LEye'][frame][1] - df['Nose'][frame][1])
    p_ymin = bodycent - (p_ymax - bodycent)
    p_xmin = c_xmin - face_width
    p_xmax = c_xmax + face_width

    return (cc_xmin, cc_xmax, cc_ymin, cc_ymax,
            c_xmin, c_xmax, c_ymin, c_ymax,
            p_xmin, p_xmax, p_ymin, p_ymax)


def get_mcneillian_mode(spaces: List[int]) -> int:
    """
    Convert subsection codes to main sections and calculate mode.

    Args:
        spaces: List of space codes (1, 2, 31-38, 41-48)

    Returns:
        Mode of the main space usage (1-4)
    """
    # Vectorized conversion to main space codes
    spaces_arr = np.array(spaces)
    mainspace = np.where(spaces_arr > 40, 4,
                         np.where(spaces_arr > 30, 3, spaces_arr))

    # Use numpy bincount for robust mode calculation (handles multimodal data)
    if len(mainspace) == 0:
        return 1  # Default to center-center

    counts = np.bincount(mainspace.astype(int))
    return int(np.argmax(counts))


def calc_mcneillian_space(
    df: pd.DataFrame,
    visibility: Optional[np.ndarray] = None,
    visibility_threshold: float = 0.5
) -> Tuple[int, int, int, int, int, int]:
    """
    Calculate McNeillian space features for both hands.

    Args:
        df: DataFrame with pose keypoints indexed by joint name
        visibility: Optional visibility scores array
        visibility_threshold: Minimum visibility to consider a joint

    Returns:
        Tuple of (space_use_L, space_use_R, mcneillian_maxL, mcneillian_maxR,
                 mcneillian_modeL, mcneillian_modeR)
    """
    Space_L = []
    Space_R = []

    for frame in range(len(df['MidHip'])):
        try:
            grid = define_mcneillian_grid(df, frame)
            (cc_xmin, cc_xmax, cc_ymin, cc_ymax,
             c_xmin, c_xmax, c_ymin, c_ymax,
             p_xmin, p_xmax, p_ymin, p_ymax) = grid

            # Process left hand if visible
            if visibility is None or visibility[frame, 15] >= visibility_threshold:
                left_hand = df['L_Hand'][frame]
                x, y = left_hand[0], left_hand[1]
                zone = _classify_hand_zone(x, y, grid)
                Space_L.append(zone)

            # Process right hand if visible
            if visibility is None or visibility[frame, 16] >= visibility_threshold:
                right_hand = df['R_Hand'][frame]
                x, y = right_hand[0], right_hand[1]
                zone = _classify_hand_zone(x, y, grid)
                Space_R.append(zone)

        except Exception as e:
            print(f"Error in frame {frame}: {str(e)}")

    # Ensure we have data
    if not Space_L:
        Space_L = [1]
    if not Space_R:
        Space_R = [1]

    # Calculate statistics
    space_use_L = len(set(Space_L))
    space_use_R = len(set(Space_R))

    mcneillian_maxL = 4 if max(Space_L) > 40 else (3 if max(Space_L) > 30 else max(Space_L))
    mcneillian_maxR = 4 if max(Space_R) > 40 else (3 if max(Space_R) > 30 else max(Space_R))

    mcneillian_modeL = get_mcneillian_mode(Space_L)
    mcneillian_modeR = get_mcneillian_mode(Space_R)

    return (space_use_L, space_use_R, mcneillian_maxL, mcneillian_maxR,
            mcneillian_modeL, mcneillian_modeR)


def _classify_hand_zone(x: float, y: float, grid: Tuple) -> int:
    """Classify hand position into McNeillian zone."""
    (cc_xmin, cc_xmax, cc_ymin, cc_ymax,
     c_xmin, c_xmax, c_ymin, c_ymax,
     p_xmin, p_xmax, p_ymin, p_ymax) = grid

    # Center-center zone
    if cc_xmin < x < cc_xmax and cc_ymin < y < cc_ymax:
        return 1

    # Center zone
    if c_xmin < x < c_xmax and c_ymin < y < c_ymax:
        return 2

    # Periphery zone
    if p_xmin < x < p_xmax and p_ymin < y < p_ymax:
        if cc_xmax < x:  # Right side
            if cc_ymax < y:
                return 31
            elif cc_ymin < y:
                return 32
            else:
                return 33
        elif cc_xmin < x:  # Center
            if c_ymax < y:
                return 38
            else:
                return 34
        else:  # Left side
            if cc_ymax < y:
                return 37
            elif cc_ymin < y:
                return 36
            else:
                return 35

    # Extra-periphery zone
    if c_xmax < x:  # Right side
        if cc_ymax < y:
            return 41
        elif cc_ymin < y:
            return 42
        else:
            return 43
    elif cc_xmin < x:  # Center
        if c_ymax < y:
            return 48
        else:
            return 44
    else:  # Left side
        if c_ymax < y:
            return 47
        elif c_ymin < y:
            return 46
        else:
            return 45

    return 1  # Default to center-center


def calc_volume_size(df: pd.DataFrame, hand: str) -> float:
    """
    Calculate the volumetric size of the gesture space.

    Args:
        df: DataFrame with pose keypoints
        hand: Which hand to analyze ('L', 'R', or 'B' for both)

    Returns:
        Volume/area of the gesture space
    """
    # Collect hand positions into arrays for vectorized operations
    hand_list = ['R_Hand', 'L_Hand'] if hand == 'B' else [hand + '_Hand']

    # Stack all positions into a single array
    all_positions = []
    for hand_key in hand_list:
        positions = np.array([df[hand_key][i] for i in range(len(df[hand_key]))])
        all_positions.append(positions)

    # Concatenate all hand positions
    all_positions = np.vstack(all_positions)

    # Vectorized min/max computation
    x_min, y_min = np.min(all_positions[:, :2], axis=0)
    x_max, y_max = np.max(all_positions[:, :2], axis=0)

    # Calculate volume/area
    if all_positions.shape[1] > 2:
        z_min = np.min(all_positions[:, 2])
        z_max = np.max(all_positions[:, 2])
        vol = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    else:
        vol = (x_max - x_min) * (y_max - y_min)

    return vol


def calc_vert_height(
    df: pd.DataFrame,
    visibility: Optional[np.ndarray] = None,
    visibility_threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Calculate vertical height separately for each hand.

    Args:
        df: DataFrame with pose keypoints
        visibility: Optional visibility scores array
        visibility_threshold: Minimum visibility to consider

    Returns:
        Tuple of (max_height_L, max_height_R)
    """
    H_L = []
    H_R = []

    for frame in range(len(df['MidHip'])):
        try:
            mid_hip_y = df['MidHip'][frame][1]
            neck_y = df['Neck'][frame][1]
            nose_y = df['Nose'][frame][1]
            left_eye_y = df['LEye'][frame][1]
            right_eye_y = df['REye'][frame][1]

            body_height = mid_hip_y - neck_y
            head_height = neck_y - nose_y

            # Process left hand
            if visibility is None or visibility[frame, 15] >= visibility_threshold:
                left_hand_y = df['L_Hand'][frame][1]
                H_L.append(_calc_normalized_height(
                    left_hand_y, mid_hip_y, neck_y, nose_y, left_eye_y,
                    body_height, head_height
                ))
            else:
                H_L.append(0)

            # Process right hand
            if visibility is None or visibility[frame, 16] >= visibility_threshold:
                right_hand_y = df['R_Hand'][frame][1]
                H_R.append(_calc_normalized_height(
                    right_hand_y, mid_hip_y, neck_y, nose_y, right_eye_y,
                    body_height, head_height
                ))
            else:
                H_R.append(0)

        except Exception as e:
            print(f"Error in frame {frame}: {str(e)}")
            H_L.append(0)
            H_R.append(0)

    max_height_L = max(H_L) if H_L else 0
    max_height_R = max(H_R) if H_R else 0

    return max_height_L, max_height_R


def _calc_normalized_height(
    hand_y: float,
    mid_hip_y: float,
    neck_y: float,
    nose_y: float,
    eye_y: float,
    body_height: float,
    head_height: float
) -> float:
    """Calculate normalized height for a single hand position."""
    if hand_y >= mid_hip_y:
        return 0
    elif hand_y >= neck_y:
        height_ratio = (mid_hip_y - hand_y) / body_height
        return 1 + height_ratio
    elif hand_y >= nose_y:
        height_ratio = (neck_y - hand_y) / head_height
        return 2 + height_ratio
    elif hand_y >= eye_y:
        height_ratio = (nose_y - hand_y) / (nose_y - eye_y)
        return 3 + height_ratio
    else:
        return 5


def find_movepauses(velocity_array: np.ndarray) -> List[int]:
    """
    Find moments when velocity is below a threshold (0.15 m/s).

    Args:
        velocity_array: Array of velocities

    Returns:
        List of indices for pause moments
    """
    pause_ix = []
    for index, velpoint in enumerate(velocity_array):
        if velpoint < 0.15:
            pause_ix.append(index)
    if len(pause_ix) == 0:
        pause_ix = 0
    return pause_ix


def calculate_distance(
    positions: List,
    fps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate distance and velocity between consecutive positions.

    Args:
        positions: List of position arrays
        fps: Frames per second

    Returns:
        Tuple of (distances, velocities) as numpy arrays
    """
    # Convert to numpy array for vectorized operations
    pos_array = np.asarray(positions)

    if len(pos_array) < 2:
        return np.array([]), np.array([])

    # Vectorized: compute differences between consecutive positions
    diffs = np.diff(pos_array, axis=0)

    # Vectorized: compute Euclidean distances
    distances = np.linalg.norm(diffs, axis=1)

    # Vectorized: compute velocities
    velocities = distances * fps

    return distances, velocities


def calc_holds(
    df: pd.DataFrame,
    subslocs_L: np.ndarray,
    subslocs_R: np.ndarray,
    fps: float,
    hand: str
) -> Tuple[int, float, float]:
    """
    Calculate hold features (pauses in movement).

    Args:
        df: DataFrame with pose keypoints
        subslocs_L: Left hand submovement peak locations
        subslocs_R: Right hand submovement peak locations
        fps: Frames per second
        hand: Which hand to analyze ('L', 'R', or 'B' for both)

    Returns:
        Tuple of (hold_count, hold_time, hold_avg_duration)
    """
    try:
        # Initialize with safe defaults
        if not isinstance(subslocs_L, (list, np.ndarray)) or len(subslocs_L) == 0:
            subslocs_L = np.array([0])
        if not isinstance(subslocs_R, (list, np.ndarray)) or len(subslocs_R) == 0:
            subslocs_R = np.array([0])

        # Calculate hold features
        _, RE_S = calculate_distance(df["RElb"], fps)
        GERix = find_movepauses(RE_S)
        _, RH_S = calculate_distance(df["R_Hand"], fps)
        GRix = find_movepauses(RH_S)

        GR = []
        GL = []

        # Process right side holds
        if isinstance(GERix, list) and isinstance(GRix, list):
            for handhold in GRix:
                for elbowhold in GERix:
                    if handhold == elbowhold:
                        GR.append(handhold)

        # Process left side
        _, LE_S = calculate_distance(df["LElb"], fps)
        GELix = find_movepauses(LE_S)
        _, LH_S = calculate_distance(df["L_Hand"], fps)
        GLix = find_movepauses(LH_S)

        if isinstance(GELix, list) and isinstance(GLix, list):
            for handhold in GLix:
                for elbowhold in GELix:
                    if handhold == elbowhold:
                        GL.append(handhold)

        # Initialize holds with safe defaults
        hold_count = 0
        hold_time = 0
        hold_avg = 0

        # Process holds based on hand selection
        if ((hand == 'B' and GL and GR) or
            (hand == 'L' and GL) or
            (hand == 'R' and GR)):

            full_hold = []
            if hand == 'B':
                for left_hold in GL:
                    for right_hold in GR:
                        if left_hold == right_hold:
                            full_hold.append(left_hold)
            elif hand == 'L':
                full_hold = GL
            elif hand == 'R':
                full_hold = GR

            if full_hold:
                # Cluster holds
                hold_cluster = [[full_hold[0]]]
                clustercount = 0
                holdcount = 1

                for idx in range(1, len(full_hold)):
                    if full_hold[idx] != hold_cluster[clustercount][holdcount - 1] + 1:
                        clustercount += 1
                        holdcount = 1
                        hold_cluster.append([full_hold[idx]])
                    else:
                        hold_cluster[clustercount].append(full_hold[idx])
                        holdcount += 1

                # Filter holds based on initial movement
                try:
                    if hand == 'B':
                        initial_move = min(np.concatenate((subslocs_L, subslocs_R)))
                    elif hand == 'L':
                        initial_move = min(subslocs_L)
                    else:
                        initial_move = min(subslocs_R)

                    hold_cluster = [cluster for cluster in hold_cluster
                                   if cluster[0] >= initial_move]
                except Exception:
                    pass

                # Calculate statistics
                hold_durations = []
                for cluster in hold_cluster:
                    if len(cluster) >= 3:
                        hold_count += 1
                        hold_time += len(cluster)
                        hold_durations.append(len(cluster))

                hold_time = hold_time / fps if fps > 0 else 0
                hold_avg = float(np.mean(hold_durations)) if hold_durations else 0

        return hold_count, hold_time, hold_avg

    except Exception as e:
        print(f"Error in calc_holds: {str(e)}")
        return 0, 0, 0


def compute_kinematic_features(
    landmarks: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    fps: float = 25.0,
    gesture_id: str = "",
    video_id: str = ""
) -> KinematicFeatures:
    """
    Compute comprehensive kinematic features for a gesture using the more active hand.

    Args:
        landmarks: Array of pose landmarks with shape (N, num_keypoints, 3)
        visibility: Optional visibility scores array
        fps: Frames per second
        gesture_id: Identifier for this gesture
        video_id: Identifier for the source video

    Returns:
        KinematicFeatures object with all computed metrics
    """
    # Convert landmarks to DataFrame format using vectorized slicing
    joint_names = ['L_Hand', 'R_Hand', 'LElb', 'RElb', 'LShoulder', 'RShoulder',
                   'Neck', 'MidHip', 'LEye', 'REye', 'Nose']
    df = pd.DataFrame({
        joint: list(landmarks[:, JOINT_MAP[joint]])
        for joint in joint_names
    })

    # Analyze movement to determine active hand
    left_hand = landmarks[:, 15]
    right_hand = landmarks[:, 16]

    left_speeds = np.linalg.norm(np.diff(left_hand, axis=0), axis=1)
    right_speeds = np.linalg.norm(np.diff(right_hand, axis=0), axis=1)

    if visibility is not None:
        visibility_threshold = 0.5
        left_vis_mask = visibility[:-1, 15] >= visibility_threshold
        right_vis_mask = visibility[:-1, 16] >= visibility_threshold

        left_visible_frames = np.sum(visibility[:, 15] >= visibility_threshold)
        right_visible_frames = np.sum(visibility[:, 16] >= visibility_threshold)

        left_speeds = left_speeds * left_vis_mask
        right_speeds = right_speeds * right_vis_mask

        left_total = np.sum(left_speeds) * (len(visibility) / max(left_visible_frames, 1))
        right_total = np.sum(right_speeds) * (len(visibility) / max(right_visible_frames, 1))
    else:
        left_total = np.sum(left_speeds)
        right_total = np.sum(right_speeds)

    active_hand = 'L' if left_total > right_total else 'R'

    # Get keys for the active hand
    hand_key = 'L_Hand' if active_hand == 'L' else 'R_Hand'
    elbow_key = 'LElb' if active_hand == 'L' else 'RElb'

    # Calculate spatial features
    mcn_space = calc_mcneillian_space(df, visibility)
    space_use = mcn_space[0] if active_hand == 'L' else mcn_space[1]
    mcneillian_max = mcn_space[2] if active_hand == 'L' else mcn_space[3]
    mcneillian_mode = mcn_space[4] if active_hand == 'L' else mcn_space[5]

    volume = calc_volume_size(df, active_hand)

    max_heights = calc_vert_height(df, visibility)
    max_height = max_heights[0] if active_hand == 'L' else max_heights[1]

    # Compute kinematics for active arm
    hand = compute_limb_kinematics(np.array([p for p in df[hand_key]]), fps)
    elbow = compute_limb_kinematics(np.array([p for p in df[elbow_key]]), fps)

    # Calculate hold features
    if active_hand == 'L':
        hold_peaks = hand.peaks
        other_peaks = np.array([])
    else:
        hold_peaks = np.array([])
        other_peaks = hand.peaks

    hold_count, hold_time, hold_avg = calc_holds(
        df, hold_peaks, other_peaks, fps, active_hand
    )

    # Safe computation helpers
    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    def safe_max(arr):
        return float(np.max(arr)) if len(arr) > 0 else 0.0

    def safe_min(arr):
        return float(np.min(arr)) if len(arr) > 0 else 0.0

    def safe_norm(arr, axis=1):
        return np.linalg.norm(arr, axis=axis) if len(arr) > 0 else np.zeros(1)

    return KinematicFeatures(
        gesture_id=gesture_id,
        video_id=video_id,
        active_hand=active_hand,

        # Spatial features
        space_use=space_use,
        mcneillian_max=mcneillian_max,
        mcneillian_mode=mcneillian_mode,
        volume=volume,
        max_height=max_height,

        # Temporal features
        duration=len(landmarks) / fps,
        hold_count=hold_count,
        hold_time=hold_time,
        hold_avg_duration=hold_avg,

        # Hand submovements
        hand_submovements=len(hand.peaks),
        hand_submovement_peaks=hand.peak_heights.tolist() if len(hand.peak_heights) > 0 else [0],
        hand_mean_submovement_amplitude=safe_mean(hand.peak_heights),

        # Elbow submovements
        elbow_submovements=len(elbow.peaks),
        elbow_mean_submovement_amplitude=safe_mean(elbow.peak_heights),

        # Hand dynamics
        hand_peak_speed=safe_max(hand.speed),
        hand_mean_speed=safe_mean(hand.speed),
        hand_peak_acceleration=safe_max(safe_norm(hand.acceleration)),
        hand_peak_deceleration=safe_min(safe_norm(hand.acceleration)),
        hand_peak_jerk=safe_max(safe_norm(hand.jerk)),

        # Elbow dynamics
        elbow_peak_speed=safe_max(elbow.speed),
        elbow_mean_speed=safe_mean(elbow.speed),
        elbow_peak_acceleration=safe_max(safe_norm(elbow.acceleration)),
        elbow_peak_deceleration=safe_min(safe_norm(elbow.acceleration)),
        elbow_peak_jerk=safe_max(safe_norm(elbow.jerk))
    )
