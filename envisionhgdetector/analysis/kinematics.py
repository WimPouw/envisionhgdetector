"""Kinematic calculations for tracked gestures."""

import statistics
from typing import NamedTuple, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy import signal


class ArmKinematics(NamedTuple):
    """Container for arm kinematic measurements."""
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    speed: np.ndarray
    peaks: np.ndarray
    peak_heights: np.ndarray


def calculate_derivatives(
    positions: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate velocity, acceleration, and jerk from position data."""
    if not isinstance(positions, np.ndarray) or positions.size == 0:
        raise ValueError("positions must be a non-empty numpy array")
    if fps <= 0:
        raise ValueError("fps must be positive")

    dt = 1 / fps
    positions = gaussian_filter1d(positions, sigma=2, axis=0)
    velocity = np.gradient(positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)
    return velocity, acceleration, jerk


def compute_limb_kinematics(positions: np.ndarray, fps: float) -> ArmKinematics:
    """Compute derivatives, speed, and submovements for a limb segment."""
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
        peak_heights=peak_heights,
    )


def find_submovements(speed_profile: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find submovements in a speed profile using peak detection.
    
    Args:
        speed_profile: Array of speeds over time
        fps: Frames per second
        
    Returns:
        Tuple of (peaks indices, peak heights)
    """
    # Handle very short sequences
    if len(speed_profile) < 3:
        # For very short sequences, just return the maximum as a peak
        if len(speed_profile) > 0:
            max_idx = np.argmax(speed_profile)
            return np.array([max_idx]), np.array([speed_profile[max_idx]])
        else:
            return np.array([0]), np.array([0])
    
    # Apply Savitzky-Golay smoothing with proper parameter handling
    if len(speed_profile) >= 15:
        # Use standard parameters for longer sequences
        smoothed = signal.savgol_filter(speed_profile, 15, 5)
    else:
        # For shorter sequences, adjust window and polyorder appropriately
        window = len(speed_profile)
        
        # Ensure window is odd
        if window % 2 == 0:
            window = window - 1
        
        # Ensure minimum window size
        if window < 3:
            window = 3
        
        # Adjust polyorder to be less than window_length
        # polyorder must be < window_length, so max polyorder = window - 1
        polyorder = min(5, window - 1)
        
        # Ensure polyorder is at least 1
        polyorder = max(1, polyorder)
        
        # Additional safety check: if window is too small, use simple smoothing
        if window < 5 or polyorder < 1:
            # For very short sequences, use simple moving average instead
            if len(speed_profile) >= 3:
                smoothed = np.convolve(speed_profile, np.ones(3)/3, mode='same')
            else:
                smoothed = speed_profile.copy()
        else:
            try:
                smoothed = signal.savgol_filter(speed_profile, window, polyorder)
            except ValueError:
                # Fallback to simple moving average if savgol still fails
                smoothed = np.convolve(speed_profile, np.ones(3)/3, mode='same')
    
    # Find peaks with prominence and distance constraints
    peaks, properties = signal.find_peaks(
        smoothed,
        distance=max(1, int(5 * fps / 25)),  # Scale distance with fps
        height=0,  # Include height to get peak heights
        prominence=max(0.01, np.std(smoothed) * 0.1)  # Adaptive prominence based on signal variability
    )
    
    # Get peak heights from the smoothed signal
    peak_heights = smoothed[peaks] if len(peaks) > 0 else np.array([0])
    
    # If no peaks found, use the maximum value as a peak
    if len(peaks) == 0:
        max_idx = np.argmax(smoothed)
        peaks = np.array([max_idx])
        peak_heights = np.array([smoothed[max_idx]])
    
    return peaks, peak_heights


def find_movepauses(velocity_array):
    """Find moments when velocity is below a threshold."""
    pause_ix = []
    for index, velpoint in enumerate(velocity_array):
        if velpoint < 0.15:
            pause_ix.append(index)
    if len(pause_ix) == 0:
        pause_ix = 0
    return pause_ix


def calculate_distance(positions, fps):
    """Calculate distance and velocity between consecutive positions."""
    distances = []
    velocities = []

    for i in range(1, len(positions)):
        dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
        distances.append(dist)
        velocities.append(dist * fps)

    return distances, velocities


def calc_holds(df, subslocs_L, subslocs_R, FPS, hand):
    """Calculate hold features with safety checks."""
    try:
        # Initialize with safe defaults
        if not isinstance(subslocs_L, (list, np.ndarray)) or len(subslocs_L) == 0:
            subslocs_L = np.array([0])
        if not isinstance(subslocs_R, (list, np.ndarray)) or len(subslocs_R) == 0:
            subslocs_R = np.array([0])
            
        # Calculate hold features with safety checks
        _, RE_S = calculate_distance(df["RElb"], FPS)
        GERix = find_movepauses(RE_S)
        _, RH_S = calculate_distance(df["R_Hand"], FPS)
        GRix = find_movepauses(RH_S)
        GFRix = GRix  # Default to hand if no finger data

        # Initialize empty lists for holds
        GR = []
        GL = []

        # Process right side holds
        if isinstance(GERix, list) and isinstance(GRix, list):
            for handhold in GRix:
                for elbowhold in GERix:
                    if handhold == elbowhold:
                        GR.append(handhold)

        # Process left side
        _, LE_S = calculate_distance(df["LElb"], FPS)
        GELix = find_movepauses(LE_S)
        _, LH_S = calculate_distance(df["L_Hand"], FPS)
        GLix = find_movepauses(LH_S)
        GFLix = GLix  # Default to hand if no finger data

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

                    hold_cluster = [cluster for cluster in hold_cluster if cluster[0] >= initial_move]
                except:
                    pass  # Keep all clusters if filtering fails

                # Calculate statistics
                hold_durations = []
                for cluster in hold_cluster:
                    if len(cluster) >= 3:
                        hold_count += 1
                        hold_time += len(cluster)
                        hold_durations.append(len(cluster))

                # Calculate final metrics with safety checks
                hold_time = hold_time / FPS if FPS > 0 else 0
                hold_avg = statistics.mean(hold_durations) if hold_durations else 0

        return hold_count, hold_time, hold_avg

    except Exception as e:
        print(f"Error in calc_holds: {str(e)}")
        return 0, 0, 0  # Return safe defaults if anything fails
