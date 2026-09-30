"""Spatial measurements for tracked gestures."""

import statistics


def define_mcneillian_grid(df, frame):
    """Define the grid based on original implementation."""
    bodycent = df['Neck'][frame][1] - (df['Neck'][frame][1] - df['MidHip'][frame][1])/2
    face_width = (df['LEye'][frame][0] - df['REye'][frame][0])*2
    # Use shoulders instead of hips for more reliable body width
    body_width = df['LShoulder'][frame][0] - df['RShoulder'][frame][0]

    # Center-center boundaries
    cc_xmin = df['RShoulder'][frame][0]
    cc_xmax = df['LShoulder'][frame][0]
    cc_len = cc_xmax - cc_xmin
    cc_ymin = bodycent - cc_len/2
    cc_ymax = bodycent + cc_len/2

    # Center boundaries
    c_xmin = df['RShoulder'][frame][0] - body_width/2
    c_xmax = df['LShoulder'][frame][0] + body_width/2
    c_len = c_xmax - c_xmin
    c_ymin = bodycent - c_len/2
    c_ymax = bodycent + c_len/2

    # Periphery boundaries
    p_ymax = df['LEye'][frame][1] + (df['LEye'][frame][1] - df['Nose'][frame][1])
    p_ymin = bodycent - (p_ymax - bodycent)
    p_xmin = c_xmin - face_width
    p_xmax = c_xmax + face_width

    return cc_xmin, cc_xmax, cc_ymin, cc_ymax, c_xmin, c_xmax, c_ymin, c_ymax, p_xmin, p_xmax, p_ymin, p_ymax


def get_mcneillian_mode(spaces):
    """Convert subsection codes to main sections and calculate mode."""
    mainspace = []
    for space in spaces:
        if space > 40:
            mainspace.append(4)
        elif space > 30:
            mainspace.append(3)
        else:
            mainspace.append(space)

    return statistics.mode(mainspace)


def calc_mcneillian_space(df, visibility=None, visibility_threshold=0.5):
    """Calculate McNeillian space features using original implementation approach."""
    Space_L = []
    Space_R = []
    
    for frame in range(len(df['MidHip'])):
        try:
            # Get grid boundaries
            cc_xmin, cc_xmax, cc_ymin, cc_ymax, c_xmin, c_xmax, c_ymin, c_ymax, p_xmin, p_xmax, p_ymin, p_ymax = \
                define_mcneillian_grid(df, frame)
            
            # Process left hand if visible
            if visibility is None or visibility[frame, 15] >= visibility_threshold:
                left_hand = df['L_Hand'][frame]
                x, y = left_hand[0], left_hand[1]
                
                # Assign zone with subsections
                if cc_xmin < x < cc_xmax and cc_ymin < y < cc_ymax:
                    Space_L.append(1)
                elif c_xmin < x < c_xmax and c_ymin < y < c_ymax:
                    Space_L.append(2)
                elif p_xmin < x < p_xmax and p_ymin < y < p_ymax:
                    # Periphery subsections
                    if cc_xmax < x:  # Right side
                        if cc_ymax < y:
                            Space_L.append(31)
                        elif cc_ymin < y:
                            Space_L.append(32)
                        else:
                            Space_L.append(33)
                    elif cc_xmin < x:  # Center
                        if c_ymax < y:
                            Space_L.append(38)
                        else:
                            Space_L.append(34)
                    else:  # Left side
                        if cc_ymax < y:
                            Space_L.append(37)
                        elif cc_ymin < y:
                            Space_L.append(36)
                        else:
                            Space_L.append(35)
                else:  # Extra-periphery subsections
                    if c_xmax < x:  # Right side
                        if cc_ymax < y:
                            Space_L.append(41)
                        elif cc_ymin < y:
                            Space_L.append(42)
                        else:
                            Space_L.append(43)
                    elif cc_xmin < x:  # Center
                        if c_ymax < y:
                            Space_L.append(48)
                        else:
                            Space_L.append(44)
                    else:  # Left side
                        if c_ymax < y:
                            Space_L.append(47)
                        elif c_ymin < y:
                            Space_L.append(46)
                        else:
                            Space_L.append(45)
            
            # Process right hand similarly
            if visibility is None or visibility[frame, 16] >= visibility_threshold:
                right_hand = df['R_Hand'][frame]
                x, y = right_hand[0], right_hand[1]
                
                # Same zone assignment logic for right hand
                if cc_xmin < x < cc_xmax and cc_ymin < y < cc_ymax:
                    Space_R.append(1)
                elif c_xmin < x < c_xmax and c_ymin < y < c_ymax:
                    Space_R.append(2)
                elif p_xmin < x < p_xmax and p_ymin < y < p_ymax:
                    if cc_xmax < x:
                        if cc_ymax < y:
                            Space_R.append(31)
                        elif cc_ymin < y:
                            Space_R.append(32)
                        else:
                            Space_R.append(33)
                    elif cc_xmin < x:
                        if c_ymax < y:
                            Space_R.append(38)
                        else:
                            Space_R.append(34)
                    else:
                        if cc_ymax < y:
                            Space_R.append(37)
                        elif cc_ymin < y:
                            Space_R.append(36)
                        else:
                            Space_R.append(35)
                else:
                    if c_xmax < x:
                        if cc_ymax < y:
                            Space_R.append(41)
                        elif cc_ymin < y:
                            Space_R.append(42)
                        else:
                            Space_R.append(43)
                    elif cc_xmin < x:
                        if c_ymax < y:
                            Space_R.append(48)
                        else:
                            Space_R.append(44)
                    else:
                        if c_ymax < y:
                            Space_R.append(47)
                        elif c_ymin < y:
                            Space_R.append(46)
                        else:
                            Space_R.append(45)
                            
        except Exception as e:
            print(f"Error in frame {frame}: {str(e)}")
    
    # Ensure we have data
    if not Space_L:
        Space_L = [1]
    if not Space_R:
        Space_R = [1]
    
    # Calculate statistics using original method
    space_use_L = len(set(Space_L))
    space_use_R = len(set(Space_R))
    
    mcneillian_maxL = 4 if max(Space_L) > 40 else (3 if max(Space_L) > 30 else max(Space_L))
    mcneillian_maxR = 4 if max(Space_R) > 40 else (3 if max(Space_R) > 30 else max(Space_R))
    
    mcneillian_modeL = get_mcneillian_mode(Space_L)
    mcneillian_modeR = get_mcneillian_mode(Space_R)
    
    return (space_use_L, space_use_R, mcneillian_maxL, mcneillian_maxR, 
            mcneillian_modeL, mcneillian_modeR)


def calc_volume_size(df, hand):
    """
    Calculate the volumetric size of the gesture space, adapted for MediaPipe landmarks.
    
    Args:
        df: DataFrame with pose keypoints
        hand: Which hand to analyze ('L', 'R', or 'B' for both)
        
    Returns:
        Volume/area of the gesture space
    """
    # Initialize boundaries from first frame
    if hand == 'B':
        x_max = max([df['R_Hand'][0][0], df['L_Hand'][0][0]])
        x_min = min([df['R_Hand'][0][0], df['L_Hand'][0][0]])
        y_max = max([df['R_Hand'][0][1], df['L_Hand'][0][1]])  # Fixed y coordinate selection
        y_min = min([df['R_Hand'][0][1], df['L_Hand'][0][1]])  # Fixed y coordinate selection
        if len(df['R_Hand'][0]) > 2:  # If 3D
            z_max = max([df['R_Hand'][0][2], df['L_Hand'][0][2]])
            z_min = min([df['R_Hand'][0][2], df['L_Hand'][0][2]])
    else:
        hand_str = hand + '_Hand'
        x_min = x_max = df[hand_str][0][0]
        y_min = y_max = df[hand_str][0][1]  # Fixed y coordinate selection
        if len(df[hand_str][0]) > 2:  # If 3D
            z_min = z_max = df[hand_str][0][2]

    # Process all frames to find extremes
    hand_list = ['R_Hand', 'L_Hand'] if hand == 'B' else [hand + '_Hand']
    
    for frame in range(len(df)):
        for hand_idx in hand_list:
            curr_pos = df[hand_idx][frame]
            x_min = min(x_min, curr_pos[0])
            x_max = max(x_max, curr_pos[0])
            y_min = min(y_min, curr_pos[1])
            y_max = max(y_max, curr_pos[1])
            if len(curr_pos) > 2:  # If 3D
                z_min = min(z_min, curr_pos[2])
                z_max = max(z_max, curr_pos[2])

    # Calculate volume/area
    if len(df[hand_list[0]][0]) > 2:  # If 3D
        vol = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    else:  # If 2D
        vol = (x_max - x_min) * (y_max - y_min)
    
    return vol


def calc_vert_height(df, visibility=None, visibility_threshold=0.5):
    """
    Calculate vertical height separately and independently for each hand.
    Corrected to handle coordinate system properly and normalize heights.
    """
    H_L = []
    H_R = []
    
    for frame in range(len(df['MidHip'])):  # Iterate using range instead of index
        # Get reference points for normalization
        try:
            # Note: MediaPipe uses y-down coordinate system, so smaller y values are higher
            mid_hip_y = df['MidHip'][frame][1]
            neck_y = df['Neck'][frame][1]
            nose_y = df['Nose'][frame][1]
            left_eye_y = df['LEye'][frame][1]
            right_eye_y = df['REye'][frame][1]
            
            # Calculate body-scaled reference heights
            body_height = mid_hip_y - neck_y
            head_height = neck_y - nose_y
            
            # Process left hand if visible
            if visibility is None or visibility[frame, 15] >= visibility_threshold:
                left_hand_y = df['L_Hand'][frame][1]
                
                # Normalize height relative to body proportions
                if left_hand_y >= mid_hip_y:  # Below hip
                    H_L.append(0)
                elif left_hand_y >= neck_y:  # Between hip and neck
                    height_ratio = (mid_hip_y - left_hand_y) / body_height
                    H_L.append(1 + height_ratio)
                elif left_hand_y >= nose_y:  # Between neck and nose
                    height_ratio = (neck_y - left_hand_y) / head_height
                    H_L.append(2 + height_ratio)
                elif left_hand_y >= left_eye_y:  # Between nose and eye
                    height_ratio = (nose_y - left_hand_y) / (nose_y - left_eye_y)
                    H_L.append(3 + height_ratio)
                else:  # Above eye
                    H_L.append(5)
            else:
                H_L.append(0)
                
            # Process right hand if visible
            if visibility is None or visibility[frame, 16] >= visibility_threshold:
                right_hand_y = df['R_Hand'][frame][1]
                
                # Normalize height relative to body proportions
                if right_hand_y >= mid_hip_y:  # Below hip
                    H_R.append(0)
                elif right_hand_y >= neck_y:  # Between hip and neck
                    height_ratio = (mid_hip_y - right_hand_y) / body_height
                    H_R.append(1 + height_ratio)
                elif right_hand_y >= nose_y:  # Between neck and nose
                    height_ratio = (neck_y - right_hand_y) / head_height
                    H_R.append(2 + height_ratio)
                elif right_hand_y >= right_eye_y:  # Between nose and eye
                    height_ratio = (nose_y - right_hand_y) / (nose_y - right_eye_y)
                    H_R.append(3 + height_ratio)
                else:  # Above eye
                    H_R.append(5)
            else:
                H_R.append(0)
                
        except Exception as e:
            print(f"Error in frame {frame}: {str(e)}")
            H_L.append(0)
            H_R.append(0)
    
    # Calculate maximum heights with proper normalization
    max_height_L = max(H_L) if H_L else 0
    max_height_R = max(H_R) if H_R else 0
    
    return max_height_L, max_height_R
