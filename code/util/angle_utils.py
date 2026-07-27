import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def determine_angle_range(image_name: str) -> Tuple[float, float]:

    image_name_upper = image_name.upper()
    
    if 'BL' in image_name_upper:
        return 90.0, 180.0
    elif 'BR' in image_name_upper:
        return 0.0, 90.0
    elif 'TR' in image_name_upper:
        return -90.0, 0.0
    elif 'TL' in image_name_upper:
        return -180.0, -90.0
    else:
        # Default to full range if quadrant cannot be determined
        return -180.0, 180.0


def normalize_angle(angle: float, to_0_360: bool = True) -> float:

    if to_0_360:
        # Convert to 0-360 scale
        return angle % 360
    else:
        # Convert to -180 to 180 scale
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle


def get_sample_angles(angle_start: float, angle_end: float, n_angles: int = 5) -> List[float]:
    return list(np.linspace(angle_start, angle_end, n_angles))


def get_cells_at_angle(
    df: pd.DataFrame,
    target_angle: float,
    tolerance: float = 5.0,
    angle_column: str = 'angle'
) -> pd.DataFrame:

    if angle_column not in df.columns:
        raise ValueError(f"Column '{angle_column}' not found in DataFrame")
    
    # Handle circular nature of angles
    # For negative angles or angles near boundaries, we need special handling
    lower_bound = target_angle - tolerance
    upper_bound = target_angle + tolerance
    
    # Simple case: no wraparound
    if lower_bound >= -180 and upper_bound <= 180:
        mask = (df[angle_column] >= lower_bound) & (df[angle_column] <= upper_bound)
        return df[mask].copy()
    
    # Handle wraparound at -180/180 boundary
    if lower_bound < -180:
        mask = (df[angle_column] >= lower_bound + 360) | (df[angle_column] <= upper_bound)
    elif upper_bound > 180:
        mask = (df[angle_column] >= lower_bound) | (df[angle_column] <= upper_bound - 360)
    else:
        mask = (df[angle_column] >= lower_bound) & (df[angle_column] <= upper_bound)
    
    return df[mask].copy()


def get_angle_bins(n_bins: int = 8, start_angle: float = 0.0, end_angle: float = 360.0) -> np.ndarray:
    return np.linspace(start_angle, end_angle, n_bins + 1)


def angle_to_radians(angle: float) -> float:
    return np.deg2rad(angle)


def angle_from_radians(radians: float) -> float:
    return np.rad2deg(radians)


def calculate_angular_statistics(angles: np.ndarray) -> dict:
    # Convert to radians
    angles_rad = np.deg2rad(angles)
    
    # Calculate circular mean
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    mean_angle_rad = np.arctan2(sin_mean, cos_mean)
    mean_angle = np.rad2deg(mean_angle_rad)
    
    # Calculate resultant length (measure of concentration)
    resultant_length = np.sqrt(sin_mean**2 + cos_mean**2)
    
    # Calculate circular standard deviation
    if resultant_length > 0:
        std_angle = np.rad2deg(np.sqrt(-2 * np.log(resultant_length)))
    else:
        std_angle = np.nan
    
    return {
        'mean_angle': mean_angle,
        'std_angle': std_angle,
        'resultant_length': resultant_length
    }
