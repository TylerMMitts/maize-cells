# Geometry read off the image itself.
#
# Plant centre, stele area and root radius. These are the measurements that
# hand-annotated species lack, which is why they drop out of some analyses.

import cv2
import numpy as np
import json
import os
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Default conversion factor: 50 µm = 77 pixels
DEFAULT_PIXEL_TO_UM = 50.0 / 77.0
DEFAULT_PIXEL_TO_UM_SQUARED = DEFAULT_PIXEL_TO_UM * DEFAULT_PIXEL_TO_UM


def get_plant_center_from_filename(filename: str) -> Optional[Tuple[int, int]]:

    base_name = os.path.basename(filename)
    
    # Placeholder: In actual use, load from a centers file or metadata
    # For now, return None to indicate center should be calculated from image
    return None


def calculate_plant_center_from_image(image: np.ndarray, method: str = 'center') -> Tuple[int, int]:

    height, width = image.shape[:2]
    
    if method == 'center':
        return width // 2, height // 2
    
    elif method == 'mass':
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate center of mass
        M = cv2.moments(gray)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        else:
            return width // 2, height // 2
    
    elif method == 'contour':
        # Convert to grayscale and threshold
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy
        
        return width // 2, height // 2
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_stele_area(inner_mask: np.ndarray, pixel_to_um: float = DEFAULT_PIXEL_TO_UM) -> float:

    if inner_mask is None or np.sum(inner_mask) == 0:
        return 0.0
    
    # Calculate area in pixels for the quarter
    quarter_stele_area_px = np.sum(inner_mask > 0)
    
    # Convert to µm² for the quarter
    quarter_stele_area_um2 = quarter_stele_area_px * (pixel_to_um ** 2)
    
    # Multiply by 4 to get full root stele area
    full_stele_area_um2 = quarter_stele_area_um2 * 4
    
    return full_stele_area_um2


def calculate_root_radius(
    inner_mask: np.ndarray,
    center_x: int,
    center_y: int,
    pixel_to_um: float = DEFAULT_PIXEL_TO_UM
) -> float:

    if inner_mask is None or np.sum(inner_mask) == 0:
        return 0.0
    
    # Get coordinates of all pixels in the mask
    y_coords, x_coords = np.where(inner_mask > 0)
    
    if len(x_coords) == 0:
        return 0.0
    
    # Calculate distances from center to all mask pixels
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    
    # Maximum distance is the root radius
    max_distance_px = np.max(distances)
    root_radius_um = max_distance_px * pixel_to_um
    
    return root_radius_um


def calculate_stele_diameter(stele_area: float) -> float:
    if stele_area <= 0:
        return 0.0
    
    # Area = π * r², so r = sqrt(Area / π)
    radius = np.sqrt(stele_area / np.pi)
    diameter = 2 * radius
    
    return diameter


def load_image_metadata(image_path: str) -> Optional[Dict[str, Any]]:

    # Try direct metadata file
    for ext in ['.jpg', '.png', '.jpeg', '.tif', '.tiff']:
        metadata_path = image_path.replace(ext, '_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata
            except Exception as e:
                logger.warning(f"Could not load metadata from {metadata_path}: {e}")
    
    # Try master metadata file
    dir_path = os.path.dirname(image_path)
    master_metadata_path = os.path.join(dir_path, 'all_crops_metadata.json')
    
    if os.path.exists(master_metadata_path):
        try:
            with open(master_metadata_path, 'r') as f:
                all_metadata = json.load(f)
                
                # Find metadata for this specific image
                image_name = os.path.basename(image_path)
                for item in all_metadata:
                    if os.path.basename(item.get('crop_path', '')) == image_name:
                        # Load the individual metadata file
                        if os.path.exists(item['metadata_path']):
                            with open(item['metadata_path'], 'r') as mf:
                                metadata = json.load(mf)
                                logger.info(f"Loaded metadata from master index for {image_name}")
                                return metadata
        except Exception as e:
            logger.warning(f"Could not load master metadata: {e}")
    
    logger.debug(f"No metadata found for {image_path}, using default conversion")
    return None


def get_conversion_factor_from_metadata(metadata: Optional[Dict[str, Any]]) -> float:

    if metadata and 'conversion_factor' in metadata:
        return metadata['conversion_factor'].get('pixels_to_um', DEFAULT_PIXEL_TO_UM)
    
    return DEFAULT_PIXEL_TO_UM


def calculate_area_from_mask(mask: np.ndarray) -> float:

    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            return float(area)
        return 0.0
    except Exception as e:
        logger.warning(f"Error calculating area from mask: {e}")
        return 0.0


def calculate_perimeter_from_mask(mask: np.ndarray) -> float:
    # Calculate perimeter from a binary mask using contour detection.
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(max_contour, True)
            return float(perimeter)
        return 0.0
    except Exception as e:
        logger.warning(f"Error calculating perimeter from mask: {e}")
        return 0.0


def calculate_centroid_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    # Calculate centroid coordinates from a binary mask.
    try:
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return float(cx), float(cy)
        return 0.0, 0.0
    except Exception as e:
        logger.warning(f"Error calculating centroid from mask: {e}")
        return 0.0, 0.0


def polar_to_cartesian(radius: float, angle: float, center_x: float, center_y: float) -> Tuple[float, float]:

    angle_rad = np.deg2rad(angle)
    x = center_x + radius * np.cos(angle_rad)
    y = center_y + radius * np.sin(angle_rad)
    return float(x), float(y)


def cartesian_to_polar(x: float, y: float, center_x: float, center_y: float) -> Tuple[float, float]:

    dx = x - center_x
    dy = y - center_y
    radius = np.sqrt(dx**2 + dy**2)
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.rad2deg(angle_rad)
    return float(radius), float(angle_deg)
