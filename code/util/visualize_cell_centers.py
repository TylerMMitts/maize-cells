# Draws detected cell centres onto the original image.
#
# The quickest way to see whether segmentation has gone wrong on an image,
# and the basis of the QC review thumbnails.

import cv2
import numpy as np
import os
import re
from ultralytics import YOLO
from code.yolo.test_robust import extract_cell_data, filter_cells_by_inner_part, get_inner_part_mask
from code.config import DATA_FOLDER, EXCLUSION_WEIGHTS, ROBUST_CELL_WEIGHTS

def get_plant_center_from_filename(filename, image_width=1080, image_height=1080):
    
    # Based on which quadrant the image is from, return the expected plant center coordinates
    match = re.search(r'(BL|BR|TL|TR)', filename.upper())
    if not match:
        return None, None, None
    
    quadrant = match.group(1)
    
    if quadrant == 'BL':
        center_x, center_y = image_width, 0
    elif quadrant == 'BR':
        center_x, center_y = 0, 0
    elif quadrant == 'TL':
        center_x, center_y = image_width, image_height
    elif quadrant == 'TR':
        center_x, center_y = 0, image_height
    else:
        return None, None, None
    
    return center_x, center_y, quadrant

def draw_red_dots_on_image(image, cell_centers, output_path, plant_center=None, quadrant=None):
    img = image.copy()
    
    for center in cell_centers:
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    
    if plant_center is not None:
        cx, cy = plant_center
        cv2.circle(img, (int(cx), int(cy)), 8, (255, 0, 0), -1)
        if quadrant:
            cv2.putText(img, f"Plant Center ({quadrant})", (int(cx) + 10, int(cy) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.imwrite(output_path, img)
    return img

def visualize_cell_centers(
    weights_path=ROBUST_CELL_WEIGHTS,
    exclusion_weights=EXCLUSION_WEIGHTS,
    image_path="test_image.jpg",
    confidence=0.99,
    exclusion_confidence=0.5,
    output_path="cell_centers.jpg",
    show_plant_center=True,
    show_result=False,
    refine=True,
    color_tolerance=15,
    filter_by_inner_part=True,
    inner_open_kernel=99,
    inner_close_kernel=155,
    outer_open_kernel=99,
    outer_close_kernel=155,
    exclusion_overlap_threshold=0.0
):
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    model = YOLO(weights_path)
    results = model(image_path, conf=confidence, max_det=3000)
    
    img = cv2.imread(image_path)
    filename = os.path.basename(image_path)
    
    plant_center = None
    quadrant = None
    if show_plant_center:
        h, w = img.shape[:2]
        center_x, center_y, quadrant = get_plant_center_from_filename(filename, w, h)
        if center_x is not None:
            plant_center = (center_x, center_y)
    
    cell_masks, cell_centers, cell_confidences = extract_cell_data(results, img, class_id=0, refine=refine, color_tolerance=color_tolerance)
    
    if filter_by_inner_part and os.path.exists(exclusion_weights):
        exclusion_model = YOLO(exclusion_weights)
        exclusion_results = exclusion_model(image_path, conf=exclusion_confidence, max_det=3000)
        
        inner_mask = get_inner_part_mask(
            exclusion_results, img,
            inner_open_kernel=inner_open_kernel,
            inner_close_kernel=inner_close_kernel,
            inner_open_iterations=1,
            inner_close_iterations=1,
            outer_open_kernel=outer_open_kernel,
            outer_close_kernel=outer_close_kernel,
            outer_open_iterations=1,
            outer_close_iterations=1,
            keep_largest_component=True
        )
        
        if np.sum(inner_mask) > 0:
            cells_before = len(cell_masks)
            cell_masks, cell_centers, cell_confidences = filter_cells_by_inner_part(
                cell_masks, cell_centers, cell_confidences, inner_mask, exclusion_overlap_threshold
            )
            cells_removed = cells_before - len(cell_centers)
            print(f"Cells removed by exclusion filter: {cells_removed}")
    
    draw_red_dots_on_image(img, cell_centers, output_path, plant_center, quadrant)
    print(f"Visualization saved to {output_path}")
    
    if show_result:
        img_display = cv2.imread(output_path)
        cv2.imshow("Cell Centers", img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def batch_visualize_cell_centers(
    weights_path=ROBUST_CELL_WEIGHTS,
    exclusion_weights=EXCLUSION_WEIGHTS,
    image_folder=DATA_FOLDER / 'cropped_images',
    confidence=0.99,
    exclusion_confidence=0.5,
    output_dir="cell_centers_output",
    show_plant_center=True,
    refine=True,
    color_tolerance=15,
    filter_by_inner_part=True,
    inner_open_kernel=99,
    inner_close_kernel=155,
    outer_open_kernel=99,
    outer_close_kernel=155,
    exclusion_overlap_threshold=0.0
):
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return
    
    if not os.path.exists(image_folder):
        print(f"Folder not found: {image_folder}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(weights_path)
    
    exclusion_model = None
    if filter_by_inner_part:
        if not os.path.exists(exclusion_weights):
            print(f"Exclusion model not found: {exclusion_weights}")
            filter_by_inner_part = False
        else:
            exclusion_model = YOLO(exclusion_weights)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]
    
    if not images:
        print(f"No images found in {image_folder}")
        return
    
    total_cells = 0
    
    for img_file in images:
        img_path = os.path.join(image_folder, img_file)
        base_name = os.path.splitext(img_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}_centers.jpg")
        
        results = model(img_path, conf=confidence, max_det=3000)
        img = cv2.imread(img_path)
        filename = os.path.basename(img_path)
        
        plant_center = None
        quadrant = None
        if show_plant_center:
            h, w = img.shape[:2]
            center_x, center_y, quadrant = get_plant_center_from_filename(filename, w, h)
            if center_x is not None:
                plant_center = (center_x, center_y)
        
        cell_masks, cell_centers, cell_confidences = extract_cell_data(results, img, class_id=0, refine=refine, color_tolerance=color_tolerance)
        
        cells_removed = 0
        if filter_by_inner_part and exclusion_model is not None:
            exclusion_results = exclusion_model(img_path, conf=exclusion_confidence, max_det=3000)
            
            inner_mask = get_inner_part_mask(
                exclusion_results, img,
                inner_open_kernel=inner_open_kernel,
                inner_close_kernel=inner_close_kernel,
                inner_open_iterations=1,
                inner_close_iterations=1,
                outer_open_kernel=outer_open_kernel,
                outer_close_kernel=outer_close_kernel,
                outer_open_iterations=1,
                outer_close_iterations=1,
                keep_largest_component=True
            )
            
            if np.sum(inner_mask) > 0:
                cells_before = len(cell_masks)
                cell_masks, cell_centers, cell_confidences = filter_cells_by_inner_part(
                    cell_masks, cell_centers, cell_confidences, inner_mask, exclusion_overlap_threshold
                )
                cells_removed = cells_before - len(cell_centers)
        
        draw_red_dots_on_image(img, cell_centers, output_path, plant_center, quadrant)
        
        total_cells += len(cell_centers)
        quadrant_str = f" (Quadrant: {quadrant})" if quadrant else ""
        removed_str = f", removed: {cells_removed}" if cells_removed > 0 else ""
        print(f"{img_file}: {len(cell_centers)} cells{quadrant_str}{removed_str}")
    
    print(f"Results saved to {output_dir}")

