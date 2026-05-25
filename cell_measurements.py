import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from test_robust import extract_cell_data, filter_cells_by_inner_part, get_inner_part_mask
from visualize_cell_centers import get_plant_center_from_filename, draw_red_dots_on_image

def extract_cell_measurements(
    weights_path="runs/robust_segmentation/robust_cell_detector/weights/best.pt",
    exclusion_weights="runs/segment/runs/exclusion_model/inner_part_detector/weights/best.pt",
    image_path="test_image.jpg",
    confidence=0.99,
    exclusion_confidence=0.5,
    output_csv="cell_measurements.csv",
    output_image="cell_centers.jpg",
    show_result=False,
    refine=True,
    color_tolerance=15,
    filter_by_inner_part=True,
    morph_open_kernel=99,
    morph_close_kernel=155
):
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return None
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    model = YOLO(weights_path)
    results = model(image_path, conf=confidence, max_det=3000)
    
    img = cv2.imread(image_path)
    filename = os.path.basename(image_path)
    
    h, w = img.shape[:2]
    center_x, center_y, quadrant = get_plant_center_from_filename(filename, w, h)
    
    if center_x is None:
        print(f"Could not determine plant center from {filename}")
        return None
    
    cell_masks, cell_centers = extract_cell_data(results, img, class_id=0, 
                                                  refine=refine, 
                                                  color_tolerance=color_tolerance)
    
    if filter_by_inner_part and os.path.exists(exclusion_weights):
        exclusion_model = YOLO(exclusion_weights)
        exclusion_results = exclusion_model(image_path, conf=exclusion_confidence, max_det=3000)
        
        inner_mask = get_inner_part_mask(
            exclusion_results, img,
            morph_open_kernel=morph_open_kernel,
            morph_close_kernel=morph_close_kernel,
            morph_open_iterations=1,
            morph_close_iterations=1,
            keep_largest_component=True
        )
        
        if np.sum(inner_mask) > 0:
            cells_before = len(cell_masks)
            cell_masks, cell_centers = filter_cells_by_inner_part(cell_masks, cell_centers, inner_mask)
            cells_removed = cells_before - len(cell_centers)
            print(f"Cells removed by inner-part filter: {cells_removed}")
    
    measurements = []
    for i, (center, mask) in enumerate(zip(cell_centers, cell_masks)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area_pixels = cv2.contourArea(contour)
        else:
            area_pixels = 0
        
        dx = center[0] - center_x
        dy = center[1] - center_y
        radius = np.sqrt(dx**2 + dy**2)
        angle = np.degrees(np.arctan2(dy, dx))
        
        measurements.append({
            'cell_id': i + 1,
            'radius_pixels': round(radius, 2),
            'angle_degrees': round(angle, 2),
            'area_pixels': round(area_pixels, 2)
        })
    
    df = pd.DataFrame(measurements)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(measurements)} cells to {output_csv}")
    
    img_output = img.copy()
    for center in cell_centers:
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(img_output, (cx, cy), 3, (0, 0, 255), -1)
    
    cv2.circle(img_output, (int(center_x), int(center_y)), 8, (255, 0, 0), -1)
    cv2.putText(img_output, f"Cells: {len(cell_centers)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imwrite(output_image, img_output)
    print(f"Visualization saved to {output_image}")
    
    if show_result:
        cv2.imshow("Cell Centers", img_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return df


def batch_extract_measurements(
    weights_path="runs/robust_segmentation/robust_cell_detector/weights/best.pt",
    exclusion_weights="runs/segment/runs/exclusion_model/inner_part_detector/weights/best.pt",
    image_folder="cropped_images",
    confidence=0.25,
    exclusion_confidence=0.5,
    output_dir="measurements",
    refine=True,
    color_tolerance=15,
    filter_by_inner_part=True,
    morph_open_kernel=99,
    morph_close_kernel=155
):
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return None
    
    if not os.path.exists(image_folder):
        print(f"Folder not found: {image_folder}")
        return None
    
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
        return None
    
    for img_file in images:
        img_path = os.path.join(image_folder, img_file)
        base_name = os.path.splitext(img_file)[0]
        csv_path = os.path.join(output_dir, f"{base_name}_measurements.csv")
        image_output_path = os.path.join(output_dir, f"{base_name}_centers.jpg")
        
        results = model(img_path, conf=confidence, max_det=3000)
        img = cv2.imread(img_path)
        filename = os.path.basename(img_path)
        
        h, w = img.shape[:2]
        center_x, center_y, quadrant = get_plant_center_from_filename(img_file, w, h)
        
        if center_x is None:
            print(f"Could not determine plant center from {img_file}")
            continue
        
        cell_masks, cell_centers = extract_cell_data(results, img, class_id=0, 
                                                      refine=refine, 
                                                      color_tolerance=color_tolerance)
        
        cells_removed = 0
        if filter_by_inner_part and exclusion_model is not None:
            exclusion_results = exclusion_model(img_path, conf=exclusion_confidence, max_det=3000)
            
            inner_mask = get_inner_part_mask(
                exclusion_results, img,
                morph_open_kernel=morph_open_kernel,
                morph_close_kernel=morph_close_kernel,
                morph_open_iterations=1,
                morph_close_iterations=1,
                keep_largest_component=True
            )
            
            if np.sum(inner_mask) > 0:
                cells_before = len(cell_masks)
                cell_masks, cell_centers = filter_cells_by_inner_part(cell_masks, cell_centers, inner_mask)
                cells_removed = cells_before - len(cell_centers)
        measurements = []
        for i, (center, mask) in enumerate(zip(cell_centers, cell_masks)):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                area_pixels = cv2.contourArea(contour)
            else:
                area_pixels = 0
            
            dx = center[0] - center_x
            dy = center[1] - center_y
            radius = np.sqrt(dx**2 + dy**2)
            angle = np.degrees(np.arctan2(dy, dx))
            
            measurements.append({
                'cell_id': i + 1,
                'radius_pixels': round(radius, 2),
                'angle_degrees': round(angle, 2),
                'area_pixels': round(area_pixels, 2)
            })
        
        df = pd.DataFrame(measurements)
        df.to_csv(csv_path, index=False)
        
        removed_str = f", removed: {cells_removed}" if cells_removed > 0 else ""
        print(f"{img_file}: {len(measurements)} cells saved to {csv_path} (Quadrant: {quadrant}){removed_str}")
        
        img_output = img.copy()
        for center in cell_centers:
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(img_output, (cx, cy), 3, (0, 0, 255), -1)
        
        cv2.circle(img_output, (int(center_x), int(center_y)), 8, (255, 0, 0), -1)
        cv2.putText(img_output, f"Cells: {len(cell_centers)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imwrite(image_output_path, img_output)
        print(f"   Visualization saved to {image_output_path}")

def visualize_measurements(
    weights_path="runs/robust_segmentation/robust_cell_detector/weights/best.pt",
    exclusion_weights="runs/segment/runs/exclusion_model/inner_part_detector/weights/best.pt",
    image_path="test_image.jpg",
    confidence=0.99,
    exclusion_confidence=0.5,
    output_path="annotated_red_dots.jpg",
    show_plant_center=True,
    show_result=False,
    refine=True,
    color_tolerance=15,
    filter_by_inner_part=True,
    morph_open_kernel=99,
    morph_close_kernel=155
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
    
    cell_masks, cell_centers = extract_cell_data(results, img, class_id=0, 
                                                  refine=refine, 
                                                  color_tolerance=color_tolerance)
    
    cell_masks, cell_centers = extract_cell_data(results, img, class_id=0, 
                                                  refine=refine, 
                                                  color_tolerance=color_tolerance)
    
    if filter_by_inner_part and os.path.exists(exclusion_weights):
        exclusion_model = YOLO(exclusion_weights)
        exclusion_results = exclusion_model(image_path, conf=exclusion_confidence, max_det=3000)
        
        inner_mask = get_inner_part_mask(
            exclusion_results, img,
            morph_open_kernel=morph_open_kernel,
            morph_close_kernel=morph_close_kernel,
            morph_open_iterations=1,
            morph_close_iterations=1,
            keep_largest_component=True
        )
        
        if np.sum(inner_mask) > 0:
            cell_masks, cell_centers = filter_cells_by_inner_part(cell_masks, cell_centers, inner_mask)
    
    draw_red_dots_on_image(img, cell_centers, output_path, plant_center, quadrant)
    print(f"Visualization saved to {output_path}")
    print(f"Total cells: {len(cell_centers)}")
    
    if show_result:
        img_display = cv2.imread(output_path)
        cv2.imshow("Cell Centers", img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
