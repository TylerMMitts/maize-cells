import cv2
import numpy as np
import pandas as pd
import os
import gc
import time
import psutil
from ultralytics import YOLO
from test_robust import extract_cell_data, filter_cells_by_inner_part, get_inner_part_mask, filter_overlapping_cells
from visualize_cell_centers import get_plant_center_from_filename, draw_red_dots_on_image

# Set environment variables for CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TORCH_DEVICE'] = 'cpu'

def get_memory_usage_mb():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def force_memory_cleanup():
    gc.collect()
    if hasattr(gc, 'garbage'):
        gc.garbage.clear()
    
    # Clear OpenCV's internal caches
    cv2.setNumThreads(0)  # Disable OpenCV multithreading

def calculate_area_from_mask(mask):
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            del contours
            return area
        return 0.0
    except Exception:
        return 0.0

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
    use_darkest_seed=False,
    filter_by_inner_part=True,
    inner_open_kernel=99,
    inner_close_kernel=155,
    outer_open_kernel=99,
    outer_close_kernel=155,
    keep_largest_component=True,
    exclusion_overlap_threshold=0.0,
    overlap_threshold=15.0,
    morph_post_process=True,
    morph_close_kernel=21,
    morph_open_kernel=21,
    sam_refiner=None
):

    print(f"Processing: {image_path}")
    print(f"Memory before: {get_memory_usage_mb():.1f} MB")
    
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return None
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    # Load models on CPU
    model = YOLO(weights_path)
    model.to('cpu')
    
    exclusion_model = None
    if filter_by_inner_part and os.path.exists(exclusion_weights):
        exclusion_model = YOLO(exclusion_weights)
        exclusion_model.to('cpu')
    else:
        filter_by_inner_part = False
    
    # Read image
    img = cv2.imread(image_path)
    filename = os.path.basename(image_path)
    
    h, w = img.shape[:2]
    center_x, center_y, quadrant = get_plant_center_from_filename(filename, w, h)
    
    if center_x is None:
        print(f"Could not determine plant center from {filename}")
        del model, exclusion_model, img
        force_memory_cleanup()
        return None
    
    # Run inference
    results = model(image_path, conf=confidence, max_det=3000, device='cpu')
    
    # Extract cell data
    cell_masks, cell_centers, cell_confidences = extract_cell_data(
        results, img, class_id=0, 
        refine=refine, 
        color_tolerance=color_tolerance,
        use_darkest_seed=use_darkest_seed,
        morph_post_process=morph_post_process,
        morph_close_kernel=morph_close_kernel,
        morph_open_kernel=morph_open_kernel,
        sam_refiner=sam_refiner
    )
    
    # Clean up results
    del results
    
    # Filter by inner part if requested
    if filter_by_inner_part and exclusion_model is not None:
        exclusion_results = exclusion_model(image_path, conf=exclusion_confidence, max_det=3000, device='cpu')
        
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
            keep_largest_component=keep_largest_component
        )
        
        del exclusion_results
        
        if np.sum(inner_mask) > 0:
            cells_before = len(cell_masks)
            cell_masks, cell_centers, cell_confidences = filter_cells_by_inner_part(
                cell_masks, cell_centers, cell_confidences, inner_mask, exclusion_overlap_threshold
            )
            cells_removed = cells_before - len(cell_centers)
            print(f"Cells removed by exclusion filter: {cells_removed}")
        
        del inner_mask
    
    # Apply overlap filtering
    cells_before_overlap = len(cell_masks)
    cell_masks, cell_centers, cell_confidences = filter_overlapping_cells(
        cell_masks, cell_centers, cell_confidences, overlap_threshold=overlap_threshold
    )
    cells_removed_overlap = cells_before_overlap - len(cell_masks)
    if cells_removed_overlap > 0:
        print(f"Cells removed by overlap filter: {cells_removed_overlap}")
    
    # Process measurements one by one to save memory
    measurements = []
    for i, (center, mask) in enumerate(zip(cell_centers, cell_masks)):
        area_pixels = calculate_area_from_mask(mask)
        
        dx = center[0] - center_x
        dy = center[1] - center_y
        radius = np.sqrt(dx**2 + dy**2)
        angle = np.degrees(np.arctan2(dy, dx))
        
        measurements.append({
            'cell_id': i + 1,
            'x_pixels': round(center[0], 2),
            'y_pixels': round(center[1], 2),
            'radius_pixels': round(radius, 2),
            'angle_degrees': round(angle, 2),
            'area_pixels': round(area_pixels, 2)
        })
        
        # Clear mask to free memory
        del mask
    
    # Save CSV
    df = pd.DataFrame(measurements)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(measurements)} cells to {output_csv}")
    
    # Create visualization
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
    
    # Cleanup
    del img, img_output, cell_masks, cell_centers, cell_confidences, df, measurements
    del model
    if exclusion_model:
        del exclusion_model
    force_memory_cleanup()
    
    print(f"Memory after: {get_memory_usage_mb():.1f} MB")
    
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
    use_darkest_seed=False,
    filter_by_inner_part=True,
    inner_open_kernel=99,
    inner_close_kernel=155,
    outer_open_kernel=99,
    outer_close_kernel=155,
    keep_largest_component=True,
    exclusion_overlap_threshold=0.0,
    overlap_threshold=15.0,
    morph_post_process=True,
    morph_close_kernel=21,
    morph_open_kernel=21,
    sam_refiner=None
):
    
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return None
    
    if not os.path.exists(image_folder):
        print(f"Folder not found: {image_folder}")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models on CPU (once for all images)
    model = YOLO(weights_path)
    model.to('cpu')
    
    exclusion_model = None
    if filter_by_inner_part:
        if not os.path.exists(exclusion_weights):
            print(f"Exclusion model not found: {exclusion_weights}")
            filter_by_inner_part = False
        else:
            exclusion_model = YOLO(exclusion_weights)
            exclusion_model.to('cpu')
    
    # Get image list
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]
    
    if not images:
        print(f"No images found in {image_folder}")
        return None
    
    successful = 0
    failed = 0
    
    # Process each image
    for idx, img_file in enumerate(images, 1):
        img_path = os.path.join(image_folder, img_file)
        base_name = os.path.splitext(img_file)[0]
        csv_path = os.path.join(output_dir, f"{base_name}_measurements.csv")
        image_output_path = os.path.join(output_dir, f"{base_name}_centers.jpg")
        
        img = None
        results = None
        cell_masks = None
        cell_centers = None
        
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_file}")
                failed += 1
                continue
            
            filename = os.path.basename(img_path)
            h, w = img.shape[:2]
            center_x, center_y, quadrant = get_plant_center_from_filename(img_file, w, h)
            
            if center_x is None:
                print(f"Could not determine plant center from {img_file}")
                failed += 1
                continue
            
            # Run inference
            results = model(img_path, conf=confidence, max_det=3000, device='cpu')
            
            # Extract cell data
            try:
                cell_masks, cell_centers, cell_confidences = extract_cell_data(
                    results, img, class_id=0, 
                    refine=refine, 
                    color_tolerance=color_tolerance,
                    use_darkest_seed=use_darkest_seed,
                    morph_post_process=morph_post_process,
                    morph_close_kernel=morph_close_kernel,
                    morph_open_kernel=morph_open_kernel,
                    sam_refiner=sam_refiner
                )
            except (cv2.error, MemoryError) as e:
                print(f"Failed to extract cell data: {e}")
                failed += 1
                continue
            
            # Clean up results
            del results
            results = None
            
            # Filter by inner part
            cells_removed = 0
            if filter_by_inner_part and exclusion_model is not None:
                try:
                    exclusion_results = exclusion_model(img_path, conf=exclusion_confidence, max_det=3000, device='cpu')
                    
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
                        keep_largest_component=keep_largest_component
                    )
                    
                    del exclusion_results
                    
                    if np.sum(inner_mask) > 0:
                        cells_before = len(cell_masks)
                        cell_masks, cell_centers, cell_confidences = filter_cells_by_inner_part(
                            cell_masks, cell_centers, cell_confidences, inner_mask, exclusion_overlap_threshold
                        )
                        cells_removed = cells_before - len(cell_centers)
                    
                    del inner_mask
                    
                except Exception as e:
                    print(f"Error in exclusion filtering: {e}")
            
            # Apply overlap filtering
            cells_before_overlap = len(cell_masks)
            cell_masks, cell_centers, cell_confidences = filter_overlapping_cells(
                cell_masks, cell_centers, cell_confidences, overlap_threshold=overlap_threshold
            )
            cells_removed_overlap = cells_before_overlap - len(cell_masks)
            
            # Process measurements
            measurements = []
            for i, (center, mask) in enumerate(zip(cell_centers, cell_masks)):
                area_pixels = calculate_area_from_mask(mask)
                
                dx = center[0] - center_x
                dy = center[1] - center_y
                radius = np.sqrt(dx**2 + dy**2)
                angle = np.degrees(np.arctan2(dy, dx))
                
                measurements.append({
                    'cell_id': i + 1,
                    'x_pixels': round(center[0], 2),
                    'y_pixels': round(center[1], 2),
                    'radius_pixels': round(radius, 2),
                    'angle_degrees': round(angle, 2),
                    'area_pixels': round(area_pixels, 2)
                })
                
                del mask  # Free mask memory
            
            # Save CSV
            df = pd.DataFrame(measurements)
            df.to_csv(csv_path, index=False)
            
            # Create visualization
            img_output = img.copy()
            for center in cell_centers:
                cx, cy = int(center[0]), int(center[1])
                cv2.circle(img_output, (cx, cy), 3, (0, 0, 255), -1)
            
            cv2.circle(img_output, (int(center_x), int(center_y)), 8, (255, 0, 0), -1)
            cv2.putText(img_output, f"Cells: {len(cell_centers)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imwrite(image_output_path, img_output)
            print(f"Visualization saved to {image_output_path}")
            
            successful += 1
            
            # Clean up for this image
            del img, img_output, cell_masks, cell_centers, cell_confidences, measurements, df
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            failed += 1
        
        finally:
            # Force garbage collection
            force_memory_cleanup()
            print(f"Memory after cleanup: {get_memory_usage_mb():.1f} MB")
            
            # Small delay for system to reclaim memory
            time.sleep(0.1)
    
    # Clean up models
    del model
    if exclusion_model:
        del exclusion_model
    force_memory_cleanup()
    
    return successful


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
    
    # Load models on CPU
    model = YOLO(weights_path)
    model.to('cpu')
    
    results = model(image_path, conf=confidence, max_det=3000, device='cpu')
    
    img = cv2.imread(image_path)
    filename = os.path.basename(image_path)
    
    plant_center = None
    quadrant = None
    if show_plant_center:
        h, w = img.shape[:2]
        center_x, center_y, quadrant = get_plant_center_from_filename(filename, w, h)
        if center_x is not None:
            plant_center = (center_x, center_y)
    
    # Extract cell data
    cell_masks, cell_centers, cell_confidences = extract_cell_data(
        results, img, class_id=0, 
        refine=refine, 
        color_tolerance=color_tolerance
    )
    
    del results
    
    # Filter by inner part if requested
    if filter_by_inner_part and os.path.exists(exclusion_weights):
        exclusion_model = YOLO(exclusion_weights)
        exclusion_model.to('cpu')
        exclusion_results = exclusion_model(image_path, conf=exclusion_confidence, max_det=3000, device='cpu')
        
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
        
        del exclusion_results
        
        if np.sum(inner_mask) > 0:
            cell_masks, cell_centers, cell_confidences = filter_cells_by_inner_part(
                cell_masks, cell_centers, cell_confidences, inner_mask, exclusion_overlap_threshold
            )
        
        del inner_mask, exclusion_model
    
    # Draw visualization
    draw_red_dots_on_image(img, cell_centers, output_path, plant_center, quadrant)
    print(f"Visualization saved to {output_path}")
    print(f"Total cells: {len(cell_centers)}")
    
    # Clean up
    del img, cell_masks, cell_centers, cell_confidences, model
    force_memory_cleanup()
    
    if show_result:
        img_display = cv2.imread(output_path)
        cv2.imshow("Cell Centers", img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        del img_display