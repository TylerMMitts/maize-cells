from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd
import gc
import torch

# Try to enable OpenCL for GPU acceleration
try:
    cv2.ocl.setUseOpenCL(True)
    if cv2.ocl.useOpenCL():
        print("OpenCL enabled for GPU acceleration (OpenCV)")
    else:
        print("OpenCL not available, using CPU")
except:
    print("OpenCL not available, using CPU")

# Check PyTorch GPU availability
if torch.cuda.is_available():
    print(f"PyTorch CUDA available: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
else:
    print("PyTorch using CPU")

# Region growing to refine masks based on color similarity
def refine_with_region_growing(image, mask, seed_point=None, tolerance=15):
    h, w = mask.shape
    refined_mask = np.zeros((h, w), dtype=np.uint8)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # If no seed point provided, use the center of the mask
    if seed_point is None:
        moments = cv2.moments(mask)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            seed_point = (cx, cy)
        else:
            return mask
    
    if not (0 <= seed_point[0] < w and 0 <= seed_point[1] < h):
        return mask
    
    if mask[seed_point[1], seed_point[0]] == 0:
        return mask
    
    # Get seed color and define range for region growing
    seed_color = hsv[seed_point[1], seed_point[0]]
    
    lower = np.array([max(0, seed_color[0] - tolerance), 
                      max(0, seed_color[1] - 40), 
                      max(0, seed_color[2] - 40)])
    upper = np.array([min(179, seed_color[0] + tolerance), 
                      min(255, seed_color[1] + 40), 
                      min(255, seed_color[2] + 40)])
    
    # Create a mask for the color range and combine with the original mask
    color_mask = cv2.inRange(hsv, lower, upper)
    combined = cv2.bitwise_and(color_mask, mask)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    
    return refined

def extract_cell_data(results, img, class_id=None, refine=True, color_tolerance=15, use_darkest_seed=False,
                     morph_post_process=True, morph_close_kernel=21, morph_open_kernel=21, sam_refiner=None):
    
    cell_masks = []
    cell_centers = []
    cell_confidences = []
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        
        for i, mask in enumerate(masks):
            if boxes is not None and class_id is not None:
                if int(boxes.cls[i]) != class_id:
                    continue
            
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
            
            # Get bounding box to work on smaller region (saves memory)
            rows = np.any(binary_mask > 0, axis=1)
            cols = np.any(binary_mask > 0, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue
                
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Add padding for morphological operations
            pad = max(morph_close_kernel, morph_open_kernel, 10) if morph_post_process else 10
            y_min = max(0, y_min - pad)
            y_max = min(img.shape[0], y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(img.shape[1], x_max + pad)
            
            # Extract region of interest (much smaller than full image)
            roi_mask = binary_mask[y_min:y_max, x_min:x_max]
            roi_img = img[y_min:y_max, x_min:x_max]
            
            if refine:
                if sam_refiner is not None:
                    refined_roi = sam_refiner.refine_with_sam(roi_img, roi_mask, fallback_to_original=True)
                else:
                    seed_point = None
                    if use_darkest_seed:
                        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                        masked_gray = np.where(roi_mask > 0, gray, 255)
                        min_loc = np.unravel_index(np.argmin(masked_gray), masked_gray.shape)
                        seed_point = (min_loc[1], min_loc[0])
                    
                    refined_roi = refine_with_region_growing(roi_img, roi_mask, 
                                                              seed_point=seed_point,
                                                              tolerance=color_tolerance)
            else:
                refined_roi = roi_mask
            
            # Apply morphological post-processing on ROI (much faster and less memory)
            if morph_post_process:
                try:
                    # Closing to fill holes
                    if morph_close_kernel > 0:
                        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                                (morph_close_kernel, morph_close_kernel))
                        refined_roi = cv2.morphologyEx(refined_roi, cv2.MORPH_CLOSE, close_kernel)
                        del close_kernel
                    
                    # Opening to smooth boundaries and fix radius
                    if morph_open_kernel > 0:
                        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                               (morph_open_kernel, morph_open_kernel))
                        refined_roi = cv2.morphologyEx(refined_roi, cv2.MORPH_OPEN, open_kernel)
                        del open_kernel
                except cv2.error as e:
                    print(f"Morphology failed on cell {i}, using without post-processing")
            
            # Reconstruct full-size mask
            refined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            refined_mask[y_min:y_max, x_min:x_max] = refined_roi
            
            # Clean up ROI variables
            del roi_mask, roi_img, refined_roi
            
            try:
                contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except cv2.error as e:
                print(f"findContours failed on cell {i}: {e}")
                del refined_mask, binary_mask, mask_resized
                continue
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    cell_centers.append((cx, cy))
                    cell_masks.append(refined_mask)
                    
                    # Get confidence score
                    if boxes is not None and hasattr(boxes, 'conf'):
                        cell_confidences.append(float(boxes.conf[i]))
                    else:
                        cell_confidences.append(1.0)
                else:
                    del refined_mask
            else:
                del refined_mask
            
            # Clean up per-cell variables to prevent memory accumulation
            del binary_mask, mask_resized
            
            # Periodic garbage collection for large batches
            if i > 0 and i % 100 == 0:
                gc.collect()
    
    return cell_masks, cell_centers, cell_confidences

def get_inner_part_mask(results, img, 
                        inner_open_kernel=99, inner_close_kernel=155,
                        inner_open_iterations=1, inner_close_iterations=1,
                        outer_open_kernel=99, outer_close_kernel=155,
                        outer_open_iterations=1, outer_close_iterations=1,
                        keep_largest_component=True):
    h, w = img.shape[:2]
    
    # Separate masks for each class
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        
        for i, mask in enumerate(masks):
            if boxes is not None:
                cls = int(boxes.cls[i])
                mask_resized = cv2.resize(mask, (w, h))
                binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                
                # Separate by class: 0 = inner-part, 1 = outer-part
                if cls == 0:
                    inner_mask = cv2.bitwise_or(inner_mask, binary_mask)
                elif cls == 1:
                    outer_mask = cv2.bitwise_or(outer_mask, binary_mask)
    
    # Process inner-part mask
    processed_inner = np.zeros((h, w), dtype=np.uint8)
    if np.sum(inner_mask) > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_open_kernel, inner_open_kernel))
        opened = cv2.morphologyEx(inner_mask, cv2.MORPH_OPEN, kernel_open, iterations=inner_open_iterations)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_close_kernel, inner_close_kernel))
        processed_inner = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=inner_close_iterations)
        
        if keep_largest_component:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed_inner, connectivity=8)
            
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                largest_inner = np.zeros_like(processed_inner)
                largest_inner[labels == largest_label] = 255
                processed_inner = largest_inner
                print(f"Inner-part: Found {num_labels - 1} components, keeping largest (area: {stats[largest_label, cv2.CC_STAT_AREA]} pixels)")
    
    # Process outer-part mask
    processed_outer = np.zeros((h, w), dtype=np.uint8)
    if np.sum(outer_mask) > 0:

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_close_kernel, outer_close_kernel))
        processed_outer = cv2.morphologyEx(outer_mask, cv2.MORPH_CLOSE, kernel_close, iterations=outer_close_iterations)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_open_kernel, outer_open_kernel))
        processed_outer = cv2.morphologyEx(processed_outer, cv2.MORPH_OPEN, kernel_open, iterations=outer_open_iterations)
        
        
        if keep_largest_component:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed_outer, connectivity=8)
            
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                largest_outer = np.zeros_like(processed_outer)
                largest_outer[labels == largest_label] = 255
                processed_outer = largest_outer
                print(f"Outer-part: Found {num_labels - 1} components, keeping largest (area: {stats[largest_label, cv2.CC_STAT_AREA]} pixels)")
    
    # Combine both processed masks
    combined_mask = cv2.bitwise_or(processed_inner, processed_outer)
    
    return combined_mask

def filter_cells_by_inner_part(cell_masks, cell_centers, cell_confidences, inner_mask, exclusion_overlap_threshold=0.0):

    filtered_masks = []
    filtered_centers = []
    filtered_confidences = []
    
    for center, mask, conf in zip(cell_centers, cell_masks, cell_confidences):
        cx, cy = int(center[0]), int(center[1])
        
        if cx >= inner_mask.shape[1] or cy >= inner_mask.shape[0]:
            filtered_masks.append(mask)
            filtered_centers.append(center)
            filtered_confidences.append(conf)
            continue
        
        # If threshold is 0, just check center point
        if exclusion_overlap_threshold <= 0.0:
            if inner_mask[cy, cx] == 0:
                filtered_masks.append(mask)
                filtered_centers.append(center)
                filtered_confidences.append(conf)
        else:
            # Calculate overlap percentage
            cell_area = np.sum(mask > 0)
            if cell_area == 0:
                continue
            
            overlap_area = np.sum(np.logical_and(mask > 0, inner_mask > 0))
            overlap_percentage = (overlap_area / cell_area) * 100
            
            # Keep cell only if overlap is below threshold
            if overlap_percentage < exclusion_overlap_threshold:
                filtered_masks.append(mask)
                filtered_centers.append(center)
                filtered_confidences.append(conf)
    
    return filtered_masks, filtered_centers, filtered_confidences

def get_bounding_box(mask):
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

def bboxes_overlap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Check if one box is to the left of the other
    if x1 + w1 < x2 or x2 + w2 < x1:
        return False
    
    # Check if one box is above the other
    if y1 + h1 < y2 or y2 + h2 < y1:
        return False
    
    return True

def calculate_overlap_percentage(mask1, mask2, bbox1=None, bbox2=None):

    if bbox1 is not None and bbox2 is not None:
        if not bboxes_overlap(bbox1, bbox2):
            return 0.0
    
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    
    if intersection == 0:
        return 0.0
    
    area1 = np.sum(mask1 > 0)
    area2 = np.sum(mask2 > 0)
    
    if area1 == 0 or area2 == 0:
        return 0.0
    
    # Calculate overlap as percentage of the smaller cell
    smaller_area = min(area1, area2)
    overlap_percentage = (intersection / smaller_area) * 100
    
    return overlap_percentage

def filter_overlapping_cells(cell_masks, cell_centers, cell_confidences, overlap_threshold=15.0):

    if len(cell_masks) == 0:
        return cell_masks, cell_centers, cell_confidences
    
    # Pre-compute bounding boxes for all masks (speeds up overlap checks)
    bboxes = [get_bounding_box(mask) for mask in cell_masks]
    
    removed_indices = set()
    
    # Sort by confidence (descending) to process higher confidence cells first
    sorted_indices = sorted(range(len(cell_confidences)), 
                          key=lambda i: cell_confidences[i], 
                          reverse=True)
    
    for i, idx_i in enumerate(sorted_indices):
        if idx_i in removed_indices:
            continue
            
        for idx_j in sorted_indices[i+1:]:
            if idx_j in removed_indices:
                continue
            
            # Fast bounding box check first
            if bboxes[idx_i] is None or bboxes[idx_j] is None:
                continue
            
            if not bboxes_overlap(bboxes[idx_i], bboxes[idx_j]):
                continue
            
            # Only do expensive mask overlap check if bounding boxes overlap
            overlap = calculate_overlap_percentage(
                cell_masks[idx_i], cell_masks[idx_j], 
                bboxes[idx_i], bboxes[idx_j]
            )
            
            # If overlap exceeds threshold, remove the one with lower confidence
            if overlap > overlap_threshold:
                # idx_i has higher confidence (processed first), so remove idx_j
                removed_indices.add(idx_j)
    
    # Keep only the cells that weren't removed
    filtered_masks = [cell_masks[i] for i in range(len(cell_masks)) if i not in removed_indices]
    filtered_centers = [cell_centers[i] for i in range(len(cell_centers)) if i not in removed_indices]
    filtered_confidences = [cell_confidences[i] for i in range(len(cell_confidences)) if i not in removed_indices]
    
    return filtered_masks, filtered_centers, filtered_confidences

def draw_cells_on_image(image, cell_masks, color=(0, 0, 255), thickness=2):
    img = image.copy()
    for mask in cell_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, thickness)
    return img

def batch_test_robust_model(
    cell_weights="runs/robust_segmentation/robust_cell_detector/weights/best.pt",
    exclusion_weights="runs/exclusion_model/inner_part_detector/weights/best.pt",
    image_folder="cropped_images",
    confidence=0.25,
    exclusion_confidence=0.5,
    output_dir="test_results_robust",
    refine=True,
    color_tolerance=15,
    use_darkest_seed=False,
    max_det=3000,
    filter_by_inner_part=True,
    inner_open_kernel=99,
    inner_close_kernel=155,
    inner_open_iterations=1,
    inner_close_iterations=1,
    outer_open_kernel=99,
    outer_close_kernel=155,
    outer_open_iterations=1,
    outer_close_iterations=1,
    keep_largest_component=True,
    exclusion_overlap_threshold=0.0,
    overlap_threshold=15.0,
    morph_post_process=False,
    morph_close_kernel=3,
    morph_open_kernel=3,
    sam_refiner=None
):
    if not os.path.exists(cell_weights):
        print(f"Cell model not found: {cell_weights}")
        return None
    
    if not os.path.exists(image_folder):
        print(f"Folder not found: {image_folder}")
        return None
    
    cell_model = YOLO(cell_weights)
    cell_model.overrides['max_det'] = max_det
    
    # Force CPU for YOLO to avoid GPU memory issues with high max_det
    exclusion_model = None
    if filter_by_inner_part:
        if not os.path.exists(exclusion_weights):
            print(f"Exclusion model not found: {exclusion_weights}")
            filter_by_inner_part = False
        else:
            exclusion_model = YOLO(exclusion_weights)
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]
    
    if not images:
        print(f"No images found in {image_folder}")
        return None
    
    
    results_summary = {}
    
    for img_file in images:
        img_path = os.path.join(image_folder, img_file)
        
        # Clear GPU cache before each image to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use CPU device to avoid GPU memory errors
        cell_results = cell_model(img_path, conf=confidence, max_det=max_det, iou=0.45, device='cpu')
        
        img = cv2.imread(img_path)
        filename = os.path.basename(img_path)
        
        cell_masks, cell_centers, cell_confidences = extract_cell_data(cell_results, img, class_id=0, 
                                                                        refine=refine,
                                                                        color_tolerance=color_tolerance,
                                                                        morph_post_process=morph_post_process,
                                                                        morph_close_kernel=morph_close_kernel,
                                                                        morph_open_kernel=morph_open_kernel,
                                                                        sam_refiner=sam_refiner)
        
        inner_mask = None
        inner_pixel_count = 0
        cells_removed_by_exclusion = 0
        cells_removed_by_overlap = 0
        
        if filter_by_inner_part and exclusion_model is not None:
            # Use CPU device to avoid GPU memory errors
            exclusion_results = exclusion_model(img_path, conf=exclusion_confidence, max_det=max_det, device='cpu')
            
            inner_mask = get_inner_part_mask(
                exclusion_results, img,
                inner_open_kernel=inner_open_kernel,
                inner_close_kernel=inner_close_kernel,
                inner_open_iterations=inner_open_iterations,
                inner_close_iterations=inner_close_iterations,
                outer_open_kernel=outer_open_kernel,
                outer_close_kernel=outer_close_kernel,
                outer_open_iterations=outer_open_iterations,
                outer_close_iterations=outer_close_iterations,
                keep_largest_component=keep_largest_component
            )
            inner_pixel_count = np.sum(inner_mask > 0)
            
            if inner_pixel_count > 0:
                cells_before = len(cell_masks)
                cell_masks, cell_centers, cell_confidences = filter_cells_by_inner_part(
                    cell_masks, cell_centers, cell_confidences, inner_mask, exclusion_overlap_threshold
                )
                cells_removed_by_exclusion = cells_before - len(cell_masks)
            else:
                print(f"No exclusion zones detected after morphology")
        
        # Apply overlap filtering
        cells_before_overlap = len(cell_masks)
        cell_masks, cell_centers, cell_confidences = filter_overlapping_cells(
            cell_masks, cell_centers, cell_confidences, overlap_threshold=overlap_threshold
        )
        cells_removed_by_overlap = cells_before_overlap - len(cell_masks)
        
        img_output = draw_cells_on_image(img, cell_masks, color=(0, 0, 255), thickness=2)
        
        if inner_pixel_count > 0 and inner_mask is not None:
            overlay = np.zeros_like(img)
            overlay[inner_mask == 255] = (0, 0, 255)
            img_output = cv2.addWeighted(img_output, 0.7, overlay, 0.3, 0)
        
        cv2.putText(img_output, f"Cells: {len(cell_masks)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        total_removed = cells_removed_by_exclusion + cells_removed_by_overlap
        if total_removed > 0:
            cv2.putText(img_output, f"Removed: {total_removed} (Excl:{cells_removed_by_exclusion}, Ovlp:{cells_removed_by_overlap})", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        output_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(output_path, img_output)
        
        if inner_pixel_count > 0 and inner_mask is not None:
            inner_viz = img.copy()
            inner_viz[inner_mask == 255] = (0, 0, 255)
            cv2.putText(inner_viz, f"Exclusion zone mask (inner + outer parts)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            inner_output_path = os.path.join(output_dir, f"exclusion_zone_{filename}")
            cv2.imwrite(inner_output_path, inner_viz)
        
        results_summary[filename] = {
            'raw_cells': len(cell_masks) + cells_removed_by_exclusion + cells_removed_by_overlap,
            'filtered_cells': len(cell_masks),
            'cells_removed_by_exclusion': cells_removed_by_exclusion,
            'cells_removed_by_overlap': cells_removed_by_overlap,
            'total_cells_removed': cells_removed_by_exclusion + cells_removed_by_overlap,
            'inner_part_area': inner_pixel_count
        }
        
        # Free memory after each image to prevent accumulation
        del img, img_output, cell_masks, cell_centers, cell_confidences, cell_results
        if inner_mask is not None:
            del inner_mask
        if 'inner_viz' in locals():
            del inner_viz
        if 'exclusion_results' in locals():
            del exclusion_results
        
        # Garbage collect every 5 images
        if len(results_summary) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    summary_df = pd.DataFrame([
        {
            'image': img, 
            'raw_cells': data['raw_cells'],
            'filtered_cells': data['filtered_cells'],
            'removed_exclusion': data['cells_removed_by_exclusion'],
            'removed_overlap': data['cells_removed_by_overlap'],
            'total_removed': data['total_cells_removed'],
            'inner_part_area': data['inner_part_area']
        }
        for img, data in results_summary.items()
    ])
    summary_path = os.path.join(output_dir, "detection_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Results saved to {output_dir}")
    
    return results_summary
