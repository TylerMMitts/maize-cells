from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd

def refine_with_region_growing(image, mask, seed_point=None, tolerance=15):
    h, w = mask.shape
    refined_mask = np.zeros((h, w), dtype=np.uint8)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
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
    
    seed_color = hsv[seed_point[1], seed_point[0]]
    
    lower = np.array([max(0, seed_color[0] - tolerance), 
                      max(0, seed_color[1] - 40), 
                      max(0, seed_color[2] - 40)])
    upper = np.array([min(179, seed_color[0] + tolerance), 
                      min(255, seed_color[1] + 40), 
                      min(255, seed_color[2] + 40)])
    
    color_mask = cv2.inRange(hsv, lower, upper)
    combined = cv2.bitwise_and(color_mask, mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    
    return refined

def extract_cell_data(results, img, class_id=None, refine=True, color_tolerance=15, use_darkest_seed=False):
    cell_masks = []
    cell_centers = []
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        
        for i, mask in enumerate(masks):
            if boxes is not None and class_id is not None:
                if int(boxes.cls[i]) != class_id:
                    continue
            
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
            
            if refine:
                seed_point = None
                if use_darkest_seed:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    masked_gray = np.where(binary_mask > 0, gray, 255)
                    min_loc = np.unravel_index(np.argmin(masked_gray), masked_gray.shape)
                    seed_point = (min_loc[1], min_loc[0])
                
                refined_mask = refine_with_region_growing(img, binary_mask, 
                                                          seed_point=seed_point,
                                                          tolerance=color_tolerance)
            else:
                refined_mask = binary_mask
            
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    cell_centers.append((cx, cy))
                    cell_masks.append(refined_mask)
    
    return cell_masks, cell_centers

def get_inner_part_mask(results, img, morph_open_kernel=99, morph_close_kernel=155, 
                        morph_open_iterations=1, morph_close_iterations=1, 
                        keep_largest_component=True):
    h, w = img.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        
        for i, mask in enumerate(masks):
            if boxes is not None and int(boxes.cls[i]) == 0:
                mask_resized = cv2.resize(mask, (w, h))
                binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                combined_mask = cv2.bitwise_or(combined_mask, binary_mask)
    
    if np.sum(combined_mask) > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_kernel, morph_open_kernel))
        opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open, iterations=morph_open_iterations)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_kernel, morph_close_kernel))
        processed_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=morph_close_iterations)
        
        if keep_largest_component:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed_mask, connectivity=8)
            
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                largest_mask = np.zeros_like(processed_mask)
                largest_mask[labels == largest_label] = 255
                
                print(f"         Found {num_labels - 1} components, keeping largest (area: {stats[largest_label, cv2.CC_STAT_AREA]} pixels)")
                
                return largest_mask
        
        return processed_mask
    
    return combined_mask

def filter_cells_by_inner_part(cell_masks, cell_centers, inner_mask):
    filtered_masks = []
    filtered_centers = []
    
    for center, mask in zip(cell_centers, cell_masks):
        cx, cy = int(center[0]), int(center[1])
        
        if cx >= inner_mask.shape[1] or cy >= inner_mask.shape[0]:
            filtered_masks.append(mask)
            filtered_centers.append(center)
            continue
        
        if inner_mask[cy, cx] == 0:
            filtered_masks.append(mask)
            filtered_centers.append(center)
    
    return filtered_masks, filtered_centers

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
    morph_open_kernel=99,
    morph_close_kernel=155,
    morph_open_iterations=1,
    morph_close_iterations=1,
    keep_largest_component=True
):
    if not os.path.exists(cell_weights):
        print(f"❌ Cell model not found: {cell_weights}")
        return None
    
    if not os.path.exists(image_folder):
        print(f"❌ Folder not found: {image_folder}")
        return None
    
    cell_model = YOLO(cell_weights)
    cell_model.overrides['max_det'] = max_det
    
    exclusion_model = None
    if filter_by_inner_part:
        if not os.path.exists(exclusion_weights):
            print(f"⚠️ Exclusion model not found: {exclusion_weights}")
            filter_by_inner_part = False
        else:
            exclusion_model = YOLO(exclusion_weights)
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]
    
    if not images:
        print(f"❌ No images found in {image_folder}")
        return None
    
    print(f"\n📷 ROBUST MODEL BATCH TESTING")
    print("="*50)
    print(f"Region growing: {refine}")
    print(f"Color tolerance: {color_tolerance}")
    print(f"Filter by inner-part: {filter_by_inner_part}")
    print(f"Exclusion confidence: {exclusion_confidence}")
    print(f"Keep largest component: {keep_largest_component}")
    print(f"Images found: {len(images)}")
    print("="*50)
    
    results_summary = {}
    
    for img_file in images:
        img_path = os.path.join(image_folder, img_file)
        
        cell_results = cell_model(img_path, conf=confidence, max_det=max_det, iou=0.45)
        
        img = cv2.imread(img_path)
        filename = os.path.basename(img_path)
        
        cell_masks, cell_centers = extract_cell_data(cell_results, img, class_id=0, 
                                                      refine=refine,
                                                      color_tolerance=color_tolerance)
        
        print(f"\n   {filename}")
        print(f"      Final cells after region growing: {len(cell_masks)}")
        
        inner_mask = None
        inner_pixel_count = 0
        cells_removed = 0
        
        if filter_by_inner_part and exclusion_model is not None:
            exclusion_results = exclusion_model(img_path, conf=exclusion_confidence, max_det=max_det)
            
            if exclusion_results[0].masks is not None:
                num_raw = len(exclusion_results[0].masks)
                print(f"      Exclusion model raw detections: {num_raw} at conf={exclusion_confidence}")
            else:
                print(f"      Exclusion model raw detections: 0 at conf={exclusion_confidence}")
            
            inner_mask = get_inner_part_mask(
                exclusion_results, img,
                morph_open_kernel=morph_open_kernel,
                morph_close_kernel=morph_close_kernel,
                morph_open_iterations=morph_open_iterations,
                morph_close_iterations=morph_close_iterations,
                keep_largest_component=keep_largest_component
            )
            inner_pixel_count = np.sum(inner_mask > 0)
            
            if inner_pixel_count > 0:
                cells_before = len(cell_masks)
                cell_masks, cell_centers = filter_cells_by_inner_part(cell_masks, cell_centers, inner_mask)
                cells_removed = cells_before - len(cell_masks)
                print(f"      Inner-part area (after morphology): {inner_pixel_count} pixels")
                print(f"      Cells removed: {cells_removed}")
                print(f"      Final cells after filtering: {len(cell_masks)}")
            else:
                print(f"      No inner-part detected after morphology")
        
        img_output = draw_cells_on_image(img, cell_masks, color=(0, 0, 255), thickness=2)
        
        if inner_pixel_count > 0 and inner_mask is not None:
            overlay = np.zeros_like(img)
            overlay[inner_mask == 255] = (0, 0, 255)
            img_output = cv2.addWeighted(img_output, 0.7, overlay, 0.3, 0)
        
        cv2.putText(img_output, f"Cells: {len(cell_masks)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if inner_pixel_count > 0:
            cv2.putText(img_output, f"Removed: {cells_removed}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        output_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(output_path, img_output)
        
        if inner_pixel_count > 0 and inner_mask is not None:
            inner_viz = img.copy()
            inner_viz[inner_mask == 255] = (0, 0, 255)
            cv2.putText(inner_viz, f"Inner-part mask (largest component only)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            inner_output_path = os.path.join(output_dir, f"inner_part_{filename}")
            cv2.imwrite(inner_output_path, inner_viz)
        
        results_summary[filename] = {
            'raw_cells': len(cell_masks) + cells_removed,
            'filtered_cells': len(cell_masks),
            'cells_removed': cells_removed,
            'inner_part_area': inner_pixel_count
        }
    
    summary_df = pd.DataFrame([
        {
            'image': img, 
            'raw_cells': data['raw_cells'],
            'filtered_cells': data['filtered_cells'],
            'cells_removed': data['cells_removed'],
            'inner_part_area': data['inner_part_area']
        }
        for img, data in results_summary.items()
    ])
    summary_path = os.path.join(output_dir, "detection_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("="*50)
    print(summary_df.to_string(index=False))
    print(f"\n✅ Complete! Results saved to {output_dir}")
    
    return results_summary

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        batch_test_robust_model(image_folder=sys.argv[1])
    else:
        print("Usage: python test_robust.py <image_folder>")
