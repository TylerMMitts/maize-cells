# Visual check of the exclusion model on one image.
#
# Shows which region the model considers valid cortex, which is the usual
# first thing to inspect when cell counts come out wrong.

import cv2
import os
import numpy as np
from ultralytics import YOLO
from code.config import DATA_FOLDER, EXCLUSION_WEIGHTS

EXCLUSION_WEIGHTS = EXCLUSION_WEIGHTS
IMAGE_PATH = DATA_FOLDER / 'cropped_images' / 'BL7.jpg'
CONFIDENCE = 0.25
OUTPUT_DIR = "test_inner_part"

# Morphological post-processing settings for inner-part
INNER_OPEN_KERNEL = 99
INNER_CLOSE_KERNEL = 155
INNER_OPEN_ITERATIONS = 1
INNER_CLOSE_ITERATIONS = 1

# Morphological post-processing settings for outer-part
OUTER_OPEN_KERNEL = 33
OUTER_CLOSE_KERNEL = 55
OUTER_OPEN_ITERATIONS = 1
OUTER_CLOSE_ITERATIONS = 1

if not os.path.exists(EXCLUSION_WEIGHTS):
    print(f"Exclusion model not found: {EXCLUSION_WEIGHTS}")
    exit()

if not os.path.exists(IMAGE_PATH):
    print(f"Test image not found: {IMAGE_PATH}")
    exit()

model = YOLO(EXCLUSION_WEIGHTS)

results = model(IMAGE_PATH, conf=CONFIDENCE)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load original image
img = cv2.imread(IMAGE_PATH)
overlay_img = img.copy()
mask_only_img = np.zeros_like(img)

# Process detections
num_detections = 0
inner_pixel_count = 0
class_names = ['inner-part', 'outer-part']
processed_inner = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
processed_outer = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

if results[0].masks is not None:
    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes
    num_detections = len(masks)
    
    print(f"\nFound {num_detections} exclusion detections (inner-part + outer-part)")
    
    # Create separate masks for each class
    inner_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    outer_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
        
        conf = float(boxes.conf[i]) if boxes is not None else 0
        cls = int(boxes.cls[i]) if boxes is not None else 0
        cls_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
        area = np.sum(binary_mask > 0)
        print(f"      Raw detection {i+1}: {cls_name}, confidence={conf:.3f}, area={area} pixels")
        
        # Separate by class
        if cls == 0:
            inner_mask = cv2.bitwise_or(inner_mask, binary_mask)
        elif cls == 1:
            outer_mask = cv2.bitwise_or(outer_mask, binary_mask)
    
    # Process inner-part mask
    if np.sum(inner_mask) > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INNER_OPEN_KERNEL, INNER_OPEN_KERNEL))
        opened_inner = cv2.morphologyEx(inner_mask, cv2.MORPH_OPEN, kernel_open, iterations=INNER_OPEN_ITERATIONS)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INNER_CLOSE_KERNEL, INNER_CLOSE_KERNEL))
        processed_inner = cv2.morphologyEx(opened_inner, cv2.MORPH_CLOSE, kernel_close, iterations=INNER_CLOSE_ITERATIONS)
        
        # Keep largest inner-part component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed_inner, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            largest_inner = np.zeros_like(processed_inner)
            largest_inner[labels == largest_label] = 255
            processed_inner = largest_inner
            print(f"Inner-part: Found {num_labels - 1} components, keeping largest (area: {stats[largest_label, cv2.CC_STAT_AREA]} pixels)")
    
    # Process outer-part mask
    if np.sum(outer_mask) > 0:

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OUTER_CLOSE_KERNEL, OUTER_CLOSE_KERNEL))
        processed_outer = cv2.morphologyEx(outer_mask, cv2.MORPH_CLOSE, kernel_close, iterations=OUTER_CLOSE_ITERATIONS)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OUTER_OPEN_KERNEL, OUTER_OPEN_KERNEL))
        opened_outer = cv2.morphologyEx(processed_outer, cv2.MORPH_OPEN, kernel_open, iterations=OUTER_OPEN_ITERATIONS)
        
        # Keep largest outer-part component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened_outer, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            largest_outer = np.zeros_like(opened_outer)
            largest_outer[labels == largest_label] = 255
            processed_outer = largest_outer
            print(f"Outer-part: Found {num_labels - 1} components, keeping largest (area: {stats[largest_label, cv2.CC_STAT_AREA]} pixels)")
    
    # Combine both processed masks
    combined_mask = cv2.bitwise_or(processed_inner, processed_outer)
    processed_mask = combined_mask
    
    # Count areas
    inner_area = np.sum(processed_inner > 0)
    outer_area = np.sum(processed_outer > 0)
    processed_area = np.sum(processed_mask > 0)
    
    print(f"Processed inner-part area: {inner_area} pixels")
    print(f"Processed outer-part area: {outer_area} pixels")
    print(f"Total combined area: {processed_area} pixels")
    
    # Find contours from processed mask
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw on overlay image
    cv2.drawContours(overlay_img, contours, -1, (0, 0, 255), 2)
    
    # Create mask-only image
    mask_only_img[processed_mask == 255] = (0, 0, 255)
    
    inner_pixel_count = processed_area
else:
    print(f"\nNo exclusion detections found")

# Add text to images
cv2.putText(overlay_img, f"Exclusion zones: {len(contours) if 'contours' in dir() else 0}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(overlay_img, f"Confidence: {CONFIDENCE}", (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.putText(mask_only_img, f"Exclusion masks (inner + outer)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Save results
overlay_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(IMAGE_PATH)}")
cv2.imwrite(overlay_path, overlay_img)

mask_path = os.path.join(OUTPUT_DIR, f"mask_{os.path.basename(IMAGE_PATH)}")
cv2.imwrite(mask_path, mask_only_img)

# Also save intermediate masks for comparison
if results[0].masks is not None:
    # Save processed inner-part mask
    if np.sum(processed_inner) > 0:
        inner_viz = np.zeros_like(img)
        inner_viz[processed_inner == 255] = (0, 255, 0)
        inner_path = os.path.join(OUTPUT_DIR, f"inner_part_{os.path.basename(IMAGE_PATH)}")
        cv2.imwrite(inner_path, inner_viz)
        print(f"Saved inner-part mask: {inner_path}")
    
    # Save processed outer-part mask
    if np.sum(processed_outer) > 0:
        outer_viz = np.zeros_like(img)
        outer_viz[processed_outer == 255] = (255, 0, 0)
        outer_path = os.path.join(OUTPUT_DIR, f"outer_part_{os.path.basename(IMAGE_PATH)}")
        cv2.imwrite(outer_path, outer_viz)
        print(f"Saved outer-part mask: {outer_path}")
    
    # Save combined processed mask
    combined_path = os.path.join(OUTPUT_DIR, f"combined_mask_{os.path.basename(IMAGE_PATH)}")
    cv2.imwrite(combined_path, processed_mask)
    print(f"Saved combined mask: {combined_path}")

# Display result
cv2.imshow(f"Exclusion Zone Detection (Inner + Outer)", overlay_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
