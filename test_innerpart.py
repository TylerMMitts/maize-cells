import cv2
import os
import numpy as np
from ultralytics import YOLO

EXCLUSION_WEIGHTS = "runs/segment/runs/exclusion_model/inner_part_detector/weights/best.pt"
IMAGE_PATH = "cropped_images/BL3.jpg"
CONFIDENCE = 0.5
OUTPUT_DIR = "test_inner_part"

# Morphological post-processing settings
MORPH_OPEN_KERNEL = 99       
MORPH_CLOSE_KERNEL = 155       
MORPH_OPEN_ITERATIONS = 1  
MORPH_CLOSE_ITERATIONS = 1

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

if results[0].masks is not None:
    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes
    num_detections = len(masks)
    
    print(f"\nFound {num_detections} inner-part detections")
    
    # Create combined mask from all detections
    combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
        
        conf = float(boxes.conf[i]) if boxes is not None else 0
        area = np.sum(binary_mask > 0)
        print(f"      Raw detection {i+1}: confidence={conf:.3f}, area={area} pixels")
        
        # Add to combined mask
        combined_mask = cv2.bitwise_or(combined_mask, binary_mask)
    
    # Apply morphological post-processing
    # Opening to remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL))
    opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open, iterations=MORPH_OPEN_ITERATIONS)
    
    # Closing to fill gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL))
    processed_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=MORPH_CLOSE_ITERATIONS)
    
    # Count areas
    raw_area = np.sum(combined_mask > 0)
    opened_area = np.sum(opened > 0)
    processed_area = np.sum(processed_mask > 0)
    
    print(f"Raw combined area: {raw_area} pixels")
    print(f"After OPEN: {opened_area} pixels")
    print(f"After CLOSE: {processed_area} pixels")
    
    # Find contours from processed mask
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw on overlay image
    cv2.drawContours(overlay_img, contours, -1, (0, 0, 255), 2)
    
    # Create mask-only image
    mask_only_img[processed_mask == 255] = (0, 0, 255)
    
    inner_pixel_count = processed_area
else:
    print(f"\nNo inner-part detections found")

# Add text to images
cv2.putText(overlay_img, f"Inner-part: {len(contours) if 'contours' in dir() else 0}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(overlay_img, f"Confidence: {CONFIDENCE}", (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.putText(mask_only_img, f"Inner-part masks", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Save results
overlay_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(IMAGE_PATH)}")
cv2.imwrite(overlay_path, overlay_img)

mask_path = os.path.join(OUTPUT_DIR, f"mask_{os.path.basename(IMAGE_PATH)}")
cv2.imwrite(mask_path, mask_only_img)

# Also save intermediate masks for comparison
if results[0].masks is not None:
    raw_mask_path = os.path.join(OUTPUT_DIR, f"raw_mask_{os.path.basename(IMAGE_PATH)}")
    cv2.imwrite(raw_mask_path, combined_mask)
    print(f"Saved raw mask: {raw_mask_path}")
    
    opened_mask_path = os.path.join(OUTPUT_DIR, f"opened_mask_{os.path.basename(IMAGE_PATH)}")
    cv2.imwrite(opened_mask_path, opened)
    print(f"Saved opened mask (after noise removal): {opened_mask_path}")

# Display result
cv2.imshow(f"Inner-part Detection", overlay_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
