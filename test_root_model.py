from ultralytics import YOLO
import cv2
import os
import numpy as np

def split_into_quadrants(image, base_name, output_dir="quadrants"):
    h, w = image.shape[:2]
    half_h, half_w = h // 2, w // 2
    
    quadrants = [
        ("TL", 0, 0, half_w, half_h),
        ("TR", half_w, 0, w, half_h),
        ("BL", 0, half_h, half_w, h),
        ("BR", half_w, half_h, w, h)
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for name, x1, y1, x2, y2 in quadrants:
        quadrant_img = image[y1:y2, x1:x2]
        output_path = os.path.join(output_dir, f"{name}_{base_name}.jpg")
        cv2.imwrite(output_path, quadrant_img)
        saved_paths.append(output_path)
    
    return saved_paths

def test_root_model(
    weights_path="runs/detect/runs/root_detection/root_detector/weights/best.pt",
    image_path="test_root.jpg",
    confidence=0.50,
    output_dir="root_results",
    save_quadrants=True
):
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return None
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    model = YOLO(weights_path)
    results = model(image_path, conf=confidence)
    
    os.makedirs(output_dir, exist_ok=True)
    detections_dir = os.path.join(output_dir, "detections")
    os.makedirs(detections_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    detections = []
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class': cls
            })
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        print(f"Found {len(detections)} root detection(s)")
        
        # Find the detection with the highest confidence
        best_idx = max(range(len(detections)), key=lambda i: detections[i]['confidence'])
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            root_crop = img[y1:y2, x1:x2]
            
            root_base = f"{base_name}_root{i+1}"
            
            # Only save quadrants for the best confidence detection
            if save_quadrants and i == best_idx:
                split_into_quadrants(root_crop, root_base, output_dir)
            
            crop_path = os.path.join(detections_dir, f"{root_base}_full.jpg")
            cv2.imwrite(crop_path, root_crop)
    
    output_path = os.path.join(detections_dir, f"detected_{base_name}.jpg")
    cv2.imwrite(output_path, img)
    
    return detections

def batch_test_root_model(
    weights_path="runs/detect/runs/root_detection/root_detector/weights/best.pt",
    image_folder="images",
    confidence=0.50,
    output_dir="root_results"
):
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return
    
    if not os.path.exists(image_folder):
        print(f"Folder not found: {image_folder}")
        return
    
    model = YOLO(weights_path)
    os.makedirs(output_dir, exist_ok=True)
    detections_dir = os.path.join(output_dir, "detections")
    os.makedirs(detections_dir, exist_ok=True)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]
    
    all_detections = {}
    
    for img_file in images:
        img_path = os.path.join(image_folder, img_file)
        results = model(img_path, conf=confidence)
        
        img = cv2.imread(img_path)
        base_name = os.path.splitext(img_file)[0]
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            print(f"\n{img_file}: {len(boxes)} detection(s)")
            
            # Find the detection with the highest confidence
            best_idx = 0
            best_conf = 0.0
            detections_data = []
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections_data.append((i, x1, y1, x2, y2, conf))
                
                if conf > best_conf:
                    best_conf = conf
                    best_idx = i
                
            
            for i, x1, y1, x2, y2, conf in detections_data:
                root_crop = img[y1:y2, x1:x2]
                root_base = f"{base_name}_root{i+1}"
                
                # Only save quadrants for the best confidence detection
                if i == best_idx:
                    split_into_quadrants(root_crop, root_base, output_dir)
                
                crop_path = os.path.join(detections_dir, f"{root_base}_full.jpg")
                cv2.imwrite(crop_path, root_crop)
                print(f"Saved: {crop_path}")
        else:
            print(f"\n{img_file}: No detections")
        
        output_path = os.path.join(detections_dir, f"detected_{img_file}")
        cv2.imwrite(output_path, img)
