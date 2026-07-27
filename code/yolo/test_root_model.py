from ultralytics import YOLO
import cv2
import os
import numpy as np
import json
import csv
from pathlib import Path

PIXEL_TO_UM = 50 / 77

def save_crop_metadata(original_image_path, crop_bbox, output_crop_path, quadrant_info=None):
    original_img = cv2.imread(original_image_path)
    h, w = original_img.shape[:2]
    
    x1, y1, x2, y2 = crop_bbox
    crop_width_px = x2 - x1
    crop_height_px = y2 - y1
    
    crop_width_um = crop_width_px * PIXEL_TO_UM
    crop_height_um = crop_height_px * PIXEL_TO_UM
    
    metadata = {
        'original_image': os.path.basename(original_image_path),
        'original_dimensions_px': {'width': w, 'height': h},
        'original_dimensions_um': {
            'width': round(w * PIXEL_TO_UM, 2),
            'height': round(h * PIXEL_TO_UM, 2)
        },
        'crop_bbox_px': {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'width': crop_width_px, 'height': crop_height_px
        },
        'crop_bbox_um': {
            'x1': round(x1 * PIXEL_TO_UM, 2),
            'y1': round(y1 * PIXEL_TO_UM, 2),
            'x2': round(x2 * PIXEL_TO_UM, 2),
            'y2': round(y2 * PIXEL_TO_UM, 2),
            'width': round(crop_width_um, 2),
            'height': round(crop_height_um, 2)
        },
        'conversion_factor': {
            'pixels_to_um': PIXEL_TO_UM,
            'formula': '50µm / 77pixels'
        },
        'quadrant': quadrant_info,
        'cropped_image': os.path.basename(output_crop_path)
    }
    
    metadata_path = output_crop_path.replace('.jpg', '_metadata.json').replace('.png', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path

def split_into_quadrants(image, base_name, output_dir="quadrants", original_image_path=None, root_bbox=None):
    
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
    metadata_paths = []
    
    for name, x1, y1, x2, y2 in quadrants:
        quadrant_img = image[y1:y2, x1:x2]
        output_path = os.path.join(output_dir, f"{name}_{base_name}.jpg")
        cv2.imwrite(output_path, quadrant_img)
        saved_paths.append(output_path)
        
        if original_image_path and root_bbox:
            root_x1, root_y1, root_x2, root_y2 = root_bbox
            abs_x1 = root_x1 + x1
            abs_y1 = root_y1 + y1
            abs_x2 = root_x1 + x2
            abs_y2 = root_y1 + y2
            
            quadrant_bbox = (abs_x1, abs_y1, abs_x2, abs_y2)
            metadata_path = save_crop_metadata(original_image_path, quadrant_bbox, output_path, name)
            metadata_paths.append(metadata_path)
    
    return saved_paths, metadata_paths

def test_root_model(
    weights_path="models/runs/detect/runs/root_detection/root_detector/weights/best.pt",
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
        
        print(f"Found {len(detections)} root detection(s)")
        
        best_idx = max(range(len(detections)), key=lambda i: detections[i]['confidence'])
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            root_crop = img[y1:y2, x1:x2]
            
            root_base = f"{base_name}_root{i+1}"
            
            crop_path = os.path.join(detections_dir, f"{root_base}_full.jpg")
            cv2.imwrite(crop_path, root_crop)
            
            save_crop_metadata(image_path, (x1, y1, x2, y2), crop_path, None)
            
            if save_quadrants and i == best_idx:
                quadrants, metadata_paths = split_into_quadrants(
                    root_crop, root_base, output_dir,
                    original_image_path=image_path, 
                    root_bbox=(x1, y1, x2, y2)
                )
                print(f"  Saved {len(quadrants)} quadrants to {output_dir}")
    
    return detections

def batch_test_root_model(
    weights_path="models/runs/detect/runs/root_detection/root_detector/weights/best.pt",
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
    
    master_metadata = []
    
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
                
                crop_path = os.path.join(detections_dir, f"{root_base}_full.jpg")
                cv2.imwrite(crop_path, root_crop)
                
                metadata_path = save_crop_metadata(img_path, (x1, y1, x2, y2), crop_path, None)
                master_metadata.append({
                    'image': img_file,
                    'detection_index': i,
                    'confidence': conf,
                    'is_best': (i == best_idx),
                    'crop_path': crop_path,
                    'metadata_path': metadata_path
                })
                
                if i == best_idx:
                    quadrants, quadrant_metadata = split_into_quadrants(
                        root_crop, root_base, output_dir,
                        original_image_path=img_path,
                        root_bbox=(x1, y1, x2, y2)
                    )
                    for q_path, q_metadata in zip(quadrants, quadrant_metadata):
                        master_metadata.append({
                            'image': img_file,
                            'detection_index': i,
                            'quadrant': os.path.basename(q_path).split('_')[0],
                            'crop_path': q_path,
                            'metadata_path': q_metadata
                        })
        else:
            print(f"\n{img_file}: No detections")
    
    master_metadata_path = os.path.join(output_dir, "all_crops_metadata.json")
    with open(master_metadata_path, 'w') as f:
        json.dump(master_metadata, f, indent=2)
    
    create_metadata_summary(master_metadata_path, output_dir)
    
    return all_detections

def create_metadata_summary(metadata_json_path, output_dir):
    
    with open(metadata_json_path, 'r') as f:
        metadata_list = json.load(f)
    
    summary_file = os.path.join(output_dir, "crops_summary.csv")
    
    with open(summary_file, 'w', newline='') as csvfile:
        fieldnames = ['image', 'crop_type', 'crop_name', 'width_um', 'height_um', 
                     'width_px', 'height_px', 'metadata_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in metadata_list:
            with open(item['metadata_path'], 'r') as mf:
                crop_meta = json.load(mf)
            
            writer.writerow({
                'image': item['image'],
                'crop_type': item.get('quadrant', 'full_root'),
                'crop_name': os.path.basename(item['crop_path']),
                'width_um': crop_meta['crop_bbox_um']['width'],
                'height_um': crop_meta['crop_bbox_um']['height'],
                'width_px': crop_meta['crop_bbox_px']['width'],
                'height_px': crop_meta['crop_bbox_px']['height'],
                'metadata_path': item['metadata_path']
            })
    
    print(f"Summary CSV saved to: {summary_file}")

def load_crop_metadata(crop_image_path):

    metadata_path = crop_image_path.replace('.jpg', '_metadata.json').replace('.png', '_metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: No metadata found for {crop_image_path}")
        return None

def get_pixel_to_um_conversion(crop_image_path):

    metadata = load_crop_metadata(crop_image_path)
    
    if metadata:
        conversion = metadata['conversion_factor']['pixels_to_um']
        print(f"Conversion for {os.path.basename(crop_image_path)}: 1 pixel = {conversion:.4f} µm")
        return conversion, metadata
    else:
        print(f"Warning: Using default conversion for {crop_image_path}")
        return PIXEL_TO_UM, None

if __name__ == "__main__":
    batch_test_root_model(
        image_folder="data/chosen_some", 
        output_dir="data/chosen_results"
    )