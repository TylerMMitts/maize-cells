# Measures every segmented cell.
#
# Turns each cell mask into area, radius and angle from the stele centre,
# in microns. Writes one _measurements.csv per quadrant image into
# results/measurements_all/.

import cv2
import numpy as np
import pandas as pd
import os
import gc
import time
import psutil
import json
import glob
import re
import traceback
import sys
from ultralytics import YOLO
from code.yolo.test_robust import extract_cell_data, filter_cells_by_inner_part, get_inner_part_mask, filter_overlapping_cells
from code.util.visualize_cell_centers import get_plant_center_from_filename, draw_red_dots_on_image
from code.measurements.density_analysis import compute_all_density_metrics
from code.measurements.mask_neighbors import get_neighbors_from_masks
from code.util.file_utils import parse_image_name, resolve_original_image_name
from code.util.image_utils import load_image_metadata, calculate_area_from_mask, calculate_stele_area, calculate_root_radius
from collections import defaultdict
from code.config import DATA_FOLDER, EXCLUSION_WEIGHTS, MASTER_SUMMARY_PATH, MEASUREMENTS_FOLDER, ROBUST_CELL_WEIGHTS

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TORCH_DEVICE'] = 'cpu'

# Try to import config for dynamic conversion factor
try:
    from code import config
    DEFAULT_PIXEL_TO_UM = getattr(config, 'DEFAULT_PIXEL_TO_UM', 50.0 / 77.0)
    DEFAULT_PIXEL_TO_UM_SQUARED = getattr(config, 'DEFAULT_PIXEL_TO_UM_SQUARED', DEFAULT_PIXEL_TO_UM * DEFAULT_PIXEL_TO_UM)
except (ImportError, AttributeError):
    # Fallback to hardcoded values if config not available
    DEFAULT_PIXEL_TO_UM = 50.0 / 77.0
    DEFAULT_PIXEL_TO_UM_SQUARED = DEFAULT_PIXEL_TO_UM * DEFAULT_PIXEL_TO_UM

# Store the current conversion factors for use throughout the module
_CURRENT_PIXEL_TO_UM = DEFAULT_PIXEL_TO_UM
_CURRENT_PIXEL_TO_UM_SQUARED = DEFAULT_PIXEL_TO_UM_SQUARED


def set_conversion_factors(pixel_to_um: float, pixel_to_um_squared: float = None):

    global _CURRENT_PIXEL_TO_UM, _CURRENT_PIXEL_TO_UM_SQUARED
    _CURRENT_PIXEL_TO_UM = pixel_to_um
    if pixel_to_um_squared is None:
        _CURRENT_PIXEL_TO_UM_SQUARED = pixel_to_um * pixel_to_um
    else:
        _CURRENT_PIXEL_TO_UM_SQUARED = pixel_to_um_squared
    print(f"Conversion factors set: pixel_to_um={_CURRENT_PIXEL_TO_UM}, pixel_to_um_squared={_CURRENT_PIXEL_TO_UM_SQUARED}")


def get_conversion_factors():
    return _CURRENT_PIXEL_TO_UM, _CURRENT_PIXEL_TO_UM_SQUARED


def get_conversion_factor_from_metadata(metadata):
    if metadata and 'conversion_factor' in metadata:
        return metadata['conversion_factor']['pixels_to_um']
    return _CURRENT_PIXEL_TO_UM


def get_memory_usage_mb():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def force_memory_cleanup():
    gc.collect()
    if hasattr(gc, 'garbage'):
        gc.garbage.clear()
    cv2.setNumThreads(0)


def add_local_densities_to_measurements(measurements, cell_centers, radius_scale=100):
    if len(cell_centers) < 3:
        for m in measurements:
            m['local_density'] = 0
        return measurements
    
    coords = np.array(cell_centers, dtype=np.float32)
    from scipy.spatial import KDTree
    tree = KDTree(coords)
    
    densities = []
    for center in coords:
        neighbors = tree.query_ball_point(center, radius_scale)
        density = len(neighbors) / (np.pi * radius_scale**2)
        densities.append(density)
    
    for i, m in enumerate(measurements):
        m['local_density'] = round(densities[i], 6)
    
    return measurements


def save_image_summary(image_path, output_dir, stele_area_um2, root_radius_um, quadrant, n_cells, conversion_factor):
    summary = {
        'image_name': os.path.basename(image_path),
        'quadrant': quadrant,
        'stele_area_um2': round(stele_area_um2, 2),
        'root_radius_um': round(root_radius_um, 2),
        'n_cells_detected': n_cells,
        'conversion_factor_pixels_to_um': conversion_factor,
        'conversion_formula': f'{conversion_factor:.6f} µm/pixel'
    }
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    summary_path = os.path.join(output_dir, f"{base_name}_image_summary.json")
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved image summary to {summary_path}")
    return summary_path


def create_master_summary(measurements_folder=MEASUREMENTS_FOLDER, output_path=MASTER_SUMMARY_PATH, 
                          species=None, population=None):
    
    # Use provided values or defaults
    if species is None:
        species = 'Zea mays'
        print(f"No species provided, using default: {species}")
    if population is None:
        population = 'IBM'
        print(f"No population provided, using default: {population}")
    
    csv_files = glob.glob(os.path.join(measurements_folder, "*_measurements.csv"))
    
    if not csv_files:
        print(f"No measurement CSV files found in {measurements_folder}")
        return None
    
    print(f"Found {len(csv_files)} measurement files")
    
    json_files = glob.glob(os.path.join(measurements_folder, "*_image_summary.json"))
    print(f"Found {len(json_files)} image summary files")
    
    # Load image summaries into a dictionary
    image_summaries = {}
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                summary = json.load(f)
                image_name = summary.get('image_name', '')
                if image_name:
                    base_name = os.path.splitext(image_name)[0]
                    # Remove _centers suffix if present
                    if base_name.endswith('_centers'):
                        base_name = base_name[:-8]
                    image_summaries[base_name] = summary
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
    
    master_data = []
    
    for csv_path in csv_files:
        base_name = os.path.basename(csv_path).replace('_measurements.csv', '')
        
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                print(f"Skipping {base_name}: empty")
                continue
            
            # Use the shared parse_image_name function from file_utils
            parsed = parse_image_name(base_name)
            summary = image_summaries.get(base_name, {})
            
            # Calculate file count
            if 'cell_file' in df.columns:
                valid_files = df[df['cell_file'] >= 0]['cell_file']
                n_files = valid_files.nunique() if not valid_files.empty else 0
            elif 'file_number' in df.columns:
                n_files = df['file_number'].nunique()
            else:
                if 'cell_file_derivative' in df.columns:
                    valid_files = df[df['cell_file_derivative'] >= 0]['cell_file_derivative']
                    n_files = valid_files.nunique() if not valid_files.empty else 0
                else:
                    n_files = 0
            
            # Calculate average cell area
            if 'area_um2' in df.columns:
                avg_area = df['area_um2'].mean()
                n_cells = len(df)
            elif 'area_pixels' in df.columns:
                conversion = summary.get('conversion_factor_pixels_to_um', _CURRENT_PIXEL_TO_UM)
                avg_area = (df['area_pixels'].mean()) * (conversion ** 2)
                n_cells = len(df)
            else:
                avg_area = 0
                n_cells = len(df)
            
            stele_area = summary.get('stele_area_um2', 0)
            root_radius = summary.get('root_radius_um', 0)
            quadrant = parsed.get('quadrant', 'unknown')
            
            master_data.append({
                # Identifiers
                'image_name': base_name,
                'quadrant': quadrant,
                'plot_number': parsed.get('plot_number'),
                'plant_number': parsed.get('plant_number'),
                'root_number': parsed.get('root_number'),
                'technical_replicate': parsed.get('technical_replicate'),
                
                # Experimental design
                'treatment': parsed.get('treatment', 'unknown'),
                'root_type': parsed.get('root_type', 'unknown'),
                'species': species,
                'population': population,
                
                # Root measurements
                'file_count': n_files,
                'average_cell_area_um2': round(avg_area, 2) if avg_area > 0 else 0,
                'n_cells': n_cells,
                'stele_area_um2': round(stele_area, 2) if stele_area > 0 else 0,
                'root_radius_um': round(root_radius, 2) if root_radius > 0 else 0,
                
                # Metadata tracking
                'measurements_csv': os.path.basename(csv_path),
                'has_image_summary': base_name in image_summaries
            })
            
            print(f"Processed {base_name}: {n_files} files, {n_cells} cells, avg area {avg_area:.1f}µm²")
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
    
    if not master_data:
        print("No data to save")
        return None
    
    master_df = pd.DataFrame(master_data)
    master_df = master_df.sort_values(['plant_number', 'root_number', 'quadrant']).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master_df.to_csv(output_path, index=False)
    
    print(f"\nMaster summary saved to: {output_path}")
    
    print("\nQuadrant distribution:")
    quadrant_counts = master_df['quadrant'].value_counts()
    for quadrant, count in quadrant_counts.items():
        print(f"  {quadrant}: {count} images")
    
    return master_df


def extract_cell_measurements(
    weights_path=ROBUST_CELL_WEIGHTS,
    exclusion_weights=EXCLUSION_WEIGHTS,
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
    sam_refiner=None,
    compute_density_analysis=True,
    density_output_dir="density_analysis",
    expansion_pixels=2,
    max_neighbors=8,
    pixel_to_um=None,
    pixel_to_um_squared=None
):

    print(f"Processing: {image_path}")
    print(f"Memory before: {get_memory_usage_mb():.1f} MB")
    
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return None
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    # Determine conversion factors
    if pixel_to_um is not None:
        print(f"Using provided conversion factor: pixel_to_um={pixel_to_um}")
        if pixel_to_um_squared is None:
            pixel_to_um_squared = pixel_to_um * pixel_to_um
    else:
        metadata = load_image_metadata(image_path)
        pixel_to_um = get_conversion_factor_from_metadata(metadata)
        pixel_to_um_squared = pixel_to_um * pixel_to_um
        print(f"Using conversion factor from metadata: pixel_to_um={pixel_to_um}")
    
    model = YOLO(weights_path)
    model.to('cpu')
    
    exclusion_model = None
    if filter_by_inner_part and os.path.exists(exclusion_weights):
        exclusion_model = YOLO(exclusion_weights)
        exclusion_model.to('cpu')
    else:
        filter_by_inner_part = False
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None
    
    filename = os.path.basename(image_path)
    h, w = img.shape[:2]
    
    # Get plant center
    center_x, center_y, quadrant = get_plant_center_from_filename(filename, w, h)
    
    if center_x is None:
        print(f"Could not determine plant center from {filename}")
        del model, exclusion_model, img
        force_memory_cleanup()
        return None
    
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
    
    del results
    
    # Store inner_mask for later analysis
    inner_mask_for_analysis = None
    
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
        
        inner_mask_for_analysis = inner_mask.copy() if inner_mask is not None else None
        
        if np.sum(inner_mask) > 0:
            cells_before = len(cell_masks)
            cell_masks, cell_centers, cell_confidences = filter_cells_by_inner_part(
                cell_masks, cell_centers, cell_confidences, inner_mask, exclusion_overlap_threshold
            )
            cells_removed = cells_before - len(cell_centers)
            print(f"Cells removed by exclusion filter: {cells_removed}")
        
        del inner_mask
    
    cells_before_overlap = len(cell_masks)
    cell_masks, cell_centers, cell_confidences = filter_overlapping_cells(
        cell_masks, cell_centers, cell_confidences, overlap_threshold=overlap_threshold
    )
    cells_removed_overlap = cells_before_overlap - len(cell_masks)
    if cells_removed_overlap > 0:
        print(f"Cells removed by overlap filter: {cells_removed_overlap}")
    
    measurements = []
    cell_objects = []
    
    center_x_f = float(center_x)
    center_y_f = float(center_y)
    
    # Create measurements from processed masks
    for i, (center, mask) in enumerate(zip(cell_centers, cell_masks)):
        area_pixels = calculate_area_from_mask(mask)
        
        dx = center[0] - center_x_f
        dy = center[1] - center_y_f
        radius_pixels = np.sqrt(dx*dx + dy*dy)
        angle = np.degrees(np.arctan2(dy, dx))
        
        area_um2 = area_pixels * pixel_to_um_squared
        radius_um = radius_pixels * pixel_to_um
        x_um = center[0] * pixel_to_um
        y_um = center[1] * pixel_to_um
        
        measurements.append({
            'cell_id': i + 1,
            'x_pixels': round(center[0], 2),
            'y_pixels': round(center[1], 2),
            'x_um': round(x_um, 2),
            'y_um': round(y_um, 2),
            'radius_pixels': round(radius_pixels, 2),
            'radius_um': round(radius_um, 2),
            'angle_degrees': round(angle, 2),
            'area_pixels': round(area_pixels, 2),
            'area_um2': round(area_um2, 2),
            'quadrant': quadrant if quadrant else 'unknown'
        })
        
        class SimpleCell:
            __slots__ = ('radius', 'angle', 'x', 'y')
            pass
        
        cell_obj = SimpleCell()
        cell_obj.radius = radius_pixels
        cell_obj.angle = angle
        cell_obj.x = center[0]
        cell_obj.y = center[1]
        cell_objects.append(cell_obj)
        
        del mask
    
    # Create deep copies of masks for neighbor detection
    if cell_masks is not None and len(cell_masks) > 0:
        processed_cell_masks = [mask.copy() for mask in cell_masks]
    else:
        processed_cell_masks = None
    
    del cell_masks
    
    # Calculate image-level statistics
    stele_area_um2 = 0.0
    root_radius_um = 0.0
    
    if inner_mask_for_analysis is not None and np.sum(inner_mask_for_analysis) > 0:
        stele_area_um2 = calculate_stele_area(inner_mask_for_analysis, pixel_to_um)
        root_radius_um = calculate_root_radius(
            inner_mask_for_analysis, center_x, center_y, pixel_to_um
        )
        
        save_image_summary(
            image_path, 
            os.path.dirname(output_csv), 
            stele_area_um2, 
            root_radius_um, 
            quadrant, 
            len(measurements),
            pixel_to_um
        )
    
    # Add image-level info to each measurement
    for m in measurements:
        m['stele_area_um2'] = round(stele_area_um2, 2)
        m['root_radius_um'] = round(root_radius_um, 2)
    
    # Add local densities
    measurements = add_local_densities_to_measurements(measurements, cell_centers, radius_scale=100)
    
    # Process neighbor detection
    if processed_cell_masks is not None and len(processed_cell_masks) > 0:
        try:
            measurements = get_neighbors_from_masks(
                measurements,
                processed_cell_masks,
                cell_centers,
                expansion_pixels=expansion_pixels,
                max_neighbors=max_neighbors,
                verbose=True
            )
        except Exception as e:
            print(f"Warning: Neighbor detection failed: {e}")
        finally:
            del processed_cell_masks
            force_memory_cleanup()
    
    df = pd.DataFrame(measurements)
    df.to_csv(output_csv, index=False)
    print(f"Measurements saved to {output_csv}")
    
    if compute_density_analysis and len(cell_objects) > 0:
        img_density_dir = os.path.join(density_output_dir, os.path.splitext(filename)[0])
        df_with_density = pd.DataFrame(measurements)
        try:
            density_metrics = compute_all_density_metrics(
                df_with_density,
                output_dir=img_density_dir,
                filename=filename
            )
            print(f"Density analysis saved to {img_density_dir}")
        except Exception as e:
            print(f"Warning: Density analysis failed: {e}")
        finally:
            del df_with_density
    
    # Create visualization
    img_output = img.copy()
    for center in cell_centers:
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(img_output, (cx, cy), 3, (0, 0, 255), -1)
    
    cv2.circle(img_output, (int(center_x), int(center_y)), 8, (255, 0, 0), -1)
    cv2.putText(img_output, f"Cells: {len(cell_centers)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img_output, f"Quadrant: {quadrant}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    cv2.imwrite(output_image, img_output)
    print(f"Visualization saved to {output_image}")
    
    if show_result:
        cv2.imshow("Cell Centers", img_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Clean up
    del img, img_output, cell_centers, cell_confidences, measurements, cell_objects, df
    if inner_mask_for_analysis is not None:
        del inner_mask_for_analysis
    del model
    if exclusion_model:
        del exclusion_model
    force_memory_cleanup()
    
    print(f"Memory after: {get_memory_usage_mb():.1f} MB")
    
    return None


def batch_extract_measurements(
    weights_path=ROBUST_CELL_WEIGHTS,
    exclusion_weights=EXCLUSION_WEIGHTS,
    image_folder=DATA_FOLDER / 'cropped_images',
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
    sam_refiner=None,
    compute_density_analysis=True,
    density_output_dir="density_analysis",
    expansion_pixels=2,
    max_neighbors=8,
    specific_images=None,
    create_master_summary_flag=False,
    species=None,
    population=None,
    pixel_to_um=None,
    pixel_to_um_squared=None,
    pixel_to_um_map=None
):

    print(f"batch_extract_measurements called with {image_folder}")

    # Set global conversion factors if provided (used as the fallback default
    # for any image not covered by pixel_to_um_map)
    if pixel_to_um is not None:
        if pixel_to_um_squared is None:
            pixel_to_um_squared = pixel_to_um * pixel_to_um
        set_conversion_factors(pixel_to_um, pixel_to_um_squared)
        print(f"Using provided conversion factor: pixel_to_um={pixel_to_um}")

    if pixel_to_um_map:
        print(f"Using per-image scale map ({len(pixel_to_um_map)} images); "
              f"falling back to pixel_to_um={pixel_to_um} for unmatched images")
    
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return None
    
    if not os.path.exists(image_folder):
        print(f"Folder not found: {image_folder}")
        return None
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if compute_density_analysis:
        os.makedirs(density_output_dir, exist_ok=True)
    
    # Get list of images
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]
    
    # Filter out _full.jpg images and detected_ images (only process quadrants)
    images = [f for f in images if not f.endswith('_full.jpg') and not f.startswith('detected_')]
    
    # If specific_images is provided, filter to only those images
    if specific_images is not None:
        if not specific_images:
            print(f"No new images to process (specific_images is empty)")
            return None
        specific_set = set(specific_images)
        original_count = len(images)
        images = [f for f in images if f in specific_set]
        print(f"Filtering to {len(images)} specific new images (from {original_count} total quadrants)")
    
    if not images:
        print(f"No quadrant images found to process in {image_folder}")
        return None
    
    print(f"Found {len(images)} quadrant images to process")
    
    successful = 0
    failed = 0
    failed_images = []
    
    # Process each image
    for idx, img_file in enumerate(images, 1):
        img_path = os.path.join(image_folder, img_file)
        base_name = os.path.splitext(img_file)[0]
        csv_path = os.path.join(output_dir, f"{base_name}_measurements.csv")
        image_output_path = os.path.join(output_dir, f"{base_name}_centers.jpg")
        
        # Skip if measurements already exist (unless specific_images was provided)
        if os.path.exists(csv_path) and specific_images is None:
            print(f"[{idx}/{len(images)}] Skipping {img_file} (measurements already exist)")
            successful += 1
            continue
        
        print(f"[{idx}/{len(images)}] Processing {img_file}...")
        print(f"Memory before: {get_memory_usage_mb():.1f} MB")

        # Resolve per-image scale from the map, if provided
        image_pixel_to_um = pixel_to_um
        image_pixel_to_um_squared = pixel_to_um_squared
        if pixel_to_um_map:
            original_name = resolve_original_image_name(base_name)
            matched_value = pixel_to_um_map.get(original_name)
            if matched_value is not None:
                image_pixel_to_um = matched_value
                image_pixel_to_um_squared = matched_value * matched_value
                print(f"  Scale for {original_name}: pixel_to_um={image_pixel_to_um:.6f} (from scale map)")
            else:
                print(f"  WARNING: No scale match for '{original_name}' in pixel_to_um_map, "
                      f"falling back to pixel_to_um={pixel_to_um}")

        try:
            extract_cell_measurements(
                weights_path=weights_path,
                exclusion_weights=exclusion_weights,
                image_path=img_path,
                confidence=confidence,
                exclusion_confidence=exclusion_confidence,
                output_csv=csv_path,
                output_image=image_output_path,
                show_result=False,
                refine=refine,
                color_tolerance=color_tolerance,
                use_darkest_seed=use_darkest_seed,
                filter_by_inner_part=filter_by_inner_part,
                inner_open_kernel=inner_open_kernel,
                inner_close_kernel=inner_close_kernel,
                outer_open_kernel=outer_open_kernel,
                outer_close_kernel=outer_close_kernel,
                keep_largest_component=keep_largest_component,
                exclusion_overlap_threshold=exclusion_overlap_threshold,
                overlap_threshold=overlap_threshold,
                morph_post_process=morph_post_process,
                morph_close_kernel=morph_close_kernel,
                morph_open_kernel=morph_open_kernel,
                sam_refiner=sam_refiner,
                compute_density_analysis=compute_density_analysis,
                density_output_dir=density_output_dir,
                expansion_pixels=expansion_pixels,
                max_neighbors=max_neighbors,
                pixel_to_um=image_pixel_to_um,
                pixel_to_um_squared=image_pixel_to_um_squared
            )
            successful += 1
            print(f"Successfully processed {img_file}")
            
        except KeyboardInterrupt:
            print("\nBatch processing interrupted by user")
            raise
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            traceback.print_exc()
            failed += 1
            failed_images.append(img_file)
        
        finally:
            force_memory_cleanup()
            print(f"Memory after cleanup: {get_memory_usage_mb():.1f} MB")
            time.sleep(0.5)
        
        print(f"Completed processing {idx}/{len(images)}")
    
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(images)}")
    
    if failed_images:
        print(f"\nFailed images:")
        for img in failed_images:
            print(f"  - {img}")
    
    # Create master summary ONLY if explicitly requested
    if create_master_summary_flag and successful > 0:
        try:
            master_df = create_master_summary(
                measurements_folder=output_dir,
                output_path=os.path.join(os.path.dirname(output_dir), "master_summary.csv"),
                species=species,
                population=population
            )
            return master_df
        except Exception as e:
            print(f"Error creating master summary: {e}")
            traceback.print_exc()
            return None
    
    return None


def visualize_measurements(
    weights_path=ROBUST_CELL_WEIGHTS,
    exclusion_weights=EXCLUSION_WEIGHTS,
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
    
    metadata = load_image_metadata(image_path)
    pixel_to_um = get_conversion_factor_from_metadata(metadata)
    
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
    
    cell_masks, cell_centers, cell_confidences = extract_cell_data(
        results, img, class_id=0, 
        refine=refine, 
        color_tolerance=color_tolerance
    )
    
    del results
    
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
    
    draw_red_dots_on_image(img, cell_centers, output_path, plant_center, quadrant)

    del img, cell_masks, cell_centers, cell_confidences, model
    force_memory_cleanup()
    
    if show_result:
        img_display = cv2.imread(output_path)
        cv2.imshow("Cell Centers", img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        del img_display