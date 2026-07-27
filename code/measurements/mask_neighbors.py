import cv2
import numpy as np
from collections import defaultdict
import warnings
import gc

warnings.filterwarnings('ignore', category=UserWarning)

def get_neighbors_from_masks(
    measurements,
    cell_masks,
    cell_centers,
    expansion_pixels=2,
    max_neighbors=None,
    verbose=True
):
    
    n = len(cell_masks)
    
    if n < 3:
        for m in measurements:
            m['neighbors'] = ''
            m['degree'] = 0
        return measurements
    
    # Use the memory-efficient method
    neighbor_sets = find_neighbors_by_dilation_efficient(
        cell_masks, expansion_pixels, verbose
    )
    
    # Optionally enforce max neighbors
    if max_neighbors is not None and max_neighbors > 0:
        coords = np.array(cell_centers)
        neighbor_sets = enforce_max_neighbors(neighbor_sets, coords, max_neighbors, verbose)
    
    # Add neighbors and degree to measurements
    for i, m in enumerate(measurements):
        if i in neighbor_sets and neighbor_sets[i]:
            neighbor_ids = [measurements[n]['cell_id'] for n in neighbor_sets[i] if n < len(measurements)]
            neighbor_ids.sort()
            m['neighbors'] = ','.join(map(str, neighbor_ids))
            m['degree'] = len(neighbor_ids)
        else:
            m['neighbors'] = ''
            m['degree'] = 0
    
    # Print statistics
    if verbose:
        degrees = [m['degree'] for m in measurements]
        total_edges = sum(degrees) // 2
        max_actual = max(degrees) if degrees else 0
        min_actual = min(degrees) if degrees else 0
        mean_actual = np.mean(degrees) if degrees else 0
        median_actual = np.median(degrees) if degrees else 0
    
    return measurements


def find_neighbors_by_dilation_efficient(cell_masks, expansion_pixels, verbose):

    n = len(cell_masks)
    neighbor_sets = {i: set() for i in range(n)}
    
    # Crop each mask to its bounding box and store info
    mask_data = []
    for i, mask in enumerate(cell_masks):
        if mask is None or np.sum(mask) == 0:
            continue
        
        # Get bounding box
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        if not np.any(rows) or not np.any(cols):
            continue
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Add padding for expansion (so dilated mask doesn't get truncated)
        pad = expansion_pixels + 2
        y_min = max(0, y_min - pad)
        y_max = min(mask.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(mask.shape[1], x_max + pad)
        
        # Crop mask
        cropped = mask[y_min:y_max, x_min:x_max]
        
        # Store data
        mask_data.append({
            'index': i,
            'mask': cropped,
            'bbox': (x_min, y_min, x_max, y_max),
            'area': np.sum(cropped > 0)
        })
    
    if len(mask_data) < 2:
        if verbose:
            print("Not enough valid masks")
        return neighbor_sets
    
    # Use cell centers for initial filtering
    coords = []
    for data in mask_data:
        x1, y1, x2, y2 = data['bbox']
        coords.append(((x1 + x2) / 2, (y1 + y2) / 2))
    
    # Dilate all cropped masks (one at a time to save memory)
    kernel_size = expansion_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Dilate each mask once and store
    dilated_data = []
    for data in mask_data:
        if data['area'] == 0:
            continue
        dilated = cv2.dilate(data['mask'].astype(np.uint8), kernel, iterations=1)
        dilated_data.append({
            'index': data['index'],
            'dilated': dilated > 0,
            'bbox': data['bbox'],
            'area': data['area']
        })
    
    # Check overlaps efficiently using spatial filtering
    overlap_count = 0
    m = len(dilated_data)
    
    # Process in batches to avoid memory issues
    batch_size = min(50, m)  # Process 50 cells at a time
    
    for batch_start in range(0, m, batch_size):
        batch_end = min(batch_start + batch_size, m)
        
        for i in range(batch_start, batch_end):
            data_i = dilated_data[i]
            if data_i is None:
                continue
            
            x1_i, y1_i, x2_i, y2_i = data_i['bbox']
            center_i = ((x1_i + x2_i) / 2, (y1_i + y2_i) / 2)
            
            # Skip if this cell has no dilated pixels
            if not np.any(data_i['dilated']):
                continue
            
            # Check against all other cells (not just future ones for symmetry)
            # We'll check i < j and add both directions
            for j in range(i + 1, m):
                data_j = dilated_data[j]
                if data_j is None:
                    continue
                
                x1_j, y1_j, x2_j, y2_j = data_j['bbox']
                
                # Quick bounding box overlap check first
                if x1_i > x2_j or x1_j > x2_i or y1_i > y2_j or y1_j > y2_i:
                    continue
                
                # Quick center distance check - if too far, skip
                center_j = ((x1_j + x2_j) / 2, (y1_j + y2_j) / 2)
                dist = np.sqrt((center_i[0] - center_j[0])**2 + (center_i[1] - center_j[1])**2)
                
                # If centers are further than 2x the expansion pixels, they can't overlap
                max_possible_dist = (abs(x2_i - x1_i) + abs(x2_j - x1_j) + abs(y2_i - y1_i) + abs(y2_j - y1_j)) / 2
                if dist > max_possible_dist * 1.5:
                    continue
                
                # Crop to overlapping region for faster check
                # Find the overlapping region in the original image coordinates
                overlap_x1 = max(x1_i, x1_j)
                overlap_x2 = min(x2_i, x2_j)
                overlap_y1 = max(y1_i, y1_j)
                overlap_y2 = min(y2_i, y2_j)
                
                if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
                    continue
                
                # Extract the overlapping region from each dilated mask
                # Convert global coordinates to local coordinates
                # For mask i
                x1_i_local = overlap_x1 - x1_i
                x2_i_local = overlap_x2 - x1_i
                y1_i_local = overlap_y1 - y1_i
                y2_i_local = overlap_y2 - y1_i
                
                # For mask j
                x1_j_local = overlap_x1 - x1_j
                x2_j_local = overlap_x2 - x1_j
                y1_j_local = overlap_y1 - y1_j
                y2_j_local = overlap_y2 - y1_j
                
                # Get the overlapping regions
                try:
                    roi_i = data_i['dilated'][y1_i_local:y2_i_local, x1_i_local:x2_i_local]
                    roi_j = data_j['dilated'][y1_j_local:y2_j_local, x1_j_local:x2_j_local]
                    
                    # Check if there's any overlap in the ROI
                    if roi_i.shape == roi_j.shape and np.any(roi_i & roi_j):
                        neighbor_sets[data_i['index']].add(data_j['index'])
                        neighbor_sets[data_j['index']].add(data_i['index'])
                        overlap_count += 1
                except Exception as e:
                    # Fallback: check whole masks (slower but safe)
                    if np.any(data_i['dilated'] & data_j['dilated']):
                        neighbor_sets[data_i['index']].add(data_j['index'])
                        neighbor_sets[data_j['index']].add(data_i['index'])
                        overlap_count += 1
        
        # Free memory after each batch
        if batch_start % (batch_size * 2) == 0:
            gc.collect()
    
    # Ensure symmetry
    symmetric_sets = {i: set() for i in range(n)}
    for i in range(n):
        for j in neighbor_sets[i]:
            if i < j:
                symmetric_sets[i].add(j)
                symmetric_sets[j].add(i)
    
    return symmetric_sets


def find_neighbors_by_border_expansion(cell_masks, expansion_pixels, verbose):

    n = len(cell_masks)
    neighbor_sets = {i: set() for i in range(n)}

    # Get border masks for each cell (memory efficient - process one at a time)
    border_masks = []
    for i, mask in enumerate(cell_masks):
        if mask is None or np.sum(mask) == 0:
            border_masks.append(None)
            continue
        
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create border mask (only the boundary pixels)
        border = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(border, contours, -1, 1, 1)  # Draw contours as lines
        
        # Expand the border
        kernel_size = expansion_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded_border = cv2.dilate(border, kernel, iterations=1)
        
        border_masks.append(expanded_border > 0)
    
    # Check overlaps (using the same efficient method as above)
    overlap_count = 0
    for i in range(n):
        if border_masks[i] is None or not np.any(border_masks[i]):
            continue
        
        for j in range(i + 1, n):
            if border_masks[j] is None or not np.any(border_masks[j]):
                continue
            
            # Quick bounding box check
            # For simplicity, just check the whole mask
            if np.any(border_masks[i] & border_masks[j]):
                neighbor_sets[i].add(j)
                neighbor_sets[j].add(i)
                overlap_count += 1
    
    return neighbor_sets


def enforce_max_neighbors(neighbor_sets, coords, max_neighbors, verbose):

    n = len(coords)
    
    # Count how many cells exceed the limit
    exceeding = sum(1 for i in range(n) if len(neighbor_sets[i]) > max_neighbors)
    
    if exceeding == 0:
        return neighbor_sets
    
    final_sets = {i: set() for i in range(n)}
    
    for i in range(n):
        if i in neighbor_sets and neighbor_sets[i]:
            neighbors = list(neighbor_sets[i])
            
            if len(neighbors) <= max_neighbors:
                final_sets[i] = set(neighbors)
            else:
                # Keep only the closest neighbors (by center distance)
                distances = [np.linalg.norm(coords[i] - coords[j]) for j in neighbors]
                sorted_pairs = sorted(zip(distances, neighbors))
                kept_neighbors = [j for _, j in sorted_pairs[:max_neighbors]]
                final_sets[i] = set(kept_neighbors)
        else:
            final_sets[i] = set()
    
    # Ensure symmetry
    symmetric_final = {i: set() for i in range(n)}
    for i in range(n):
        for j in final_sets[i]:
            if i < j:
                symmetric_final[i].add(j)
                symmetric_final[j].add(i)
    
    return symmetric_final


def batch_calculate_neighbors(
    measurements_folder, 
    expansion_pixels=20, 
    max_neighbors=None, 
    verbose=True, 
    specific_images=None,
    force_recalc=False  # ADD THIS PARAMETER
):
    """
    Calculate neighbors for all measurement files in a folder.
    
    Args:
        measurements_folder: Path to folder containing measurement CSV files
        expansion_pixels: Number of pixels to expand cell masks when checking for neighbors
        max_neighbors: Maximum number of neighbors per cell (None for unlimited)
        verbose: Whether to print progress information
        specific_images: List of specific image names to process (if None, process all)
        force_recalc: If True, recalculate neighbors even if they already exist
    
    Returns:
        Number of files successfully processed
    """
    import glob
    import pandas as pd
    from pathlib import Path
    import os
    import ast
    
    measurements_folder = Path(measurements_folder)
    csv_files = list(measurements_folder.glob("*_measurements.csv"))
    
    # If specific_images is provided, filter to only those images
    if specific_images is not None:
        if not specific_images:  # Empty list - nothing to process
            if verbose:
                print(f"No new images to process for neighbor calculation (specific_images is empty)")
            return 0
        specific_set = set(os.path.splitext(img)[0] for img in specific_images)
        filtered_csv_files = []
        for csv_file in csv_files:
            base_name = csv_file.stem.replace('_measurements', '')
            if base_name in specific_set:
                filtered_csv_files.append(csv_file)
        original_count = len(csv_files)
        csv_files = filtered_csv_files
        if verbose:
            print(f"Filtering to {len(csv_files)} specific images for neighbor calculation (from {original_count} total)")
    
    if verbose:
        print(f"Found {len(csv_files)} measurement files to process")
    
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    
    for csv_path in csv_files:
        try:
            # Load measurements
            df = pd.read_csv(csv_path)
            
            # Check if neighbors already calculated and valid
            if not force_recalc and 'neighbors' in df.columns and 'degree' in df.columns:
                # Check if ANY cell has a neighbor (degree > 0)
                has_valid_neighbors = (df['degree'] > 0).any()
                
                # Also check if the neighbor data looks valid (not all empty strings)
                has_non_empty = df['neighbors'].notna().any() and (df['neighbors'].astype(str) != '').any()
                
                if has_valid_neighbors and has_non_empty:
                    if verbose:
                        print(f"  Skipping {csv_path.stem} (valid neighbors already calculated)")
                    skipped_count += 1
                    processed_count += 1
                    continue
                else:
                    if verbose:
                        print(f"  Recalculating {csv_path.stem} (existing neighbor data is empty or invalid)")
            
            # Load measurements as dict
            measurements = df.to_dict('records')
            
            # Get cell centers
            if 'center_x' not in df.columns or 'center_y' not in df.columns:
                if verbose:
                    print(f"  Skipping {csv_path.stem} (no center coordinates)")
                continue
            
            # Try to get cell masks from the measurements
            # Check if mask data is stored as a column
            cell_masks = []
            cell_centers = []
            
            for i, row in enumerate(measurements):
                # Try to parse mask if stored as string
                mask = None
                if 'mask' in row and row['mask']:
                    try:
                        # If mask is stored as a string of coordinates or numpy array
                        if isinstance(row['mask'], str):
                            # Try to parse as list of lists
                            mask_data = ast.literal_eval(row['mask'])
                            if isinstance(mask_data, list):
                                # Create a binary mask from coordinates
                                # This depends on how your masks are stored
                                # For now, skip mask loading and use borders
                                pass
                    except:
                        pass
                
                cell_centers.append((row['center_x'], row['center_y']))
            
            # If we don't have masks, we need to load them from the image files
            # or from the _centers.jpg files
            if not cell_masks or len(cell_masks) == 0:
                # Try to load masks from the corresponding image file
                # Look for the original quadrant image
                base_name = csv_path.stem.replace('_measurements', '')
                image_path = measurements_folder / f"{base_name}.jpg"
                
                if not image_path.exists():
                    # Try with different extensions
                    for ext in ['.png', '.jpeg', '.tif', '.tiff']:
                        alt_path = measurements_folder / f"{base_name}{ext}"
                        if alt_path.exists():
                            image_path = alt_path
                            break
                
                if image_path.exists():
                    if verbose:
                        print(f"  Loading cell masks from: {image_path}")
                    # Load the image and create masks from cell center data
                    # For now, we'll use a simplified approach with borders
                    # You may need to implement proper mask loading here
                    cell_masks = create_masks_from_centers(cell_centers, image_path)
                else:
                    # If we can't load masks, use border expansion method with estimated cell sizes
                    if verbose:
                        print(f"  No mask data available, using estimated cell sizes")
                    cell_masks = create_estimated_masks(cell_centers)
            
            # Calculate neighbors using the cell masks
            if len(cell_masks) > 1:
                # Use the neighbor finding function
                from code.measurements.mask_neighbors import get_neighbors_from_masks
                
                measurements_with_neighbors = get_neighbors_from_masks(
                    measurements=measurements,
                    cell_masks=cell_masks,
                    cell_centers=cell_centers,
                    expansion_pixels=expansion_pixels,
                    max_neighbors=max_neighbors,
                    verbose=verbose
                )
                
                # Update the measurements
                measurements = measurements_with_neighbors
            else:
                # Not enough cells, set empty neighbors
                if verbose:
                    print(f"  Not enough cells ({len(cell_masks)}), setting empty neighbors")
                for m in measurements:
                    m['neighbors'] = ''
                    m['degree'] = 0
            
            # Save updated measurements
            updated_df = pd.DataFrame(measurements)
            updated_df.to_csv(csv_path, index=False)
            
            # Calculate degree statistics
            degrees = [m.get('degree', 0) for m in measurements]
            degree_counts = {}
            for d in set(degrees):
                degree_counts[d] = degrees.count(d)
            
            if verbose:
                avg_degree = sum(degrees) / len(degrees) if degrees else 0
                max_degree = max(degrees) if degrees else 0
                print(f"  Processed {csv_path.stem}: {len(measurements)} cells, avg degree: {avg_degree:.2f}, max: {max_degree}")
            
            processed_count += 1
            
        except Exception as e:
            if verbose:
                print(f"  Error processing {csv_path.stem}: {e}")
            failed_count += 1
    
    if verbose:
        print(f"\nProcessed {processed_count} files, {failed_count} failed, {skipped_count} skipped")
    
    return processed_count


def create_masks_from_centers(cell_centers, image_path):
    """Create binary masks from cell center coordinates."""
    import cv2
    import numpy as np
    
    # Load the image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    h, w = img.shape[:2]
    
    # Create masks - for each cell, create a small mask around its center
    # This is a simplified approach - you'd want to use the actual segmentation masks
    masks = []
    for (cx, cy) in cell_centers:
        mask = np.zeros((h, w), dtype=np.uint8)
        # Create a small circle around the center (approximate cell size)
        cv2.circle(mask, (int(cx), int(cy)), 15, 1, -1)  # 15 pixel radius
        masks.append(mask)
    
    return masks


def create_estimated_masks(cell_centers):
    """Create estimated masks when no real masks are available."""
    import numpy as np
    
    # Use the same approach but without image dimensions
    # Just create a list of masks with estimated sizes
    # This will only work with the border expansion method
    masks = []
    for (cx, cy) in cell_centers:
        # Create a dummy mask - the actual size will be determined by the neighbor function
        masks.append(np.ones((100, 100), dtype=np.uint8))
    
    return masks