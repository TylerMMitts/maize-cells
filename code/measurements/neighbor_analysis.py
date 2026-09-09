# Analyses the neighbour graph across the whole dataset.
#
# Uses the neighbour relationships to trace paths outward through the cortex
# and produce the averaged radial patterns. This is the largest analysis
# module and holds most of the pattern plots.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from scipy import stats
from scipy.interpolate import interp1d
import cv2
from collections import deque
import json
from tqdm import tqdm

# Get the project root directory (go up from code/measurements/ to project root)
SCRIPT_DIR = Path(__file__).parent.absolute()  # code/measurements/
CODE_DIR = SCRIPT_DIR.parent  # code/
PROJECT_ROOT = CODE_DIR.parent  # project root (DNA_to_Display/)

# Add code directory to path for imports
import sys
sys.path.insert(0, str(CODE_DIR))

# Import utility functions
from code.util import load_measurement_csv, load_image_summary, find_all_measurement_files, load_cell_assignments, determine_angle_range, get_sample_angles, normalize_angle, get_cells_at_angle

# Use absolute paths based on project root
MEASUREMENTS_FOLDER = str(PROJECT_ROOT / "results" / "measurements_all")
CELL_FILE_COUNTS_FOLDER = str(PROJECT_ROOT / "results" / "cell_file" / "cell_file_counting")
OUTPUT_FOLDER = str(PROJECT_ROOT / "results" / "neighbor_analysis")

# Binning parameters
RADIUS_BINS = 15
AREA_BINS = 15
NORM_POINTS = 20

# Visualization options
COLOR_MAP = 'tab20'
ALPHA = 0.7
LINE_WIDTH = 2
MARKER_SIZE = 6
LEGEND_THRESHOLD = 20

# Angle sampling parameters
ANGLE_TOLERANCE = 5
N_ANGLES = 5

# File to track processed images for neighbor paths
PROCESSED_PATHS_LOG = os.path.join(OUTPUT_FOLDER, "processed_paths_log.csv")


def load_measurement_files(measurements_folder, specific_images=None):

    csv_files = glob.glob(str(Path(measurements_folder) / "*_measurements.csv"))
    
    print(f"Found {len(csv_files)} CSV files in {measurements_folder}")
    
    if not csv_files:
        print(f"No measurement files found in {measurements_folder}")
        return None
    
    all_data = []
    loaded_count = 0
    
    for csv_path in csv_files:
        image_name = Path(csv_path).stem.replace('_measurements', '')
        
        # If specific_images is provided, skip images not in the list
        if specific_images is not None and image_name not in specific_images:
            continue
        
        df = load_measurement_csv(csv_path)
        if df is not None and not df.empty:
            df['image_name'] = image_name
            all_data.append(df)
            loaded_count += 1
            print(f"  Loaded {image_name}: {len(df)} cells")
    
    if not all_data:
        print("No matching measurement files found")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total cells loaded: {len(combined_df)} from {loaded_count} images")
    return combined_df


def load_existing_processed_images(log_path):

    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            return set(df['image_name'].tolist())
        except Exception as e:
            print(f"Warning: Could not load processed paths log: {e}")
            return set()
    return set()


def save_processed_images_log(log_path, processed_images):

    df = pd.DataFrame({'image_name': list(processed_images)})
    df.to_csv(log_path, index=False)
    print(f"  Saved processed paths log: {log_path}")


def merge_cell_file_data(combined_df, cell_file_counts_folder):

    print("LOADING CELL FILE ASSIGNMENTS")

    # Get unique image names from the combined_df
    image_names = combined_df['image_name'].unique()

    loaded_count = 0
    failed_count = 0
    all_assignments = []

    for image_name in image_names:
        # Load cell assignments for this image
        assignments_df = load_cell_assignments(cell_file_counts_folder, image_name)

        if assignments_df is None:
            failed_count += 1
            continue

        assignments_df = assignments_df[['cell_id', 'cell_file_derivative']].copy()
        assignments_df['image_name'] = image_name
        all_assignments.append(assignments_df)
        loaded_count += 1

    print(f"Loaded cell_file data for {loaded_count} images, failed for {failed_count} images")

    # Drop any stale placeholder column before merging
    combined_df = combined_df.drop(columns=['cell_file'], errors='ignore')

    if all_assignments:
        assignments_combined = pd.concat(all_assignments, ignore_index=True)
        assignments_combined = assignments_combined.rename(columns={'cell_file_derivative': 'cell_file'})
        combined_df = combined_df.merge(
            assignments_combined, on=['image_name', 'cell_id'], how='left'
        )
        combined_df['cell_file'] = combined_df['cell_file'].fillna(-1).astype(int)
    else:
        combined_df['cell_file'] = -1

    # Check how many cells have valid cell_file values
    valid_count = len(combined_df[combined_df['cell_file'] >= 0])
    print(f"Cells with valid cell_file: {valid_count} out of {len(combined_df)}")

    return combined_df


def find_outermost_cell(image_df, target_angle, debug=False):

    angle_df = get_cells_at_angle(image_df, target_angle, angle_column='angle_degrees')
    
    if angle_df.empty:
        if debug:
            print(f"    No cells found at angle {target_angle:.1f}° (±{ANGLE_TOLERANCE}°)")
        return None
    
    outer_idx = angle_df['radius_pixels'].idxmax()
    outer_cell = angle_df.loc[outer_idx]
    
    if debug:
        print(f"    Angle {target_angle:.1f}°: outer radius={outer_cell['radius_pixels']:.1f}px")
    
    return outer_cell


def find_min_cell_file_for_angle(image_df, target_angle, debug=False):

    angle_df = get_cells_at_angle(image_df, target_angle, angle_column='angle_degrees')
    
    if angle_df.empty:
        return None
    
    # Filter out cells with invalid cell_file (-1)
    valid_df = angle_df[angle_df['cell_file'] >= 0]
    
    if valid_df.empty:
        return None
    
    # Find the minimum cell file number in this angle sample
    min_cell_file = valid_df['cell_file'].min()
    
    if debug:
        print(f"    Min cell file at angle {target_angle:.1f}°: {min_cell_file}")
    
    return min_cell_file


def trace_path_to_stele(image_df, start_cell, adjacency, debug=False):

    cell_to_df_idx = {row['cell_id']: idx for idx, row in image_df.iterrows()}
    
    # Get the start index
    current_idx = cell_to_df_idx.get(start_cell['cell_id'])
    if current_idx is None:
        return None, None
    
    path_indices = [current_idx]
    
    # Keep track of visited cells to avoid cycles
    visited = set([current_idx])
    
    while True:
        current_cell = image_df.loc[current_idx]
        current_radius = current_cell['radius_pixels']
        
        # Find the neighbor with the smallest radius
        best_neighbor_idx = None
        best_radius = current_radius
        
        for neighbor_idx in adjacency.get(current_idx, []):
            if neighbor_idx in visited:
                continue
            
            neighbor_cell = image_df.loc[neighbor_idx]
            neighbor_radius = neighbor_cell['radius_pixels']
            
            # Check if this neighbor has a smaller radius
            if neighbor_radius < best_radius:
                best_radius = neighbor_radius
                best_neighbor_idx = neighbor_idx
        
        # If no neighbor has a smaller radius, we've reached the local minimum (stele)
        if best_neighbor_idx is None:
            if debug:
                print(f"    Reached stele at cell {current_cell['cell_id']} with radius {current_radius:.1f}px")
            break
        
        # Move to the best neighbor
        current_idx = best_neighbor_idx
        visited.add(current_idx)
        path_indices.append(current_idx)
        
        if debug:
            print(f"    Moving to neighbor with radius {best_radius:.1f}px")
    
    # Get the final cell (the one with the smallest radius reached)
    final_cell = image_df.loc[current_idx]
    
    if debug:
        path_length = len(path_indices)
        print(f"    Path complete: {path_length} cells, final radius={final_cell['radius_pixels']:.1f}px")
    
    return path_indices, final_cell


def build_neighbor_graph(image_df, neighbor_col='neighbors'):

    cell_to_df_idx = {row['cell_id']: idx for idx, row in image_df.iterrows()}
    
    adjacency = {idx: [] for idx in image_df.index}
    
    neighbor_count = 0
    for idx, row in image_df.iterrows():
        neighbor_ids = row.get(neighbor_col, '')
        if pd.isna(neighbor_ids) or neighbor_ids == '':
            continue
        
        try:
            neighbor_str = str(neighbor_ids).strip('"').strip("'")
            neighbors = [int(n.strip()) for n in neighbor_str.split(',') if n.strip()]
            for n_id in neighbors:
                if n_id in cell_to_df_idx:
                    neighbor_df_idx = cell_to_df_idx[n_id]
                    adjacency[idx].append(neighbor_df_idx)
                    neighbor_count += 1
        except Exception as e:
            continue
    
    return adjacency, cell_to_df_idx


def calculate_physical_distance(path_cells, use_um=True):

    if len(path_cells) < 2:
        return 0
    
    total_distance = 0
    x_key = 'x_um' if use_um else 'x_pixels'
    y_key = 'y_um' if use_um else 'y_pixels'
    
    for i in range(len(path_cells) - 1):
        dx = path_cells[i+1][x_key] - path_cells[i][x_key]
        dy = path_cells[i+1][y_key] - path_cells[i][y_key]
        total_distance += np.sqrt(dx**2 + dy**2)
    
    return total_distance


def save_path_metadata(image_name, angle_paths, output_folder):

    if not angle_paths:
        return
    
    # Create metadata folder
    metadata_folder = os.path.join(output_folder, 'metadata')
    os.makedirs(metadata_folder, exist_ok=True)
    
    # Prepare data for CSV
    path_data = []
    travel_distances_steps = []
    physical_distances_um = []
    
    for i, path_info in enumerate(angle_paths):
        if path_info is None or len(path_info['path']) < 2:
            continue
        
        display_angle = normalize_angle(path_info['angle'])
        
        # Get path cell IDs
        path_cell_ids = [cell['cell_id'] for cell in path_info['path']]
        
        # Calculate physical distance in micrometers
        physical_um = calculate_physical_distance(path_info['path'], use_um=True)
        
        path_data.append({
            'image_name': image_name,
            'angle_index': i + 1,
            'angle_degrees': display_angle,
            'angle_raw': path_info['angle'],
            'travel_distance_steps': path_info['travel_distance'],
            'physical_distance_um': physical_um,
            'path_length_cells': path_info['path_length'],
            'target_cell_file': path_info.get('target_cell_file', -1),
            'outer_cell_id': path_info['outer_cell']['cell_id'],
            'outer_radius_px': path_info['outer_cell']['radius_pixels'],
            'outer_radius_um': path_info['outer_cell']['radius_um'],
            'final_cell_id': path_info['final_cell']['cell_id'],
            'final_radius_px': path_info['final_cell']['radius_pixels'],
            'final_radius_um': path_info['final_cell']['radius_um'],
            'path_cell_ids': ';'.join(map(str, path_cell_ids))
        })
        
        travel_distances_steps.append(path_info['travel_distance'])
        physical_distances_um.append(physical_um)
    
    if not path_data:
        return
    
    # Save individual path data
    df_paths = pd.DataFrame(path_data)
    csv_path = os.path.join(metadata_folder, f'{image_name}_travel_paths.csv')
    df_paths.to_csv(csv_path, index=False)
    print(f"    Saved metadata: {Path(csv_path).name}")
    
    # Calculate and save summary statistics
    summary_data = {
        'image_name': image_name,
        'total_paths_found': len(path_data),
        'total_angles_analyzed': len(angle_paths),
        'avg_travel_distance_steps': np.mean(travel_distances_steps) if travel_distances_steps else 0,
        'std_travel_distance_steps': np.std(travel_distances_steps) if travel_distances_steps else 0,
        'min_travel_distance_steps': np.min(travel_distances_steps) if travel_distances_steps else 0,
        'max_travel_distance_steps': np.max(travel_distances_steps) if travel_distances_steps else 0,
        'median_travel_distance_steps': np.median(travel_distances_steps) if travel_distances_steps else 0,
        'sum_travel_distance_steps': np.sum(travel_distances_steps) if travel_distances_steps else 0,
        'avg_physical_distance_um': np.mean(physical_distances_um) if physical_distances_um else 0,
        'std_physical_distance_um': np.std(physical_distances_um) if physical_distances_um else 0,
        'min_physical_distance_um': np.min(physical_distances_um) if physical_distances_um else 0,
        'max_physical_distance_um': np.max(physical_distances_um) if physical_distances_um else 0,
        'median_physical_distance_um': np.median(physical_distances_um) if physical_distances_um else 0,
        'sum_physical_distance_um': np.sum(physical_distances_um) if physical_distances_um else 0,
        'avg_path_length_cells': np.mean([p['path_length_cells'] for p in path_data]) if path_data else 0,
        'min_path_length_cells': np.min([p['path_length_cells'] for p in path_data]) if path_data else 0,
        'max_path_length_cells': np.max([p['path_length_cells'] for p in path_data]) if path_data else 0,
        'avg_final_radius_um': np.mean([p['final_radius_um'] for p in path_data]) if path_data else 0
    }
    
    # Add per-angle distances
    for i, dist in enumerate(travel_distances_steps):
        summary_data[f'angle_{i+1}_travel_steps'] = dist
    for i, dist in enumerate(physical_distances_um):
        summary_data[f'angle_{i+1}_physical_um'] = dist
    
    df_summary = pd.DataFrame([summary_data])
    summary_csv_path = os.path.join(metadata_folder, f'{image_name}_summary.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    
    # Master summary
    master_summary_path = os.path.join(metadata_folder, 'all_images_summary.csv')
    if os.path.exists(master_summary_path):
        master_df = pd.read_csv(master_summary_path)
        master_df = master_df[master_df['image_name'] != image_name]
        master_df = pd.concat([master_df, df_summary], ignore_index=True)
    else:
        master_df = df_summary
    master_df.to_csv(master_summary_path, index=False)
    
    return df_paths, df_summary


def draw_paths_on_image(image_path, angle_paths, output_folder):

    img = cv2.imread(image_path)
    if img is None:
        print(f"  Warning: Could not load image {image_path}")
        return False
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(img_rgb)
    
    colors = ['#FF0000', '#FF8C00', '#FFD700', '#00CC00', '#0066FF']
    
    plotted_any = False
    
    for i, path_info in enumerate(angle_paths):
        if path_info is None or len(path_info['path']) < 2:
            continue
        
        color = colors[i % len(colors)]
        path = path_info['path']
        outer_cell = path_info['outer_cell']
        final_cell = path_info['final_cell']
        angle = path_info['angle']
        travel_distance = path_info['travel_distance']
        physical_um = path_info.get('physical_distance_um', 0)
        
        display_angle = normalize_angle(angle)
        
        x_coords = [cell['x_pixels'] for cell in path]
        y_coords = [cell['y_pixels'] for cell in path]
        
        ax.plot(x_coords, y_coords, '-', color=color, linewidth=3, alpha=0.9,
                label=f'{display_angle:.1f}° (steps={travel_distance}, {physical_um:.1f}µm)')
        
        ax.plot(outer_cell['x_pixels'], outer_cell['y_pixels'], 'o', 
                color=color, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        ax.plot(final_cell['x_pixels'], final_cell['y_pixels'], 's', 
                color=color, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        
        plotted_any = True
    
    if not plotted_any:
        print(f"  No paths to draw for {Path(image_path).stem}")
        plt.close()
        return False
    
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f'Travel Paths: Cortex to Stele (Greedy Descent)\nImage: {Path(image_path).stem}',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f'{Path(image_path).stem}_travel_paths.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {Path(output_path).name}")
    return True


def process_image_neighbor_paths(df, image_name, measurements_folder, output_folder):

    angle_start, angle_end = determine_angle_range(image_name)
    
    display_start = normalize_angle(angle_start)
    display_end = normalize_angle(angle_end)
    
    print(f"\nProcessing: {image_name} (angle range: {display_start}°-{display_end}°)")
    
    image_df = df[df['image_name'] == image_name].copy()
    
    if image_df.empty:
        print(f"  No cells found for {image_name}")
        return
    
    print(f"  Image has {len(image_df)} cells")
    
    # Check if cell_file column exists
    if 'cell_file' not in image_df.columns:
        print(f"  ERROR: 'cell_file' column not found!")
        print(f"  Available columns: {list(image_df.columns)}")
        return
    
    # Check how many cells have valid cell_file
    valid_cells = image_df[image_df['cell_file'] >= 0]
    print(f"  Cells with valid cell_file: {len(valid_cells)} out of {len(image_df)}")
    
    if len(valid_cells) == 0:
        print(f"  WARNING: No valid cell_file values for {image_name}")
        return
    
    data_min_angle = image_df['angle_degrees'].min()
    data_max_angle = image_df['angle_degrees'].max()
    print(f"  Data angle range: {data_min_angle:.1f}° - {data_max_angle:.1f}°")
    
    if 'neighbors' not in image_df.columns:
        print(f"  ERROR: 'neighbors' column not found!")
        print(f"  Available columns: {list(image_df.columns)}")
        return
    
    # Build neighbor graph
    adjacency, cell_to_df_idx = build_neighbor_graph(image_df)
    print(f"  Built neighbor graph: {len(adjacency)} nodes")
    
    sample_angles = get_sample_angles(angle_start, angle_end)
    display_angles = [normalize_angle(a) for a in sample_angles]
    print(f"  Sampling angles: {[f'{a:.1f}°' for a in display_angles]}")
    
    angle_paths = []
    
    for target_angle in sample_angles:
        # Step 1: Find the outermost cell at this angle
        outer_cell = find_outermost_cell(image_df, target_angle, debug=True)
        
        if outer_cell is None:
            angle_paths.append(None)
            continue
        
        # Step 2: Find the minimum cell file number in this angle sample
        min_cell_file = find_min_cell_file_for_angle(image_df, target_angle, debug=True)
        
        if min_cell_file is None:
            display_angle = normalize_angle(target_angle)
            print(f"    No valid cell_file found in angle {display_angle:.1f}° sample")
            angle_paths.append(None)
            continue
        
        # Step 3: Find the closest cell (by BFS) that has this min cell file number
        path_to_cell_file, target_cell = find_closest_cell_with_min_cell_file(
            image_df, outer_cell, min_cell_file, adjacency, debug=True
        )
        
        if path_to_cell_file is None or target_cell is None:
            display_angle = normalize_angle(target_angle)
            print(f"    No path found to cell_file {min_cell_file} for angle {display_angle:.1f}°")
            angle_paths.append(None)
            continue
        
        # Step 4: From the target cell, trace the path to the stele
        path_to_stele_indices, final_cell = trace_path_to_stele(
            image_df, target_cell, adjacency, debug=True
        )
        
        if path_to_stele_indices is None or final_cell is None:
            display_angle = normalize_angle(target_angle)
            print(f"    No stele path found from cell_file {min_cell_file} for angle {display_angle:.1f}°")
            angle_paths.append(None)
            continue
        
        # Combine the paths
        path_cell_file_cells = []
        for idx in path_to_cell_file:
            cell = image_df.loc[idx]
            path_cell_file_cells.append({
                'cell_id': cell['cell_id'],
                'x_pixels': cell['x_pixels'],
                'y_pixels': cell['y_pixels'],
                'x_um': cell['x_um'],
                'y_um': cell['y_um'],
                'radius_pixels': cell['radius_pixels'],
                'radius_um': cell['radius_um'],
                'cell_file': cell['cell_file']
            })
        
        path_stele_cells = []
        for idx in path_to_stele_indices[1:]:
            cell = image_df.loc[idx]
            path_stele_cells.append({
                'cell_id': cell['cell_id'],
                'x_pixels': cell['x_pixels'],
                'y_pixels': cell['y_pixels'],
                'x_um': cell['x_um'],
                'y_um': cell['y_um'],
                'radius_pixels': cell['radius_pixels'],
                'radius_um': cell['radius_um'],
                'cell_file': cell['cell_file']
            })
        
        full_path_cells = path_cell_file_cells + path_stele_cells
        
        # Calculate distances
        travel_distance = len(full_path_cells) - 1
        physical_um = calculate_physical_distance(full_path_cells, use_um=True)
        display_angle = normalize_angle(target_angle)
        
        print(f"    Angle {display_angle:.1f}°: path length = {len(full_path_cells)} cells, steps = {travel_distance}, distance = {physical_um:.1f}µm")
        print(f"      Outer radius: {outer_cell['radius_um']:.1f}µm, Final radius: {final_cell['radius_um']:.1f}µm")
        print(f"      Min cell_file: {min_cell_file}, Final cell_file: {final_cell['cell_file']}")
        
        angle_paths.append({
            'angle': target_angle,
            'path': full_path_cells,
            'outer_cell': outer_cell,
            'final_cell': final_cell,
            'target_cell_file': min_cell_file,
            'travel_distance': travel_distance,
            'physical_distance_um': physical_um,
            'path_length': len(full_path_cells),
            'cell_file_path_length': len(path_cell_file_cells),
            'stele_path_length': len(path_stele_cells)
        })
    
    # Find the corresponding image file
    image_path = None
    possible_patterns = [
        f"{image_name}_centers.jpg",
        f"{image_name}_centers.png",
        f"{image_name}.jpg",
        f"{image_name}.png",
        f"{image_name}_centers.jpeg",
        f"{image_name}.jpeg",
        f"{image_name}_centers.tif",
        f"{image_name}.tif",
    ]
    
    for pattern in possible_patterns:
        test_path = os.path.join(measurements_folder, pattern)
        if os.path.exists(test_path):
            image_path = test_path
            break
    
    if image_path is None:
        possible_files = glob.glob(os.path.join(measurements_folder, f"{image_name}*"))
        if possible_files:
            for ext in ['.jpg', '.png', '.jpeg', '.tif', '.tiff']:
                for f in possible_files:
                    if f.endswith(ext):
                        image_path = f
                        break
                if image_path is not None:
                    break
    
    # Save metadata before drawing paths
    save_path_metadata(image_name, angle_paths, output_folder)
    
    if image_path is None:
        print(f"  WARNING: No image file found for {image_name}")
        return
    
    print(f"  Found image: {Path(image_path).name}")
    draw_paths_on_image(image_path, angle_paths, output_folder)


def find_closest_cell_with_min_cell_file(image_df, start_cell, target_cell_file, adjacency, debug=False):

    cell_to_df_idx = {row['cell_id']: idx for idx, row in image_df.iterrows()}
    
    start_idx = cell_to_df_idx.get(start_cell['cell_id'])
    if start_idx is None:
        return None, None
    
    visited = {start_idx: None}
    queue = deque([start_idx])
    target_idx = None
    
    while queue:
        current = queue.popleft()
        current_cell = image_df.loc[current]
        
        if current_cell['cell_file'] == target_cell_file:
            if current != start_idx:
                target_idx = current
                break
        
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)
    
    if target_idx is None:
        if debug:
            print(f"    No cell found with cell file {target_cell_file}")
        return None, None
    
    path = []
    current = target_idx
    while current is not None:
        path.append(current)
        current = visited[current]
    path.reverse()
    
    target_cell = image_df.loc[target_idx]
    
    if debug:
        print(f"Found target cell: ID={target_cell['cell_id']}, cell_file={target_cell['cell_file']}, radius={target_cell['radius_pixels']:.1f}px, travel_distance={len(path)-1}")
    
    return path, target_cell


def analyze_all_images_neighbor_paths(
    df, 
    measurements_folder, 
    output_folder, 
    specific_images=None,
    force_rebuild=False
):

    # Determine which images to process
    if specific_images is not None:
        image_names = [img for img in df['image_name'].unique() if img in specific_images]
        print(f"\nProcessing {len(image_names)} specific images (out of {len(df['image_name'].unique())} total)")
    else:
        image_names = list(df['image_name'].unique())
        print(f"\nProcessing all {len(image_names)} images")
    
    if not image_names:
        print("No images to process")
        return 0
    
    # Load existing processed images log
    log_path = os.path.join(output_folder, "processed_paths_log.csv")
    existing_processed = load_existing_processed_images(log_path)
    
    # Filter to only new images (unless force_rebuild)
    if not force_rebuild and existing_processed:
        images_to_process = [img for img in image_names if img not in existing_processed]
        print(f"Skipping {len(existing_processed)} already processed images")
        print(f"New images to process: {len(images_to_process)}")
    else:
        if force_rebuild:
            print("FORCE REBUILD: Processing all images from scratch")
        images_to_process = image_names
    
    if not images_to_process:
        print("No new images to process for neighbor paths")
        return 0
    
    print(f"ANALYZING NEIGHBOR PATHS FOR {len(images_to_process)} IMAGES")
    
    path_output_folder = os.path.join(output_folder, 'travel_paths')
    os.makedirs(path_output_folder, exist_ok=True)
    
    processed_count = 0
    processed_set = set(existing_processed)  # Start with existing
    
    for image_name in tqdm(images_to_process, desc="Processing neighbor paths"):
        process_image_neighbor_paths(df, image_name, measurements_folder, path_output_folder)
        processed_count += 1
        processed_set.add(image_name)
    
    # Save updated processed images log
    save_processed_images_log(log_path, processed_set)
    
    print(f"\nProcessed {processed_count} new images")
    print(f"Total images processed: {len(processed_set)}")
    
    return processed_count


def create_radius_vs_neighbors_plot(df, output_folder):

    df_valid = df[df['degree'] >= 0].copy()
    
    if df_valid.empty:
        print("No valid neighbor data found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    ax1 = axes[0]
    
    df_valid['radius_bin'] = pd.qcut(df_valid['radius_pixels'], q=RADIUS_BINS, duplicates='drop')
    binned = df_valid.groupby('radius_bin', observed=False)['degree'].agg(['mean', 'std', 'count']).reset_index()
    binned['radius_center'] = [interval.mid for interval in binned['radius_bin']]
    binned = binned.sort_values('radius_center')
    binned = binned[binned['count'] >= 3]
    
    if len(binned) > 0:
        ax1.errorbar(binned['radius_center'], binned['mean'], 
                    yerr=binned['std'], fmt='o-', capsize=5, 
                    capthick=2, color='steelblue', markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH, alpha=ALPHA,
                    label='Mean ± Std Dev')
        
        for _, row in binned.iterrows():
            ax1.annotate(f'n={int(row["count"])}', 
                        (row['radius_center'], row['mean']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        if len(binned) > 2:
            z = np.polyfit(binned['radius_center'], binned['mean'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(binned['radius_center'].min(), binned['radius_center'].max(), 100)
            ax1.plot(x_line, p(x_line), 'r--', linewidth=2, 
                    label=f'Trend: slope={z[0]:.3f}')
            
            corr, p_val = stats.pearsonr(binned['radius_center'], binned['mean'])
            ax1.text(0.05, 0.95, f'Correlation: r={corr:.3f} (p={p_val:.4f})', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Radius (pixels)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Number of Neighbors', fontsize=12, fontweight='bold')
    ax1.set_title(f'Average Neighbors vs Radius - Combined\n({len(df_valid)} cells, {RADIUS_BINS} bins)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    
    images = df_valid['image_name'].unique()
    cmap = plt.get_cmap(COLOR_MAP)
    colors = cmap(np.linspace(0, 1, len(images)))
    
    for idx, image_name in enumerate(images):
        image_df = df_valid[df_valid['image_name'] == image_name].copy()
        
        if len(image_df) < 10:
            continue
        
        n_bins = min(8, len(image_df) // 5)
        if n_bins < 2:
            continue
        
        try:
            image_df['radius_bin'] = pd.qcut(image_df['radius_pixels'], q=n_bins, duplicates='drop')
            binned_img = image_df.groupby('radius_bin', observed=False)['degree'].mean().reset_index()
            binned_img['radius_center'] = [interval.mid for interval in binned_img['radius_bin']]
            binned_img = binned_img.sort_values('radius_center')
            
            color = colors[idx % len(colors)]
            short_name = image_name[:20] + '...' if len(image_name) > 20 else image_name
            ax2.plot(binned_img['radius_center'], binned_img['degree'], 'o-', 
                    color=color, alpha=ALPHA, linewidth=LINE_WIDTH, 
                    markersize=MARKER_SIZE-1, label=short_name)
                    
        except Exception as e:
            continue
    
    ax2.set_xlabel('Radius (pixels)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Number of Neighbors', fontsize=12, fontweight='bold')
    ax2.set_title(f'Average Neighbors vs Radius - Individual Images\n({len(images)} images)', 
                 fontsize=14, fontweight='bold')
    
    if len(images) <= LEGEND_THRESHOLD:
        ax2.legend(loc='best', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    
    norm_x = np.linspace(0, 1, NORM_POINTS)
    all_norm_curves = []
    individual_curves_data = []
    
    for idx, image_name in enumerate(images):
        image_df = df_valid[df_valid['image_name'] == image_name].copy()
        
        if len(image_df) < 10:
            continue
        
        n_bins = min(8, len(image_df) // 5)
        if n_bins < 2:
            continue
        
        try:
            image_df['radius_bin'] = pd.qcut(image_df['radius_pixels'], q=n_bins, duplicates='drop')
            binned_img = image_df.groupby('radius_bin', observed=False)['degree'].mean().reset_index()
            binned_img['radius_center'] = [interval.mid for interval in binned_img['radius_bin']]
            binned_img = binned_img.sort_values('radius_center')
            
            min_r = binned_img['radius_center'].min()
            max_r = binned_img['radius_center'].max()
            if max_r > min_r:
                x_norm = (binned_img['radius_center'] - min_r) / (max_r - min_r)
            else:
                x_norm = np.zeros_like(binned_img['radius_center'])
            
            min_y = binned_img['degree'].min()
            max_y = binned_img['degree'].max()
            if max_y > min_y:
                y_norm = (binned_img['degree'] - min_y) / (max_y - min_y)
            else:
                y_norm = np.ones_like(binned_img['degree']) * 0.5
            
            if len(x_norm) > 1:
                interp_func = interp1d(x_norm, y_norm, kind='linear', fill_value='extrapolate')
                y_interp = interp_func(norm_x)
                all_norm_curves.append(y_interp)
                individual_curves_data.append((x_norm, y_norm, image_name, colors[idx % len(colors)]))
                    
        except Exception as e:
            continue
    
    if all_norm_curves:
        all_curves = np.array(all_norm_curves)
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        n_curves = len(all_curves)
        
        ax3.plot(norm_x, mean_curve, 'k-', linewidth=3, 
                label=f'Mean (n={n_curves} images)')
        ax3.fill_between(norm_x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color='gray', label='±1 Std Dev')
        
        slope = np.polyfit(norm_x, mean_curve, 1)[0]
        ax3.text(0.05, 0.05, f'Slope: {slope:.3f}', 
                transform=ax3.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Normalized Radius (0=inner, 1=outer)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Normalized Number of Neighbors (0-1 scale)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Normalized: Average Neighbors vs Radius\n({len(images)} images, each scaled 0-1)', 
                 fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'radius_vs_neighbors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: radius_vs_neighbors.png")
    
    fig_individual, ax_individual = plt.subplots(1, 1, figsize=(10, 8))
    
    n_images_plotted = 0
    for x_norm, y_norm, image_name, color in individual_curves_data:
        short_name = image_name[:15] + '...' if len(image_name) > 15 else image_name
        ax_individual.plot(x_norm, y_norm, 'o-', 
                          color=color, alpha=0.7, linewidth=1.5, 
                          markersize=4, label=short_name)
        n_images_plotted += 1
    
    ax_individual.set_xlabel('Normalized Radius (0=inner, 1=outer)', fontsize=12, fontweight='bold')
    ax_individual.set_ylabel('Normalized Number of Neighbors (0-1 scale)', fontsize=12, fontweight='bold')
    ax_individual.set_title(f'Individual Normalized Patterns - Radius\n({n_images_plotted} images, each scaled 0-1)', 
                           fontsize=14, fontweight='bold')
    
    if n_images_plotted <= LEGEND_THRESHOLD:
        ax_individual.legend(loc='best', fontsize=7, ncol=2)
    ax_individual.grid(True, alpha=0.3)
    ax_individual.set_xlim(-0.05, 1.05)
    ax_individual.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'radius_individual_patterns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: radius_individual_patterns.png (individual normalized curves)")
    
    if all_norm_curves:
        fig_avg, ax_avg = plt.subplots(1, 1, figsize=(10, 8))
        
        all_curves = np.array(all_norm_curves)
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        n_curves = len(all_curves)
        
        ax_avg.plot(norm_x, mean_curve, 'k-', linewidth=3, 
                   label=f'Mean (n={n_curves} images)')
        ax_avg.fill_between(norm_x, mean_curve - std_curve, mean_curve + std_curve,
                           alpha=0.2, color='gray', label='±1 Std Dev')
        
        slope = np.polyfit(norm_x, mean_curve, 1)[0]
        ax_avg.text(0.05, 0.05, f'Slope: {slope:.3f}', 
                   transform=ax_avg.transAxes, fontsize=11, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_avg.text(0.05, 0.95, f'n={n_curves} images', 
                   transform=ax_avg.transAxes, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_avg.set_xlabel('Normalized Radius (0=inner, 1=outer)', fontsize=12, fontweight='bold')
        ax_avg.set_ylabel('Normalized Number of Neighbors (0-1 scale)', fontsize=12, fontweight='bold')
        ax_avg.set_title('Averaged Normalized Pattern - Radius\n(Mean ± Std Dev)', 
                        fontsize=14, fontweight='bold')
        ax_avg.legend(loc='best')
        ax_avg.grid(True, alpha=0.3)
        ax_avg.set_xlim(-0.05, 1.05)
        ax_avg.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'radius_averaged_pattern.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: radius_averaged_pattern.png (averaged normalized curve)")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    
    ax_left = axes2[0]
    if all_norm_curves:
        all_curves = np.array(all_norm_curves)
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        n_curves = len(all_curves)
        
        ax_left.plot(norm_x, mean_curve, 'k-', linewidth=3, 
                    label=f'Mean (n={n_curves} images)')
        ax_left.fill_between(norm_x, mean_curve - std_curve, mean_curve + std_curve,
                            alpha=0.2, color='gray', label='±1 Std Dev')
        
        slope = np.polyfit(norm_x, mean_curve, 1)[0]
        ax_left.text(0.05, 0.05, f'Slope: {slope:.3f}', 
                    transform=ax_left.transAxes, fontsize=11, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_left.text(0.05, 0.95, f'n={n_curves} images', 
                    transform=ax_left.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_left.set_xlabel('Normalized Radius (0=inner, 1=outer)', fontsize=12, fontweight='bold')
    ax_left.set_ylabel('Normalized Number of Neighbors (0-1 scale)', fontsize=12, fontweight='bold')
    ax_left.set_title('Averaged Normalized Pattern\n(Mean ± Std Dev)', fontsize=14, fontweight='bold')
    ax_left.legend(loc='best')
    ax_left.grid(True, alpha=0.3)
    ax_left.set_xlim(-0.05, 1.05)
    ax_left.set_ylim(-0.05, 1.05)
    
    ax_right = axes2[1]
    
    for idx, (x_norm, y_norm, image_name, color) in enumerate(individual_curves_data):
        short_name = image_name[:15] + '...' if len(image_name) > 15 else image_name
        ax_right.plot(x_norm, y_norm, 'o-', 
                    color=color, alpha=0.7, linewidth=1.5, 
                    markersize=4, label=short_name)
    
    ax_right.set_xlabel('Normalized Radius (0=inner, 1=outer)', fontsize=12, fontweight='bold')
    ax_right.set_ylabel('Normalized Number of Neighbors (0-1 scale)', fontsize=12, fontweight='bold')
    ax_right.set_title(f'Individual Normalized Patterns\n({len(individual_curves_data)} images, each scaled 0-1)', 
                      fontsize=14, fontweight='bold')
    
    if len(individual_curves_data) <= LEGEND_THRESHOLD:
        ax_right.legend(loc='best', fontsize=7, ncol=2)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_xlim(-0.05, 1.05)
    ax_right.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'radius_normalized_patterns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: radius_normalized_patterns.png (side-by-side)")


def create_area_vs_neighbors_plot(df, output_folder):

    df_valid = df[df['degree'] >= 0].copy()
    
    if df_valid.empty:
        print("No valid neighbor data found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    ax1 = axes[0]
    
    df_valid['area_bin'] = pd.qcut(df_valid['area_pixels'], q=AREA_BINS, duplicates='drop')
    binned = df_valid.groupby('area_bin', observed=False)['degree'].agg(['mean', 'std', 'count']).reset_index()
    binned['area_center'] = [interval.mid for interval in binned['area_bin']]
    binned = binned.sort_values('area_center')
    binned = binned[binned['count'] >= 3]
    
    if len(binned) > 0:
        ax1.errorbar(binned['area_center'], binned['mean'], 
                    yerr=binned['std'], fmt='o-', capsize=5, 
                    capthick=2, color='steelblue', markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH, alpha=ALPHA,
                    label='Mean ± Std Dev')
        
        for _, row in binned.iterrows():
            ax1.annotate(f'n={int(row["count"])}', 
                        (row['area_center'], row['mean']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        if len(binned) > 2:
            z = np.polyfit(binned['area_center'], binned['mean'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(binned['area_center'].min(), binned['area_center'].max(), 100)
            ax1.plot(x_line, p(x_line), 'r--', linewidth=2, 
                    label=f'Trend: slope={z[0]:.3f}')
            
            corr, p_val = stats.pearsonr(binned['area_center'], binned['mean'])
            ax1.text(0.05, 0.95, f'Correlation: r={corr:.3f} (p={p_val:.4f})', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Cell Area (pixels²)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Number of Neighbors', fontsize=12, fontweight='bold')
    ax1.set_title(f'Average Neighbors vs Area - Combined\n({len(df_valid)} cells, {AREA_BINS} bins)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    
    images = df_valid['image_name'].unique()
    cmap = plt.get_cmap(COLOR_MAP)
    colors = cmap(np.linspace(0, 1, len(images)))
    
    for idx, image_name in enumerate(images):
        image_df = df_valid[df_valid['image_name'] == image_name].copy()
        
        if len(image_df) < 10:
            continue
        
        n_bins = min(8, len(image_df) // 5)
        if n_bins < 2:
            continue
        
        try:
            image_df['area_bin'] = pd.qcut(image_df['area_pixels'], q=n_bins, duplicates='drop')
            binned_img = image_df.groupby('area_bin', observed=False)['degree'].mean().reset_index()
            binned_img['area_center'] = [interval.mid for interval in binned_img['area_bin']]
            binned_img = binned_img.sort_values('area_center')
            
            color = colors[idx % len(colors)]
            short_name = image_name[:20] + '...' if len(image_name) > 20 else image_name
            ax2.plot(binned_img['area_center'], binned_img['degree'], 'o-', 
                    color=color, alpha=ALPHA, linewidth=LINE_WIDTH, 
                    markersize=MARKER_SIZE-1, label=short_name)
                    
        except Exception as e:
            continue
    
    ax2.set_xlabel('Cell Area (pixels²)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Number of Neighbors', fontsize=12, fontweight='bold')
    ax2.set_title(f'Average Neighbors vs Area - Individual Images\n({len(images)} images)', 
                 fontsize=14, fontweight='bold')
    
    if len(images) <= LEGEND_THRESHOLD:
        ax2.legend(loc='best', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    
    norm_x = np.linspace(0, 1, NORM_POINTS)
    all_norm_curves = []
    individual_curves_data = []
    
    for idx, image_name in enumerate(images):
        image_df = df_valid[df_valid['image_name'] == image_name].copy()
        
        if len(image_df) < 10:
            continue
        
        n_bins = min(8, len(image_df) // 5)
        if n_bins < 2:
            continue
        
        try:
            image_df['area_bin'] = pd.qcut(image_df['area_pixels'], q=n_bins, duplicates='drop')
            binned_img = image_df.groupby('area_bin', observed=False)['degree'].mean().reset_index()
            binned_img['area_center'] = [interval.mid for interval in binned_img['area_bin']]
            binned_img = binned_img.sort_values('area_center')
            
            min_a = binned_img['area_center'].min()
            max_a = binned_img['area_center'].max()
            if max_a > min_a:
                x_norm = (binned_img['area_center'] - min_a) / (max_a - min_a)
            else:
                x_norm = np.zeros_like(binned_img['area_center'])
            
            min_y = binned_img['degree'].min()
            max_y = binned_img['degree'].max()
            if max_y > min_y:
                y_norm = (binned_img['degree'] - min_y) / (max_y - min_y)
            else:
                y_norm = np.ones_like(binned_img['degree']) * 0.5
            
            if len(x_norm) > 1:
                interp_func = interp1d(x_norm, y_norm, kind='linear', fill_value='extrapolate')
                y_interp = interp_func(norm_x)
                all_norm_curves.append(y_interp)
                individual_curves_data.append((x_norm, y_norm, image_name, colors[idx % len(colors)]))
                    
        except Exception as e:
            continue
    
    if all_norm_curves:
        all_curves = np.array(all_norm_curves)
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        n_curves = len(all_curves)
        
        ax3.plot(norm_x, mean_curve, 'k-', linewidth=3, 
                label=f'Mean (n={n_curves} images)')
        ax3.fill_between(norm_x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color='gray', label='±1 Std Dev')
        
        slope = np.polyfit(norm_x, mean_curve, 1)[0]
        ax3.text(0.05, 0.05, f'Slope: {slope:.3f}', 
                transform=ax3.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Normalized Cell Area (0=smallest, 1=largest)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Normalized Number of Neighbors (0-1 scale)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Normalized: Average Neighbors vs Area\n({len(images)} images, each scaled 0-1)', 
                 fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'area_vs_neighbors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: area_vs_neighbors.png")
    
    fig_individual, ax_individual = plt.subplots(1, 1, figsize=(10, 8))
    
    n_images_plotted = 0
    for x_norm, y_norm, image_name, color in individual_curves_data:
        short_name = image_name[:15] + '...' if len(image_name) > 15 else image_name
        ax_individual.plot(x_norm, y_norm, 'o-', 
                          color=color, alpha=0.7, linewidth=1.5, 
                          markersize=4, label=short_name)
        n_images_plotted += 1
    
    ax_individual.set_xlabel('Normalized Cell Area (0=smallest, 1=largest)', fontsize=12, fontweight='bold')
    ax_individual.set_ylabel('Normalized Number of Neighbors (0-1 scale)', fontsize=12, fontweight='bold')
    ax_individual.set_title(f'Individual Normalized Patterns - Area\n({n_images_plotted} images, each scaled 0-1)', 
                           fontsize=14, fontweight='bold')
    
    if n_images_plotted <= LEGEND_THRESHOLD:
        ax_individual.legend(loc='best', fontsize=7, ncol=2)
    ax_individual.grid(True, alpha=0.3)
    ax_individual.set_xlim(-0.05, 1.05)
    ax_individual.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'area_individual_patterns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: area_individual_patterns.png (individual normalized curves)")
    
    if all_norm_curves:
        fig_avg, ax_avg = plt.subplots(1, 1, figsize=(10, 8))
        
        all_curves = np.array(all_norm_curves)
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        n_curves = len(all_curves)
        
        ax_avg.plot(norm_x, mean_curve, 'k-', linewidth=3, 
                   label=f'Mean (n={n_curves} images)')
        ax_avg.fill_between(norm_x, mean_curve - std_curve, mean_curve + std_curve,
                           alpha=0.2, color='gray', label='±1 Std Dev')
        
        slope = np.polyfit(norm_x, mean_curve, 1)[0]
        ax_avg.text(0.05, 0.05, f'Slope: {slope:.3f}', 
                   transform=ax_avg.transAxes, fontsize=11, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_avg.text(0.05, 0.95, f'n={n_curves} images', 
                   transform=ax_avg.transAxes, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_avg.set_xlabel('Normalized Cell Area (0=smallest, 1=largest)', fontsize=12, fontweight='bold')
        ax_avg.set_ylabel('Normalized Number of Neighbors (0-1 scale)', fontsize=12, fontweight='bold')
        ax_avg.set_title('Averaged Normalized Pattern - Area\n(Mean ± Std Dev)', 
                        fontsize=14, fontweight='bold')
        ax_avg.legend(loc='best')
        ax_avg.grid(True, alpha=0.3)
        ax_avg.set_xlim(-0.05, 1.05)
        ax_avg.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'area_averaged_pattern.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: area_averaged_pattern.png (averaged normalized curve)")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    
    ax_left = axes2[0]
    if all_norm_curves:
        all_curves = np.array(all_norm_curves)
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        n_curves = len(all_curves)
        
        ax_left.plot(norm_x, mean_curve, 'k-', linewidth=3, 
                    label=f'Mean (n={n_curves} images)')
        ax_left.fill_between(norm_x, mean_curve - std_curve, mean_curve + std_curve,
                            alpha=0.2, color='gray', label='±1 Std Dev')
        
        slope = np.polyfit(norm_x, mean_curve, 1)[0]
        ax_left.text(0.05, 0.05, f'Slope: {slope:.3f}', 
                    transform=ax_left.transAxes, fontsize=11, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_left.text(0.05, 0.95, f'n={n_curves} images', 
                    transform=ax_left.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_left.set_xlabel('Normalized Cell Area (0=smallest, 1=largest)', fontsize=12, fontweight='bold')
    ax_left.set_ylabel('Normalized Number of Neighbors (0-1 scale)', fontsize=12, fontweight='bold')
    ax_left.set_title('Averaged Normalized Pattern\n(Mean ± Std Dev)', fontsize=14, fontweight='bold')
    ax_left.legend(loc='best')
    ax_left.grid(True, alpha=0.3)
    ax_left.set_xlim(-0.05, 1.05)
    ax_left.set_ylim(-0.05, 1.05)
    
    ax_right = axes2[1]
    
    for idx, (x_norm, y_norm, image_name, color) in enumerate(individual_curves_data):
        short_name = image_name[:15] + '...' if len(image_name) > 15 else image_name
        ax_right.plot(x_norm, y_norm, 'o-', 
                    color=color, alpha=0.7, linewidth=1.5, 
                    markersize=4, label=short_name)
    
    ax_right.set_xlabel('Normalized Cell Area (0=smallest, 1=largest)', fontsize=12, fontweight='bold')
    ax_right.set_ylabel('Normalized Number of Neighbors (0-1 scale)', fontsize=12, fontweight='bold')
    ax_right.set_title(f'Individual Normalized Patterns\n({len(individual_curves_data)} images, each scaled 0-1)', 
                      fontsize=14, fontweight='bold')
    
    if len(individual_curves_data) <= LEGEND_THRESHOLD:
        ax_right.legend(loc='best', fontsize=7, ncol=2)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_xlim(-0.05, 1.05)
    ax_right.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'area_normalized_patterns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: area_normalized_patterns.png (side-by-side)")

def create_radius_vs_neighbors_partial_norm_plot(df, output_folder):

    df_valid = df[df['degree'] >= 0].copy()
    
    if df_valid.empty:
        print("No valid neighbor data found")
        return
    
    images = df_valid['image_name'].unique()
    norm_x = np.linspace(0, 1, NORM_POINTS)
    all_curves = []
    individual_curves_data = []
    
    cmap = plt.get_cmap(COLOR_MAP)
    colors = cmap(np.linspace(0, 1, len(images)))
    
    for idx, image_name in enumerate(images):
        image_df = df_valid[df_valid['image_name'] == image_name].copy()
        
        if len(image_df) < 10:
            continue
        
        n_bins = min(8, len(image_df) // 5)
        if n_bins < 2:
            continue
        
        try:
            image_df['radius_bin'] = pd.qcut(image_df['radius_pixels'], q=n_bins, duplicates='drop')
            binned_img = image_df.groupby('radius_bin', observed=False)['degree'].mean().reset_index()
            binned_img['radius_center'] = [interval.mid for interval in binned_img['radius_bin']]
            binned_img = binned_img.sort_values('radius_center')
            
            min_r = binned_img['radius_center'].min()
            max_r = binned_img['radius_center'].max()
            if max_r > min_r:
                x_norm = (binned_img['radius_center'] - min_r) / (max_r - min_r)
            else:
                x_norm = np.zeros_like(binned_img['radius_center'])
            
            y_values = binned_img['degree'].values
            
            if len(x_norm) > 1:
                interp_func = interp1d(x_norm, y_values, kind='linear', fill_value='extrapolate')
                y_interp = interp_func(norm_x)
                all_curves.append(y_interp)
                individual_curves_data.append((x_norm, y_values, image_name, colors[idx % len(colors)]))
                    
        except Exception as e:
            continue
    
    if not all_curves:
        print("No valid curves could be generated for radius partial normalization")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax_left = axes[0]
    all_curves = np.array(all_curves)
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)
    n_curves = len(all_curves)
    
    ax_left.plot(norm_x, mean_curve, 'k-', linewidth=3, 
                label=f'Mean (n={n_curves} images)')
    ax_left.fill_between(norm_x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color='gray', label='±1 Std Dev')
    
    slope = np.polyfit(norm_x, mean_curve, 1)[0]
    ax_left.text(0.05, 0.95, f'Slope: {slope:.3f}', 
                transform=ax_left.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_left.text(0.05, 0.05, f'n={n_curves} images', 
                transform=ax_left.transAxes, fontsize=11, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_left.set_xlabel('Normalized Radius (0=inner, 1=outer)', fontsize=12, fontweight='bold')
    ax_left.set_ylabel('Average Number of Neighbors', fontsize=12, fontweight='bold')
    ax_left.set_title('Radius vs Neighbors (Averaged)\nRadius normalized, neighbors NOT normalized', 
                     fontsize=14, fontweight='bold')
    ax_left.legend(loc='best')
    ax_left.grid(True, alpha=0.3)
    ax_left.set_xlim(-0.05, 1.05)
    
    ax_right = axes[1]
    
    for x_norm, y_values, image_name, color in individual_curves_data:
        short_name = image_name[:15] + '...' if len(image_name) > 15 else image_name
        ax_right.plot(x_norm, y_values, 'o-', 
                    color=color, alpha=0.7, linewidth=1.5, 
                    markersize=4, label=short_name)
    
    ax_right.set_xlabel('Normalized Radius (0=inner, 1=outer)', fontsize=12, fontweight='bold')
    ax_right.set_ylabel('Average Number of Neighbors', fontsize=12, fontweight='bold')
    ax_right.set_title(f'Individual Patterns: Radius Normalized Only\n({len(individual_curves_data)} images)', 
                      fontsize=14, fontweight='bold')
    
    if len(individual_curves_data) <= LEGEND_THRESHOLD:
        ax_right.legend(loc='best', fontsize=7, ncol=2)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, 'radius_partial_norm_patterns.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: radius_partial_norm_patterns.png (radius normalized only)")
    
    return all_curves


def create_area_vs_neighbors_partial_norm_plot(df, output_folder):

    df_valid = df[df['degree'] >= 0].copy()
    
    if df_valid.empty:
        print("No valid neighbor data found")
        return
    
    images = df_valid['image_name'].unique()
    norm_x = np.linspace(0, 1, NORM_POINTS)
    all_curves = []
    individual_curves_data = []
    
    cmap = plt.get_cmap(COLOR_MAP)
    colors = cmap(np.linspace(0, 1, len(images)))
    
    for idx, image_name in enumerate(images):
        image_df = df_valid[df_valid['image_name'] == image_name].copy()
        
        if len(image_df) < 10:
            continue
        
        n_bins = min(8, len(image_df) // 5)
        if n_bins < 2:
            continue
        
        try:
            image_df['area_bin'] = pd.qcut(image_df['area_pixels'], q=n_bins, duplicates='drop')
            binned_img = image_df.groupby('area_bin', observed=False)['degree'].mean().reset_index()
            binned_img['area_center'] = [interval.mid for interval in binned_img['area_bin']]
            binned_img = binned_img.sort_values('area_center')
            
            min_a = binned_img['area_center'].min()
            max_a = binned_img['area_center'].max()
            if max_a > min_a:
                x_norm = (binned_img['area_center'] - min_a) / (max_a - min_a)
            else:
                x_norm = np.zeros_like(binned_img['area_center'])
            
            y_values = binned_img['degree'].values
            
            if len(x_norm) > 1:
                interp_func = interp1d(x_norm, y_values, kind='linear', fill_value='extrapolate')
                y_interp = interp_func(norm_x)
                all_curves.append(y_interp)
                individual_curves_data.append((x_norm, y_values, image_name, colors[idx % len(colors)]))
                    
        except Exception as e:
            continue
    
    if not all_curves:
        print("No valid curves could be generated for area partial normalization")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax_left = axes[0]
    all_curves = np.array(all_curves)
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)
    n_curves = len(all_curves)
    
    ax_left.plot(norm_x, mean_curve, 'k-', linewidth=3, 
                label=f'Mean (n={n_curves} images)')
    ax_left.fill_between(norm_x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color='gray', label='±1 Std Dev')
    
    slope = np.polyfit(norm_x, mean_curve, 1)[0]
    ax_left.text(0.05, 0.95, f'Slope: {slope:.3f}', 
                transform=ax_left.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_left.text(0.05, 0.05, f'n={n_curves} images', 
                transform=ax_left.transAxes, fontsize=11, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_left.set_xlabel('Normalized Cell Area (0=smallest, 1=largest)', fontsize=12, fontweight='bold')
    ax_left.set_ylabel('Average Number of Neighbors', fontsize=12, fontweight='bold')
    ax_left.set_title('Area vs Neighbors (Averaged)\nArea normalized, neighbors NOT normalized', 
                     fontsize=14, fontweight='bold')
    ax_left.legend(loc='best')
    ax_left.grid(True, alpha=0.3)
    ax_left.set_xlim(-0.05, 1.05)
    
    ax_right = axes[1]
    
    for x_norm, y_values, image_name, color in individual_curves_data:
        short_name = image_name[:15] + '...' if len(image_name) > 15 else image_name
        ax_right.plot(x_norm, y_values, 'o-', 
                    color=color, alpha=0.7, linewidth=1.5, 
                    markersize=4, label=short_name)
    
    ax_right.set_xlabel('Normalized Cell Area (0=smallest, 1=largest)', fontsize=12, fontweight='bold')
    ax_right.set_ylabel('Average Number of Neighbors', fontsize=12, fontweight='bold')
    ax_right.set_title(f'Individual Patterns: Area Normalized Only\n({len(individual_curves_data)} images)', 
                      fontsize=14, fontweight='bold')
    
    if len(individual_curves_data) <= LEGEND_THRESHOLD:
        ax_right.legend(loc='best', fontsize=7, ncol=2)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, 'area_partial_norm_patterns.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: area_partial_norm_patterns.png (area normalized only)")
    
    return all_curves


def create_combined_plot(df, output_folder):

    df_valid = df[df['degree'] >= 0].copy()
    
    if df_valid.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    
    df_valid['radius_bin'] = pd.qcut(df_valid['radius_pixels'], q=RADIUS_BINS, duplicates='drop')
    binned_r = df_valid.groupby('radius_bin', observed=False)['degree'].agg(['mean', 'std', 'count']).reset_index()
    binned_r['radius_center'] = [interval.mid for interval in binned_r['radius_bin']]
    binned_r = binned_r.sort_values('radius_center')
    binned_r = binned_r[binned_r['count'] >= 3]
    
    if len(binned_r) > 0:
        ax1.errorbar(binned_r['radius_center'], binned_r['mean'], 
                    yerr=binned_r['std'], fmt='o-', capsize=5, 
                    capthick=2, color='#2E86AB', markersize=8,
                    linewidth=2, label='Mean ± Std Dev')
        
        if len(binned_r) > 2:
            z = np.polyfit(binned_r['radius_center'], binned_r['mean'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(binned_r['radius_center'].min(), binned_r['radius_center'].max(), 100)
            ax1.plot(x_line, p(x_line), 'r--', linewidth=2, 
                    label=f'Slope: {z[0]:.3f}')
            
            corr, p_val = stats.pearsonr(binned_r['radius_center'], binned_r['mean'])
            ax1.text(0.05, 0.95, f'r = {corr:.3f} (p={p_val:.4f})', 
                    transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Radius (pixels)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Neighbors', fontsize=11, fontweight='bold')
    ax1.set_title('Average Neighbors vs Radius', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    
    df_valid['area_bin'] = pd.qcut(df_valid['area_pixels'], q=AREA_BINS, duplicates='drop')
    binned_a = df_valid.groupby('area_bin', observed=False)['degree'].agg(['mean', 'std', 'count']).reset_index()
    binned_a['area_center'] = [interval.mid for interval in binned_a['area_bin']]
    binned_a = binned_a.sort_values('area_center')
    binned_a = binned_a[binned_a['count'] >= 3]
    
    if len(binned_a) > 0:
        ax2.errorbar(binned_a['area_center'], binned_a['mean'], 
                    yerr=binned_a['std'], fmt='o-', capsize=5, 
                    capthick=2, color='#A23B72', markersize=8,
                    linewidth=2, label='Mean ± Std Dev')
        
        if len(binned_a) > 2:
            z = np.polyfit(binned_a['area_center'], binned_a['mean'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(binned_a['area_center'].min(), binned_a['area_center'].max(), 100)
            ax2.plot(x_line, p(x_line), 'r--', linewidth=2, 
                    label=f'Slope: {z[0]:.3f}')
            
            corr, p_val = stats.pearsonr(binned_a['area_center'], binned_a['mean'])
            ax2.text(0.05, 0.95, f'r = {corr:.3f} (p={p_val:.4f})', 
                    transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Cell Area (pixels²)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Neighbors', fontsize=11, fontweight='bold')
    ax2.set_title('Average Neighbors vs Cell Area', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    
    norm_x = np.linspace(0, 1, NORM_POINTS)
    all_radius_curves = []
    
    for image_name in df_valid['image_name'].unique():
        image_df = df_valid[df_valid['image_name'] == image_name].copy()
        if len(image_df) < 10:
            continue
        
        n_bins = min(8, len(image_df) // 5)
        if n_bins < 2:
            continue
        
        try:
            image_df['radius_bin'] = pd.qcut(image_df['radius_pixels'], q=n_bins, duplicates='drop')
            binned_img = image_df.groupby('radius_bin', observed=False)['degree'].mean().reset_index()
            binned_img['radius_center'] = [interval.mid for interval in binned_img['radius_bin']]
            binned_img = binned_img.sort_values('radius_center')
            
            min_r, max_r = binned_img['radius_center'].min(), binned_img['radius_center'].max()
            x_norm = (binned_img['radius_center'] - min_r) / (max_r - min_r) if max_r > min_r else np.zeros_like(binned_img['radius_center'])
            
            min_y, max_y = binned_img['degree'].min(), binned_img['degree'].max()
            y_norm = (binned_img['degree'] - min_y) / (max_y - min_y) if max_y > min_y else np.ones_like(binned_img['degree']) * 0.5
            
            if len(x_norm) > 1:
                interp_func = interp1d(x_norm, y_norm, kind='linear', fill_value='extrapolate')
                all_radius_curves.append(interp_func(norm_x))
        except:
            continue
    
    if all_radius_curves:
        all_curves = np.array(all_radius_curves)
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        
        ax3.plot(norm_x, mean_curve, 'k-', linewidth=3, label=f'Mean (n={len(all_radius_curves)})')
        ax3.fill_between(norm_x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color='gray', label='±1 Std Dev')
        
        slope = np.polyfit(norm_x, mean_curve, 1)[0]
        ax3.text(0.05, 0.05, f'Slope: {slope:.3f}', transform=ax3.transAxes, 
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Normalized Radius (0=inner, 1=outer)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Normalized Neighbors', fontsize=11, fontweight='bold')
    ax3.set_title('Normalized: Radius vs Neighbors', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    
    ax4 = axes[1, 1]
    
    all_area_curves = []
    
    for image_name in df_valid['image_name'].unique():
        image_df = df_valid[df_valid['image_name'] == image_name].copy()
        if len(image_df) < 10:
            continue
        
        n_bins = min(8, len(image_df) // 5)
        if n_bins < 2:
            continue
        
        try:
            image_df['area_bin'] = pd.qcut(image_df['area_pixels'], q=n_bins, duplicates='drop')
            binned_img = image_df.groupby('area_bin', observed=False)['degree'].mean().reset_index()
            binned_img['area_center'] = [interval.mid for interval in binned_img['area_bin']]
            binned_img = binned_img.sort_values('area_center')
            
            min_a, max_a = binned_img['area_center'].min(), binned_img['area_center'].max()
            x_norm = (binned_img['area_center'] - min_a) / (max_a - min_a) if max_a > min_a else np.zeros_like(binned_img['area_center'])
            
            min_y, max_y = binned_img['degree'].min(), binned_img['degree'].max()
            y_norm = (binned_img['degree'] - min_y) / (max_y - min_y) if max_y > min_y else np.ones_like(binned_img['degree']) * 0.5
            
            if len(x_norm) > 1:
                interp_func = interp1d(x_norm, y_norm, kind='linear', fill_value='extrapolate')
                all_area_curves.append(interp_func(norm_x))
        except:
            continue
    
    if all_area_curves:
        all_curves = np.array(all_area_curves)
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        
        ax4.plot(norm_x, mean_curve, 'k-', linewidth=3, label=f'Mean (n={len(all_area_curves)})')
        ax4.fill_between(norm_x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color='gray', label='±1 Std Dev')
        
        slope = np.polyfit(norm_x, mean_curve, 1)[0]
        ax4.text(0.05, 0.05, f'Slope: {slope:.3f}', transform=ax4.transAxes, 
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax4.set_xlabel('Normalized Cell Area (0=smallest, 1=largest)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Normalized Neighbors', fontsize=11, fontweight='bold')
    ax4.set_title('Normalized: Area vs Neighbors', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'neighbor_analysis_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: neighbor_analysis_summary.png")


def main(specific_images=None, force_rebuild=False):
    
    # Create output folder
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load measurement files (only specific images if provided)
    combined_df = load_measurement_files(MEASUREMENTS_FOLDER, specific_images=specific_images)
    
    if combined_df is None or combined_df.empty:
        print("No measurement data found")
        return
    
    # Merge cell_file data from cell assignments
    combined_df = merge_cell_file_data(combined_df, CELL_FILE_COUNTS_FOLDER)
    
    # Filter for valid neighbor data
    df_valid = combined_df[combined_df['degree'] >= 0]
    print(f"Cells with valid neighbor data: {len(df_valid)}")
    
    # Check if neighbors column exists
    if 'neighbors' not in combined_df.columns:
        print("\nERROR: 'neighbors' column not found in the data!")
        print(f"Available columns: {list(combined_df.columns)}")
        return
    
    # Pass specific_images to only process new images
    processed_count = analyze_all_images_neighbor_paths(
        combined_df, 
        MEASUREMENTS_FOLDER, 
        OUTPUT_FOLDER,
        specific_images=specific_images,
        force_rebuild=force_rebuild
    )
    
    if processed_count == 0 and specific_images is not None:
        print("\nNo new images to process for neighbor paths.")
        print("Skipping plot generation (no new data).")
        return
    
    # Create summary plots (using ALL data, not just new)
    create_radius_vs_neighbors_plot(combined_df, OUTPUT_FOLDER)
    create_area_vs_neighbors_plot(combined_df, OUTPUT_FOLDER)
    
    # Create partial normalization plots
    create_radius_vs_neighbors_partial_norm_plot(combined_df, OUTPUT_FOLDER)
    create_area_vs_neighbors_partial_norm_plot(combined_df, OUTPUT_FOLDER)
    
    create_combined_plot(combined_df, OUTPUT_FOLDER)
    
    print(f"Results saved to: {OUTPUT_FOLDER}/")
    print(f"Travel path images saved to: {OUTPUT_FOLDER}/travel_paths/")
    print(f"Travel path metadata saved to: {OUTPUT_FOLDER}/metadata/")
    print(f"Processed paths log: {OUTPUT_FOLDER}/processed_paths_log.csv")
    
    if processed_count > 0:
        print(f"\nNew images processed: {processed_count}")
    else:
        print("\nNo new images were processed (already up to date)")
    

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    specific_images = None
    force_rebuild = False
    
    if len(sys.argv) > 1:
        if '--force-rebuild' in sys.argv or '-f' in sys.argv:
            force_rebuild = True
        
        # Check for specific images argument
        for arg in sys.argv:
            if arg.startswith('--images='):
                specific_images = arg.split('=')[1].split(',')
                break
    
    main(specific_images=specific_images, force_rebuild=force_rebuild)