import os
import pandas as pd
import numpy as np
import json
import glob
import re
from pathlib import Path
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

from code.util.file_utils import parse_image_name

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_BASE = PROJECT_ROOT / "results"

MASTER_SUMMARY_PATH = str(RESULTS_BASE / "master_summary.csv")
MEASUREMENTS_FOLDER = str(RESULTS_BASE / "measurements_all")
OUTPUT_PATH = str(RESULTS_BASE / "feature_table.csv")
OUTPUT_FOLDER = str(RESULTS_BASE / "feature_analysis")
CELL_ASSIGNMENTS_FOLDER = str(RESULTS_BASE / "cell_file" / "cell_file_counting")

SPLINE_SMOOTHING = 0.1
MIN_CELLS_FOR_SPLINE = 5
MIN_QUADRANTS_FOR_ROOT = 2

# Features to analyze (for both radius and file-based approaches)
FEATURES = [
    'peak_height',
    'peak_position',
    'rise_slope',
    'decay_slope',
    'outer_rise_magnitude'
]

FEATURE_LABELS = {
    'peak_height': 'Peak Height',
    'peak_position': 'Peak Position',
    'rise_slope': 'Rise Slope',
    'decay_slope': 'Decay Slope',
    'outer_rise_magnitude': 'Outer-Rise Magnitude'
}

def extract_root_identifier(image_name):
    if not image_name:
        return image_name
    
    # Check if it starts with a quadrant prefix (BL_, BR_, TL_, TR_)
    quadrant_patterns = ['BL_', 'BR_', 'TL_', 'TR_']
    for prefix in quadrant_patterns:
        if image_name.startswith(prefix):
            return image_name[len(prefix):]
    
    # If no quadrant prefix found, return the original name
    return image_name


def extract_plot_number_from_root(root_identifier):
    if not root_identifier:
        return 0
    
    # Try to find plot number at the beginning
    match = re.match(r'^(\d+)', root_identifier)
    if match:
        return int(match.group(1))
    
    return 0


def extract_plant_number_from_root(root_identifier):
    if not root_identifier:
        return 0
    
    # Try to find P{number}
    match = re.search(r'P(\d+)', root_identifier)
    if match:
        return int(match.group(1))
    
    return 0


def load_measurement_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        # Check if it has the expected columns
        if 'area_um2' in df.columns and 'radius_pixels' in df.columns:
            return df
        elif 'area_pixels' in df.columns and 'radius_pixels' in df.columns:
            return df
        else:
            logger.warning(f"Missing required columns in {csv_path}")
            return None
    except Exception as e:
        logger.warning(f"Error loading {csv_path}: {e}")
        return None

def fit_spline_and_extract_features(x, y, smoothing=SPLINE_SMOOTHING):
    if len(x) < MIN_CELLS_FOR_SPLINE:
        return None
    
    try:
        # Fit smoothing spline
        spline = UnivariateSpline(x, y, s=smoothing * len(x), k=3)
        
        # Generate smooth curve
        x_smooth = np.linspace(0, 1, 200)
        y_smooth = spline(x_smooth)
        
        # Find peak
        peak_idx = np.argmax(y_smooth)
        peak_height = y_smooth[peak_idx]
        peak_position = x_smooth[peak_idx]
        
        # Calculate derivative for slopes
        derivative = spline.derivative()
        y_deriv = derivative(x_smooth)
        
        # Rise slope (from start to peak)
        rise_x = x_smooth[:peak_idx + 1]
        rise_y = y_smooth[:peak_idx + 1]
        if len(rise_x) > 2:
            rise_slope = (rise_y[-1] - rise_y[0]) / (rise_x[-1] - rise_x[0] + 0.001)
        else:
            rise_slope = 0
        
        # Decay slope (from peak to end)
        decay_x = x_smooth[peak_idx:]
        decay_y = y_smooth[peak_idx:]
        if len(decay_x) > 2:
            decay_slope = (decay_y[-1] - decay_y[0]) / (decay_x[-1] - decay_x[0] + 0.001)
        else:
            decay_slope = 0
        
        # Outer-rise magnitude (the 'hump' near the outside)
        outer_start_idx = int(0.7 * len(x_smooth))
        outer_end_idx = int(0.95 * len(x_smooth))
        
        if outer_start_idx < 0:
            outer_start_idx = 0
        if outer_end_idx > len(x_smooth):
            outer_end_idx = len(x_smooth)
        if outer_start_idx >= outer_end_idx:
            outer_start_idx = max(0, outer_end_idx - 10)
        
        outer_region_x = x_smooth[outer_start_idx:outer_end_idx]
        outer_region_y = y_smooth[outer_start_idx:outer_end_idx]
        
        if len(outer_region_y) > 5:
            outer_rise_magnitude = np.max(outer_region_y) - y_smooth[-1]
            if outer_rise_magnitude < 0:
                outer_rise_magnitude = 0
        else:
            outer_rise_magnitude = 0
        
        # Area under the curve (AUC)
        area_under_curve = np.trapz(y_smooth, x_smooth)
        
        # Average normalized cell size
        avg_normalized_cell_size = np.mean(y_smooth)
        
        return {
            'peak_height': peak_height,
            'peak_position': peak_position,
            'rise_slope': rise_slope,
            'decay_slope': decay_slope,
            'outer_rise_magnitude': outer_rise_magnitude,
            'area_under_curve': area_under_curve,
            'avg_normalized_cell_size': avg_normalized_cell_size,
            'x_smooth': x_smooth,
            'y_smooth': y_smooth,
            'spline': spline
        }
        
    except Exception as e:
        logger.warning(f"Spline fitting failed: {e}")
        return None

def process_single_image_radius(csv_path, summary_info, debug=False):
    df = load_measurement_csv(csv_path)
    if df is None or df.empty:
        return None
    
    if 'area_um2' in df.columns:
        area_col = 'area_um2'
    else:
        area_col = 'area_pixels'
    
    df = df[df[area_col] > 0]
    df = df[df['radius_pixels'] > 0]
    
    if len(df) < MIN_CELLS_FOR_SPLINE:
        return None
    
    min_radius = df['radius_pixels'].min()
    max_radius = df['radius_pixels'].max()
    if max_radius > min_radius:
        df['radius_normalized'] = (df['radius_pixels'] - min_radius) / (max_radius - min_radius)
    else:
        return None
    
    min_area = df[area_col].min()
    max_area = df[area_col].max()
    if max_area > min_area:
        df['area_normalized'] = (df[area_col] - min_area) / (max_area - min_area)
    else:
        return None
    
    n_bins = min(20, len(df) // 3)
    if n_bins < 3:
        return None
    
    df['radius_bin'] = pd.cut(df['radius_normalized'], bins=n_bins, labels=False)
    binned = df.groupby('radius_bin').agg({
        'radius_normalized': 'mean',
        'area_normalized': 'mean'
    }).reset_index()
    binned = binned.sort_values('radius_normalized')
    
    x = binned['radius_normalized'].values
    y = binned['area_normalized'].values
    
    features = fit_spline_and_extract_features(x, y)
    
    if features is None:
        return None
    
    result = {
        'image_name': summary_info.get('image_name', ''),
        'quadrant': summary_info.get('quadrant', 'unknown'),
        'root_identifier': summary_info.get('root_identifier'),
        'plant_number': summary_info.get('plant_number'),
        'root_number': summary_info.get('root_number'),
        'biological_replicate': summary_info.get('biological_replicate'),
        'technical_replicate': summary_info.get('technical_replicate'),
        'treatment': summary_info.get('treatment', 'unknown'),
        'root_type': summary_info.get('root_type', 'unknown'),
        'species': summary_info.get('species', 'Zea mays'),
        'population': summary_info.get('population', 'IBM'),
        'n_files': summary_info.get('file_count', 0),
        'stele_area_um2': summary_info.get('stele_area_um2', 0),
        'root_radius_um': summary_info.get('root_radius_um', 0),
        'avg_cell_area_um2': summary_info.get('average_cell_area_um2', 0),
        'n_cells': summary_info.get('n_cells', 0),
        # Radius-based features
        'radius_peak_height': features.get('peak_height'),
        'radius_peak_position': features.get('peak_position'),
        'radius_rise_slope': features.get('rise_slope'),
        'radius_decay_slope': features.get('decay_slope'),
        'radius_outer_rise_magnitude': features.get('outer_rise_magnitude'),
        'radius_area_under_curve': features.get('area_under_curve'),
        'radius_avg_normalized_cell_size': features.get('avg_normalized_cell_size'),
        # Store spline data for plotting
        'radius_x_smooth': features.get('x_smooth'),
        'radius_y_smooth': features.get('y_smooth')
    }
    
    return result


def process_single_image_file_based(image_name, cell_assignments_folder, summary_info, debug=False):
    # Check root_identifier
    if summary_info.get('root_identifier') is None:
        if debug:
            print(f"Skipping {image_name}: root_identifier is None")
        return None
    
    # Try different possible paths
    possible_paths = [
        os.path.join(cell_assignments_folder, image_name, "cell_assignments.csv"),
        os.path.join(cell_assignments_folder, image_name, f"{image_name}_method_derivative.csv"),
        os.path.join(cell_assignments_folder, image_name, "cell_assignments_all_methods.csv"),
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                break
            except Exception as e:
                continue
    
    if df is None:
        return None
    
    # Check for required columns
    if 'cell_file_derivative' not in df.columns:
        return None
    
    if 'area_um2' not in df.columns:
        if 'area_pixels' in df.columns:
            area_col = 'area_pixels'
        else:
            return None
    else:
        area_col = 'area_um2'
    
    # Filter valid cells
    df = df[df[area_col] > 0]
    df = df[df['cell_file_derivative'] >= 0]
    
    if len(df) < MIN_CELLS_FOR_SPLINE:
        return None
    
    # Group by cell file and calculate average area
    file_groups = df.groupby('cell_file_derivative').agg({
        area_col: ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    file_groups.columns = ['cell_file_derivative', 'avg_area_um2', 'std_area_um2', 'n_cells']
    file_groups = file_groups.sort_values('cell_file_derivative')
    
    # Get values for spline fitting
    file_numbers = file_groups['cell_file_derivative'].values
    areas = file_groups['avg_area_um2'].values
    
    n_files = len(file_numbers)
    if n_files < 3:
        return None
    
    # Normalize file numbers to 0-1
    min_file = file_numbers.min()
    max_file = file_numbers.max()
    if max_file > min_file:
        file_norm = (file_numbers - min_file) / (max_file - min_file)
    else:
        return None
    
    # Normalize areas to 0-1
    min_area = areas.min()
    max_area = areas.max()
    if max_area > min_area:
        area_norm = (areas - min_area) / (max_area - min_area)
    else:
        return None
    
    # Fit spline and extract features
    features = fit_spline_and_extract_features(file_norm, area_norm)
    
    if features is None:
        return None
    
    # Also get the actual file data for the result
    result = {
        'image_name': image_name,
        'quadrant': summary_info.get('quadrant', 'unknown'),
        'root_identifier': summary_info.get('root_identifier'),
        'plant_number': summary_info.get('plant_number'),
        'root_number': summary_info.get('root_number'),
        'biological_replicate': summary_info.get('biological_replicate'),
        'technical_replicate': summary_info.get('technical_replicate'),
        'treatment': summary_info.get('treatment', 'unknown'),
        'root_type': summary_info.get('root_type', 'unknown'),
        'species': summary_info.get('species', 'Zea mays'),
        'population': summary_info.get('population', 'IBM'),
        'n_files': n_files,
        'stele_area_um2': summary_info.get('stele_area_um2', 0),
        'root_radius_um': summary_info.get('root_radius_um', 0),
        'avg_cell_area_um2': summary_info.get('average_cell_area_um2', 0),
        'n_cells': summary_info.get('n_cells', 0),
        # File-based features
        'file_peak_height': features.get('peak_height'),
        'file_peak_position': features.get('peak_position'),
        'file_rise_slope': features.get('rise_slope'),
        'file_decay_slope': features.get('decay_slope'),
        'file_outer_rise_magnitude': features.get('outer_rise_magnitude'),
        'file_area_under_curve': features.get('area_under_curve'),
        'file_avg_normalized_cell_size': features.get('avg_normalized_cell_size'),
        # Store raw data for reference
        'file_numbers': list(file_numbers),
        'file_areas': list(areas),
        'file_std_areas': list(file_groups['std_area_um2'].values),
        'file_cell_counts': list(file_groups['n_cells'].values)
    }
    
    return result

def combine_quarters_to_roots(image_features):
    roots = {}
    
    for features in image_features:
        if features is None:
            continue
        
        root_id = features.get('root_identifier')
        if root_id is None:
            continue
        
        if root_id not in roots:
            roots[root_id] = {
                'root_identifier': root_id,
                'plant_number': features.get('plant_number'),
                'root_number': features.get('root_number'),
                'biological_replicate': features.get('biological_replicate'),
                'technical_replicate': features.get('technical_replicate'),
                'treatment': features.get('treatment', 'unknown'),
                'root_type': features.get('root_type', 'unknown'),
                'species': features.get('species', 'Zea mays'),
                'population': features.get('population', 'IBM'),
                'genotype': 'unknown',
                # Initialize as lists to collect all values
                'n_files_values': [],
                'stele_area_um2_values': [],
                'root_radius_um_values': [],
                'avg_cell_area_um2_values': [],
                'n_cells_values': [],
                # Radius-based features - initialize as empty lists
                'radius_peak_height_values': [],
                'radius_peak_position_values': [],
                'radius_rise_slope_values': [],
                'radius_decay_slope_values': [],
                'radius_outer_rise_magnitude_values': [],
                'radius_area_under_curve_values': [],
                'radius_avg_normalized_size_values': [],
                # File-based features - initialize as empty lists
                'file_peak_height_values': [],
                'file_peak_position_values': [],
                'file_rise_slope_values': [],
                'file_decay_slope_values': [],
                'file_outer_rise_magnitude_values': [],
                'file_area_under_curve_values': [],
                'file_avg_normalized_size_values': [],
                'quadrants': set()
            }
        
        # Collect numeric values from each quadrant
        numeric_fields = {
            'n_files': 'n_files_values',
            'stele_area_um2': 'stele_area_um2_values',
            'root_radius_um': 'root_radius_um_values',
            'avg_cell_area_um2': 'avg_cell_area_um2_values',
            'n_cells': 'n_cells_values'
        }
        
        for field, list_key in numeric_fields.items():
            if field in features and features[field] is not None and features[field] > 0:
                roots[root_id][list_key].append(features[field])
        
        # Add radius features
        radius_feature_keys = {
            'radius_peak_height': 'radius_peak_height_values',
            'radius_peak_position': 'radius_peak_position_values',
            'radius_rise_slope': 'radius_rise_slope_values',
            'radius_decay_slope': 'radius_decay_slope_values',
            'radius_outer_rise_magnitude': 'radius_outer_rise_magnitude_values',
            'radius_area_under_curve': 'radius_area_under_curve_values',
            'radius_avg_normalized_cell_size': 'radius_avg_normalized_size_values'
        }
        
        for feature_key, list_key in radius_feature_keys.items():
            if feature_key in features and features[feature_key] is not None:
                roots[root_id][list_key].append(features[feature_key])
        
        # Add file-based features
        file_feature_keys = {
            'file_peak_height': 'file_peak_height_values',
            'file_peak_position': 'file_peak_position_values',
            'file_rise_slope': 'file_rise_slope_values',
            'file_decay_slope': 'file_decay_slope_values',
            'file_outer_rise_magnitude': 'file_outer_rise_magnitude_values',
            'file_area_under_curve': 'file_area_under_curve_values',
            'file_avg_normalized_cell_size': 'file_avg_normalized_size_values'
        }
        
        for feature_key, list_key in file_feature_keys.items():
            if feature_key in features and features[feature_key] is not None:
                roots[root_id][list_key].append(features[feature_key])
        
        roots[root_id]['quadrants'].add(features.get('quadrant'))
    
    # Average features across quarters
    for root_id, root_data in roots.items():
        # Average numeric fields
        numeric_fields = {
            'n_files_values': 'n_files',
            'stele_area_um2_values': 'stele_area_um2',
            'root_radius_um_values': 'root_radius_um',
            'avg_cell_area_um2_values': 'avg_cell_area_um2',
            'n_cells_values': 'n_cells'
        }
        
        for values_key, result_key in numeric_fields.items():
            values = [v for v in root_data[values_key] if v is not None and not np.isnan(v) and v > 0]
            if values:
                root_data[result_key] = np.mean(values)
            else:
                root_data[result_key] = 0
        
        # Radius features
        radius_keys = {
            'radius_peak_height_values': 'radius_peak_height',
            'radius_peak_position_values': 'radius_peak_position',
            'radius_rise_slope_values': 'radius_rise_slope',
            'radius_decay_slope_values': 'radius_decay_slope',
            'radius_outer_rise_magnitude_values': 'radius_outer_rise_magnitude',
            'radius_area_under_curve_values': 'radius_area_under_curve',
            'radius_avg_normalized_size_values': 'radius_avg_normalized_cell_size'
        }
        
        for values_key, result_key in radius_keys.items():
            values = [v for v in root_data[values_key] if v is not None and not np.isnan(v)]
            if values:
                root_data[result_key] = np.mean(values)
            else:
                root_data[result_key] = np.nan
        
        # File-based features
        file_keys = {
            'file_peak_height_values': 'file_peak_height',
            'file_peak_position_values': 'file_peak_position',
            'file_rise_slope_values': 'file_rise_slope',
            'file_decay_slope_values': 'file_decay_slope',
            'file_outer_rise_magnitude_values': 'file_outer_rise_magnitude',
            'file_area_under_curve_values': 'file_area_under_curve',
            'file_avg_normalized_size_values': 'file_avg_normalized_cell_size'
        }
        
        for values_key, result_key in file_keys.items():
            values = [v for v in root_data[values_key] if v is not None and not np.isnan(v)]
            if values:
                root_data[result_key] = np.mean(values)
            else:
                root_data[result_key] = np.nan
        
        root_data['quadrants'] = ';'.join(sorted(root_data['quadrants']))
    
    return roots

def analyze_feature_distributions(feature_df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Separate radius and file features
    radius_features = [f for f in FEATURES if f'radius_{f}' in feature_df.columns]
    file_features = [f for f in FEATURES if f'file_{f}' in feature_df.columns]
    
    # Calculate statistics for both
    all_stats = []
    
    for feature_type, prefix, label_prefix in [('radius', 'radius_', 'Radius'), 
                                                ('file', 'file_', 'File #')]:
        features = radius_features if feature_type == 'radius' else file_features
        
        for feature in features:
            col = f'{prefix}{feature}'
            if col not in feature_df.columns:
                continue
                
            values = feature_df[col].dropna()
            
            if len(values) == 0:
                continue
            
            stats_data = {
                'approach': label_prefix,
                'feature': feature,
                'label': f"{label_prefix} {FEATURE_LABELS.get(feature, feature)}",
                'n': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'variance': np.var(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
                'iqr': np.percentile(values, 75) - np.percentile(values, 25),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.nan
            }
            all_stats.append(stats_data)
    
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(os.path.join(output_folder, 'feature_distributions_both.csv'), index=False)
    print(f"\nSaved: feature_distributions_both.csv")
    
    # Print comparison table
    print(f"{'Feature':<25} {'Radius Mean':>12} {'File # Mean':>12} {'Radius CV':>10} {'File # CV':>10}")

    for feature in FEATURES:
        radius_row = stats_df[stats_df['approach'] == 'Radius']
        file_row = stats_df[stats_df['approach'] == 'File #']
        
        radius_mean = radius_row[radius_row['feature'] == feature]['mean'].values
        file_mean = file_row[file_row['feature'] == feature]['mean'].values
        radius_cv = radius_row[radius_row['feature'] == feature]['cv'].values
        file_cv = file_row[file_row['feature'] == feature]['cv'].values
        
        if len(radius_mean) > 0 and len(file_mean) > 0:
            print(f"{FEATURE_LABELS.get(feature, feature):<25} {radius_mean[0]:>12.3f} {file_mean[0]:>12.3f} {radius_cv[0]:>10.3f} {file_cv[0]:>10.3f}")
    
    return stats_df


def get_existing_root_identifiers(existing_df):

    if existing_df is None:
        return set()
    
    if 'root_identifier' in existing_df.columns:
        return set(existing_df['root_identifier'].unique())
    elif 'root_id' in existing_df.columns:
        return set(existing_df['root_id'].unique())
    return set()


def build_feature_table_incremental(
    master_summary_path, 
    measurements_folder, 
    output_path, 
    force_rebuild=False, 
    debug=False
):

    print(f"Feature table path: {output_path}")
    
    # Load existing feature table if it exists and we're not forcing a rebuild
    existing_df = None
    existing_roots = set()
    
    if os.path.exists(output_path) and not force_rebuild:
        try:
            existing_df = pd.read_csv(output_path)
            existing_roots = get_existing_root_identifiers(existing_df)
            print(f"Loaded existing feature table with {len(existing_df)} rows, {len(existing_roots)} unique roots")
        except Exception as e:
            print(f"Warning: Could not load existing feature table: {e}")
            existing_df = None
            existing_roots = set()
    else:
        if force_rebuild:
            print("Force rebuild: Starting from scratch")
        else:
            print("No existing feature table found, building from scratch")
    
    # Load master summary
    if not os.path.exists(master_summary_path):
        print(f"Error: {master_summary_path} not found")
        return None
    
    master_df = pd.read_csv(master_summary_path)
    print(f"Loaded {len(master_df)} entries from master summary")
    
    # Filter to images with cells
    master_df = master_df[master_df['n_cells'] > 0]
    print(f"Filtered to {len(master_df)} entries with cells")
    
    if len(master_df) == 0:
        print("No images with cells!")
        return None
    
    # Determine which images need processing
    images_to_process = []
    
    if not force_rebuild and existing_df is not None:
        # Only process images that haven't been processed yet
        # We identify images by their root_identifier
        processed_roots = set(existing_roots)
        
        # Also get the quadrant-level image names that are already in the feature table
        # Since we store at root level, we need to check which quadrants belong to existing roots
        processed_quadrants = set()
        
        # For each row in master_df, check if its root is already processed
        for idx, row in master_df.iterrows():
            image_name = row['image_name']
            root_id = extract_root_identifier(image_name)
            
            if root_id in processed_roots:
                processed_quadrants.add(image_name)
        
        # Now filter master_df to only unprocessed images
        unprocessed_mask = ~master_df['image_name'].isin(processed_quadrants)
        master_to_process = master_df[unprocessed_mask]
        
        print(f"Found {len(master_to_process)} new images to process ({len(master_df) - len(master_to_process)} already processed)")
        
        if len(master_to_process) == 0:
            print("No new images to process. Feature table is up to date.")
            return existing_df
        
        images_to_process = list(master_to_process.iterrows())
    else:
        # Process all images
        print(f"Processing all {len(master_df)} images (force rebuild or no existing table)")
        images_to_process = list(master_df.iterrows())
    
    # If nothing to process, return existing
    if not images_to_process:
        return existing_df
    
    # Use the global CELL_ASSIGNMENTS_FOLDER
    cell_assignments_folder = CELL_ASSIGNMENTS_FOLDER
    print(f"Cell assignments folder: {cell_assignments_folder}")
    
    # Check if cell assignments folder exists
    if os.path.exists(cell_assignments_folder):
        subdirs = [d for d in os.listdir(cell_assignments_folder) if os.path.isdir(os.path.join(cell_assignments_folder, d))]
        print(f"Found {len(subdirs)} image folders in cell assignments")
    else:
        print(f"Cell assignments folder NOT found: {cell_assignments_folder}")
    
    image_features_radius = []
    image_features_file = []
    failed_images = []
    
    for idx, row in tqdm(images_to_process, desc="Processing new images", unit="image"):
        image_name = row['image_name']
        
        # Remove quadrant prefix (BL_, BR_, TL_, TR_) to get root identifier
        root_identifier = extract_root_identifier(image_name)
        
        # Also try to parse with the existing function for additional fields
        parsed = parse_image_name(image_name)
        
        quadrant = row.get('quadrant', 'unknown')
        if quadrant == 'unknown' or quadrant is None:
            quadrant = parsed.get('quadrant', 'unknown')
        
        # Get plant number from parsed data or try to extract from root_identifier
        plant_number = parsed.get('plant_number')
        if plant_number is None:
            plant_number = extract_plant_number_from_root(root_identifier)
        
        # Get root number from parsed data
        root_number = parsed.get('root_number')
        if root_number is None:
            match = re.search(r'root(\d+)', root_identifier)
            if match:
                root_number = int(match.group(1))
        
        # Get technical replicate from parsed data
        tech_rep = parsed.get('technical_replicate')
        
        summary_info = {
            'image_name': image_name,
            'quadrant': quadrant,
            'root_identifier': root_identifier,
            'file_count': row.get('file_count', 0),
            'stele_area_um2': row.get('stele_area_um2', 0),
            'root_radius_um': row.get('root_radius_um', 0),
            'average_cell_area_um2': row.get('average_cell_area_um2', 0),
            'n_cells': row.get('n_cells', 0),
            'plant_number': plant_number,
            'root_number': root_number,
            'biological_replicate': parsed.get('biological_replicate'),
            'technical_replicate': tech_rep,
            'treatment': parsed.get('treatment', 'unknown'),
            'root_type': parsed.get('root_type', 'unknown'),
            'species': row.get('species', 'Zea mays'),
            'population': row.get('population', 'IBM'),
        }
        
        # RADIUS APPROACH - From measurement CSV
        csv_path = os.path.join(measurements_folder, f"{image_name}_measurements.csv")
        if not os.path.exists(csv_path):
            alt_path = os.path.join(measurements_folder, f"{image_name}_centers_measurements.csv")
            if os.path.exists(alt_path):
                csv_path = alt_path
            else:
                failed_images.append((image_name, "CSV not found"))
                # Still try file-based approach
                features_file = process_single_image_file_based(
                    image_name, cell_assignments_folder, summary_info, debug=False
                )
                if features_file is not None:
                    image_features_file.append(features_file)
                continue
        
        # Process radius approach
        features_radius = process_single_image_radius(csv_path, summary_info, debug=(debug and idx < 5))
        if features_radius is not None:
            image_features_radius.append(features_radius)
        elif debug and idx < 10:
            print(f"[DEBUG] Radius approach failed for: {image_name}")
        
        # FILE NUMBER APPROACH - From cell assignment files
        features_file = process_single_image_file_based(
            image_name, cell_assignments_folder, summary_info, debug=False
        )
        if features_file is not None:
            image_features_file.append(features_file)
        
        # Track failures
        if features_radius is None and features_file is None:
            failed_images.append((image_name, "Both approaches failed"))
    
    print(f"\nRadius approach: {len(image_features_radius)} new images processed")
    print(f"File # approach: {len(image_features_file)} new images processed")
    print(f"Failed: {len(failed_images)} images")
    
    # If no features extracted, return existing
    if not image_features_radius and not image_features_file:
        print("No features extracted from new images!")
        if existing_df is not None:
            return existing_df
        return None
    
    # Combine both sets of features - start with radius features
    all_features = []
    file_dict = {f['image_name']: f for f in image_features_file}
    
    # If radius approach has features, combine with file approach
    if image_features_radius:
        for radius_feat in image_features_radius:
            image_name = radius_feat['image_name']
            combined = radius_feat.copy()
            
            if image_name in file_dict:
                file_feat = file_dict[image_name]
                # Add file-based features
                for key in ['file_peak_height', 'file_peak_position', 'file_rise_slope', 
                            'file_decay_slope', 'file_outer_rise_magnitude', 
                            'file_area_under_curve', 'file_avg_normalized_cell_size']:
                    if key in file_feat:
                        combined[key] = file_feat[key]
            
            all_features.append(combined)
    else:
        # If only file approach has features, use those
        all_features = list(file_dict.values())
    
    print(f"Combined new features: {len(all_features)} images")
    
    if not all_features:
        print("No features extracted!")
        if existing_df is not None:
            return existing_df
        return None
    
    # Combine quarters to roots for new images only
    new_roots = combine_quarters_to_roots(all_features)
    print(f"Combined into {len(new_roots)} new roots")
    
    # Create new rows for the feature table
    new_rows = []
    for root_id, root_data in new_roots.items():
        # Skip if this root already exists in the feature table
        if root_id in existing_roots:
            print(f"  Skipping {root_id} (already in feature table)")
            continue
            
        quadrants = root_data.get('quadrants', '')
        n_quadrants = len(quadrants.split(';')) if quadrants else 0
        
        plant_number = root_data.get('plant_number')
        if plant_number is None:
            plant_number = extract_plant_number_from_root(root_id)
        
        root_number = root_data.get('root_number')
        if root_number is None:
            match = re.search(r'root(\d+)', root_id)
            if match:
                root_number = int(match.group(1))
        
        new_rows.append({
            'root_id': root_id,
            'root_identifier': root_id,
            'plant_number': plant_number,
            'root_number': root_number,
            'biological_replicate': root_data.get('biological_replicate'),
            'technical_replicate': root_data.get('technical_replicate'),
            'n_quadrants': n_quadrants,
            'quadrants': quadrants,
            'species': root_data.get('species', 'Zea mays'),
            'population': root_data.get('population', 'IBM'),
            'genotype': root_data.get('genotype', 'unknown'),
            'root_type': root_data.get('root_type', 'unknown'),
            'treatment': root_data.get('treatment', 'unknown'),
            'n_files': root_data.get('n_files', 0),
            'stele_area_um2': root_data.get('stele_area_um2', 0),
            'root_radius_um': root_data.get('root_radius_um', 0),
            'avg_cell_area_um2': root_data.get('avg_cell_area_um2', 0),
            'n_cells': root_data.get('n_cells', 0),
            # Radius-based features
            'radius_peak_height': root_data.get('radius_peak_height', np.nan),
            'radius_peak_position': root_data.get('radius_peak_position', np.nan),
            'radius_rise_slope': root_data.get('radius_rise_slope', np.nan),
            'radius_decay_slope': root_data.get('radius_decay_slope', np.nan),
            'radius_outer_rise_magnitude': root_data.get('radius_outer_rise_magnitude', np.nan),
            'radius_area_under_curve': root_data.get('radius_area_under_curve', np.nan),
            'radius_avg_normalized_cell_size': root_data.get('radius_avg_normalized_cell_size', np.nan),
            # File-based features
            'file_peak_height': root_data.get('file_peak_height', np.nan),
            'file_peak_position': root_data.get('file_peak_position', np.nan),
            'file_rise_slope': root_data.get('file_rise_slope', np.nan),
            'file_decay_slope': root_data.get('file_decay_slope', np.nan),
            'file_outer_rise_magnitude': root_data.get('file_outer_rise_magnitude', np.nan),
            'file_area_under_curve': root_data.get('file_area_under_curve', np.nan),
            'file_avg_normalized_cell_size': root_data.get('file_avg_normalized_cell_size', np.nan)
        })
    
    if not new_rows:
        print("No new roots to add (all were already in feature table)")
        return existing_df
    
    # Create DataFrame for new rows
    new_df = pd.DataFrame(new_rows)
    print(f"Created {len(new_df)} new rows for feature table")
    
    # Calculate stele_diameter_um for new rows
    if 'stele_area_um2' in new_df.columns:
        valid_mask = new_df['stele_area_um2'] > 0
        if valid_mask.any():
            new_df.loc[valid_mask, 'stele_diameter_um'] = 2 * np.sqrt(new_df.loc[valid_mask, 'stele_area_um2'] / np.pi)
            new_df.loc[~valid_mask, 'stele_diameter_um'] = np.nan
            print(f"Calculated stele_diameter_um for {valid_mask.sum()} new roots")
        else:
            new_df['stele_diameter_um'] = np.nan
    else:
        new_df['stele_diameter_um'] = np.nan
    
    # Combine with existing data
    if existing_df is not None and len(existing_df) > 0:
        # Combine existing and new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"Combined: {len(existing_df)} existing + {len(new_df)} new = {len(combined_df)} total")
        
        # Sort
        if 'plant_number' in combined_df.columns:
            combined_df['plant_number'] = combined_df['plant_number'].fillna(0)
            combined_df = combined_df.sort_values(['plant_number', 'root_identifier']).reset_index(drop=True)
        else:
            combined_df = combined_df.sort_values(['root_identifier']).reset_index(drop=True)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        print(f"\nFeature table saved to: {output_path}")
        print(f"Total roots: {len(combined_df)}")
        
        # Run analysis on the complete dataset (this should use ALL data)
        output_folder = os.path.join(os.path.dirname(output_path), "feature_analysis")
        stats_df = analyze_feature_distributions(combined_df, output_folder)
        
        return combined_df
    else:
        # Just save the new data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_df.to_csv(output_path, index=False)
        print(f"\nFeature table saved to: {output_path}")
        print(f"Total roots: {len(new_df)}")
        
        return new_df


def build_feature_table(master_summary_path, measurements_folder, output_path, debug=False):

    return build_feature_table_incremental(
        master_summary_path=master_summary_path,
        measurements_folder=measurements_folder,
        output_path=output_path,
        force_rebuild=True,
        debug=debug
    )


def main():
    feature_df = build_feature_table(
        master_summary_path=MASTER_SUMMARY_PATH,
        measurements_folder=MEASUREMENTS_FOLDER,
        output_path=OUTPUT_PATH,
        debug=True
    )


if __name__ == "__main__":
    main()