# Fits the spline to each root and extracts the six shape features.
#
# This is where peak height, peak position, rise slope, decay slope, minima
# position and outer-rise magnitude are defined. Everything downstream that
# talks about the pattern is reading numbers produced here. Also writes the
# per-root spline plots and the pooled cubic fit.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import re
from scipy.optimize import curve_fit
from scipy import stats
from scipy.interpolate import UnivariateSpline
import warnings
from tqdm import tqdm

# Import utility functions
from code.util import (
    parse_image_name,
    load_measurement_csv,
    normalize_radius_and_area,
    bin_by_radius
)
from code.measurements.build_feature_table import extract_root_identifier
from code.config import CELL_FILE_COUNTS_FOLDER, MASTER_SUMMARY_PATH, MEASUREMENTS_FOLDER, RESULTS_FOLDER

warnings.filterwarnings('ignore')


MASTER_SUMMARY_PATH = MASTER_SUMMARY_PATH
MEASUREMENTS_FOLDER = MEASUREMENTS_FOLDER
CELL_ASSIGNMENTS_FOLDER = CELL_FILE_COUNTS_FOLDER

# These are the actual paths where plots are saved
OUTPUT_FOLDER = RESULTS_FOLDER / 'combined_cubic_fit'
INDIVIDUAL_PLOTS_RADIUS = os.path.join(OUTPUT_FOLDER, "individual_plots_radius")
INDIVIDUAL_PLOTS_FILE = os.path.join(OUTPUT_FOLDER, "individual_plots_file")

N_BINS = 20
MIN_CELLS_FOR_SPLINE = 5
SPLINE_SMOOTHING = 0.1

# File to track which images have been processed for individual plots
PROCESSED_IMAGES_LOG = os.path.join(OUTPUT_FOLDER, "processed_images_for_plots.csv")
FEATURE_SUMMARY_PATH = os.path.join(OUTPUT_FOLDER, "individual_features_summary.csv")


def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


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
        
        # Minima position - the post-peak local minimum where the outer-rise begins
        decay_x = x_smooth[peak_idx:]
        decay_y = y_smooth[peak_idx:]
        if len(decay_y) > 1:
            minima_idx = np.argmin(decay_y)
            minima_position = decay_x[minima_idx]
            minima_height = decay_y[minima_idx]
        else:
            minima_idx = 0
            minima_position = peak_position
            minima_height = peak_height

        # Decay slope (from peak to minima - excludes the outer-rise portion)
        if len(decay_y) > 1 and minima_idx > 0:
            decay_slope = (minima_height - peak_height) / (minima_position - peak_position + 0.001)
        else:
            decay_slope = 0

        # Outer-rise magnitude - the increase in the curve after the minima
        post_minima_y = decay_y[minima_idx:]
        if len(post_minima_y) > 1:
            outer_rise_magnitude = np.max(post_minima_y) - minima_height
            if outer_rise_magnitude < 0:
                outer_rise_magnitude = 0
        else:
            outer_rise_magnitude = 0
        
        # Area under the curve
        area_under_curve = np.trapz(y_smooth, x_smooth)
        avg_normalized_cell_size = np.mean(y_smooth)
        
        return {
            'peak_height': peak_height,
            'peak_position': peak_position,
            'rise_slope': rise_slope,
            'decay_slope': decay_slope,
            'minima_position': minima_position,
            'minima_height': minima_height,
            'outer_rise_magnitude': outer_rise_magnitude,
            'area_under_curve': area_under_curve,
            'avg_normalized_cell_size': avg_normalized_cell_size,
            'x_smooth': x_smooth,
            'y_smooth': y_smooth,
            'spline': spline,
            'peak_idx': peak_idx
        }
        
    except Exception as e:
        return None


def load_measurement_csv(csv_path):

    try:
        df = pd.read_csv(csv_path)
        if 'area_um2' in df.columns and 'radius_pixels' in df.columns:
            return df
        elif 'area_pixels' in df.columns and 'radius_pixels' in df.columns:
            return df
        else:
            return None
    except Exception as e:
        return None


def find_existing_plots(radius_folder, file_folder):

    existing_images = set()
    
    # Check radius plots
    if os.path.exists(radius_folder):
        radius_plots = glob.glob(os.path.join(radius_folder, "*_spline.png"))
        radius_names = {os.path.basename(f).replace('_spline.png', '') for f in radius_plots}
    else:
        radius_names = set()
        os.makedirs(radius_folder, exist_ok=True)
    
    # Check file plots
    if os.path.exists(file_folder):
        file_plots = glob.glob(os.path.join(file_folder, "*_file_spline.png"))
        file_names = {os.path.basename(f).replace('_file_spline.png', '') for f in file_plots}
    else:
        file_names = set()
        os.makedirs(file_folder, exist_ok=True)
    
    # An image is fully processed if it has BOTH radius and file plots
    existing_images = radius_names & file_names
    
    print(f"Found {len(radius_names)} existing radius plots")
    print(f"Found {len(file_names)} existing file plots")
    print(f"Found {len(existing_images)} images with BOTH plot types")
    
    return existing_images


def save_processed_images_log(log_path, processed_images):

    # Ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    df = pd.DataFrame({'image_name': list(processed_images)})
    df.to_csv(log_path, index=False)
    print(f"Saved processed images log: {log_path} ({len(processed_images)} images)")


def load_existing_feature_summary(summary_path):

    if os.path.exists(summary_path):
        try:
            return pd.read_csv(summary_path)
        except Exception as e:
            print(f"Warning: Could not load existing feature summary: {e}")
            return None
    return None


def save_feature_summary(summary_path, feature_summary_data, append=True):

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    if not feature_summary_data:
        return
    
    new_df = pd.DataFrame(feature_summary_data)
    
    if append and os.path.exists(summary_path):
        try:
            existing_df = pd.read_csv(summary_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(summary_path, index=False)
            print(f"Appended {len(new_df)} new features to existing summary ({len(combined_df)} total)")
            return
        except Exception as e:
            print(f"Warning: Could not append to existing summary: {e}")
    
    new_df.to_csv(summary_path, index=False)
    print(f"Saved feature summary: {summary_path} ({len(new_df)} rows)")


def create_individual_spline_plots_from_master(
    master_summary_path, 
    measurements_folder, 
    output_radius_folder, 
    output_file_folder,
    force_rebuild=False,
    debug=False
):
    
    # Create output folders
    os.makedirs(output_radius_folder, exist_ok=True)
    os.makedirs(output_file_folder, exist_ok=True)
    
    # Load master summary
    master_df = pd.read_csv(master_summary_path)
    master_df = master_df[master_df['n_cells'] > 0]
    master_df = master_df[master_df['file_count'] > 0]
    
    # Filter to images that have per_file_avg_areas_um2 data
    master_df_with_files = master_df[master_df['per_file_avg_areas_um2'] != '']
    
    print(f"Total images in master summary: {len(master_df)}")
    print(f"Images with per-file area data: {len(master_df_with_files)}")
    
    # Find existing plots
    existing_processed = find_existing_plots(output_radius_folder, output_file_folder)
    
    if not force_rebuild and existing_processed:
        print(f"Found {len(existing_processed)} images with existing plots")
        
        # Filter to only new images
        master_df_to_process = master_df_with_files[~master_df_with_files['image_name'].isin(existing_processed)]
        
        print(f"New images to process: {len(master_df_to_process)}")
        
        if len(master_df_to_process) == 0:
            print("No new images to process. All images already have individual plots.")
            return 0, 0, []
    else:
        if force_rebuild:
            print("FORCE REBUILD: Processing all images from scratch")
            # Remove existing plots if force_rebuild
            import shutil
            if os.path.exists(output_radius_folder):
                shutil.rmtree(output_radius_folder)
                os.makedirs(output_radius_folder, exist_ok=True)
            if os.path.exists(output_file_folder):
                shutil.rmtree(output_file_folder)
                os.makedirs(output_file_folder, exist_ok=True)
        else:
            print("No existing plots found. Processing all images.")
        master_df_to_process = master_df_with_files.copy()
    
    if len(master_df_to_process) == 0:
        print("No new images to process.")
        return 0, 0, []
    
    print(f"Processing {len(master_df_to_process)} new images...")
    
    # Load existing feature summary to append to
    existing_features = load_existing_feature_summary(FEATURE_SUMMARY_PATH)
    
    radius_count = 0
    file_count = 0
    feature_summary_data = []
    processed_image_names = set(existing_processed)  # Start with existing
    
    for idx, row in tqdm(master_df_to_process.iterrows(), total=len(master_df_to_process), desc="Creating individual plots"):
        image_name = row['image_name']
        
        parsed = parse_image_name(image_name)
        treatment = parsed.get('treatment', 'unknown')
        root_type = parsed.get('root_type', 'unknown')
        n_files = row.get('file_count', 0)
        
        # Track that we're processing this image
        processed_image_names.add(image_name)
        
        csv_path = os.path.join(measurements_folder, f"{image_name}_measurements.csv")
        if not os.path.exists(csv_path):
            alt_path = os.path.join(measurements_folder, f"{image_name}_centers_measurements.csv")
            if os.path.exists(alt_path):
                csv_path = alt_path
            else:
                continue
        
        df = load_measurement_csv(csv_path)
        if df is None or df.empty:
            continue
        
        if 'area_um2' in df.columns:
            area_col = 'area_um2'
        else:
            area_col = 'area_pixels'
        
        df = df[df[area_col] > 0]
        df = df[df['radius_pixels'] > 0]
        
        if len(df) < MIN_CELLS_FOR_SPLINE:
            continue
        
        # Normalize radius
        min_radius = df['radius_pixels'].min()
        max_radius = df['radius_pixels'].max()
        if max_radius <= min_radius:
            continue
        df['radius_normalized'] = (df['radius_pixels'] - min_radius) / (max_radius - min_radius)
        
        # Normalize area
        min_area = df[area_col].min()
        max_area = df[area_col].max()
        if max_area <= min_area:
            continue
        df['area_normalized'] = (df[area_col] - min_area) / (max_area - min_area)
        
        # Bin data
        n_bins = min(N_BINS, len(df) // 3)
        if n_bins < 3:
            continue
        
        df['radius_bin'] = pd.cut(df['radius_normalized'], bins=n_bins, labels=False)
        binned = df.groupby('radius_bin').agg({
            'radius_normalized': 'mean',
            'area_normalized': 'mean'
        }).reset_index()
        binned = binned.sort_values('radius_normalized')
        
        x = binned['radius_normalized'].values
        y = binned['area_normalized'].values
        
        # Fit spline and extract features
        features = fit_spline_and_extract_features(x, y)
        
        if features is not None:
            # Create clean radius plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            x_smooth = features['x_smooth']
            y_smooth = features['y_smooth']
            ax.plot(x_smooth, y_smooth, 'b-', linewidth=3, label='Spline fit')
            
            peak_idx = features['peak_idx']
            peak_x = x_smooth[peak_idx]
            peak_y = y_smooth[peak_idx]
            ax.scatter(peak_x, peak_y, s=150, color='red', marker='*',
                      zorder=10, label=f"Peak: ({peak_x:.3f}, {peak_y:.3f})")

            minima_x = features['minima_position']
            minima_y = features['minima_height']
            ax.scatter(minima_x, minima_y, s=150, color='green', marker='v',
                      zorder=10, label=f"Minima: ({minima_x:.3f}, {minima_y:.3f})")

            feature_text = (
                f"Features:\n"
                f"  Peak Height: {features['peak_height']:.3f}\n"
                f"  Peak Position: {features['peak_position']:.3f}\n"
                f"  Rise Slope: {features['rise_slope']:.3f}\n"
                f"  Decay Slope: {features['decay_slope']:.3f}\n"
                f"  Minima Position: {features['minima_position']:.3f}\n"
                f"  Outer Rise: {features['outer_rise_magnitude']:.3f}\n"
                f"  AUC: {features['area_under_curve']:.3f}"
            )
            
            ax.text(0.02, 0.98, feature_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
            
            ax.set_xlabel('Normalized Radius (0=center, 1=edge)', fontsize=12)
            ax.set_ylabel('Normalized Cell Area', fontsize=12)
            ax.set_title(f'{image_name} (Radius-based)\nTreatment: {treatment}, Root Type: {root_type}, Files: {n_files}', 
                        fontsize=11)
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_radius_folder, f'{image_name}_spline.png'), dpi=150)
            plt.close()
            radius_count += 1
            
            # Store feature summary for this image
            feature_summary_data.append({
                'image_name': image_name,
                'approach': 'radius',
                'treatment': treatment,
                'root_type': root_type,
                'n_files': n_files,
                'peak_height': features['peak_height'],
                'peak_position': features['peak_position'],
                'rise_slope': features['rise_slope'],
                'decay_slope': features['decay_slope'],
                'minima_position': features['minima_position'],
                'outer_rise_magnitude': features['outer_rise_magnitude'],
                'area_under_curve': features['area_under_curve'],
                'avg_normalized_cell_size': features['avg_normalized_cell_size']
            })

        per_file_areas_str = row.get('per_file_avg_areas_um2', '')
        
        if per_file_areas_str:
            try:
                per_file_areas = [float(x) for x in per_file_areas_str.split(',') if x]
                
                if len(per_file_areas) >= MIN_CELLS_FOR_SPLINE:
                    file_numbers = np.arange(len(per_file_areas))
                    areas = np.array(per_file_areas)
                    
                    min_file = file_numbers.min()
                    max_file = file_numbers.max()
                    if max_file > min_file:
                        file_norm = (file_numbers - min_file) / (max_file - min_file)
                    else:
                        continue
                    
                    min_area_file = areas.min()
                    max_area_file = areas.max()
                    if max_area_file > min_area_file:
                        area_norm_file = (areas - min_area_file) / (max_area_file - min_area_file)
                    else:
                        continue
                    
                    features_file = fit_spline_and_extract_features(file_norm, area_norm_file)
                    
                    if features_file is not None:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        x_smooth_file = features_file['x_smooth']
                        y_smooth_file = features_file['y_smooth']
                        ax.plot(x_smooth_file, y_smooth_file, 'b-', linewidth=3, label='Spline fit')
                        
                        peak_idx_file = features_file['peak_idx']
                        peak_x_file = x_smooth_file[peak_idx_file]
                        peak_y_file = y_smooth_file[peak_idx_file]
                        ax.scatter(peak_x_file, peak_y_file, s=150, color='red', marker='*',
                                  zorder=10, label=f"Peak: ({peak_x_file:.3f}, {peak_y_file:.3f})")

                        minima_x_file = features_file['minima_position']
                        minima_y_file = features_file['minima_height']
                        ax.scatter(minima_x_file, minima_y_file, s=150, color='green', marker='v',
                                  zorder=10, label=f"Minima: ({minima_x_file:.3f}, {minima_y_file:.3f})")

                        feature_text_file = (
                            f"Features:\n"
                            f"  Peak Height: {features_file['peak_height']:.3f}\n"
                            f"  Peak Position: {features_file['peak_position']:.3f}\n"
                            f"  Rise Slope: {features_file['rise_slope']:.3f}\n"
                            f"  Decay Slope: {features_file['decay_slope']:.3f}\n"
                            f"  Minima Position: {features_file['minima_position']:.3f}\n"
                            f"  Outer Rise: {features_file['outer_rise_magnitude']:.3f}\n"
                            f"  AUC: {features_file['area_under_curve']:.3f}"
                        )
                        
                        ax.text(0.02, 0.98, feature_text_file, transform=ax.transAxes, 
                               fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
                        
                        treatment = parsed.get('treatment', 'unknown')
                        root_type = parsed.get('root_type', 'unknown')
                        n_files_file = len(per_file_areas)
                        
                        ax.set_xlabel('Normalized File Number (0=first file)', fontsize=12)
                        ax.set_ylabel('Normalized Cell Area', fontsize=12)
                        ax.set_title(f'{image_name} (File-based)\nTreatment: {treatment}, Root Type: {root_type}, Files: {n_files_file}', 
                                    fontsize=11)
                        ax.legend(loc='lower right', fontsize=10)
                        ax.grid(True, alpha=0.3)
                        ax.set_xlim(-0.05, 1.05)
                        ax.set_ylim(-0.05, 1.05)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_file_folder, f'{image_name}_file_spline.png'), dpi=150)
                        plt.close()
                        file_count += 1
                        
                        feature_summary_data.append({
                            'image_name': image_name,
                            'approach': 'file',
                            'treatment': treatment,
                            'root_type': root_type,
                            'n_files': n_files_file,
                            'peak_height': features_file['peak_height'],
                            'peak_position': features_file['peak_position'],
                            'rise_slope': features_file['rise_slope'],
                            'decay_slope': features_file['decay_slope'],
                            'minima_position': features_file['minima_position'],
                            'outer_rise_magnitude': features_file['outer_rise_magnitude'],
                            'area_under_curve': features_file['area_under_curve'],
                            'avg_normalized_cell_size': features_file['avg_normalized_cell_size']
                        })
                        
            except Exception as e:
                if debug:
                    print(f"  Error processing file data for {image_name}: {e}")
    
    # Save processed images log
    save_processed_images_log(PROCESSED_IMAGES_LOG, processed_image_names)
    
    # Save feature summary (append to existing)
    if feature_summary_data:
        save_feature_summary(FEATURE_SUMMARY_PATH, feature_summary_data, append=True)
    
    print(f"\nCreated {radius_count} NEW radius-based spline plots in: {output_radius_folder}")
    print(f"Created {file_count} NEW file-based spline plots in: {output_file_folder}")
    print(f"Total images processed so far: {len(processed_image_names)}")
    
    return radius_count, file_count, feature_summary_data


def process_all_images(master_summary_path, measurements_folder):
    
    master_df = pd.read_csv(master_summary_path)
    master_df = master_df[master_df['n_cells'] > 0]
    master_df = master_df[master_df['file_count'] > 0]
    print(f"Processing {len(master_df)} images for combined cubic fit...")
    
    all_binned_data = []
    root_info = {}
    
    for idx, row in tqdm(master_df.iterrows(), total=len(master_df), desc="Processing all images for cubic fit"):
        image_name = row['image_name']
        
        parsed = parse_image_name(image_name)
        root_id = extract_root_identifier(image_name)
        
        if root_id is None:
            continue
        
        csv_path = os.path.join(measurements_folder, f"{image_name}_measurements.csv")
        if not os.path.exists(csv_path):
            alt_path = os.path.join(measurements_folder, f"{image_name}_centers_measurements.csv")
            if os.path.exists(alt_path):
                csv_path = alt_path
            else:
                continue
        
        df = load_measurement_csv(csv_path)
        if df is None or df.empty:
            continue
        
        if 'area_um2' in df.columns:
            area_col = 'area_um2'
        else:
            area_col = 'area_pixels'
        
        df = df[df[area_col] > 0]
        df = df[df['radius_pixels'] > 0]
        
        if len(df) < MIN_CELLS_FOR_SPLINE:
            continue
        
        # Normalize radius to 0-1 for this root
        min_radius = df['radius_pixels'].min()
        max_radius = df['radius_pixels'].max()
        if max_radius <= min_radius:
            continue
        df['radius_normalized'] = (df['radius_pixels'] - min_radius) / (max_radius - min_radius)
        
        # Normalize cell area to 0-1 for this root
        min_area = df[area_col].min()
        max_area = df[area_col].max()
        if max_area <= min_area:
            continue
        df['area_normalized'] = (df[area_col] - min_area) / (max_area - min_area)
        
        # Bin by normalized radius
        n_bins = min(N_BINS, len(df) // 3)
        if n_bins < 3:
            continue
        
        df['radius_bin'] = pd.cut(df['radius_normalized'], bins=n_bins, labels=False)
        binned = df.groupby('radius_bin').agg({
            'radius_normalized': 'mean',
            'area_normalized': 'mean'
        }).reset_index()
        binned = binned.sort_values('radius_normalized')
        
        # Add root identifier
        binned['root_id'] = root_id
        binned['root_type'] = parsed.get('root_type', 'unknown')
        binned['treatment'] = parsed.get('treatment', 'unknown')
        binned['species'] = row.get('species', 'unknown')
        binned['population'] = row.get('population', 'unknown')
        binned['n_files'] = row.get('file_count', 0)

        all_binned_data.append(binned)

        # Store root info
        if root_id not in root_info:
            root_info[root_id] = {
                'root_type': parsed.get('root_type', 'unknown'),
                'treatment': parsed.get('treatment', 'unknown'),
                'species': row.get('species', 'unknown'),
                'population': row.get('population', 'unknown'),
                'n_files': row.get('file_count', 0),
                'plant_number': parsed.get('plant_number')
            }
    
    if not all_binned_data:
        print("No data processed!")
        return None, None
    
    combined_df = pd.concat(all_binned_data, ignore_index=True)
    print(f"Combined data: {len(combined_df)} rows from {len(all_binned_data)} images")
    print(f"Unique roots: {combined_df['root_id'].nunique()}")
    
    return combined_df, root_info

def fit_combined_cubic(combined_df):
    
    x_data = combined_df['radius_normalized'].values
    y_data = combined_df['area_normalized'].values
    
    print(f"Data points: {len(x_data)}")
    
    try:
        p0 = [0.1, -0.1, 1.0, 0.5]
        popt, pcov = curve_fit(cubic, x_data, y_data, p0=p0, maxfev=10000)
        
        y_pred = cubic(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        n = len(x_data)
        p = len(popt)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        perr = np.sqrt(np.diag(pcov))
        t_stats = popt / perr
        dof = n - p
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
        
        alpha = 0.05
        t_val = stats.t.ppf(1 - alpha/2, dof)
        ci_lower = popt - t_val * perr
        ci_upper = popt + t_val * perr
        
        ss_reg = ss_tot - ss_res
        f_stat = (ss_reg / p) / (ss_res / (n - p - 1))
        f_p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
        residual_std = np.sqrt(ss_res / (n - p))
        
        x_smooth = np.linspace(0, 1, 200)
        y_smooth = cubic(x_smooth, *popt)
        
        ci = t_val * residual_std * np.sqrt(1/n + (x_smooth - np.mean(x_data))**2 / np.sum((x_data - np.mean(x_data))**2))
        
        results = {
            'params': popt,
            'param_names': ['a', 'b', 'c', 'd'],
            'param_errors': perr,
            't_stats': t_stats,
            'p_values': p_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'r2': r2,
            'r2_adj': r2_adj,
            'f_stat': f_stat,
            'f_p_value': f_p_value,
            'residual_std': residual_std,
            'n': n,
            'dof': dof,
            'x_smooth': x_smooth,
            'y_smooth': y_smooth,
            'ci': ci,
            'x_data': x_data,
            'y_data': y_data,
            'y_pred': y_pred
        }
        
        print(f"\nCubic fit results:")
        print(f"  R² = {r2:.4f}")
        print(f"  Adjusted R² = {r2_adj:.4f}")
        print(f"  F-statistic = {f_stat:.2f}")
        print(f"  F-test p-value = {f_p_value:.6f}")
        print(f"  Residual std: {residual_std:.4f}")
        print(f"  Degrees of freedom: {dof}")
        print(f"\nParameters:")
        param_names = ['a', 'b', 'c', 'd']
        for i, name in enumerate(param_names):
            sig = '***' if p_values[i] < 0.001 else ('**' if p_values[i] < 0.01 else ('*' if p_values[i] < 0.05 else 'ns'))
            print(f"  {name} = {popt[i]:.6f} ± {perr[i]:.6f}  (t={t_stats[i]:.3f}, p={p_values[i]:.6f}) {sig}")
        
        print(f"\nEquation:")
        print(f"  y = {popt[0]:.6f}x³ + {popt[1]:.6f}x² + {popt[2]:.6f}x + {popt[3]:.6f}")
        
        return results
        
    except Exception as e:
        print(f"Error fitting cubic: {e}")
        return None


def fit_cubic_by_group(combined_df, group_col):
    # Fit a cubic curve separately for each value of group_col (e.g. root_type,
    # species, population). Peak and minima are located on the fitted curve the
    # same way fit_spline_and_extract_features does (minima = the lowest point
    # after the peak), so cubic and spline comparisons report the same kind of
    # landmarks. Returns {group_value: {params, r2, r2_adj, n, peak_position,
    # peak_height, minima_position, minima_height}}.

    groups = combined_df[group_col].unique()
    results_by_group = {}
    x_smooth = np.linspace(0, 1, 200)

    for group_value in groups:
        if group_value == 'unknown' or pd.isna(group_value):
            continue

        group_df = combined_df[combined_df[group_col] == group_value]
        x_data = group_df['radius_normalized'].values
        y_data = group_df['area_normalized'].values

        try:
            p0 = [0.1, -0.1, 1.0, 0.5]
            popt, pcov = curve_fit(cubic, x_data, y_data, p0=p0, maxfev=10000)

            y_pred = cubic(x_data, *popt)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            n = len(x_data)
            p = len(popt)
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

            y_smooth = cubic(x_smooth, *popt)
            peak_idx = np.argmax(y_smooth)
            peak_position = x_smooth[peak_idx]
            peak_height = y_smooth[peak_idx]

            decay_x = x_smooth[peak_idx:]
            decay_y = y_smooth[peak_idx:]
            if len(decay_y) > 1:
                minima_idx = np.argmin(decay_y)
                minima_position = decay_x[minima_idx]
                minima_height = decay_y[minima_idx]
            else:
                minima_position = peak_position
                minima_height = peak_height

            results_by_group[group_value] = {
                'params': popt,
                'r2': r2,
                'r2_adj': r2_adj,
                'n': n,
                'peak_position': peak_position,
                'peak_height': peak_height,
                'minima_position': minima_position,
                'minima_height': minima_height,
            }

            print(f"\n{group_value}:")
            print(f"  R² = {r2:.4f}")
            print(f"  Adjusted R² = {r2_adj:.4f}")
            print(f"  peak={peak_position:.4f} at height={peak_height:.4f}")
            print(f"  minima={minima_position:.4f} at height={minima_height:.4f}")
            print(f"  n = {n}")

        except Exception as e:
            print(f"Error fitting {group_value}: {e}")

    return results_by_group


def fit_spline_by_group(combined_df, group_col):
    # Fit a smoothing spline (same fit used per-individual-root) to the pooled
    # points of each value of group_col. Returns {group_value: features_dict},
    # where features_dict is whatever fit_spline_and_extract_features returns
    # (peak_height, peak_position, rise_slope, decay_slope, minima_position,
    # outer_rise_magnitude, area_under_curve, x_smooth, y_smooth, ...).

    groups = combined_df[group_col].unique()
    results_by_group = {}

    for group_value in groups:
        if group_value == 'unknown' or pd.isna(group_value):
            continue

        group_df = combined_df[combined_df[group_col] == group_value]
        x_data = group_df['radius_normalized'].values
        y_data = group_df['area_normalized'].values

        # Re-bin the pooled points so the spline sees a smooth mean curve
        # rather than every individual root's raw binned points stacked on
        # top of each other.
        n_bins = min(N_BINS, len(x_data) // 3)
        if n_bins < 3:
            print(f"Skipping spline for {group_value}: not enough pooled points")
            continue

        pooled = pd.DataFrame({'x': x_data, 'y': y_data})
        pooled['bin'] = pd.cut(pooled['x'], bins=n_bins, labels=False)
        binned = pooled.groupby('bin').agg({'x': 'mean', 'y': 'mean'}).reset_index()
        binned = binned.sort_values('x')

        features = fit_spline_and_extract_features(binned['x'].values, binned['y'].values)
        if features is None:
            print(f"Spline fit failed for {group_value}")
            continue

        features['n'] = len(x_data)
        results_by_group[group_value] = features

        print(f"\n{group_value} (spline):")
        print(f"  peak_height={features['peak_height']:.4f}  peak_position={features['peak_position']:.4f}")
        print(f"  rise_slope={features['rise_slope']:.4f}  decay_slope={features['decay_slope']:.4f}")
        print(f"  minima_position={features['minima_position']:.4f}  outer_rise_magnitude={features['outer_rise_magnitude']:.4f}")
        print(f"  n={features['n']}")

    return results_by_group


def plot_cubic_comparison(results_by_group, title, save_path, group_order=None):
    # Overlay one cubic fit curve per group (species, population, root_type,
    # ...) on a single axes, colored distinctly, with each curve's peak marked
    # and its R²/n/peak position in the legend.

    if not results_by_group:
        print(f"  Skipped (no groups): {save_path}")
        return

    groups = group_order if group_order is not None else sorted(results_by_group.keys())
    groups = [g for g in groups if g in results_by_group]
    if len(groups) < 2:
        print(f"  Skipped (need >=2 groups to compare, got {len(groups)}): {save_path}")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    group_to_color = {g: colors[i] for i, g in enumerate(groups)}
    x_smooth = np.linspace(0, 1, 200)

    for group_value in groups:
        result = results_by_group[group_value]
        params = result['params']
        y_smooth = cubic(x_smooth, *params)
        color = group_to_color[group_value]

        ax.plot(x_smooth, y_smooth, '-', linewidth=2.5, color=color,
                label=f"{group_value} (R²={result['r2']:.3f}, n={result['n']}, "
                      f"peak={result['peak_position']:.2f}, minima={result['minima_position']:.2f})")
        ax.scatter(result['peak_position'], result['peak_height'], s=120, color=color, marker='*', zorder=10)
        ax.scatter(result['minima_position'], result['minima_height'], s=100, color=color, marker='v', zorder=10)

    ax.set_xlabel('Normalized Radius (0=center, 1=edge)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Cell Size', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_spline_comparison(results_by_group, title, save_path, group_order=None):
    # Overlay one spline fit curve per group on a single axes, with each
    # curve's peak (star) and minima (triangle) marked in matching color, and
    # the six shape features in the legend.

    if not results_by_group:
        print(f"  Skipped (no groups): {save_path}")
        return

    groups = group_order if group_order is not None else sorted(results_by_group.keys())
    groups = [g for g in groups if g in results_by_group]
    if len(groups) < 2:
        print(f"  Skipped (need >=2 groups to compare, got {len(groups)}): {save_path}")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    group_to_color = {g: colors[i] for i, g in enumerate(groups)}

    for group_value in groups:
        f = results_by_group[group_value]
        color = group_to_color[group_value]

        ax.plot(f['x_smooth'], f['y_smooth'], '-', linewidth=2.5, color=color,
                label=f"{group_value} (peak={f['peak_position']:.2f}, "
                      f"minima={f['minima_position']:.2f}, n={f['n']})")
        ax.scatter(f['peak_position'], f['peak_height'], s=140, color=color, marker='*', zorder=10)
        ax.scatter(f['minima_position'], f['minima_height'], s=110, color=color, marker='v', zorder=10)

    ax.set_xlabel('Normalized Radius (0=center, 1=edge)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Cell Size', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def save_group_cubic_results(results_by_group, group_col_name, save_path):
    if not results_by_group:
        return
    rows = []
    for group_value, result in results_by_group.items():
        rows.append({
            group_col_name: group_value,
            'a': result['params'][0],
            'b': result['params'][1],
            'c': result['params'][2],
            'd': result['params'][3],
            'r2': result['r2'],
            'r2_adj': result['r2_adj'],
            'peak_position': result['peak_position'],
            'peak_height': result['peak_height'],
            'minima_position': result['minima_position'],
            'minima_height': result['minima_height'],
            'n': result['n']
        })
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")


def save_group_spline_results(results_by_group, group_col_name, save_path):
    if not results_by_group:
        return
    rows = []
    for group_value, f in results_by_group.items():
        rows.append({
            group_col_name: group_value,
            'peak_height': f['peak_height'],
            'peak_position': f['peak_position'],
            'rise_slope': f['rise_slope'],
            'decay_slope': f['decay_slope'],
            'minima_position': f['minima_position'],
            'minima_height': f['minima_height'],
            'outer_rise_magnitude': f['outer_rise_magnitude'],
            'area_under_curve': f['area_under_curve'],
            'avg_normalized_cell_size': f['avg_normalized_cell_size'],
            'n': f['n']
        })
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")


def create_visualizations(combined_results, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # PLOT 1: Combined data with cubic fit
    fig, ax = plt.subplots(figsize=(12, 8))
    x_data = combined_results['x_data']
    y_data = combined_results['y_data']
    
    if len(x_data) > 10000:
        sample_idx = np.random.choice(len(x_data), 10000, replace=False)
        ax.scatter(x_data[sample_idx], y_data[sample_idx], alpha=0.1, s=10, color='gray', label='Data (sampled)')
    else:
        ax.scatter(x_data, y_data, alpha=0.2, s=5, color='gray', label='Data')
    
    x_smooth = combined_results['x_smooth']
    y_smooth = combined_results['y_smooth']
    ci = combined_results['ci']
    
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=3, label='Cubic fit')
    ax.fill_between(x_smooth, y_smooth - ci, y_smooth + ci, alpha=0.2, color='red', label='95% Confidence Band')
    
    peak_idx = np.argmax(y_smooth)
    peak_x = x_smooth[peak_idx]
    peak_y = y_smooth[peak_idx]
    ax.scatter(peak_x, peak_y, s=150, color='green', marker='*', zorder=10, label=f"Peak: ({peak_x:.3f}, {peak_y:.3f})")
    
    params = combined_results['params']
    eq_text = f"y = {params[0]:.4f}x³ + {params[1]:.4f}x² + {params[2]:.4f}x + {params[3]:.4f}"
    stats_text = f"R² = {combined_results['r2']:.4f}\nAdj. R² = {combined_results['r2_adj']:.4f}\np < {combined_results['f_p_value']:.2e}\nn = {combined_results['n']}"
    
    ax.text(0.05, 0.95, f"{eq_text}\n\n{stats_text}\n\nPeak at x={peak_x:.3f}, y={peak_y:.3f}",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Normalized Radius (0=center, 1=edge)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Cell Size', fontsize=12, fontweight='bold')
    ax.set_title('Cubic Fit: All Images Combined', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'combined_cubic_fit.png'), dpi=300)
    plt.close()
    print("  Saved: combined_cubic_fit.png")
    
    # PLOT 2: Residual plot
    fig, ax = plt.subplots(figsize=(12, 6))
    residuals = combined_results['y_data'] - combined_results['y_pred']
    
    if len(residuals) > 10000:
        sample_idx = np.random.choice(len(residuals), 10000, replace=False)
        ax.scatter(combined_results['x_data'][sample_idx], residuals[sample_idx], alpha=0.3, s=10)
    else:
        ax.scatter(combined_results['x_data'], residuals, alpha=0.3, s=5)
    
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Normalized Radius', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax.set_title('Residual Plot: Cubic Fit', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cubic_fit_residuals.png'), dpi=300)
    plt.close()
    print("  Saved: cubic_fit_residuals.png")
    
    # PLOT 4: Q-Q plot of residuals
    fig, ax = plt.subplots(figsize=(8, 8))
    from scipy import stats as scipy_stats
    scipy_stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Residuals', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cubic_fit_qqplot.png'), dpi=300)
    plt.close()
    print("  Saved: cubic_fit_qqplot.png")
    
    # PLOT 5: Density plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if len(x_data) > 1000:
        hb = ax.hexbin(x_data, y_data, gridsize=50, cmap='YlOrRd', mincnt=1)
        plt.colorbar(hb, ax=ax, label='Count')
    else:
        ax.scatter(x_data, y_data, alpha=0.3, s=5)
    
    ax.plot(x_smooth, y_smooth, 'b-', linewidth=3, label='Cubic fit')
    ax.set_xlabel('Normalized Radius (0=center, 1=edge)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Cell Size', fontsize=12, fontweight='bold')
    ax.set_title('Density Plot with Cubic Fit', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cubic_fit_density.png'), dpi=300)
    plt.close()
    print("  Saved: cubic_fit_density.png")
    
    # PLOT 6: Mean binned curve
    fig, ax = plt.subplots(figsize=(12, 8))
    
    combined_df = pd.DataFrame({
        'radius': combined_results['x_data'],
        'area': combined_results['y_data']
    })
    n_bins = 20
    combined_df['radius_bin'] = pd.cut(combined_df['radius'], bins=n_bins, labels=False) / n_bins
    binned_mean = combined_df.groupby('radius_bin')['area'].agg(['mean', 'std', 'count']).reset_index()
    binned_mean['radius_center'] = binned_mean['radius_bin'] + 1/(2*n_bins)
    binned_mean['se'] = binned_mean['std'] / np.sqrt(binned_mean['count'])
    
    ax.errorbar(binned_mean['radius_center'], binned_mean['mean'], 
                yerr=binned_mean['se'], fmt='o-', color='gray', 
                capsize=3, capthick=1, alpha=0.7, label='Binned mean ± SE')
    
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=3, label='Cubic fit')
    ax.fill_between(x_smooth, y_smooth - ci, y_smooth + ci, alpha=0.2, color='red', label='95% Confidence Band')
    
    ax.set_xlabel('Normalized Radius (0=center, 1=edge)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Cell Size', fontsize=12, fontweight='bold')
    ax.set_title('Binned Mean with Cubic Fit', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cubic_fit_binned.png'), dpi=300)
    plt.close()
    print("  Saved: cubic_fit_binned.png")


def save_results(combined_results, output_folder):
    # Save results to CSV.
    print("SAVING RESULTS")
    
    params_df = pd.DataFrame({
        'parameter': combined_results['param_names'],
        'value': combined_results['params'],
        'std_error': combined_results['param_errors'],
        't_stat': combined_results['t_stats'],
        'p_value': combined_results['p_values'],
        'lower_ci_95': combined_results['ci_lower'],
        'upper_ci_95': combined_results['ci_upper'],
        'significant': combined_results['p_values'] < 0.05
    })
    params_df.to_csv(os.path.join(output_folder, 'combined_cubic_parameters.csv'), index=False)
    print("  Saved: combined_cubic_parameters.csv")
    
    summary_df = pd.DataFrame({
        'metric': ['R²', 'Adjusted R²', 'F-statistic', 'F-test p-value', 'n', 'dof', 'residual_std'],
        'value': [combined_results['r2'], combined_results['r2_adj'], 
                  combined_results['f_stat'], combined_results['f_p_value'],
                  combined_results['n'], combined_results['dof'], combined_results['residual_std']]
    })
    summary_df.to_csv(os.path.join(output_folder, 'combined_cubic_summary.csv'), index=False)
    print("  Saved: combined_cubic_summary.csv")


def _slug(value):
    return re.sub(r'[^a-zA-Z0-9]+', '_', str(value)).strip('_')


def create_grouped_comparisons(combined_df, output_folder):
    # Compare cell-area curves (both the cubic fit and the spline fit) across
    # root type, across treatment, across species, across population (pooled
    # over all species), and separately across populations within each
    # species. Skips any comparison that would only have one group -- there's
    # nothing to compare a single curve against.

    print("ROOT TYPE COMPARISON")

    type_values = sorted(v for v in combined_df['root_type'].unique()
                         if v != 'unknown' and pd.notna(v))

    if len(type_values) < 2:
        print(f"  Only one root type ({type_values}); skipping root type comparison.")
    else:
        cubic_by_type = fit_cubic_by_group(combined_df, 'root_type')
        spline_by_type = fit_spline_by_group(combined_df, 'root_type')

        plot_cubic_comparison(
            cubic_by_type, 'Cubic Fit by Root Type',
            os.path.join(output_folder, 'cubic_fit_by_type.png'), group_order=type_values)
        plot_spline_comparison(
            spline_by_type, 'Spline Fit by Root Type',
            os.path.join(output_folder, 'spline_fit_by_type.png'), group_order=type_values)

        save_group_cubic_results(cubic_by_type, 'root_type',
                                  os.path.join(output_folder, 'cubic_fit_by_type.csv'))
        save_group_spline_results(spline_by_type, 'root_type',
                                   os.path.join(output_folder, 'spline_fit_by_type.csv'))

    print("TREATMENT COMPARISON")

    treatment_values = sorted(v for v in combined_df['treatment'].unique()
                              if v != 'unknown' and pd.notna(v))

    if len(treatment_values) < 2:
        print(f"  Only one treatment ({treatment_values}); skipping treatment comparison.")
    else:
        cubic_by_treatment = fit_cubic_by_group(combined_df, 'treatment')
        spline_by_treatment = fit_spline_by_group(combined_df, 'treatment')

        plot_cubic_comparison(
            cubic_by_treatment, 'Cubic Fit by Treatment',
            os.path.join(output_folder, 'cubic_fit_by_treatment.png'), group_order=treatment_values)
        plot_spline_comparison(
            spline_by_treatment, 'Spline Fit by Treatment',
            os.path.join(output_folder, 'spline_fit_by_treatment.png'), group_order=treatment_values)

        save_group_cubic_results(cubic_by_treatment, 'treatment',
                                  os.path.join(output_folder, 'cubic_fit_by_treatment.csv'))
        save_group_spline_results(spline_by_treatment, 'treatment',
                                   os.path.join(output_folder, 'spline_fit_by_treatment.csv'))

    print("SPECIES COMPARISON")

    species_values = sorted(v for v in combined_df['species'].unique()
                            if v != 'unknown' and pd.notna(v))

    if len(species_values) < 2:
        print(f"  Only one species ({species_values}); skipping species comparison.")
    else:
        cubic_by_species = fit_cubic_by_group(combined_df, 'species')
        spline_by_species = fit_spline_by_group(combined_df, 'species')

        plot_cubic_comparison(
            cubic_by_species, 'Cubic Fit by Species',
            os.path.join(output_folder, 'cubic_fit_by_species.png'), group_order=species_values)
        plot_spline_comparison(
            spline_by_species, 'Spline Fit by Species',
            os.path.join(output_folder, 'spline_fit_by_species.png'), group_order=species_values)

        save_group_cubic_results(cubic_by_species, 'species',
                                  os.path.join(output_folder, 'cubic_fit_by_species.csv'))
        save_group_spline_results(spline_by_species, 'species',
                                   os.path.join(output_folder, 'spline_fit_by_species.csv'))

    print("POPULATION COMPARISON (pooled across all species)")

    all_pop_values = sorted(v for v in combined_df['population'].unique()
                            if v != 'unknown' and pd.notna(v))

    if len(all_pop_values) < 2:
        print(f"  Only one population ({all_pop_values}); skipping pooled population comparison.")
    else:
        cubic_by_pop_all = fit_cubic_by_group(combined_df, 'population')
        spline_by_pop_all = fit_spline_by_group(combined_df, 'population')

        plot_cubic_comparison(
            cubic_by_pop_all, 'Cubic Fit by Population (all species)',
            os.path.join(output_folder, 'cubic_fit_by_population.png'), group_order=all_pop_values)
        plot_spline_comparison(
            spline_by_pop_all, 'Spline Fit by Population (all species)',
            os.path.join(output_folder, 'spline_fit_by_population.png'), group_order=all_pop_values)

        save_group_cubic_results(cubic_by_pop_all, 'population',
                                  os.path.join(output_folder, 'cubic_fit_by_population.csv'))
        save_group_spline_results(spline_by_pop_all, 'population',
                                   os.path.join(output_folder, 'spline_fit_by_population.csv'))

    print("POPULATION COMPARISON (within each species)")

    pop_output_folder = os.path.join(output_folder, 'by_population')
    os.makedirs(pop_output_folder, exist_ok=True)

    for species_value in combined_df['species'].unique():
        if species_value == 'unknown' or pd.isna(species_value):
            continue

        species_df = combined_df[combined_df['species'] == species_value]
        pop_values = sorted(v for v in species_df['population'].unique()
                            if v != 'unknown' and pd.notna(v))

        print(f"\n--- {species_value}: {len(pop_values)} population(s) {pop_values} ---")
        if len(pop_values) < 2:
            print(f"  Only one population for {species_value}; skipping population comparison.")
            continue

        slug = _slug(species_value)
        cubic_by_pop = fit_cubic_by_group(species_df, 'population')
        spline_by_pop = fit_spline_by_group(species_df, 'population')

        plot_cubic_comparison(
            cubic_by_pop, f'Cubic Fit by Population ({species_value})',
            os.path.join(pop_output_folder, f'cubic_fit_by_population_{slug}.png'), group_order=pop_values)
        plot_spline_comparison(
            spline_by_pop, f'Spline Fit by Population ({species_value})',
            os.path.join(pop_output_folder, f'spline_fit_by_population_{slug}.png'), group_order=pop_values)

        save_group_cubic_results(cubic_by_pop, 'population',
                                  os.path.join(pop_output_folder, f'cubic_fit_by_population_{slug}.csv'))
        save_group_spline_results(spline_by_pop, 'population',
                                   os.path.join(pop_output_folder, f'spline_fit_by_population_{slug}.csv'))


def main(force_rebuild=False):

    # Ensure output folders exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(INDIVIDUAL_PLOTS_RADIUS, exist_ok=True)
    os.makedirs(INDIVIDUAL_PLOTS_FILE, exist_ok=True)
    
    # Create individual spline plots with feature annotations (INCREMENTAL)
    radius_count, file_count, feature_summary = create_individual_spline_plots_from_master(
        master_summary_path=MASTER_SUMMARY_PATH,
        measurements_folder=MEASUREMENTS_FOLDER,
        output_radius_folder=INDIVIDUAL_PLOTS_RADIUS,
        output_file_folder=INDIVIDUAL_PLOTS_FILE,
        force_rebuild=force_rebuild,
        debug=False
    )
    
    # Process all images for combined cubic fit (ALWAYS ALL IMAGES)
    combined_df, root_info = process_all_images(MASTER_SUMMARY_PATH, MEASUREMENTS_FOLDER)
    
    if combined_df is None:
        print("No data processed for combined fit!")
        return
    
    # Fit combined cubic
    combined_results = fit_combined_cubic(combined_df)
    
    if combined_results is None:
        print("Cubic fit failed!")
        return
    
    # Create visualizations (ALWAYS ALL DATA)
    create_visualizations(combined_results, OUTPUT_FOLDER)

    # Save results
    save_results(combined_results, OUTPUT_FOLDER)

    # Root type, species, pooled population, and within-species population comparisons
    create_grouped_comparisons(combined_df, OUTPUT_FOLDER)


if __name__ == "__main__":
    import sys
    force_rebuild = '--force-rebuild' in sys.argv or '-f' in sys.argv
    main(force_rebuild=force_rebuild)