
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

warnings.filterwarnings('ignore')


MASTER_SUMMARY_PATH = "results/master_summary.csv"
MEASUREMENTS_FOLDER = "results/measurements_all"
CELL_ASSIGNMENTS_FOLDER = "results/cell_file/cell_file_counting"

# These are the actual paths where plots are saved
OUTPUT_FOLDER = "results/combined_cubic_fit"
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
        
        # Decay slope (from peak to end)
        decay_x = x_smooth[peak_idx:]
        decay_y = y_smooth[peak_idx:]
        if len(decay_x) > 2:
            decay_slope = (decay_y[-1] - decay_y[0]) / (decay_x[-1] - decay_x[0] + 0.001)
        else:
            decay_slope = 0
        
        # Outer-rise magnitude (for feature extraction only, not plotted)
        outer_start_idx = int(0.7 * len(x_smooth))
        outer_end_idx = int(0.95 * len(x_smooth))
        
        if outer_start_idx < 0:
            outer_start_idx = 0
        if outer_end_idx > len(x_smooth):
            outer_end_idx = len(x_smooth)
        if outer_start_idx >= outer_end_idx:
            outer_start_idx = max(0, outer_end_idx - 10)
        
        outer_region_y = y_smooth[outer_start_idx:outer_end_idx]
        
        if len(outer_region_y) > 5:
            outer_rise_magnitude = np.max(outer_region_y) - y_smooth[-1]
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
            
            feature_text = (
                f"Features:\n"
                f"  Peak Height: {features['peak_height']:.3f}\n"
                f"  Peak Position: {features['peak_position']:.3f}\n"
                f"  Rise Slope: {features['rise_slope']:.3f}\n"
                f"  Decay Slope: {features['decay_slope']:.3f}\n"
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
                        
                        feature_text_file = (
                            f"Features:\n"
                            f"  Peak Height: {features_file['peak_height']:.3f}\n"
                            f"  Peak Position: {features_file['peak_position']:.3f}\n"
                            f"  Rise Slope: {features_file['rise_slope']:.3f}\n"
                            f"  Decay Slope: {features_file['decay_slope']:.3f}\n"
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
        root_id = parsed.get('root_identifier')
        
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
        binned['n_files'] = row.get('file_count', 0)
        
        all_binned_data.append(binned)
        
        # Store root info
        if root_id not in root_info:
            root_info[root_id] = {
                'root_type': parsed.get('root_type', 'unknown'),
                'treatment': parsed.get('treatment', 'unknown'),
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


def fit_cubic_by_root_type(combined_df):
    
    root_types = combined_df['root_type'].unique()
    results_by_type = {}
    
    for root_type in root_types:
        if root_type == 'unknown' or pd.isna(root_type):
            continue
        
        type_df = combined_df[combined_df['root_type'] == root_type]
        x_data = type_df['radius_normalized'].values
        y_data = type_df['area_normalized'].values
        
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
            
            results_by_type[root_type] = {
                'params': popt,
                'r2': r2,
                'r2_adj': r2_adj,
                'n': n
            }
            
            print(f"\n{root_type}:")
            print(f"  R² = {r2:.4f}")
            print(f"  Adjusted R² = {r2_adj:.4f}")
            print(f"  n = {n}")
            
        except Exception as e:
            print(f"Error fitting {root_type}: {e}")
    
    return results_by_type


def create_visualizations(combined_results, results_by_type, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    sns.set_style("whitegrid")
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
    
    # PLOT 2: By root type
    if results_by_type:
        fig, ax = plt.subplots(figsize=(12, 8))
        root_types = sorted(results_by_type.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(root_types)))
        type_to_color = {t: colors[i] for i, t in enumerate(root_types)}
        x_smooth = np.linspace(0, 1, 200)
        
        for root_type in root_types:
            result = results_by_type[root_type]
            params = result['params']
            y_smooth_type = cubic(x_smooth, *params)
            
            peak_idx_type = np.argmax(y_smooth_type)
            peak_x_type = x_smooth[peak_idx_type]
            peak_y_type = y_smooth_type[peak_idx_type]
            
            ax.plot(x_smooth, y_smooth_type, '-', linewidth=2.5, 
                   color=type_to_color[root_type],
                   label=f"{root_type} (R²={result['r2']:.3f}, n={result['n']}, peak={peak_x_type:.2f})")
            ax.scatter(peak_x_type, peak_y_type, s=80, color=type_to_color[root_type], marker='*', zorder=10)
        
        ax.set_xlabel('Normalized Radius (0=center, 1=edge)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Cell Size', fontsize=12, fontweight='bold')
        ax.set_title('Cubic Fit by Root Type', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'cubic_fit_by_type.png'), dpi=300)
        plt.close()
        print("  Saved: cubic_fit_by_type.png")
    
    # PLOT 3: Residual plot
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


def save_results(combined_results, results_by_type, output_folder):
    """Save results to CSV."""
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
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
    
    if results_by_type:
        type_results = []
        for root_type, result in results_by_type.items():
            type_results.append({
                'root_type': root_type,
                'a': result['params'][0],
                'b': result['params'][1],
                'c': result['params'][2],
                'd': result['params'][3],
                'r2': result['r2'],
                'r2_adj': result['r2_adj'],
                'n': result['n']
            })
        type_df = pd.DataFrame(type_results)
        type_df.to_csv(os.path.join(output_folder, 'cubic_fit_by_type.csv'), index=False)
        print("  Saved: cubic_fit_by_type.csv")


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
    
    # Fit by root type
    results_by_type = fit_cubic_by_root_type(combined_df)
    
    # Create visualizations (ALWAYS ALL DATA)
    create_visualizations(combined_results, results_by_type, OUTPUT_FOLDER)
    
    # Save results
    save_results(combined_results, results_by_type, OUTPUT_FOLDER)



if __name__ == "__main__":
    import sys
    force_rebuild = '--force-rebuild' in sys.argv or '-f' in sys.argv
    main(force_rebuild=force_rebuild)