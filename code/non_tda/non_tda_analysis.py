# The pooled cell-size profile across all images.
#
# Loads every measurement file, normalises each root onto a common axis and
# overlays them, which is the plot the whole project is built around.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
import json
import os
from code.util.file_utils import load_image_summary
from code.config import MASTER_SUMMARY_PATH, RESULTS_FOLDER

def load_and_filter_cells(csv_path, angle_tolerance=5):

    df = pd.read_csv(csv_path)
    
    if df.empty:
        return None
    
    # Find the angle range for this image
    min_angle = df['angle_degrees'].min()
    max_angle = df['angle_degrees'].max()
    angle_range = max_angle - min_angle
    
    # Sample at 5 different angles across the 90-degree range
    sample_angles = [0, 22.5, 45, 67.5, 90]
    target_angles = [min_angle + offset for offset in sample_angles]
    
    # Collect cells from all 5 angles
    all_filtered = []
    
    for angle_offset, target_angle in zip(sample_angles, target_angles):
        # Filter cells within tolerance of this target angle
        angle_lower = target_angle - angle_tolerance
        angle_upper = target_angle + angle_tolerance
        
        angle_filtered = df[
            (df['angle_degrees'] >= angle_lower) & 
            (df['angle_degrees'] <= angle_upper)
        ].copy()
        
        if not angle_filtered.empty:
            angle_filtered['sampled_angle'] = angle_offset
            all_filtered.append(angle_filtered)
    
    if not all_filtered:
        return None
    
    # Combine all angles
    filtered_df = pd.concat(all_filtered, ignore_index=True)
    
    # Add metadata
    filtered_df['filename'] = Path(csv_path).stem
    filtered_df['angle_range'] = angle_range
    
    return filtered_df


def process_all_measurements(measurements_folder, angle_tolerance=5):

    csv_files = glob.glob(str(Path(measurements_folder) / "*_measurements.csv"))
    
    print(f"Found {len(csv_files)} measurement files")
    
    all_data = []
    files_processed = 0
    files_with_data = 0
    
    for csv_path in csv_files:
        filtered_data = load_and_filter_cells(csv_path, angle_tolerance)
        if filtered_data is not None and not filtered_data.empty:
            all_data.append(filtered_data)
            files_with_data += 1
        files_processed += 1
    
    if not all_data:
        print("No data found")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df


def fit_spline_and_extract_features(x, y, smoothing_factor=0.5):

    # Remove any NaN values
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < 3:
        return None
    
    # Sort by x
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    
    # Fit spline
    try:
        spline = UnivariateSpline(x, y, s=smoothing_factor, ext='extrapolate')
    except Exception:
        return None
    
    # Generate smooth curve
    x_smooth = np.linspace(0, 1, 200)
    y_smooth = spline(x_smooth)
    
    # 1. Peak height and position
    peak_idx = np.argmax(y_smooth)
    peak_height = y_smooth[peak_idx]
    peak_position = x_smooth[peak_idx]
    
    # 2. Rise slope (stele to peak)
    stele_to_peak_mask = x_smooth <= peak_position
    x_rise = x_smooth[stele_to_peak_mask]
    y_rise = y_smooth[stele_to_peak_mask]
    
    if len(x_rise) > 1:
        slope, _, _, _, _ = linregress(x_rise, y_rise)
        rise_slope = slope
    else:
        derivative = spline.derivative()
        rise_slope = derivative(peak_position / 2) if peak_position > 0 else 0
    
    # 3. Decay slope (locally just past the peak)
    decay_window = 0.10  # 10% of radius after peak
    decay_mask = (x_smooth >= peak_position) & (x_smooth <= peak_position + decay_window)
    x_decay = x_smooth[decay_mask]
    y_decay = y_smooth[decay_mask]
    
    if len(x_decay) > 1:
        slope, _, _, _, _ = linregress(x_decay, y_decay)
        decay_slope = slope
    else:
        derivative = spline.derivative()
        decay_slope = derivative(peak_position) if peak_position < 1 else 0
    
    # 4. Post-peak minimum
    post_peak_mask = x_smooth >= peak_position
    y_post_peak = y_smooth[post_peak_mask]
    x_post_peak = x_smooth[post_peak_mask]
    
    if len(y_post_peak) > 0:
        post_peak_min = np.min(y_post_peak)
        post_peak_min_idx = np.argmin(y_post_peak)
        post_peak_min_x = x_post_peak[post_peak_min_idx]
    else:
        post_peak_min = peak_height
        post_peak_min_x = peak_position
    
    # 5. Outer-rise magnitude
    outer_edge_value = y_smooth[-1]  # value at x=1.0
    outer_rise_magnitude = outer_edge_value - post_peak_min
    
    # 6. Area under the curve
    area_under_curve = np.trapz(y_smooth, x_smooth)
    
    # 7. Average cell size (mean of the curve)
    avg_cell_size = np.mean(y_smooth)
    
    return {
        'peak_height': peak_height,
        'peak_position': peak_position,
        'rise_slope': rise_slope,
        'decay_slope': decay_slope,
        'outer_rise_magnitude': outer_rise_magnitude,
        'post_peak_min': post_peak_min,
        'post_peak_min_x': post_peak_min_x,
        'outer_edge_value': outer_edge_value,
        'area_under_curve': area_under_curve,
        'avg_cell_size': avg_cell_size,
        'spline_smoothing': smoothing_factor,
        'x_smooth': x_smooth.tolist(),
        'y_smooth': y_smooth.tolist()
    }


def create_master_summary_with_features(data, measurements_folder, output_path=MASTER_SUMMARY_PATH):
    
    # Get unique images
    images = data['filename'].unique()
    print(f"Processing {len(images)} images...")
    
    master_data = []
    
    for idx, filename in enumerate(images):
        print(f"  [{idx+1}/{len(images)}] Processing {filename}...")
        
        # Get image data
        image_data = data[data['filename'] == filename].copy()
        
        # Load image summary for metadata
        summary = load_image_summary(filename, measurements_folder)
        
        # Get quadrant from summary or infer from filename
        quadrant = 'unknown'
        if summary and 'quadrant' in summary:
            quadrant = summary['quadrant']
        else:
            # Try to infer from filename
            if 'BL' in filename:
                quadrant = 'BL'
            elif 'BR' in filename:
                quadrant = 'BR'
            elif 'TL' in filename:
                quadrant = 'TL'
            elif 'TR' in filename:
                quadrant = 'TR'
        
        # Get stele area and root radius
        stele_area_um2 = summary.get('stele_area_um2', 0) if summary else 0
        root_radius_um = summary.get('root_radius_um', 0) if summary else 0
        
        # Get cell file count (if available from cell assignments)
        # For now, use a placeholder or infer from data
        file_count = 0
        if 'cell_file' in image_data.columns:
            valid_files = image_data[image_data['cell_file'] >= 0]['cell_file']
            file_count = valid_files.nunique() if not valid_files.empty else 0
        elif 'cell_file_derivative' in image_data.columns:
            valid_files = image_data[image_data['cell_file_derivative'] >= 0]['cell_file_derivative']
            file_count = valid_files.nunique() if not valid_files.empty else 0
        
        # Calculate average cell area
        if 'area_um2' in image_data.columns:
            avg_area = image_data['area_um2'].mean()
        elif 'area_pixels' in image_data.columns:
            avg_area = image_data['area_pixels'].mean()
        else:
            avg_area = 0
        
        n_cells = len(image_data)
        
        # Normalize radius to 0-1
        min_radius = image_data['radius_pixels'].min()
        max_radius = image_data['radius_pixels'].max()
        if max_radius > min_radius:
            image_data['radius_normalized'] = (image_data['radius_pixels'] - min_radius) / (max_radius - min_radius)
        else:
            image_data['radius_normalized'] = 0.5
        
        # Normalize cell area to 0-1
        if 'area_um2' in image_data.columns:
            area_col = 'area_um2'
        else:
            area_col = 'area_pixels'
        
        min_area = image_data[area_col].min()
        max_area = image_data[area_col].max()
        if max_area > min_area:
            image_data['area_normalized'] = (image_data[area_col] - min_area) / (max_area - min_area)
        else:
            image_data['area_normalized'] = 0.5
        
        # Bin by normalized radius
        n_bins = min(20, len(image_data) // 3)
        if n_bins >= 3:
            image_data['radius_bin'] = pd.cut(image_data['radius_normalized'], bins=n_bins, labels=False)
            binned = image_data.groupby('radius_bin').agg({
                'radius_normalized': 'mean',
                'area_normalized': 'mean'
            }).reset_index()
            binned = binned.sort_values('radius_normalized')
            
            x = binned['radius_normalized'].values
            y = binned['area_normalized'].values
            
            # Fit spline and extract features
            features = fit_spline_and_extract_features(x, y, smoothing_factor=0.5)
            
            if features is not None:
                # Add features to master data
                master_entry = {
                    'image_name': filename,
                    'quadrant': quadrant,
                    'file_count': file_count,
                    'average_cell_area_um2': round(avg_area, 2) if avg_area else 0,
                    'n_cells': n_cells,
                    'stele_area_um2': round(stele_area_um2, 2) if stele_area_um2 else 0,
                    'root_radius_um': round(root_radius_um, 2) if root_radius_um else 0,
                    'has_image_summary': summary is not None,
                    # Curve features
                    'peak_height': round(features['peak_height'], 4),
                    'peak_position': round(features['peak_position'], 4),
                    'rise_slope': round(features['rise_slope'], 4),
                    'decay_slope': round(features['decay_slope'], 4),
                    'outer_rise_magnitude': round(features['outer_rise_magnitude'], 4),
                    'area_under_curve': round(features['area_under_curve'], 4),
                    'avg_normalized_cell_size': round(features['avg_cell_size'], 4),
                    'post_peak_min': round(features['post_peak_min'], 4),
                    'outer_edge_value': round(features['outer_edge_value'], 4)
                }
            else:
                # Fallback: no features
                master_entry = {
                    'image_name': filename,
                    'quadrant': quadrant,
                    'file_count': file_count,
                    'average_cell_area_um2': round(avg_area, 2) if avg_area else 0,
                    'n_cells': n_cells,
                    'stele_area_um2': round(stele_area_um2, 2) if stele_area_um2 else 0,
                    'root_radius_um': round(root_radius_um, 2) if root_radius_um else 0,
                    'has_image_summary': summary is not None,
                    'peak_height': np.nan,
                    'peak_position': np.nan,
                    'rise_slope': np.nan,
                    'decay_slope': np.nan,
                    'outer_rise_magnitude': np.nan,
                    'area_under_curve': np.nan,
                    'avg_normalized_cell_size': np.nan,
                    'post_peak_min': np.nan,
                    'outer_edge_value': np.nan
                }
        else:
            # Too few data points for features
            master_entry = {
                'image_name': filename,
                'quadrant': quadrant,
                'file_count': file_count,
                'average_cell_area_um2': round(avg_area, 2) if avg_area else 0,
                'n_cells': n_cells,
                'stele_area_um2': round(stele_area_um2, 2) if stele_area_um2 else 0,
                'root_radius_um': round(root_radius_um, 2) if root_radius_um else 0,
                'has_image_summary': summary is not None,
                'peak_height': np.nan,
                'peak_position': np.nan,
                'rise_slope': np.nan,
                'decay_slope': np.nan,
                'outer_rise_magnitude': np.nan,
                'area_under_curve': np.nan,
                'avg_normalized_cell_size': np.nan,
                'post_peak_min': np.nan,
                'outer_edge_value': np.nan
            }
        
        master_data.append(master_entry)
    
    # Create DataFrame and save
    master_df = pd.DataFrame(master_data)
    
    # Sort by image name
    master_df = master_df.sort_values('image_name').reset_index(drop=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    master_df.to_csv(output_path, index=False)
    print(f"\nMaster summary saved to: {output_path}")
    print(f"Total images summarized: {len(master_df)}")
    
    # Print summary statistics
    print(f"Total images: {len(master_df)}")
    print(f"Images with features: {master_df['peak_height'].notna().sum()}")
    print(f"Average peak height: {master_df['peak_height'].mean():.3f}")
    print(f"Average peak position: {master_df['peak_position'].mean():.3f}")
    print(f"Average rise slope: {master_df['rise_slope'].mean():.3f}")
    print(f"Average decay slope: {master_df['decay_slope'].mean():.3f}")
    
    return master_df


def create_visualizations(data, output_folder=RESULTS_FOLDER / 'non_tda_results'):

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Average cell size per image
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Process each image separately
    images = data['filename'].unique()
    
    # Use a color palette for better distinction
    colors = plt.cm.tab20(np.linspace(0, 1, len(images)))
    
    for idx, filename in enumerate(images):
        image_data = data[data['filename'] == filename].copy()
        
        # Bin radius values
        image_data['radius_binned'] = (image_data['radius_pixels'] // 20) * 20
        
        # Average cell areas for each radius bin
        averaged = image_data.groupby('radius_binned')['area_pixels'].mean().reset_index()
        averaged = averaged.sort_values('radius_binned')
        
        # Plot line for this image
        ax.plot(averaged['radius_binned'], averaged['area_pixels'], 
                alpha=0.7, linewidth=1.5, marker='o', markersize=3,
                color=colors[idx], label=filename if len(images) <= 15 else '')
    
    ax.set_xlabel('Radius from Center (pixels)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Cell Size (pixels²)', fontsize=14, fontweight='bold')
    ax.set_title('Cell Size vs Radius: All Images Combined\n(5-angle sampling: 0°, 22.5°, 45°, 67.5°, 90°)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend only if not too many images
    if len(images) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=1)
    else:
        ax.text(0.02, 0.98, f'{len(images)} images shown', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / 'all_images_lines_combined.png', dpi=300, bbox_inches='tight')
    print(f"Saved: all_images_lines_combined.png")
    plt.close()
    
    # Cleaner version with smoothing (using larger bins)
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for idx, filename in enumerate(images):
        image_data = data[data['filename'] == filename].copy()
        
        # Use larger bins (50 pixels) for smoother lines
        image_data['radius_binned'] = (image_data['radius_pixels'] // 50) * 50
        
        # Average cell areas for each radius bin
        averaged = image_data.groupby('radius_binned')['area_pixels'].mean().reset_index()
        averaged = averaged.sort_values('radius_binned')
        
        # Plot line for this image
        ax.plot(averaged['radius_binned'], averaged['area_pixels'], 
                alpha=0.6, linewidth=2, 
                color=colors[idx], label=filename if len(images) <= 15 else '')
    
    ax.set_xlabel('Radius from Center (pixels)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Cell Size (pixels²)', fontsize=14, fontweight='bold')
    ax.set_title('Cell Size vs Radius: All Images (Smoothed Lines)\n(5-angle sampling, 50-pixel radius bins)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if len(images) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=1)
    else:
        ax.text(0.02, 0.98, f'{len(images)} images shown', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / 'all_images_lines_smoothed.png', dpi=300, bbox_inches='tight')
    print(f"Saved: all_images_lines_smoothed.png")
    plt.close()
    
    # Single line showing average pattern
    fig, ax = plt.subplots(figsize=(14, 9))
    
    all_normalized_data = []
    
    for filename in images:
        image_data = data[data['filename'] == filename].copy()
        
        # Normalize both radius and cell size to 0-1 scale for this image
        min_radius = image_data['radius_pixels'].min()
        max_radius = image_data['radius_pixels'].max()
        if max_radius > min_radius:
            image_data['radius_normalized'] = (image_data['radius_pixels'] - min_radius) / (max_radius - min_radius)
        else:
            image_data['radius_normalized'] = 0.5
        
        # Bin normalized radius into 20 equal intervals (0-1 scale)
        image_data['radius_binned'] = pd.cut(image_data['radius_normalized'], bins=20, labels=False) / 20.0
        
        # Average cell size per radius bin for this image
        averaged = image_data.groupby('radius_binned')['area_pixels'].mean().reset_index()
        
        if len(averaged) > 0:
            # Normalize cell size to 0-1 for this image
            min_area = averaged['area_pixels'].min()
            max_area = averaged['area_pixels'].max()
            if max_area > min_area:
                averaged['normalized_area'] = (averaged['area_pixels'] - min_area) / (max_area - min_area)
            else:
                averaged['normalized_area'] = 0.5  # Handle case where all areas are same
            
            all_normalized_data.append(averaged[['radius_binned', 'normalized_area']])
    
    if all_normalized_data:
        # Pool all normalized data across images
        combined_normalized = pd.concat(all_normalized_data, ignore_index=True)
        
        # Average normalized values by radius bin
        avg_pattern = combined_normalized.groupby('radius_binned').agg({
            'normalized_area': ['mean', 'std', 'count']
        }).reset_index()
        avg_pattern.columns = ['radius_binned', 'mean_normalized', 'std_normalized', 'count']
        avg_pattern = avg_pattern.sort_values('radius_binned').reset_index(drop=True)
        
        # Apply rolling average for smoothing
        if len(avg_pattern) >= 3:
            avg_pattern['mean_smoothed'] = avg_pattern['mean_normalized'].rolling(window=3, center=True, min_periods=1).mean()
        else:
            avg_pattern['mean_smoothed'] = avg_pattern['mean_normalized']
        
        ax.plot(avg_pattern['radius_binned'], avg_pattern['mean_smoothed'], 
                linewidth=3, color='#2E86AB', marker='o', markersize=6)
        
        ax.set_xlabel('Normalized Radius from Center (0-1 scale)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Cell Size (0-1 scale)', fontsize=14, fontweight='bold')
        ax.set_title(f'Normalized Average Cell Size vs Radius', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        ax.text(0.02, 0.98, f'Based on {len(images)} images\n{len(combined_normalized)} data points', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path / 'normalized_average_pattern.png', dpi=300, bbox_inches='tight')
    print(f"Saved: normalized_average_pattern.png")
    plt.close()
    
    # All images with 0-1 normalized scales
    fig, ax = plt.subplots(figsize=(14, 9))
    
    for idx, filename in enumerate(images):
        image_data = data[data['filename'] == filename].copy()
        
        # Normalize both radius and cell size to 0-1 scale for this image
        min_radius = image_data['radius_pixels'].min()
        max_radius = image_data['radius_pixels'].max()
        if max_radius > min_radius:
            image_data['radius_normalized'] = (image_data['radius_pixels'] - min_radius) / (max_radius - min_radius)
        else:
            image_data['radius_normalized'] = 0.5
        
        # Bin normalized radius into 20 equal intervals (0-1 scale)
        image_data['radius_binned'] = pd.cut(image_data['radius_normalized'], bins=20, labels=False) / 20.0
        
        # Average cell size per radius bin for this image
        averaged = image_data.groupby('radius_binned')['area_pixels'].mean().reset_index()
        
        if len(averaged) > 0:
            # Normalize cell size to 0-1 for this image
            min_area = averaged['area_pixels'].min()
            max_area = averaged['area_pixels'].max()
            if max_area > min_area:
                averaged['normalized_area'] = (averaged['area_pixels'] - min_area) / (max_area - min_area)
            else:
                averaged['normalized_area'] = 0.5
            
            averaged = averaged.sort_values('radius_binned')
            
            # Plot line for this image
            ax.plot(averaged['radius_binned'], averaged['normalized_area'], 
                    alpha=0.6, linewidth=2, 
                    color=colors[idx], label=filename if len(images) <= 15 else '')
    
    ax.set_xlabel('Normalized Radius from Center (0-1 scale)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Cell Size (0-1 scale)', fontsize=14, fontweight='bold')
    ax.set_title('Cell Size vs Radius: All Images (Normalized Scales)\n(5-angle sampling, both axes 0-1 normalized per-image)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    if len(images) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=1)
    else:
        ax.text(0.02, 0.98, f'{len(images)} images shown', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / 'all_images_lines_normalized.png', dpi=300, bbox_inches='tight')
    print(f"Saved: all_images_lines_normalized.png")
    plt.close()
    
    # Cell size vs radius
    fig, ax = plt.subplots(figsize=(14, 8))
    scatter = ax.scatter(data['radius_pixels'], data['area_pixels'], 
                        alpha=0.5, s=30, c=data['radius_pixels'], 
                        cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(data['radius_pixels'], data['area_pixels'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(data['radius_pixels'].min(), data['radius_pixels'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Polynomial fit (degree 2)')
    
    ax.set_xlabel('Radius from Center (pixels)', fontsize=12)
    ax.set_ylabel('Cell Area (pixels²)', fontsize=12)
    ax.set_title('Cell Size vs Distance from Center (5-Angle Sampling)', fontsize=14, fontweight='bold')
    ax.legend()
    plt.colorbar(scatter, label='Radius (pixels)')
    plt.tight_layout()
    plt.savefig(output_path / 'cell_size_vs_radius_scatter.png', dpi=300)
    print(f"Saved: cell_size_vs_radius_scatter.png")
    plt.close()
    
    # Binned average plot with error bars
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create radius bins
    n_bins = 20
    data['radius_bin'] = pd.cut(data['radius_pixels'], bins=n_bins)
    binned_stats = data.groupby('radius_bin')['area_pixels'].agg(['mean', 'std', 'count'])
    binned_stats['se'] = binned_stats['std'] / np.sqrt(binned_stats['count'])
    binned_stats['radius_center'] = [interval.mid for interval in binned_stats.index]
    
    ax.errorbar(binned_stats['radius_center'], binned_stats['mean'], 
                yerr=binned_stats['se'], fmt='o-', linewidth=2, markersize=8,
                capsize=5, capthick=2, label='Mean ± SE')
    
    ax.set_xlabel('Radius from Center (pixels)', fontsize=12)
    ax.set_ylabel('Average Cell Area (pixels²)', fontsize=12)
    ax.set_title('Average Cell Size by Distance from Center (Binned)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'cell_size_vs_radius_binned.png', dpi=300)
    print(f"Saved: cell_size_vs_radius_binned.png")
    plt.close()
    
    # Distribution plots at different radii
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Divide data into quartiles based on radius
    quartiles = data['radius_pixels'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    quartile_labels = ['Q1: Near Center', 'Q2', 'Q3', 'Q4: Far from Center']
    
    for idx, (ax, label) in enumerate(zip(axes.flatten(), quartile_labels)):
        q_data = data[(data['radius_pixels'] >= quartiles[idx]) & 
                     (data['radius_pixels'] < quartiles[idx + 1])]
        
        ax.hist(q_data['area_pixels'], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Cell Area (pixels²)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{label}\n(Radius: {quartiles[idx]:.0f}-{quartiles[idx+1]:.0f} px, n={len(q_data)})', 
                    fontsize=11)
        ax.axvline(q_data['area_pixels'].median(), color='red', linestyle='--', 
                  linewidth=2, label=f'Median: {q_data["area_pixels"].median():.0f}')
        ax.legend()
    
    plt.suptitle('Cell Size Distribution by Distance from Center', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'cell_size_distribution_by_radius.png', dpi=300)
    print(f"Saved: cell_size_distribution_by_radius.png")
    plt.close()
    
    # Hexbin plot for density
    fig, ax = plt.subplots(figsize=(14, 8))
    hexbin = ax.hexbin(data['radius_pixels'], data['area_pixels'], 
                       gridsize=30, cmap='YlOrRd', mincnt=1)
    ax.set_xlabel('Radius from Center (pixels)', fontsize=12)
    ax.set_ylabel('Cell Area (pixels²)', fontsize=12)
    ax.set_title('Cell Size vs Radius Density Plot (5-Angle Sampling)', fontsize=14, fontweight='bold')
    plt.colorbar(hexbin, label='Cell Count')
    plt.tight_layout()
    plt.savefig(output_path / 'cell_size_vs_radius_density.png', dpi=300)
    print(f"Saved: cell_size_vs_radius_density.png")
    plt.close()
    
    # Box plots by radius bins
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create 10 bins for clearer visualization
    data['radius_bin_10'] = pd.qcut(data['radius_pixels'], q=10, duplicates='drop')
    bin_labels = [f"{int(interval.left)}-{int(interval.right)}" 
                  for interval in data['radius_bin_10'].cat.categories]
    
    data['radius_bin_label'] = data['radius_bin_10'].cat.rename_categories(bin_labels)
    
    sns.boxplot(data=data, x='radius_bin_label', y='area_pixels', ax=ax)
    ax.set_xlabel('Radius Range (pixels)', fontsize=12)
    ax.set_ylabel('Cell Area (pixels²)', fontsize=12)
    ax.set_title('Cell Size Distribution Across Radius Ranges', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'cell_size_boxplot_by_radius.png', dpi=300)
    print(f"Saved: cell_size_boxplot_by_radius.png")
    plt.close()
    
    # Individual file comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sample up to 20 files for clarity
    unique_files = data['filename'].unique()
    if len(unique_files) > 20:
        sampled_files = np.random.choice(unique_files, 20, replace=False)
        plot_data = data[data['filename'].isin(sampled_files)]
    else:
        plot_data = data
    
    for filename in plot_data['filename'].unique():
        file_data = plot_data[plot_data['filename'] == filename]
        ax.scatter(file_data['radius_pixels'], file_data['area_pixels'], 
                  alpha=0.6, s=20, label=filename if len(plot_data['filename'].unique()) <= 10 else '')
    
    ax.set_xlabel('Radius from Center (pixels)', fontsize=12)
    ax.set_ylabel('Cell Area (pixels²)', fontsize=12)
    ax.set_title('Cell Size vs Radius: Individual Images', fontsize=14, fontweight='bold')
    if len(plot_data['filename'].unique()) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path / 'cell_size_vs_radius_by_file.png', dpi=300, bbox_inches='tight')
    print(f"Saved: cell_size_vs_radius_by_file.png")
    plt.close()


def main():
    # Configuration
    measurements_folder = RESULTS_FOLDER / 'measurements'
    output_folder = RESULTS_FOLDER / 'non_tda_results'
    angle_tolerance = 5  # degrees around each sampled angle
    
    # Process all measurements
    combined_data = process_all_measurements(measurements_folder, angle_tolerance)
    
    if combined_data is None:
        print("No data found!")
        return
    
    # Create visualizations
    create_visualizations(combined_data, output_folder)
    
    # Save processed data
    output_path = Path(output_folder)
    data_file = output_path / 'processed_data.csv'
    combined_data.to_csv(data_file, index=False)
    print(f"Saved processed data to: {data_file}")
    
    # Create master summary with curve features
    master_summary_path = MASTER_SUMMARY_PATH
    create_master_summary_with_features(
        combined_data, 
        measurements_folder, 
        master_summary_path
    )
    
    print(f"\nAll results saved to: {output_folder}/")
    print(f"Master summary saved to: {master_summary_path}")


if __name__ == "__main__":
    main()