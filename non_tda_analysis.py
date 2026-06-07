import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy import stats

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

def create_visualizations(data, output_folder="non_tda_results"):

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
    measurements_folder = "measurements"
    output_folder = "non_tda_results"
    angle_tolerance = 5  # degrees around each sampled angle
    
    # Process all measurements
    combined_data = process_all_measurements(measurements_folder, angle_tolerance)
    
    create_visualizations(combined_data, output_folder)
    
    # Save processed data
    output_path = Path(output_folder)
    data_file = output_path / 'processed_data.csv'
    combined_data.to_csv(data_file, index=False)

    print(f"All results saved to: {output_folder}/")

if __name__ == "__main__":
    main()
