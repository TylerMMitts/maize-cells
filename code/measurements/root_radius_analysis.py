# Does the pattern change with overall root size?
#
# Bins roots by radius and overlays the normalised profiles, to separate a
# genuine shape difference from the trivial effect of a bigger root having
# more room.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d
import json
import statsmodels.api as sm
import sys
from pathlib import Path

# Get project root
SCRIPT_DIR = Path(__file__).parent.absolute()  # code/measurements/
CODE_DIR = SCRIPT_DIR.parent  # code/
PROJECT_ROOT = CODE_DIR.parent  # project root

# Add code directory to path for imports
sys.path.insert(0, str(CODE_DIR))

# Import utility functions
from code.util import load_measurement_csv, load_image_summary, find_all_measurement_files, normalize_radius_and_area, determine_angle_range

# Use absolute paths
MEASUREMENTS_FOLDER = str(PROJECT_ROOT / "results" / "measurements_all")
OUTPUT_FOLDER = str(PROJECT_ROOT / "results" / "root_radius_analysis")
N_BINS = 5
MIN_CELLS_PER_IMAGE = 10
N_PROFILE_BINS = 20

def find_all_measurement_files_with_radius(measurements_folder):

    all_files = find_all_measurement_files(measurements_folder, min_cells=MIN_CELLS_PER_IMAGE)
    
    image_data = []
    for file_info in all_files:
        image_name = file_info['name']
        csv_path = file_info['csv_path']
        
        # Load the image summary
        summary = load_image_summary(image_name, measurements_folder)
        
        if summary is not None and summary.get('root_radius_um', 0) > 0:
            df = load_measurement_csv(csv_path)
            if df is not None and len(df) >= MIN_CELLS_PER_IMAGE:
                image_data.append({
                    'name': image_name,
                    'csv_path': csv_path,
                    'summary': summary,
                    'df': df,
                    'stele_area_um2': summary.get('stele_area_um2', 0),
                    'root_radius_um': summary['root_radius_um'],
                    'quadrant': summary.get('quadrant', 'unknown')
                })
    
    return image_data

def bin_images_by_root_radius(image_data, n_bins=N_BINS):
    radii = [img['root_radius_um'] for img in image_data if img['root_radius_um'] > 0]
    
    if len(radii) < n_bins:
        print(f"Only {len(radii)} images with root radius, cannot create {n_bins} bins")
        return {}, []
    
    # Create bins based on percentiles (evenly distributed)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(radii, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # Assign images to bins
    binned_images = {}
    bin_ranges = {}
    
    bin_names = [f'Radius {i+1}' for i in range(n_bins)]
    
    for i in range(n_bins):
        bin_name = bin_names[i]
        low = bin_edges[i]
        high = bin_edges[i + 1]
        bin_ranges[bin_name] = (low, high)
        binned_images[bin_name] = []
        
        for img in image_data:
            if img['root_radius_um'] > 0 and low <= img['root_radius_um'] < high:
                binned_images[bin_name].append(img)
    
    return binned_images, bin_ranges

def create_normalized_profile(df, root_radius, n_bins=N_PROFILE_BINS):
    # Normalize radius
    df_norm = df.copy()
    df_norm['norm_radius'] = df_norm['radius_um'] / root_radius
    
    # Create bins from 0 to 1
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    mean_areas = []
    std_areas = []
    
    for i in range(n_bins):
        in_bin = (df_norm['norm_radius'] >= bins[i]) & (df_norm['norm_radius'] < bins[i+1])
        areas_in_bin = df_norm.loc[in_bin, 'area_um2']
        
        if len(areas_in_bin) > 0:
            mean_areas.append(areas_in_bin.mean())
            std_areas.append(areas_in_bin.std())
        else:
            mean_areas.append(np.nan)
            std_areas.append(np.nan)
    
    # Convert to numpy arrays
    mean_areas = np.array(mean_areas)
    std_areas = np.array(std_areas)
    
    # Find first and last valid indices
    valid = ~np.isnan(mean_areas)
    if not np.any(valid):
        return None, None, None
    
    first_valid = np.where(valid)[0][0]
    last_valid = np.where(valid)[0][-1]
    
    # Fill NaN values at the ends with the nearest valid value
    for i in range(first_valid):
        mean_areas[i] = mean_areas[first_valid]
        std_areas[i] = std_areas[first_valid]
    
    for i in range(last_valid + 1, len(mean_areas)):
        mean_areas[i] = mean_areas[last_valid]
        std_areas[i] = std_areas[last_valid]
    
    return bin_centers, mean_areas, std_areas

def plot_continuous_color_profiles(binned_images, bin_ranges, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Use the same color scheme as stele area analysis for consistency
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(binned_images)))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, (bin_name, images) in enumerate(binned_images.items()):
        if not images:
            continue
        
        color = colors[idx]
        low, high = bin_ranges[bin_name]
        
        all_profiles = []
        all_std = []
        
        for img in images:
            bin_centers, mean_areas, std_areas = create_normalized_profile(
                img['df'], img['root_radius_um']
            )
            if bin_centers is not None:
                all_profiles.append(mean_areas)
                all_std.append(std_areas)
        
        if not all_profiles:
            continue
        
        # Average across images
        avg_profile = np.nanmean(all_profiles, axis=0)
        avg_std = np.nanmean(all_std, axis=0)
        
        # Format label
        if np.isinf(low):
            label = f"{bin_name}\n< {high:.0f} µm\n({len(images)} images)"
        elif np.isinf(high):
            label = f"{bin_name}\n> {low:.0f} µm\n({len(images)} images)"
        else:
            label = f"{bin_name}\n{low:.0f}-{high:.0f} µm\n({len(images)} images)"
        
        x_vals = np.linspace(0, 1, len(avg_profile))  # 0 to 1
        ax.plot(x_vals, avg_profile, '-', linewidth=2.5, color=color, label=label)
        ax.fill_between(x_vals, avg_profile - avg_std, avg_profile + avg_std,
                        alpha=0.2, color=color)
    
    ax.set_xlabel('Normalized Distance from Center (0 = center, 1 = root edge)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Cell Area (µm²)', fontsize=12, fontweight='bold')
    ax.set_title('Cell Size vs Normalized Distance from Center\n'
                'Colored by Root Radius (5 equal bins)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '01_continuous_color_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: 01_continuous_color_profiles.png")

def perform_regression_analysis(image_data, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Combine all cells from all images
    all_data = []
    for img in image_data:
        df = img['df'].copy()
        df['root_radius'] = img['root_radius_um']
        df['image_name'] = img['name']
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Step 1: Calculate the means
    mean_radius = combined_df['radius_um'].mean()
    mean_root_radius = combined_df['root_radius'].mean()
    
    # Step 2: Create centered variables
    combined_df['radius_centered'] = combined_df['radius_um'] - mean_radius
    combined_df['root_radius_centered'] = combined_df['root_radius'] - mean_root_radius
    
    # Step 3: Create interaction term from centered variables
    combined_df['radius_root_centered'] = combined_df['radius_centered'] * combined_df['root_radius_centered']
    
    # Step 4: Build the X matrix with centered variables
    X_centered = combined_df[['radius_centered', 'root_radius_centered', 'radius_root_centered']].copy()
    X_centered = sm.add_constant(X_centered)  # Adds the intercept column (all 1s)
    y = combined_df['area_um2']
    
    # Step 5: Fit the centered model
    model_centered = sm.OLS(y, X_centered).fit()
        
    with open(os.path.join(output_folder, '02_regression_results_centered.txt'), 'w', encoding='utf-8') as f:
        f.write("REGRESSION ANALYSIS: Cell Area vs Radius and Root Radius\n")
        f.write("MODEL: area = b0 + b1*(radius - mean_radius) + b2*(root_radius - mean_root_radius) + b3*(radius - mean_radius)*(root_radius - mean_root_radius)\n")
        
        f.write(f"CENTERING VALUES:\n")
        f.write(f"  Mean cell radius: {mean_radius:.4f} µm\n")
        f.write(f"  Mean root radius: {mean_root_radius:.4f} µm\n\n")
        
        f.write(model_centered.summary().as_text())
        f.write("COEFFICIENT INTERPRETATIONS (CENTERED MODEL):\n")
        
        coeffs = model_centered.params
        pvals = model_centered.pvalues
        
        # Intercept
        f.write(f"\nIntercept: {coeffs['const']:.4f} (p={pvals['const']:.4e})\n")
        f.write(f"  -> Predicted cell area when cell radius and root radius are at their AVERAGE values: {coeffs['const']:.2f} µm²\n")
        
        # Cell radius effect
        if 'radius_centered' in coeffs.index:
            f.write(f"\nb1 (Cell radius effect, at average root radius): {coeffs['radius_centered']:.4f} (p={pvals['radius_centered']:.4e})\n")
            f.write(f"  -> At the average root radius, a 1 µm increase in cell radius is associated with a {coeffs['radius_centered']:.4f} µm² change in cell area\n")
            if coeffs['radius_centered'] < 0:
                f.write(f"  -> NOTE: The negative coefficient suggests that at average root radius, larger individual cell radii are associated with smaller cell areas—this is likely an artifact of collinearity between cell radius and root radius.\n")
        
        # Root radius effect
        if 'root_radius_centered' in coeffs.index:
            f.write(f"\nb2 (Root radius effect, at average cell radius): {coeffs['root_radius_centered']:.4f} (p={pvals['root_radius_centered']:.4e})\n")
            f.write(f"  -> At the average cell radius, a 100 µm increase in root radius is associated with a {coeffs['root_radius_centered']*100:.2f} µm² increase in cell area\n")
        
        # Interaction effect
        if 'radius_root_centered' in coeffs.index:
            f.write(f"\nb3 (Interaction effect): {coeffs['radius_root_centered']:.4e} (p={pvals['radius_root_centered']:.4e})\n")
            if pvals['radius_root_centered'] < 0.05:
                f.write(f"  -> Significant interaction: The effect of cell radius on area changes as root radius changes\n")
                f.write(f"  -> For every 1 µm increase in root radius (above average), the effect of cell radius changes by {coeffs['radius_root_centered']:.4e} µm²\n")
            else:
                f.write(f"  -> No significant interaction: The effect of cell radius does not depend on root radius\n")
        
        # Model fit
        f.write(f"\nR-squared: {model_centered.rsquared:.4f}\n")
        f.write(f"Adjusted R-squared: {model_centered.rsquared_adj:.4f}\n")
        f.write(f"Condition Number: {model_centered.condition_number:.2e}\n")
        
        if model_centered.condition_number > 1000:
            f.write(f"  WARNING: Condition number > 1000 suggests multicollinearity may still be an issue.\n")
        else:
            f.write(f"  Condition number is < 1000, suggesting multicollinearity has been reduced.\n")
    

    # Visualization of regression results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.scatter(y, model_centered.fittedvalues, alpha=0.3, s=5)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    ax1.set_xlabel('Actual Cell Area (µm²)', fontsize=12)
    ax1.set_ylabel('Predicted Cell Area (µm²)', fontsize=12)
    ax1.set_title(f'Actual vs Predicted (R² = {model_centered.rsquared:.3f})', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.scatter(model_centered.fittedvalues, model_centered.resid, alpha=0.3, s=5)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Cell Area (µm²)', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Regression Analysis Results (Centered Model)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '02_regression_visualization_centered.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: 02_regression_results_centered.txt")
    print(f"Saved: 02_regression_visualization_centered.png")
    
    return model_centered

def main():
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    image_data = find_all_measurement_files_with_radius(MEASUREMENTS_FOLDER)
    
    if not image_data:
        print("No image data found with root radius measurements!")
        print("Please check that measurement files and summaries exist.")
        return

    binned_images, bin_ranges = bin_images_by_root_radius(image_data, N_BINS)
    for bin_name, images in binned_images.items():
        low, high = bin_ranges[bin_name]
        if np.isinf(low):
            print(f"  {bin_name}: {len(images)} images (< {high:.0f} µm)")
        elif np.isinf(high):
            print(f"  {bin_name}: {len(images)} images (> {low:.0f} µm)")
        else:
            print(f"  {bin_name}: {len(images)} images ({low:.0f}-{high:.0f} µm)")
    
    plot_continuous_color_profiles(binned_images, bin_ranges, OUTPUT_FOLDER)
    
    regression_model = perform_regression_analysis(image_data, OUTPUT_FOLDER)
    

if __name__ == "__main__":
    main()