# Assigns each cell to a concentric cell file.
#
# The derivative method is the one in use: cell area is walked outward along
# the radius and file boundaries are placed where the slope changes sign.
# Two other methods are kept for comparison. Writes cell_assignments.csv per
# image under results/cell_file/cell_file_counting/.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import KMeans
import glob
from tqdm import tqdm
from code.config import CELL_FILE_COUNTS_FOLDER, CHOSEN_RESULTS_FOLDER, DATA_FOLDER, RESULTS_FOLDER

# Processing mode: 'single' for one image, 'batch' for multiple images
MODE = 'batch'

# For SINGLE image mode:
SINGLE_IMAGE_PATH = DATA_FOLDER / 'chosen_results' / 'BL_example.jpg'
SINGLE_CSV_PATH = 'measurements/BL_example_measurements.csv'

# For BATCH mode:
MEASUREMENTS_FOLDER = RESULTS_FOLDER / 'measurements'
IMAGE_FOLDER = CHOSEN_RESULTS_FOLDER

# Output directory for results
OUTPUT_DIR = CELL_FILE_COUNTS_FOLDER

# Visualization:
CREATE_VISUALIZATIONS = True

# Method parameters
DERIVATIVE_WINDOW = 17  
CUMULATIVE_PEAK_HEIGHT = 0.4
DERIVATIVE_CHANGE_THRESHOLD = 1.7
MIN_CELLS_FOR_ANALYSIS = 15

try:
    from code.util.visualize_cell_centers import get_plant_center_from_filename
except ImportError:
    def get_plant_center_from_filename(filename, w, h):
        filename_upper = filename.upper()
        if 'BL' in filename_upper:
            return w, 0, 'BL'
        elif 'BR' in filename_upper:
            return 0, 0, 'BR'
        elif 'TL' in filename_upper:
            return w, h, 'TL'
        elif 'TR' in filename_upper:
            return 0, h, 'TR'
        else:
            return w//2, h//2, None


def load_all_cells(csv_path, center_x, center_y):
    df = pd.read_csv(csv_path)
    
    if df.empty:
        return None
    
    # Calculate radius and angle for all cells
    if 'radius_pixels' not in df.columns:
        dx = df['x_pixels'] - center_x
        dy = df['y_pixels'] - center_y
        df['radius_pixels'] = np.sqrt(dx**2 + dy**2)
    
    if 'angle_degrees' not in df.columns:
        dx = df['x_pixels'] - center_x
        dy = df['y_pixels'] - center_y
        df['angle_degrees'] = np.degrees(np.arctan2(dy, dx))
    
    return df


def count_files_by_derivative(df):
    radii = df['radius_pixels'].values
    areas = df['area_pixels'].values
    
    # Sort by radius (distance from center)
    sort_idx = np.argsort(radii)
    radii_sorted = radii[sort_idx]
    areas_sorted = areas[sort_idx]
    
    if len(radii_sorted) < MIN_CELLS_FOR_ANALYSIS:
        return 1, np.array([]), np.array([]), None
    
    # Ensure window is odd and appropriate size
    window = DERIVATIVE_WINDOW
    if window % 2 == 0:
        window -= 1
    window = min(window, len(radii_sorted) // 3)
    if window < 5:
        window = 5
    
    # Calculate rolling slope
    slopes = []
    slope_positions = []
    for i in range(len(radii_sorted) - window):
        x = radii_sorted[i:i+window]
        y = areas_sorted[i:i+window]
        slope, _ = np.polyfit(x, y, 1)
        slopes.append(slope)
        slope_positions.append(np.mean(x))
    
    slopes = np.array(slopes)
    
    if len(slopes) > 10:
        # Smooth slopes
        if len(slopes) >= 7:
            window_slope = min(7, len(slopes) - (len(slopes) % 2) - 1)
            if window_slope >= 3:
                slopes_smooth = savgol_filter(slopes, window_slope, 2)
            else:
                slopes_smooth = slopes
        else:
            slopes_smooth = slopes
        
        # Find significant changes in slope
        slope_changes = np.diff(slopes_smooth)
        change_threshold = np.std(slope_changes) * DERIVATIVE_CHANGE_THRESHOLD
        significant_changes = np.abs(slope_changes) > change_threshold
        
        # Minimum distance between changes
        min_distance = 2
        filtered_changes = np.zeros_like(significant_changes, dtype=bool)
        last_change = -min_distance
        for i, is_change in enumerate(significant_changes):
            if is_change and (i - last_change) >= min_distance:
                filtered_changes[i] = True
                last_change = i
        
        n_phases = max(1, np.sum(filtered_changes) + 1)
    else:
        n_phases = 1
    
    # Cap at reasonable maximum (but higher than before)
    n_phases = min(n_phases, 30)
    
    return n_phases, slopes_smooth if len(slopes) > 10 else slopes, slope_positions, significant_changes if len(slopes) > 10 else None


def count_files_by_cumulative(df):
    radii = np.array(sorted(df['radius_pixels'].values))
    
    if len(radii) < MIN_CELLS_FOR_ANALYSIS:
        return 1, np.array([]), np.array([]), None
    
    # Calculate cumulative proportion (empirical CDF)
    cumulative = np.arange(1, len(radii) + 1) / len(radii)
    
    # Calculate derivative of cumulative distribution
    diffs = np.diff(cumulative)
    diff_radii = (radii[:-1] + radii[1:]) / 2
    
    if len(diffs) > 15:
        # Smooth the derivative - but not too aggressively
        window = min(9, len(diffs) - (len(diffs) % 2) - 1)
        if window >= 5:
            diffs_smooth = savgol_filter(diffs, window, 2)
        else:
            diffs_smooth = diffs
        
        # Find peaks - using more reasonable thresholds
        # If we have many cells, expect multiple peaks
        peak_height_threshold = np.max(diffs_smooth) * CUMULATIVE_PEAK_HEIGHT
        
        peaks, properties = find_peaks(diffs_smooth, 
                                       height=peak_height_threshold,
                                       distance=8,  # Reasonable minimum distance between peaks
                                       prominence=np.max(diffs_smooth) * 0.1)  # Add prominence filter
        
        # If no peaks found but we have many cells, try a lower threshold
        if len(peaks) == 0 and len(radii) > 50:
            # Fallback: use a lower threshold
            peaks, properties = find_peaks(diffs_smooth, 
                                           height=np.max(diffs_smooth) * 0.15,
                                           distance=8)
        
        # Filter peaks to ensure they represent real clusters
        valid_peaks = []
        for peak_idx in peaks:
            if peak_idx < len(diff_radii):
                peak_radius = diff_radii[peak_idx]
                # Count cells in this region
                radius_bin_width = (radii.max() - radii.min()) / 20  # 20 bins total
                cells_near_peak = np.sum(np.abs(radii - peak_radius) < radius_bin_width)
                # Require at least 3 cells for a valid peak (lowered from 5)
                if cells_near_peak >= 3:
                    valid_peaks.append(peak_idx)
        
        # Number of clusters = number of valid peaks + 1
        n_clusters = max(1, len(valid_peaks) + 1)
        
        # Ensure reasonable range (typical plant roots have 5-15 files)
        n_clusters = min(n_clusters, 25)
        
    else:
        n_clusters = 1
    
    return n_clusters, diff_radii, diffs_smooth if len(diffs) > 15 else diffs, np.array(valid_peaks) if len(diffs) > 15 and len(valid_peaks) > 0 else None


def assign_files_by_consensus(df, n_files, method_name):
    if n_files <= 1:
        df[f'cell_file_{method_name}'] = 0
        return df
    
    radii = df['radius_pixels'].values.reshape(-1, 1)
    
    # Use K-means clustering to assign cells to files
    kmeans = KMeans(n_clusters=n_files, random_state=42, n_init=10, max_iter=300)
    df[f'cell_file_{method_name}'] = kmeans.fit_predict(radii)
    
    # Reorder clusters by mean radius (file 0 = closest to center)
    cluster_means = df.groupby(f'cell_file_{method_name}')['radius_pixels'].mean().sort_values()
    cluster_map = {old: new for new, old in enumerate(cluster_means.index)}
    df[f'cell_file_{method_name}'] = df[f'cell_file_{method_name}'].map(cluster_map)
    
    return df


def visualize_cell_files_on_image(image_path, df, method_name, cell_file_col, n_files, output_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Adjust figure size based on number of cells
    n_cells = len(df)
    if n_cells > 300:
        figsize = (20, 20)
    elif n_cells > 150:
        figsize = (18, 18)
    else:
        figsize = (16, 16)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_rgb)
    
    valid_cells = df[df[cell_file_col] >= 0]
    
    if len(valid_cells) == 0:
        ax.set_title(f'{method_name} - No assignments')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    file_numbers = sorted(valid_cells[cell_file_col].unique())
    max_file = max(file_numbers) if file_numbers else 0
    
    # Create colormap
    if max_file <= 20:
        colormap = cm.get_cmap('tab20', max_file + 2)
    else:
        colormap = cm.get_cmap('gist_ncar', max_file + 2)
    
    # Adjust sizes based on cell count
    if n_cells > 300:
        circle_size = 4
        font_size = 5
        label_offset = 6
    elif n_cells > 150:
        circle_size = 5
        font_size = 6
        label_offset = 7
    else:
        circle_size = 6
        font_size = 7
        label_offset = 8
    
    # Draw all cells with their file numbers
    for _, cell in valid_cells.iterrows():
        file_num = cell[cell_file_col]
        color = colormap(file_num % colormap.N)
        
        circle = Circle((cell['x_pixels'], cell['y_pixels']), circle_size, 
                       color=color, alpha=0.8, linewidth=1.5, fill=True)
        ax.add_patch(circle)
        
        ax.text(cell['x_pixels'], cell['y_pixels'] - label_offset, str(file_num),
               fontsize=font_size, ha='center', va='center', 
               color='white', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.7))
    
    # Draw center
    center_x, center_y, _ = get_plant_center_from_filename(os.path.basename(image_path), w, h)
    if center_x is not None:
        ax.plot(center_x, center_y, 'b*', markersize=20, label='Root Center')
    
    ax.set_title(f'{method_name}: {n_files} Cell Files Detected\n({len(valid_cells)} total cells)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X pixels')
    ax.set_ylabel('Y pixels')
    ax.legend(loc='upper right')
    
    if max_file > 0:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(0, max_file))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.02)
        cbar.set_label('Cell File Number (0=closest to center)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    

def create_method_explanation_plot(image_name, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Method 1 explanation
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(0.1, 0.9, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    ax1.set_title('Derivative Method Explained', fontsize=12, fontweight='bold')
    
    # Method 2 explanation
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(0.1, 0.9, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    ax2.set_title('Cumulative Method Explained', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Cell File Detection Methods - {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_explanations.png'), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_method_analysis(df_all, results, output_dir, image_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    radii = df_all['radius_pixels'].values
    areas = df_all['area_pixels'].values
    
    # Sort by radius
    sort_idx = np.argsort(radii)
    radii_sorted = radii[sort_idx]
    areas_sorted = areas[sort_idx]
    
    # Plot 1: Derivative method - Raw data with slope visualization
    ax1 = axes[0, 0]
    ax1.scatter(radii, areas, alpha=0.5, s=10, c='steelblue')
    
    # Get derivative data
    _, slopes_smooth, slope_positions, significant_changes = count_files_by_derivative(df_all)
    
    if slopes_smooth is not None and len(slopes_smooth) > 0:
        # Create second y-axis for slopes
        ax1_twin = ax1.twinx()
        ax1_twin.plot(slope_positions, slopes_smooth, 'r-', linewidth=2, alpha=0.7)
        ax1_twin.set_ylabel('Growth Rate (dA/dR)', color='red', fontsize=10)
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        # Mark significant changes
        if significant_changes is not None:
            change_positions = [(slope_positions[i] + slope_positions[i+1])/2 
                               for i in range(len(slope_positions)-1)]
            for i, is_change in enumerate(significant_changes):
                if is_change and i < len(change_positions):
                    ax1.axvline(x=change_positions[i], color='green', 
                               linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Radius (pixels)')
    ax1.set_ylabel('Cell Area (pixels²)')
    ax1.set_title(f'Derivative Method: {results["derivative"]} files\n(Green lines = file boundaries)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative method - CDF and derivative
    ax2 = axes[0, 1]
    
    # Calculate cumulative
    radii_cumul = np.array(sorted(radii))
    cumulative = np.arange(1, len(radii_cumul) + 1) / len(radii_cumul)
    ax2.plot(radii_cumul, cumulative, 'b-', linewidth=2, label='CDF')
    
    # Get cumulative data
    n_clusters, diff_radii, diffs_smooth, peaks = count_files_by_cumulative(df_all)
    
    if len(diff_radii) > 0:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(diff_radii, diffs_smooth, 'r-', linewidth=2, alpha=0.7, label='Derivative')
        ax2_twin.set_ylabel('Derivative', color='red', fontsize=10)
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # Mark peaks
        if peaks is not None and len(peaks) > 0:
            for peak_idx in peaks:
                if peak_idx < len(diff_radii):
                    ax2.axvline(x=diff_radii[peak_idx], color='green', 
                               linestyle='--', alpha=0.5, linewidth=1)
                    ax2.text(diff_radii[peak_idx], 0.5, f'Peak', 
                            rotation=90, fontsize=8, color='green')
    
    ax2.set_xlabel('Radius (pixels)')
    ax2.set_ylabel('Cumulative Proportion')
    ax2.set_title(f'Cumulative Method: {results["cumulative"]} files\n(Green lines = peaks = file boundaries)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Comparison of the two methods
    ax3 = axes[1, 0]
    methods = ['Derivative', 'Cumulative', 'Consensus']
    values = [results['derivative'], results['cumulative'], results['consensus']]
    colors_method = ['steelblue' if v == results['consensus'] else 'lightgray' for v in values]
    bars = ax3.bar(methods, values, color=colors_method, edgecolor='black')
    ax3.set_ylabel('Number of Cell Files')
    ax3.set_title('Method Comparison')
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Final consensus assignment visualization
    ax4 = axes[1, 1]
    
    # Show colored cells by consensus file number
    if 'cell_file_consensus' in df_all.columns:
        valid_cells = df_all[df_all['cell_file_consensus'] >= 0]
        if len(valid_cells) > 0:
            max_file = valid_cells['cell_file_consensus'].max()
            if max_file <= 20:
                colormap = cm.get_cmap('tab20', max_file + 2)
            else:
                colormap = cm.get_cmap('gist_ncar', max_file + 2)
            
            # Sample points for scatter (to avoid overcrowding)
            sample_size = min(500, len(valid_cells))
            sampled = valid_cells.sample(n=sample_size) if len(valid_cells) > sample_size else valid_cells
            
            for _, cell in sampled.iterrows():
                file_num = cell['cell_file_consensus']
                color = colormap(file_num % colormap.N)
                ax4.scatter(cell['radius_pixels'], cell['area_pixels'], 
                          c=[color], alpha=0.6, s=15)
    
    ax4.set_xlabel('Radius (pixels)')
    ax4.set_ylabel('Cell Area (pixels²)')
    ax4.set_title(f'Consensus Assignment: {results["consensus"]} files\n(Colors = file numbers)')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Cell File Detection Analysis - {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{image_name}_method_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    

def count_cell_files_for_image(csv_path, image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    filename = os.path.basename(image_path)
    center_x, center_y, quadrant = get_plant_center_from_filename(filename, w, h)
    
    if center_x is None:
        return None
    
    # Load ALL cells
    df_all = load_all_cells(csv_path, center_x, center_y)
    
    if df_all is None or df_all.empty:
        return None
    
    print(f"Loaded {len(df_all)} total cells")
    
    results = {}
    
    # Method: Derivative (Growth Rate Analysis) - ONLY METHOD FOR EFFICIENCY
    n_deriv, _, _, _ = count_files_by_derivative(df_all)
    results['derivative'] = n_deriv
    df_all = assign_files_by_consensus(df_all, n_deriv, 'derivative')
    
    # Store only derivative method for efficiency (skipping cumulative and consensus)
    results['cumulative'] = n_deriv  # Store same as derivative for compatibility
    results['consensus'] = n_deriv   # Use derivative as consensus for efficiency
    
    # Cell statistics
    results['mean_cell_area'] = df_all['area_pixels'].mean()
    results['std_cell_area'] = df_all['area_pixels'].std()
    results['median_cell_area'] = df_all['area_pixels'].median()
    results['total_cells'] = len(df_all)
    results['radius_range'] = (df_all['radius_pixels'].min(), df_all['radius_pixels'].max())
    
    return results, df_all


def visualize_all_methods_on_image(image_path, df_all, results, output_dir):
    image_name = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
    
    # Only visualize derivative method for efficiency - skip explanation and analysis plots
    method_name = 'Derivative (Growth Rate)'
    col_name = 'derivative'
    n_files = results['derivative']
    
    # Create visualization
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    center_x, center_y, _ = get_plant_center_from_filename(os.path.basename(image_path), w, h)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_rgb)
    
    cell_file_col = f'cell_file_{col_name}'
    if cell_file_col in df_all.columns:
        valid_cells = df_all[df_all[cell_file_col] >= 0]
        if len(valid_cells) > 0:
            max_file = valid_cells[cell_file_col].max()
            if max_file <= 20:
                colormap = cm.get_cmap('tab20', max_file + 2)
            else:
                colormap = cm.get_cmap('gist_ncar', max_file + 2)
            
            circle_size = 3
            for _, cell in valid_cells.iterrows():
                file_num = cell[cell_file_col]
                color = colormap(file_num % colormap.N)
                circle = Circle((cell['x_pixels'], cell['y_pixels']), circle_size, 
                               color=color, alpha=0.7, linewidth=0.5, fill=True)
                ax.add_patch(circle)
    
    if center_x is not None:
        ax.plot(center_x, center_y, 'b*', markersize=15)
    
    ax.set_title(f'{method_name} - {n_files} Cell Files\n({len(df_all)} total cells)', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{image_name}_cell_files.png'), dpi=150, bbox_inches='tight')
    plt.close()
    

def batch_count_cell_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    csv_files = glob.glob(str(Path(MEASUREMENTS_FOLDER) / "*_measurements.csv"))
    
    all_results = []
    
    for csv_path in csv_files:
        image_name = Path(csv_path).stem.replace('_measurements', '')
        
        # Find image
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            test_path = os.path.join(IMAGE_FOLDER, f"{image_name}{ext}")
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        if image_path is None:
            continue
        
        results, df_all = count_cell_files_for_image(csv_path, image_path)
        
        if results is None:
            print(f"  Failed to process {image_name}")
            continue
        
        # Create output directory
        img_output_dir = os.path.join(OUTPUT_DIR, image_name)
        os.makedirs(img_output_dir, exist_ok=True)
        
        # Create visualizations
        if CREATE_VISUALIZATIONS:
            visualize_all_methods_on_image(image_path, df_all, results, img_output_dir)
        
        # Save the full data
        df_all.to_csv(os.path.join(img_output_dir, 'cell_assignments.csv'), index=False)
        
        all_results.append({
            'image': image_name,
            'total_cells': results['total_cells'],
            'derivative': results['derivative'],
            'cumulative': results['cumulative'],
            'consensus': results['consensus'],
            'mean_area': results['mean_cell_area'],
            'std_area': results['std_cell_area'],
            'median_area': results['median_cell_area']
        })
    
    # Create summary across all images
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, 'all_images_summary.csv'), index=False)
        
        # Correlation analysis
        corr = stats.pearsonr(summary_df['consensus'], summary_df['mean_area'])
        print(f"\nCorrelation between cell files and cell size: r={corr[0]:.3f} (p={corr[1]:.4f})")
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot both methods
        ax.scatter(summary_df['consensus'], summary_df['mean_area'], 
                  s=120, alpha=0.7, c='steelblue', edgecolors='black', label='Consensus')
        
        # Add error bars showing range between the two methods
        for _, row in summary_df.iterrows():
            ax.errorbar(row['consensus'], row['mean_area'],
                       xerr=[[row['consensus'] - min(row['derivative'], row['cumulative'])],
                             [max(row['derivative'], row['cumulative']) - row['consensus']]],
                       fmt='none', ecolor='gray', alpha=0.5, capsize=3)
        
        # Add trend line
        if len(summary_df) > 2:
            z = np.polyfit(summary_df['consensus'], summary_df['mean_area'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(summary_df['consensus'].min(), summary_df['consensus'].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, 
                   label=f'Trend: slope={z[0]:.1f}')
        
        ax.set_xlabel('Number of Cell Files (Consensus)', fontsize=12)
        ax.set_ylabel('Mean Cell Area (pixels²)', fontsize=12)
        ax.set_title(f'Relationship: Cell Files vs Cell Size (n={len(summary_df)} images)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add image labels
        for _, row in summary_df.iterrows():
            ax.annotate(row['image'], (row['consensus'], row['mean_area']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'cell_files_vs_cell_size.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    return all_results


def process_single_image():
    if not os.path.exists(SINGLE_CSV_PATH) or not os.path.exists(SINGLE_IMAGE_PATH):
        print("Files not found")
        return None
    
    results, df_all = count_cell_files_for_image(SINGLE_CSV_PATH, SINGLE_IMAGE_PATH)
    
    if results is None:
        print("Failed to process image")
        return None
    
    # Create output directory
    image_name = os.path.basename(SINGLE_IMAGE_PATH).replace('.jpg', '').replace('.png', '')
    img_output_dir = os.path.join(OUTPUT_DIR, image_name)
    os.makedirs(img_output_dir, exist_ok=True)
    
    # Create visualizations
    visualize_all_methods_on_image(SINGLE_IMAGE_PATH, df_all, results, img_output_dir)
    
    return results


def batch_analyze_cell_files(
    measurements_folder, 
    output_folder, 
    method='derivative', 
    specific_images=None,
    image_folder=None,
    force_rebuild=False
):
    # Analyze cell files for images.
    import glob
    from pathlib import Path
    
    # If image_folder is not provided, try to find it
    if image_folder is None:
        possible_folders = [
            Path(measurements_folder).parent / 'root_results' / 'detections',
            Path(measurements_folder).parent.parent / 'root_results' / 'detections',
            Path(RESULTS_FOLDER / 'root_results' / 'detections'),
            Path(RESULTS_FOLDER / 'root_results' / 'detections'),
            Path(measurements_folder)  # Also check measurements_all for _centers.jpg
        ]
        for folder in possible_folders:
            if folder.exists():
                image_folder = str(folder)
                break
        
        if image_folder is None:
            print(f"⚠️  Warning: Could not find image folder. Using default: {IMAGE_FOLDER}")
            image_folder = IMAGE_FOLDER
    
    os.makedirs(output_folder, exist_ok=True)
    
    csv_files = glob.glob(str(Path(measurements_folder) / "*_measurements.csv"))
    
    # Extract all image names from CSV files
    all_image_names = [Path(csv_file).stem.replace('_measurements', '') for csv_file in csv_files]
    
    # If specific_images is provided, filter to only those images
    if specific_images is not None:
        if not specific_images:
            print(f"No new images to process for cell file analysis (specific_images is empty)")
            return []
        specific_set = set(specific_images)
        images_to_process = [img for img in all_image_names if img in specific_set]
        print(f"Filtering to {len(images_to_process)} specific new images for cell file analysis (from {len(all_image_names)} total)")
    else:
        images_to_process = all_image_names
        print(f"Processing all {len(images_to_process)} images for cell file analysis")
    
    # If not force_rebuild, check which images already have cell file data
    if not force_rebuild:
        existing_data = set()
        for img in images_to_process:
            # Check for derivative method file
            output_path = os.path.join(output_folder, img, f"{img}_method_derivative.csv")
            if os.path.exists(output_path):
                existing_data.add(img)
            # Also check for cell_assignments.csv (legacy)
            elif os.path.exists(os.path.join(output_folder, img, "cell_assignments.csv")):
                existing_data.add(img)
        
        if existing_data:
            print(f"  Skipping {len(existing_data)} images that already have cell file data")
            images_to_process = [img for img in images_to_process if img not in existing_data]
    
    if not images_to_process:
        print("No new images need cell file analysis")
        return []
    
    print(f"Processing {len(images_to_process)} images for cell file analysis...")
    
    all_results = []
    
    for image_name in tqdm(images_to_process, desc="Analyzing cell files"):
        csv_path = os.path.join(measurements_folder, f"{image_name}_measurements.csv")
        
        if not os.path.exists(csv_path):
            print(f"  Warning: CSV not found for {image_name}")
            continue
        
        # Find the image in the image_folder
        image_path = None
        
        # First try: {image_name}_centers.jpg (if image_folder is measurements_all)
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = os.path.join(image_folder, f"{image_name}_centers{ext}")
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        # Second try: {image_name}.jpg (if image_folder is data/chosen_results)
        if image_path is None:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                test_path = os.path.join(image_folder, f"{image_name}{ext}")
                if os.path.exists(test_path):
                    image_path = test_path
                    break
        
        if image_path is None:
            # Debug: print what we're looking for
            print(f"  Warning: Could not find image for {image_name} in {image_folder}")
            print(f"    Tried: {image_name}_centers.jpg and {image_name}.jpg")
            continue
        
        # Run the analysis
        results, df_all = count_cell_files_for_image(csv_path, image_path)
        
        if results is None:
            print(f"  Failed to process {image_name}")
            continue
        
        print(f"  Processed {image_name}: {results['derivative']} files, {results['total_cells']} cells")
        
        # Create output directory
        img_output_dir = os.path.join(output_folder, image_name)
        os.makedirs(img_output_dir, exist_ok=True)
        
        # Save the full data
        df_all.to_csv(os.path.join(img_output_dir, 'cell_assignments.csv'), index=False)
        
        # Also save method-specific file for compatibility
        if method == 'derivative':
            # Save derivative-specific columns
            derivative_cols = ['cell_file_derivative'] + [c for c in df_all.columns if 'derivative' in c.lower()]
            existing_cols = [c for c in derivative_cols if c in df_all.columns]
            if existing_cols:
                df_all[existing_cols].to_csv(os.path.join(img_output_dir, f'{image_name}_method_derivative.csv'), index=False)
        
        # Create visualizations if enabled
        if CREATE_VISUALIZATIONS and image_path is not None:
            try:
                visualize_all_methods_on_image(image_path, df_all, results, img_output_dir)
            except Exception as e:
                print(f"  Warning: Could not create visualization for {image_name}: {e}")
        
        all_results.append({
            'image': image_name,
            'total_cells': results['total_cells'],
            'derivative': results['derivative'],
            'cumulative': results['cumulative'],
            'consensus': results['consensus'],
            'mean_area': results['mean_cell_area'],
            'std_area': results['std_cell_area'],
            'median_area': results['median_cell_area']
        })
    
    # Create summary across all images
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(os.path.join(output_folder, 'all_images_summary.csv'), index=False)
        print(f"✅ Processed {len(all_results)} new images successfully")
    
    return all_results


if __name__ == "__main__":
    if MODE == 'single':
        process_single_image()
    elif MODE == 'batch':
        batch_count_cell_files()
        
    print("\nAnalysis complete")