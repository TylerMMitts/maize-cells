import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
import cv2
from pathlib import Path
import sys

# Get project root
SCRIPT_DIR = Path(__file__).parent.absolute()  # code/measurements/
CODE_DIR = SCRIPT_DIR.parent  # code/
PROJECT_ROOT = CODE_DIR.parent  # project root

def find_measurement_files(measurements_dir=None):
    if measurements_dir is None:
        measurements_dir = str(PROJECT_ROOT / "results" / "measurements_all")
    pattern = os.path.join(measurements_dir, "*_measurements.csv")
    files = glob.glob(pattern)
    return sorted(files)


def find_center_image(measurements_dir, base_name):
    img_path = os.path.join(measurements_dir, f"{base_name}_centers.jpg")
    if os.path.exists(img_path):
        return img_path
    return None


def get_angle_range_from_filename(filename):
    filename_upper = filename.upper()
    
    if 'BL' in filename_upper:
        return 90, 180, 'Bottom Left', 'BL (90°-180°)'
    elif 'BR' in filename_upper:
        return 0, 90, 'Bottom Right', 'BR (0°-90°)'
    elif 'TL' in filename_upper:
        return 180, 270, 'Top Left', 'TL (180°-270°)'
    elif 'TR' in filename_upper:
        return 270, 360, 'Top Right', 'TR (270°-360°)'
    else:
        return 0, 360, 'Full Circle', 'Full Circle (0°-360°)'


def load_measurement_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def parse_neighbors(neighbors_str):
    if pd.isna(neighbors_str) or neighbors_str == '':
        return []
    return [int(x) for x in str(neighbors_str).split(',')]


def build_neighbor_edges(df):
    edges = []
    
    # Create a dictionary mapping cell_id to coordinates
    id_to_coords = {}
    for _, row in df.iterrows():
        cell_id = int(row['cell_id'])
        x = row['x_pixels']
        y = row['y_pixels']
        id_to_coords[cell_id] = (x, y)
    
    # Build edges from neighbor lists
    for _, row in df.iterrows():
        cell_id = int(row['cell_id'])
        neighbors = parse_neighbors(row['neighbors'])
        
        if cell_id in id_to_coords:
            x1, y1 = id_to_coords[cell_id]
            for neighbor_id in neighbors:
                if neighbor_id in id_to_coords:
                    x2, y2 = id_to_coords[neighbor_id]
                    # Add edge only once (when cell_id < neighbor_id)
                    if cell_id < neighbor_id:
                        edges.append((x1, y1, x2, y2))
    
    return edges


def compute_binned_degree_by_area(df, num_bins=20):
    df_clean = df.dropna(subset=['area_pixels', 'degree'])
    
    if len(df_clean) == 0:
        return None, None, None, None
    
    area_min = df_clean['area_pixels'].min()
    area_max = df_clean['area_pixels'].max()
    
    bins = np.linspace(area_min, area_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    avg_degree = []
    std_degree = []
    counts = []
    
    for i in range(num_bins):
        bin_mask = (df_clean['area_pixels'] >= bins[i]) & (df_clean['area_pixels'] < bins[i + 1])
        cells_in_bin = df_clean[bin_mask]
        
        if len(cells_in_bin) > 0:
            avg_degree.append(cells_in_bin['degree'].mean())
            std_degree.append(cells_in_bin['degree'].std())
            counts.append(len(cells_in_bin))
        else:
            avg_degree.append(np.nan)
            std_degree.append(np.nan)
            counts.append(0)
    
    return bin_centers, np.array(avg_degree), np.array(std_degree), np.array(counts)


def compute_rolling_average_degree(df, window_size=50):
    df_clean = df.dropna(subset=['area_pixels', 'degree'])
    df_sorted = df_clean.sort_values('area_pixels')
    
    sorted_areas = df_sorted['area_pixels'].values
    degrees = df_sorted['degree'].values
    
    rolling_degree = np.zeros_like(degrees)
    for i in range(len(degrees)):
        start = max(0, i - window_size // 2)
        end = min(len(degrees), i + window_size // 2 + 1)
        rolling_degree[i] = np.mean(degrees[start:end])
    
    return sorted_areas, rolling_degree


def compute_correlation_stats(df):
    df_clean = df.dropna(subset=['area_pixels', 'degree'])
    
    if len(df_clean) < 3:
        return None
    
    pearson_r, pearson_p = pearsonr(df_clean['area_pixels'], df_clean['degree'])
    spearman_r, spearman_p = spearmanr(df_clean['area_pixels'], df_clean['degree'])
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_cells': len(df_clean)
    }


def plot_degree_vs_area_single(df, image_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scatter plot
    ax1 = axes[0]
    scatter = ax1.scatter(df['area_pixels'], df['degree'], 
                         c=df['radius_pixels'], cmap='viridis', 
                         alpha=0.6, s=20)
    ax1.set_xlabel('Cell Area (pixels²)')
    ax1.set_ylabel('Degree (Number of Neighbors)')
    ax1.set_title(f'{image_name}\nScatter Plot (color = radius)')
    plt.colorbar(scatter, ax=ax1, label='Radius (pixels)')
    
    # Binned average
    ax2 = axes[1]
    bin_centers, avg_degree, std_degree, counts = compute_binned_degree_by_area(df, num_bins=15)
    
    if bin_centers is not None:
        ax2.errorbar(bin_centers, avg_degree, yerr=std_degree, 
                    fmt='o-', capsize=3, color='blue', alpha=0.7)
        ax2.set_xlabel('Cell Area (pixels²)')
        ax2.set_ylabel('Average Degree')
        ax2.set_title('Binned Average Degree vs Area')
        ax2.grid(True, alpha=0.3)
    
    # Rolling average
    ax3 = axes[2]
    sorted_areas, rolling_degree = compute_rolling_average_degree(df, window_size=30)
    ax3.plot(sorted_areas, rolling_degree, 'r-', linewidth=2)
    ax3.set_xlabel('Cell Area (pixels²)')
    ax3.set_ylabel('Rolling Average Degree')
    ax3.set_title('Rolling Average Degree (window=30 cells)')
    ax3.grid(True, alpha=0.3)
    
    stats = compute_correlation_stats(df)
    if stats:
        fig.suptitle(f'{image_name} | Pearson r={stats["pearson_r"]:.3f} (p={stats["pearson_p"]:.3e}) | '
                    f'Spearman ρ={stats["spearman_r"]:.3f} | n={stats["n_cells"]}', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{image_name}_degree_vs_area.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return stats


def create_neighbor_graph_on_center_image(df, center_image_path, image_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the center image (already has red dots)
    img = cv2.imread(center_image_path)
    if img is None:
        print(f"Could not load image {center_image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Display the center image
    ax.imshow(img_rgb, aspect='auto')
    
    # Get quadrant info for title
    angle_min, angle_max, quadrant_name, display_name = get_angle_range_from_filename(image_name)
    
    # Build edges from neighbor lists
    edges = build_neighbor_edges(df)
    
    # Draw edges (neighbor connections) as cyan lines (visible over red dots)
    for x1, y1, x2, y2 in edges:
        ax.plot([x1, x2], [y1, y2], 'cyan', linewidth=1.5, alpha=0.7, zorder=1)
    
    # Add degree labels next to each cell (offset slightly to not cover the red dot)
    for _, row in df.iterrows():
        ax.annotate(str(int(row['degree'])), 
                   (row['x_pixels'] + 5, row['y_pixels'] - 5),
                   textcoords="offset points", xytext=(0, 0),
                   ha='left', va='bottom', fontsize=7, color='yellow', 
                   fontweight='bold', zorder=3,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'{image_name} ({display_name})\nNeighbor Network on Cell Centers\n'
                f'Cyan lines = Neighbor connections | Yellow numbers = Degree')
    ax.set_aspect('auto')
    ax.grid(True, alpha=0.2)
    
    # Add statistics box
    stats_text = f"Total cells: {len(df)}\n"
    stats_text += f"Avg degree: {df['degree'].mean():.2f}\n"
    stats_text += f"Max degree: {df['degree'].max()}\n"
    stats_text += f"Min degree: {df['degree'].min()}\n"
    stats_text += f"Total edges: {len(edges)}\n"
    stats_text += f"Quadrant: {display_name}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{image_name}_neighbor_network.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_degree_network_on_center_image(df, center_image_path, image_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the center image
    img = cv2.imread(center_image_path)
    if img is None:
        print(f"Could not load image {center_image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Display the center image
    ax.imshow(img_rgb, aspect='auto')
    
    # Get quadrant info for title
    angle_min, angle_max, quadrant_name, display_name = get_angle_range_from_filename(image_name)
    
    # Build edges
    edges = build_neighbor_edges(df)
    
    # Normalize area for node sizing
    area_min = df['area_pixels'].min()
    area_max = df['area_pixels'].max()
    if area_max > area_min:
        node_sizes = 30 + 100 * (df['area_pixels'] - area_min) / (area_max - area_min)
    else:
        node_sizes = 50 * np.ones(len(df))
    
    # Draw edges first
    for x1, y1, x2, y2 in edges:
        ax.plot([x1, x2], [y1, y2], 'cyan', linewidth=1.2, alpha=0.5, zorder=1)
    
    # Draw nodes - size by area, color by degree
    scatter = ax.scatter(df['x_pixels'], df['y_pixels'], 
                        c=df['degree'], cmap='plasma', 
                        s=node_sizes, alpha=0.6, edgecolors='white', linewidth=0.5, zorder=2)
    
    plt.colorbar(scatter, ax=ax, label='Degree (Number of Neighbors)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'{image_name} ({display_name})\nDegree Network on Cell Centers\n'
                f'Circle size = Cell area | Color = Degree | Cyan lines = Neighbor connections')
    ax.set_aspect('auto')
    ax.grid(True, alpha=0.2)
    
    # Add legend for node sizes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='Small area'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Medium area'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=15, label='Large area'),
        Line2D([0], [0], color='cyan', linewidth=1.5, label='Neighbor connection')
    ]
    ax.legend(handles=legend_elements, loc='upper right', title='Legend')
    
    # Add statistics box
    stats_text = f"Total cells: {len(df)}\n"
    stats_text += f"Avg degree: {df['degree'].mean():.2f}\n"
    stats_text += f"Max degree: {df['degree'].max()}\n"
    stats_text += f"Min degree: {df['degree'].min()}\n"
    stats_text += f"Quadrant: {display_name}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{image_name}_degree_network.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_degree_vs_area(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: All scatter plots overlaid
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for (image_name, df), color in zip(all_results.items(), colors):
        df_clean = df.dropna(subset=['area_pixels', 'degree'])
        if len(df_clean) > 0:
            sample = df_clean.sample(min(500, len(df_clean)))
            ax1.scatter(sample['area_pixels'], sample['degree'], 
                       s=10, alpha=0.4, color=color, label=image_name[:20])
    
    ax1.set_xlabel('Cell Area (pixels²)')
    ax1.set_ylabel('Degree')
    ax1.set_title('All Images: Degree vs Area')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Binned average comparison
    ax2 = axes[0, 1]
    
    for (image_name, df), color in zip(all_results.items(), colors):
        bin_centers, avg_degree, _, _ = compute_binned_degree_by_area(df, num_bins=15)
        if bin_centers is not None:
            ax2.plot(bin_centers, avg_degree, 'o-', color=color, 
                    markersize=4, linewidth=1, alpha=0.7, label=image_name[:20])
    
    ax2.set_xlabel('Cell Area (pixels²)')
    ax2.set_ylabel('Average Degree')
    ax2.set_title('Binned Average Degree vs Area (All Images)')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correlation coefficients bar chart
    ax3 = axes[1, 0]
    
    correlations = []
    for image_name, df in all_results.items():
        stats = compute_correlation_stats(df)
        if stats:
            correlations.append({
                'image': image_name[:30],
                'pearson_r': stats['pearson_r'],
                'spearman_r': stats['spearman_r'],
                'n_cells': stats['n_cells']
            })
    
    if correlations:
        corr_df = pd.DataFrame(correlations)
        x = np.arange(len(corr_df))
        width = 0.35
        
        ax3.bar(x - width/2, corr_df['pearson_r'], width, label='Pearson r', color='blue', alpha=0.7)
        ax3.bar(x + width/2, corr_df['spearman_r'], width, label='Spearman ρ', color='red', alpha=0.7)
        ax3.set_xlabel('Image')
        ax3.set_ylabel('Correlation Coefficient')
        ax3.set_title('Area vs Degree Correlation by Image')
        ax3.set_xticks(x)
        ax3.set_xticklabels(corr_df['image'], rotation=45, ha='right', fontsize=8)
        ax3.legend()
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if correlations:
        summary_text = "Summary Statistics\n" + "\n\n"
        for corr in correlations[:10]:
            summary_text += f"{corr['image']}:\n"
            summary_text += f"  Pearson r = {corr['pearson_r']:.3f}\n"
            summary_text += f"  Spearman ρ = {corr['spearman_r']:.3f}\n"
            summary_text += f"  n = {corr['n_cells']}\n\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=8, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Degree vs Area Analysis Across All Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "combined_degree_vs_area.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved combined plot: {output_path}")


def create_degree_distribution_by_radius(df, image_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get quadrant info
    angle_min, angle_max, quadrant_name, display_name = get_angle_range_from_filename(image_name)
    
    # Scatter: degree vs radius
    ax1 = axes[0]
    scatter = ax1.scatter(df['radius_pixels'], df['degree'], 
                         c=df['area_pixels'], cmap='hot', 
                         alpha=0.6, s=20)
    ax1.set_xlabel('Radius (pixels)')
    ax1.set_ylabel('Degree')
    ax1.set_title(f'{image_name} ({display_name})\nDegree vs Radius (color = area)')
    plt.colorbar(scatter, ax=ax1, label='Area (pixels²)')
    ax1.grid(True, alpha=0.3)
    
    # Binned degree by radius
    ax2 = axes[1]
    num_radial_bins = 15
    radius_min = df['radius_pixels'].min()
    radius_max = df['radius_pixels'].max()
    radius_bins = np.linspace(radius_min, radius_max, num_radial_bins + 1)
    radius_centers = (radius_bins[:-1] + radius_bins[1:]) / 2
    
    avg_degree = []
    std_degree = []
    
    for i in range(num_radial_bins):
        bin_mask = (df['radius_pixels'] >= radius_bins[i]) & (df['radius_pixels'] < radius_bins[i + 1])
        cells_in_bin = df[bin_mask]
        
        if len(cells_in_bin) > 0:
            avg_degree.append(cells_in_bin['degree'].mean())
            std_degree.append(cells_in_bin['degree'].std())
        else:
            avg_degree.append(np.nan)
            std_degree.append(np.nan)
    
    ax2.errorbar(radius_centers, avg_degree, yerr=std_degree, 
                fmt='o-', capsize=3, color='blue', alpha=0.7)
    ax2.set_xlabel('Radius (pixels)')
    ax2.set_ylabel('Average Degree')
    ax2.set_title(f'Average Degree by Radius ({display_name})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{image_name}_degree_vs_radius.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_heatmap_degree_spatial(df, image_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get quadrant info
    angle_min, angle_max, quadrant_name, display_name = get_angle_range_from_filename(image_name)
    
    scatter = ax.scatter(df['x_pixels'], df['y_pixels'], 
                        c=df['degree'], cmap='viridis', 
                        s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.colorbar(scatter, ax=ax, label='Degree (Number of Neighbors)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'{image_name} ({display_name})\nSpatial Distribution of Cell Degree')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{image_name}_degree_spatial.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_report(all_stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    summary_df = pd.DataFrame(all_stats)
    output_path = os.path.join(output_dir, "degree_correlation_summary.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved summary report: {output_path}")

    return summary_df


def find_existing_processed_images(output_dir):

    if not os.path.exists(output_dir):
        return set()
    existing_plots = glob.glob(os.path.join(output_dir, "*_degree_vs_area.png"))
    return {os.path.basename(f).replace('_degree_vs_area.png', '') for f in existing_plots}


def analyze_all_measurements(measurements_dir=None, output_dir=None, force_rebuild=False):

    if measurements_dir is None:
        measurements_dir = str(PROJECT_ROOT / "results" / "measurements_all")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "results" / "degree_analysis")

    os.makedirs(output_dir, exist_ok=True)

    csv_files = find_measurement_files(measurements_dir)

    if not csv_files:
        print(f"No measurement files found in {measurements_dir}")
        return None, None

    existing_processed = set() if force_rebuild else find_existing_processed_images(output_dir)
    if existing_processed:
        print(f"Found {len(existing_processed)} images with existing per-image plots")
    newly_processed = set()

    all_results = {}
    all_stats = []

    for csv_path in csv_files:
        # Load data
        df = load_measurement_data(csv_path)
        image_name = os.path.basename(csv_path).replace("_measurements.csv", "")

        # Get quadrant info
        angle_min, angle_max, quadrant_name, display_name = get_angle_range_from_filename(image_name)

        df_valid = df.dropna(subset=['degree'])

        if len(df_valid) < 10:
            print(f"Not enough cells with degree data")
            continue

        all_results[image_name] = df_valid

        stats = compute_correlation_stats(df_valid)
        if stats:
            stats['image'] = image_name
            stats['quadrant'] = display_name
            all_stats.append(stats)
            print(f"Pearson r: {stats['pearson_r']:.3f} (p={stats['pearson_p']:.3e})")
            print(f"Spearman ρ: {stats['spearman_r']:.3f}")

        if image_name in existing_processed:
            continue

        # Generate individual plots (only for images not already plotted)
        plot_degree_vs_area_single(df_valid, image_name, output_dir)
        create_degree_distribution_by_radius(df_valid, image_name, output_dir)
        create_heatmap_degree_spatial(df_valid, image_name, output_dir)

        # Find the center image and overlay network
        center_image_path = find_center_image(measurements_dir, image_name)
        if center_image_path and os.path.exists(center_image_path):
            create_neighbor_graph_on_center_image(df_valid, center_image_path, image_name, output_dir)
            create_degree_network_on_center_image(df_valid, center_image_path, image_name, output_dir)
        else:
            print(f"Center image not found for {image_name}")

        newly_processed.add(image_name)

    print(f"\nGenerated new per-image plots for {len(newly_processed)} images "
          f"({len(existing_processed)} already had plots)")

    if all_results:
        plot_combined_degree_vs_area(all_results, output_dir)
        summary_df = create_summary_report(all_stats, output_dir)

        if summary_df is not None:
            cols = ['image', 'quadrant', 'pearson_r', 'spearman_r', 'n_cells']
            available_cols = [c for c in cols if c in summary_df.columns]
            print(summary_df[available_cols].to_string(index=False))

    print(f"Results saved to: {output_dir}")

    return all_results, all_stats


def plot_degree_distribution_histogram(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for image_name, df in all_results.items():
        if len(df) > 0:
            ax.hist(df['degree'], bins=range(0, int(df['degree'].max()) + 2), 
                   alpha=0.5, label=image_name[:20], density=True)
    
    ax.set_xlabel('Degree (Number of Neighbors)')
    ax.set_ylabel('Density')
    ax.set_title('Degree Distribution Across All Images')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "degree_distribution_histogram.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved degree distribution histogram: {output_path}")


def main():
    # Parse command line arguments (pipeline.py sets sys.argv before calling this)
    measurements_dir = None
    output_dir = None
    force_rebuild = '--force-rebuild' in sys.argv or '-f' in sys.argv

    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        measurements_dir = sys.argv[1]
    if len(sys.argv) > 2 and not sys.argv[2].startswith('-'):
        output_dir = sys.argv[2]

    results, stats = analyze_all_measurements(measurements_dir, output_dir, force_rebuild=force_rebuild)

    if results:
        plot_degree_distribution_histogram(results,
                                          output_dir or str(PROJECT_ROOT / "results" / "degree_analysis"))


if __name__ == "__main__":
    main()