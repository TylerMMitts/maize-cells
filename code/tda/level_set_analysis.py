import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import gudhi as gd
from scipy import ndimage
from scipy.ndimage import zoom
import glob


def find_density_files(measurements_dir="measurements"):
    pattern = os.path.join(measurements_dir, "density_analysis", "*", "polar_density_grid.csv")
    files = glob.glob(pattern)
    return sorted(files)


def load_density_grid(csv_path):
    df = pd.read_csv(csv_path, header=None)
    return df.values


def load_metadata(density_csv_path):
    base_dir = os.path.dirname(density_csv_path)
    
    radial_csv = os.path.join(base_dir, "radial_profile.csv")
    angular_csv = os.path.join(base_dir, "angular_profile.csv")
    
    radial_centers = None
    angular_centers = None
    
    if os.path.exists(radial_csv):
        radial_df = pd.read_csv(radial_csv)
        radial_centers = radial_df['radius_pixels'].values
    
    if os.path.exists(angular_csv):
        angular_df = pd.read_csv(angular_csv)
        angular_centers = angular_df['angle_degrees'].values
    
    return radial_centers, angular_centers


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


def upsample_grid(grid, factor=10):
    zoom_factor = (factor, factor)
    upsampled = zoom(grid.astype(float), zoom_factor, order=3)
    return upsampled


def compute_persistence_diagram(density_grid, use_periodic_angle=True):
    grid = density_grid.astype(np.float64).copy()
    
    np.random.seed(42)
    grid = grid + np.random.randn(*grid.shape) * 1e-8
    
    try:
        if use_periodic_angle:
            cc = gd.PeriodicCubicalComplex(
                top_dimensional_cells=grid,
                periodic_dimensions=[True, False]
            )
        else:
            cc = gd.CubicalComplex(top_dimensional_cells=grid)
        
        persistence = cc.persistence()
        
        diagrams = {}
        for interval in persistence:
            dim = interval[0]
            birth = interval[1][0]
            death = interval[1][1]
            
            if np.isinf(birth) or np.isinf(death) or np.isnan(birth) or np.isnan(death):
                continue
            if birth == death:
                continue
                
            if dim not in diagrams:
                diagrams[dim] = []
            diagrams[dim].append({
                'birth': birth,
                'death': death,
                'persistence': death - birth
            })
        
        return persistence, diagrams, cc
    
    except Exception as e:
        print(f"        GUDHI error: {e}")
        return [], {}, None


def visualize_persistence_diagram(diagrams, output_path, title="Persistence Diagram"):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    labels = {0: 'H0 (Connected Components)', 1: 'H1 (Loops/Holes)', 2: 'H2 (Cavities)'}
    
    has_features = False
    all_births = []
    all_deaths = []
    
    for dim, intervals in diagrams.items():
        if len(intervals) > 0:
            has_features = True
            births = [i['birth'] for i in intervals]
            deaths = [i['death'] for i in intervals]
            all_births.extend(births)
            all_deaths.extend(deaths)
            
            ax.scatter(births, deaths, c=colors.get(dim, 'gray'), 
                      s=80, alpha=0.7, label=labels.get(dim, f'H{dim}'), 
                      zorder=5, edgecolors='black', linewidth=0.5)
    
    if not has_features:
        ax.text(0.5, 0.5, 'No persistent features detected', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        max_val = max(max(all_births), max(all_deaths))
        min_val = min(min(all_births), min(all_deaths))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Death = Birth', linewidth=2)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
    
    ax.set_xlabel('Birth Threshold', fontsize=12)
    ax.set_ylabel('Death Threshold', fontsize=12)
    ax.set_title(f'{title}\nPoints above diagonal = persistent features', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"        Persistence diagram saved to: {output_path}")


def visualize_persistence_barcode(diagrams, output_path, title="Persistence Barcode"):
    if len(diagrams) == 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.text(0.5, 0.5, 'No persistent features detected', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"        No persistent features to plot")
        return
    
    n_dims = len(diagrams)
    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 5 * n_dims))
    
    if n_dims == 1:
        axes = [axes]
    
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    labels = {0: 'H0 (Connected Components)', 1: 'H1 (Loops/Holes)', 2: 'H2 (Cavities)'}
    
    for idx, (dim, intervals) in enumerate(sorted(diagrams.items())):
        ax = axes[idx]
        
        if len(intervals) > 0:
            valid_intervals = [i for i in intervals if i['birth'] != i['death'] and not np.isinf(i['death'])]
            
            if len(valid_intervals) > 0:
                births = [i['birth'] for i in valid_intervals]
                deaths = [i['death'] for i in valid_intervals]
                
                sorted_idx = np.argsort(births)
                births = np.array(births)[sorted_idx]
                deaths = np.array(deaths)[sorted_idx]
                
                for i, (b, d) in enumerate(zip(births, deaths)):
                    ax.hlines(y=i, xmin=b, xmax=d, linewidth=3, color=colors.get(dim, 'gray'))
                
                ax.set_xlim(min(births), max(deaths))
                ax.set_ylim(-1, len(valid_intervals))
                ax.set_title(f'{labels.get(dim, f"H{dim}")} - {len(valid_intervals)} features', fontsize=12)
                ax.set_xlabel('Threshold', fontsize=10)
                ax.set_ylabel('Feature Index', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No valid intervals', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{labels.get(dim, f"H{dim}")} - 0 features')
        else:
            ax.text(0.5, 0.5, f'No {labels.get(dim, f"H{dim}")} features detected', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{labels.get(dim, f"H{dim}")} - 0 features')
        
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Persistence barcode saved to: {output_path}")


def compute_persistence_statistics(diagrams):
    stats = {}
    
    for dim, intervals in diagrams.items():
        if len(intervals) > 0:
            valid_intervals = [i for i in intervals if not np.isinf(i['death'])]
            
            if len(valid_intervals) > 0:
                persistences = [i['persistence'] for i in valid_intervals]
                births = [i['birth'] for i in valid_intervals]
                deaths = [i['death'] for i in valid_intervals]
                
                stats[f'H{dim}_num_features'] = len(valid_intervals)
                stats[f'H{dim}_max_persistence'] = max(persistences) if persistences else 0
                stats[f'H{dim}_mean_persistence'] = np.mean(persistences) if persistences else 0
                stats[f'H{dim}_median_persistence'] = np.median(persistences) if persistences else 0
                stats[f'H{dim}_min_birth'] = min(births) if births else 0
                stats[f'H{dim}_max_death'] = max(deaths) if deaths else 0
            else:
                stats[f'H{dim}_num_features'] = 0
                stats[f'H{dim}_max_persistence'] = 0
                stats[f'H{dim}_mean_persistence'] = 0
                stats[f'H{dim}_median_persistence'] = 0
                stats[f'H{dim}_min_birth'] = 0
                stats[f'H{dim}_max_death'] = 0
        else:
            stats[f'H{dim}_num_features'] = 0
            stats[f'H{dim}_max_persistence'] = 0
            stats[f'H{dim}_mean_persistence'] = 0
            stats[f'H{dim}_median_persistence'] = 0
            stats[f'H{dim}_min_birth'] = 0
            stats[f'H{dim}_max_death'] = 0
    
    return stats


def compute_level_set_metrics(density_grid, threshold_percentile):
    positive_vals = density_grid[density_grid > 0]
    
    if len(positive_vals) > 0:
        threshold = np.percentile(positive_vals, threshold_percentile)
    else:
        threshold = np.percentile(density_grid, threshold_percentile)
    
    binary_grid = (density_grid >= threshold).astype(np.uint8)
    labeled, num_features = ndimage.label(binary_grid)
    
    component_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
    
    return {
        'percentile': threshold_percentile,
        'threshold': threshold,
        'num_components': num_features,
        'total_cells': int(np.sum(binary_grid)),
        'fraction': float(np.sum(binary_grid) / binary_grid.size),
        'max_component_size': max(component_sizes) if component_sizes else 0,
        'component_sizes': component_sizes
    }


def create_polar_projection_visualization(density_grid, radial_centers, angular_centers, output_path,
                                          angle_min=0, angle_max=360, quadrant_name="Full Circle",
                                          display_name="Full Circle (0°-360°)", title="Polar Density Heatmap"):
    
    if density_grid is None:
        print(f"Cannot visualize polar projection: density_grid is None")
        return
    
    if np.all(density_grid == 0):
        print(f"Cannot visualize polar projection: all values are zero")
        return
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': 'polar'})
        
        density_grid_polar = density_grid.T
        R, Theta = np.meshgrid(radial_centers, np.radians(angular_centers))
        
        contour = ax.contourf(Theta, R, density_grid_polar, levels=20, cmap='hot')
        plt.colorbar(contour, ax=ax, label='Density (cells/px²)', shrink=0.8)
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        angular_range = angle_max - angle_min
        if angular_range < 360:
            ax.set_thetamin(angle_min)
            ax.set_thetamax(angle_max)
            
            theta_ticks = np.radians(np.linspace(angle_min, angle_max, 5))
            theta_labels = [f'{int(t)}°' for t in np.linspace(angle_min, angle_max, 5)]
            ax.set_xticks(theta_ticks)
            ax.set_xticklabels(theta_labels)
        
        ax.set_title(f'{title}\n{display_name}', fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Polar projection saved to: {output_path}")
    except Exception as e:
        print(f"Error creating polar projection: {e}")


def analyze_single_density_grid(density_grid, image_name, output_dir, radial_centers, angular_centers):

    print(f"\nProcessing: {image_name}")
    print(f"Density grid shape: {density_grid.shape}")
    print(f" Density range: {density_grid.min():.4f} - {density_grid.max():.4f}")
    print(f"Non-zero cells: {np.sum(density_grid > 0)} / {density_grid.size}")
    
    # Get angle range from filename
    angle_min, angle_max, quadrant_name, display_name = get_angle_range_from_filename(image_name)
    print(f"Angle range: {angle_min}° - {angle_max}° ({display_name})")
    
    results = {}
    
    # Create high-resolution polar projection using the working method
    polar_path = os.path.join(output_dir, f"{image_name}_polar_projection.png")
    create_polar_projection_visualization(
        density_grid, radial_centers, angular_centers, polar_path,
        angle_min, angle_max, quadrant_name, display_name
    )
    
    # Upsample grid for smoother display
    density_grid_upsampled = upsample_grid(density_grid, factor=10)
    
    # 1. Compute persistence diagram
    print(f"Computing persistence diagram...")
    persistence, diagrams, cc = compute_persistence_diagram(density_grid, use_periodic_angle=True)
    
    # Save persistence diagram
    diag_path = os.path.join(output_dir, f"{image_name}_persistence_diagram.png")
    visualize_persistence_diagram(diagrams, diag_path, title=f"Persistence Diagram - {image_name}")
    
    # Save persistence barcode
    barcode_path = os.path.join(output_dir, f"{image_name}_persistence_barcode.png")
    visualize_persistence_barcode(diagrams, barcode_path, title=f"Persistence Barcode - {image_name}")
    
    # Compute persistence statistics
    stats = compute_persistence_statistics(diagrams)
    results['persistence_stats'] = stats
    
    print(f"H0 features (components): {stats.get('H0_num_features', 0)}")
    print(f"H1 features (loops/holes): {stats.get('H1_num_features', 0)}")
    print(f"H1 max persistence: {stats.get('H1_max_persistence', 0):.4f}")
    
    # 2. Multi-threshold level set analysis
    print(f"Running multi-threshold level set analysis...")
    percentiles = [50, 60, 70, 80, 85, 90, 92.5, 95, 97.5, 99]
    level_set_results = []
    
    for p in percentiles:
        ls_result = compute_level_set_metrics(density_grid, p)
        level_set_results.append(ls_result)
        print(f"{p}th percentile: {ls_result['num_components']} components, {ls_result['fraction']*100:.1f}% area")
    
    results['level_set'] = level_set_results
    
    # Save level set comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot([r['percentile'] for r in level_set_results], 
                 [r['num_components'] for r in level_set_results], 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Percentile Threshold')
    axes[0].set_ylabel('Number of Connected Components')
    axes[0].set_title('Components vs Threshold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot([r['percentile'] for r in level_set_results], 
                 [r['fraction']*100 for r in level_set_results], 'o-', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Percentile Threshold')
    axes[1].set_ylabel('High Density Area (%)')
    axes[1].set_title('Area vs Threshold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot([r['percentile'] for r in level_set_results], 
                 [r['max_component_size'] for r in level_set_results], 'o-', linewidth=2, markersize=8, color='red')
    axes[2].set_xlabel('Percentile Threshold')
    axes[2].set_ylabel('Max Component Size (pixels)')
    axes[2].set_title('Largest Component vs Threshold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Level Set Analysis - {image_name}')
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f"{image_name}_threshold_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Threshold comparison saved to: {comparison_path}")
    
    # Save level set results to CSV
    ls_df = pd.DataFrame(level_set_results)
    ls_csv_path = os.path.join(output_dir, f"{image_name}_level_set_results.csv")
    ls_df.to_csv(ls_csv_path, index=False)
    
    # 3. Create combined visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Original density (upsampled)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(density_grid_upsampled, aspect='auto', origin='lower', cmap='hot', interpolation='bilinear')
    ax1.set_title(f'Density Heatmap (Upsampled)\nMin: {density_grid.min():.4f}, Max: {density_grid.max():.4f}', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Density (cells/px²)')
    ax1.set_xlabel('Angle (bins)')
    ax1.set_ylabel('Radius (bins)')
    
    # 95th percentile level set
    ax2 = axes[0, 1]
    ls_95 = compute_level_set_metrics(density_grid, 95)
    binary_95 = (density_grid >= ls_95['threshold']).astype(np.uint8)
    binary_95_upsampled = upsample_grid(binary_95.astype(float), factor=10)
    im2 = ax2.imshow(binary_95_upsampled, aspect='auto', origin='lower', cmap='gray', interpolation='nearest')
    ax2.set_title(f'95th Percentile Level Set\nThreshold: {ls_95["threshold"]:.4f}\n'
                  f'{ls_95["num_components"]} components, {ls_95["fraction"]*100:.1f}% area', fontsize=12)
    ax2.set_xlabel('Angle (bins)')
    ax2.set_ylabel('Radius (bins)')
    
    # Persistence diagram summary
    ax3 = axes[1, 0]
    has_features = False
    all_births = []
    all_deaths = []
    
    for dim, intervals in diagrams.items():
        if len(intervals) > 0:
            has_features = True
            births = [i['birth'] for i in intervals]
            deaths = [i['death'] for i in intervals]
            all_births.extend(births)
            all_deaths.extend(deaths)
            color = {0: 'blue', 1: 'red', 2: 'green'}.get(dim, 'gray')
            label = {0: 'H0 (Components)', 1: 'H1 (Loops)', 2: 'H2 (Cavities)'}.get(dim, f'H{dim}')
            ax3.scatter(births, deaths, c=color, s=60, alpha=0.7, label=label, edgecolors='black', linewidth=0.5)
    
    if has_features:
        max_val = max(max(all_births), max(all_deaths))
        min_val = min(min(all_births), min(all_deaths))
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Death = Birth')
        ax3.set_xlim(min_val, max_val)
        ax3.set_ylim(min_val, max_val)
    else:
        ax3.text(0.5, 0.5, 'No persistent features', ha='center', va='center', transform=ax3.transAxes, fontsize=14)
    
    ax3.set_xlabel('Birth Threshold', fontsize=12)
    ax3.set_ylabel('Death Threshold', fontsize=12)
    ax3.set_title(f'Persistence Diagram\nH0: {stats.get("H0_num_features", 0)} components, H1: {stats.get("H1_num_features", 0)} loops', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    plt.suptitle(f'Complete Topological Analysis - {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    summary_path = os.path.join(output_dir, f"{image_name}_topology_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Topology summary saved to: {summary_path}")
    
    return results


def process_single_density_grid(csv_path, output_base_dir="level_set_analysis"):
    density_grid = load_density_grid(csv_path)
    image_name = os.path.basename(os.path.dirname(csv_path))
    output_dir = os.path.join(output_base_dir, image_name)
    os.makedirs(output_dir, exist_ok=True)
    
    radial_centers, angular_centers = load_metadata(csv_path)
    
    # If metadata not found, create defaults based on grid shape
    if radial_centers is None:
        radial_centers = np.linspace(0, density_grid.shape[0], density_grid.shape[0])
    if angular_centers is None:
        angular_centers = np.linspace(0, 360, density_grid.shape[1], endpoint=False)
    
    results = analyze_single_density_grid(density_grid, image_name, output_dir, radial_centers, angular_centers)
    
    return results


def run_all_analyses(measurements_dir="measurements", output_base_dir="level_set_analysis"):
    
    density_files = find_density_files(measurements_dir)
    
    if not density_files:
        print(f"No density grid files found in {measurements_dir}/density_analysis/")
        return None
    
    print(f"Found {len(density_files)} density grid files")
    
    all_summaries = []
    
    for csv_path in density_files:
        results = process_single_density_grid(csv_path, output_base_dir)
        image_name = os.path.basename(os.path.dirname(csv_path))
        
        stats = results['persistence_stats']
        ls_95 = [r for r in results['level_set'] if r['percentile'] == 95][0]
        
        all_summaries.append({
            'image': image_name,
            'H0_num_features': stats.get('H0_num_features', 0),
            'H1_num_features': stats.get('H1_num_features', 0),
            'H1_max_persistence': stats.get('H1_max_persistence', 0),
            'threshold_95': ls_95['threshold'],
            'num_components_95': ls_95['num_components'],
            'fraction_high_density_95': ls_95['fraction']
        })
    
    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(output_base_dir, "topology_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    return summary_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            sys.exit(1)
        
        process_single_density_grid(csv_path, "level_set_analysis")
        
    else:
        run_all_analyses("measurements", "level_set_analysis")