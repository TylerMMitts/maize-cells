# Local cell density and the air pockets between cells.
#
# Computes density on a polar grid and finds the gaps that are too large to
# be a cell, which is what the air-space measurements are built from.

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial import KDTree

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

def compute_local_density_for_cells(df, radius_scale=100):
    coords = df[['x_pixels', 'y_pixels']].values
    tree = KDTree(coords)
    
    densities = []
    for center in coords:
        neighbors = tree.query_ball_point(center, radius_scale)
        density = len(neighbors) / (np.pi * radius_scale**2)
        densities.append(density)
    
    df['local_density'] = densities
    return df

def compute_polar_density_grid(df, angle_min=0, angle_max=360, num_radial_bins=20, num_angular_bins=36):
    if len(df) == 0:
        return None, None, None, None, None
    
    radii = df['radius_pixels'].values
    angles = df['angle_degrees'].values
    
    max_radius = max(radii)
    
    if max_radius == 0:
        return None, None, None, None, None
    
    radial_bins = np.linspace(0, max_radius, num_radial_bins + 1)
    angular_bins = np.linspace(angle_min, angle_max, num_angular_bins + 1)
    
    radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
    angular_centers = (angular_bins[:-1] + angular_bins[1:]) / 2
    
    density_grid = np.zeros((num_radial_bins, num_angular_bins))
    area_grid = np.zeros((num_radial_bins, num_angular_bins))
    count_grid = np.zeros((num_radial_bins, num_angular_bins))
    
    for r_idx in range(num_radial_bins):
        r_min, r_max = radial_bins[r_idx], radial_bins[r_idx + 1]
        
        for a_idx in range(num_angular_bins):
            a_min, a_max = angular_bins[a_idx], angular_bins[a_idx + 1]
            
            cells_in_bin = df[(df['radius_pixels'] >= r_min) & (df['radius_pixels'] < r_max) &
                              (df['angle_degrees'] >= a_min) & (df['angle_degrees'] < a_max)]
            
            sector_area = np.pi * (r_max**2 - r_min**2) * (a_max - a_min) / 360
            
            count_grid[r_idx, a_idx] = len(cells_in_bin)
            
            if len(cells_in_bin) > 0 and sector_area > 0:
                density_grid[r_idx, a_idx] = len(cells_in_bin) / sector_area
                area_grid[r_idx, a_idx] = cells_in_bin['area_pixels'].mean()
    
    return density_grid, radial_centers, angular_centers, area_grid, count_grid

def compute_radial_density_profile(df, num_bins=20):
    if len(df) == 0:
        return None, None, None
    
    radii = df['radius_pixels'].values
    max_radius = max(radii)
    
    if max_radius == 0:
        return None, None, None
    
    bins = np.linspace(0, max_radius, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    density = []
    avg_area = []
    
    for i in range(num_bins):
        r_min, r_max = bins[i], bins[i + 1]
        cells_in_bin = df[(df['radius_pixels'] >= r_min) & (df['radius_pixels'] < r_max)]
        
        ring_area = np.pi * (r_max**2 - r_min**2)
        
        if len(cells_in_bin) > 0 and ring_area > 0:
            density.append(len(cells_in_bin) / ring_area)
            avg_area.append(cells_in_bin['area_pixels'].mean())
        else:
            density.append(0.0)
            avg_area.append(0.0)
    
    return bin_centers, density, avg_area

def compute_angular_density_profile(df, angle_min=0, angle_max=360, num_bins=36):
    if len(df) == 0:
        return None, None, None
    
    angles = df['angle_degrees'].values
    
    bins = np.linspace(angle_min, angle_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    cell_count = []
    avg_area = []
    
    for i in range(num_bins):
        a_min, a_max = bins[i], bins[i + 1]
        cells_in_bin = df[(df['angle_degrees'] >= a_min) & (df['angle_degrees'] < a_max)]
        
        cell_count.append(len(cells_in_bin))
        if len(cells_in_bin) > 0:
            avg_area.append(cells_in_bin['area_pixels'].mean())
        else:
            avg_area.append(0.0)
    
    return bin_centers, cell_count, avg_area

def find_air_pockets(density_grid, area_grid, count_grid, density_percentile=10, area_percentile=90):
    if density_grid is None:
        return []
    
    positive_density = density_grid[density_grid > 0]
    positive_area = area_grid[area_grid > 0]
    
    if len(positive_density) == 0 or len(positive_area) == 0:
        return []
    
    density_threshold = np.percentile(positive_density, density_percentile)
    area_threshold = np.percentile(positive_area, area_percentile)
    
    candidates = []
    for r_idx in range(density_grid.shape[0]):
        for a_idx in range(density_grid.shape[1]):
            if (density_grid[r_idx, a_idx] < density_threshold and 
                area_grid[r_idx, a_idx] > area_threshold and
                count_grid[r_idx, a_idx] > 0):
                candidates.append({
                    'radius_bin': r_idx,
                    'angle_bin': a_idx,
                    'density': density_grid[r_idx, a_idx],
                    'avg_area': area_grid[r_idx, a_idx],
                    'cell_count': count_grid[r_idx, a_idx]
                })
    
    return candidates

def visualize_polar_projection(density_grid, radial_centers, angular_centers, output_path,
                                angle_min=0, angle_max=360, quadrant_name="Full Circle",
                                display_name="Full Circle (0°-360°)", cmap='hot', title="Cell Density Polar Projection"):
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
        
        contour = ax.contourf(Theta, R, density_grid_polar, levels=20, cmap=cmap)
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
        print(f"Saved polar projection: {output_path}")
    except Exception as e:
        print(f"Error creating polar projection: {e}")

def visualize_radial_density_profile(radii, density, avg_area, output_path):
    if radii is None:
        print(f"Cannot visualize radial profile: radii is None")
        return
    
    if len(radii) == 0:
        print(f"Cannot visualize radial profile: no data")
        return
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(radii, density, 'b-', linewidth=2)
        axes[0].set_xlabel('Radius (pixels)')
        axes[0].set_ylabel('Density (cells/px²)')
        axes[0].set_title('Radial Density Profile')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(radii, avg_area, 'r-', linewidth=2)
        axes[1].set_xlabel('Radius (pixels)')
        axes[1].set_ylabel('Average Cell Area (px²)')
        axes[1].set_title('Cell Size by Radius')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved radial profile visualization: {output_path}")
    except Exception as e:
        print(f"Error creating radial profile plot: {e}")

def visualize_angular_density_profile(angles, cell_count, avg_area, output_path, 
                                       angle_min=0, angle_max=360, display_name="Full Circle"):
    if angles is None:
        print(f"Cannot visualize angular profile: angles is None")
        return
    
    if len(angles) == 0:
        print(f"Cannot visualize angular profile: no data")
        return
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(angles, cell_count, 'b-', linewidth=2)
        axes[0].set_xlabel(f'Angle (degrees)')
        axes[0].set_ylabel('Cell Count')
        axes[0].set_title(f'Angular Cell Distribution - {display_name}')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(angle_min, angle_max)
        
        axes[1].plot(angles, avg_area, 'r-', linewidth=2)
        axes[1].set_xlabel(f'Angle (degrees)')
        axes[1].set_ylabel('Average Cell Area (px²)')
        axes[1].set_title(f'Cell Size by Angle - {display_name}')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(angle_min, angle_max)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved angular profile visualization: {output_path}")
    except Exception as e:
        print(f"Error creating angular profile plot: {e}")

def compute_all_density_metrics(df, output_dir, filename="unknown"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    angle_min, angle_max, quadrant_name, display_name = get_angle_range_from_filename(filename)
    
    results = {}
    results['angle_min'] = angle_min
    results['angle_max'] = angle_max
    results['quadrant_name'] = quadrant_name
    results['display_name'] = display_name
    
    # Compute per-cell local density and save
    df_with_density = compute_local_density_for_cells(df.copy())
    density_csv_path = os.path.join(output_dir, "cell_densities.csv")
    df_with_density[['cell_id', 'radius_pixels', 'angle_degrees', 'area_pixels', 'local_density']].to_csv(density_csv_path, index=False)
    print(f"Saved per-cell densities to: {density_csv_path}")
    
    density_grid, radial_centers, angular_centers, area_grid, count_grid = compute_polar_density_grid(
        df, angle_min, angle_max
    )
    
    if density_grid is not None and not np.all(density_grid == 0):
        visualize_polar_projection(
            density_grid, radial_centers, angular_centers,
            os.path.join(output_dir, "polar_projection.png"),
            angle_min, angle_max, quadrant_name, display_name
        )
        
        pd.DataFrame(density_grid).to_csv(os.path.join(output_dir, "polar_density_grid.csv"), index=False)
        pd.DataFrame(area_grid).to_csv(os.path.join(output_dir, "polar_area_grid.csv"), index=False)
        pd.DataFrame(count_grid).to_csv(os.path.join(output_dir, "polar_count_grid.csv"), index=False)
        
        air_pockets = find_air_pockets(density_grid, area_grid, count_grid)
        if air_pockets:
            pd.DataFrame(air_pockets).to_csv(os.path.join(output_dir, "air_pockets.csv"), index=False)
            results['air_pockets'] = air_pockets
        
        results['density_grid'] = density_grid
        results['area_grid'] = area_grid
        results['count_grid'] = count_grid
    else:
        print(f"Polar density grid is empty or all zeros")
    
    radii, radial_density, radial_avg_area = compute_radial_density_profile(df)
    if radii is not None and len(radii) > 0 and max(radial_density) > 0:
        visualize_radial_density_profile(radii, radial_density, radial_avg_area,
                                          os.path.join(output_dir, "radial_profile.png"))
        
        radial_df = pd.DataFrame({
            'radius_pixels': radii,
            'density_cells_per_px2': radial_density,
            'avg_area_pixels': radial_avg_area
        })
        radial_df.to_csv(os.path.join(output_dir, "radial_profile.csv"), index=False)
        
        results['radial_radii'] = radii
        results['radial_density'] = radial_density
        results['radial_avg_area'] = radial_avg_area
        
    else:
        print(f"  Warning: Radial profile is empty or all zeros")
    
    angles, angular_count, angular_avg_area = compute_angular_density_profile(df, angle_min, angle_max)
    if angles is not None and len(angles) > 0 and max(angular_count) > 0:
        visualize_angular_density_profile(angles, angular_count, angular_avg_area,
                                           os.path.join(output_dir, "angular_profile.png"),
                                           angle_min, angle_max, display_name)
        
        angular_df = pd.DataFrame({
            'angle_degrees': angles,
            'cell_count': angular_count,
            'avg_area_pixels': angular_avg_area
        })
        angular_df.to_csv(os.path.join(output_dir, "angular_profile.csv"), index=False)
        
        results['angular_angles'] = angles
        results['angular_count'] = angular_count
        results['angular_avg_area'] = angular_avg_area
        
        if max(angular_count) > 0:
            print(f"Angular uniformity: {min(angular_count) / max(angular_count):.2f}")
        else:
            print(f"Angular uniformity: N/A")
    else:
        print(f"Angular profile is empty or all zeros")
    
    return results