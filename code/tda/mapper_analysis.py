import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

def find_measurement_files(measurements_dir="measurements"):
    pattern = os.path.join(measurements_dir, "*_measurements.csv")
    files = glob.glob(pattern)
    return sorted(files)

def load_cell_data(csv_path):
    
    df = pd.read_csv(csv_path)
    for col in ['radius_pixels', 'area_pixels', 'x_pixels', 'y_pixels']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['radius_pixels', 'area_pixels'])
    return df

def prepare_mapper_data_radius_area(df):
    
    radii = df['radius_pixels'].values.astype(np.float64)
    areas = df['area_pixels'].values.astype(np.float64)
    
    valid = np.isfinite(radii) & np.isfinite(areas) & (radii > 0) & (areas > 0)
    radii = radii[valid]
    areas = areas[valid]
    
    if len(radii) < 10:
        return None, None, None, None, None, None
    
    data = np.column_stack([radii, areas])
    lens = radii.copy()
    feature_names = ['radius_pixels', 'area_pixels']
    
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    return data_normalized, lens, feature_names, scaler, data, radii, areas

def prepare_mapper_data_radius_density(df):
    
    radii = df['radius_pixels'].values.astype(np.float64)
    
    if 'local_density' not in df.columns:
        print(f"    Warning: 'local_density' column not found. Computing on the fly...")
        from scipy.spatial import KDTree
        coords = df[['x_pixels', 'y_pixels']].values
        tree = KDTree(coords)
        densities = []
        for center in coords:
            neighbors = tree.query_ball_point(center, 100)
            density = len(neighbors) / (np.pi * 100**2)
            densities.append(density)
        densities = np.array(densities)
    else:
        densities = df['local_density'].values.astype(np.float64)
    
    valid = np.isfinite(radii) & np.isfinite(densities) & (radii > 0) & (densities > 0)
    radii = radii[valid]
    densities = densities[valid]
    
    if len(radii) < 10:
        return None, None, None, None, None, None
    
    data = np.column_stack([radii, densities])
    lens = radii.copy()
    feature_names = ['radius_pixels', 'local_density_cells_per_px2']
    
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    return data_normalized, lens, feature_names, scaler, data, radii, densities

def prepare_mapper_data_radius_area_weighted_by_density(df):
    
    radii = df['radius_pixels'].values.astype(np.float64)
    areas = df['area_pixels'].values.astype(np.float64)
    
    if 'local_density' not in df.columns:
        from scipy.spatial import KDTree
        coords = df[['x_pixels', 'y_pixels']].values
        tree = KDTree(coords)
        densities = []
        for center in coords:
            neighbors = tree.query_ball_point(center, 100)
            density = len(neighbors) / (np.pi * 100**2)
            densities.append(density)
        densities = np.array(densities)
    else:
        densities = df['local_density'].values.astype(np.float64)
    
    area_density_ratio = areas / (densities + 0.0001)
    
    valid = np.isfinite(radii) & np.isfinite(area_density_ratio) & (radii > 0)
    radii = radii[valid]
    area_density_ratio = area_density_ratio[valid]
    
    if len(radii) < 10:
        return None, None, None, None, None, None
    
    data = np.column_stack([radii, area_density_ratio])
    lens = radii.copy()
    feature_names = ['radius_pixels', 'area_to_density_ratio']
    
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    return data_normalized, lens, feature_names, scaler, data, radii, area_density_ratio

def calculate_bins(radius_min, radius_max, n_cubes, overlap):

    bin_width = (radius_max - radius_min) / n_cubes
    step = bin_width * (1 - overlap)
    
    total_span = step * (n_cubes - 1) + bin_width
    
    if total_span > (radius_max - radius_min) * 1.1:
        step = (radius_max - radius_min) / (n_cubes - overlap * (n_cubes - 1))
        bin_width = step / (1 - overlap) if overlap < 1 else (radius_max - radius_min) / n_cubes
    
    bin_edges = []
    bin_centers = []
    
    for i in range(n_cubes):
        left = radius_min + i * step
        right = left + bin_width
        
        if i == n_cubes - 1:
            right = radius_max
        
        bin_edges.append((left, right))
        bin_centers.append((left + right) / 2)
    
    return bin_edges, bin_centers, step, bin_width

def get_cluster_ellipse(points):

    if len(points) < 3:
        return None
    
    center = np.mean(points, axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * np.sqrt(eigenvalues[0]) * 2.5
    height = 2 * np.sqrt(eigenvalues[1]) * 2.5
    
    return Ellipse(center, width, height, angle=angle, alpha=0.3, facecolor='none', edgecolor='black', linewidth=2)

def visualize_mapper_clustering(data, radii, feature2, lens, n_cubes, overlap, eps, min_samples, output_path, title="Mapper Clustering Process"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    radius_min, radius_max = radii.min(), radii.max()
    bin_edges, bin_centers, step, bin_width = calculate_bins(radius_min, radius_max, n_cubes, overlap)
    
    print(f"    Radius range: {radius_min:.1f} - {radius_max:.1f} pixels")
    print(f"    Bin widths: {[(round(right-left, 1)) for left, right in bin_edges]}")
    
    bin_colors = plt.cm.tab10(np.linspace(0, 1, n_cubes))
    
    ax1 = axes[0]
    scatter1 = ax1.scatter(radii, feature2, c=radii, cmap='viridis', s=25, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Radius (pixels)', fontsize=12)
    ax1.set_ylabel(title.split(' - ')[0] if ' - ' in title else 'Feature', fontsize=12)
    ax1.set_title(f'Raw Data: {len(radii)} Cells\nColor = Radius (pixels)', fontsize=14)
    cbar = plt.colorbar(scatter1, ax=ax1)
    cbar.set_label('Radius (pixels)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    for left, right in bin_edges:
        ax1.axvline(x=left, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.axvline(x=radius_max, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    cluster_counter = 0
    cluster_info = []
    
    for bin_idx, (left, right) in enumerate(bin_edges):
        in_bin = (radii >= left) & (radii < right)
        bin_data = data[in_bin]
        bin_radii = radii[in_bin]
        bin_feature2 = feature2[in_bin]
        bin_indices = np.where(in_bin)[0]
        
        if len(bin_data) < min_samples:
            continue
        
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(bin_data)
        
        unique_labels = set(cluster_labels)
        for label in unique_labels:
            if label == -1:
                continue
            
            cluster_mask = cluster_labels == label
            cluster_points = bin_data[cluster_mask]
            cluster_radii = bin_radii[cluster_mask]
            cluster_feature2 = bin_feature2[cluster_mask]
            cluster_indices_list = bin_indices[cluster_mask]
            
            center = np.mean(cluster_points, axis=0)
            cluster_color = bin_colors[bin_idx % len(bin_colors)]
            
            if len(cluster_points) >= 3:
                ellipse = get_cluster_ellipse(cluster_points)
                if ellipse:
                    ellipse.set_facecolor(cluster_color)
                    ellipse.set_alpha(0.15)
                    ellipse.set_edgecolor(cluster_color)
                    ellipse.set_linewidth(2)
                    ax1.add_patch(ellipse)
                    ax1.text(center[0], center[1], f'C{cluster_counter+1}', ha='center', va='center',
                            fontsize=8, fontweight='bold', color='black',
                            bbox=dict(facecolor=cluster_color, alpha=0.7, boxstyle='circle,pad=0.3'))
            
            cluster_info.append({
                'id': cluster_counter,
                'bin_idx': bin_idx,
                'label': label,
                'center': center,
                'color': cluster_color,
                'points': cluster_points,
                'radii': cluster_radii,
                'feature2': cluster_feature2,
                'indices': cluster_indices_list
            })
            cluster_counter += 1
    
    ax2 = axes[1]
    scatter2 = ax2.scatter(radii, feature2, c=radii, cmap='viridis', s=25, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Radius (pixels)', fontsize=12)
    ax2.set_ylabel(title.split(' - ')[0] if ' - ' in title else 'Feature', fontsize=12)
    ax2.set_title(f'Overlaps Between Clusters (Creates Mapper Edges)\n'
                  f'Red lines = Edges | Gold regions = Bin overlaps', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for left, right in bin_edges:
        ax2.axvline(x=left, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.axvline(x=radius_max, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    for i in range(n_cubes - 1):
        current_right = bin_edges[i][1]
        next_left = bin_edges[i + 1][0]
        if current_right > next_left:
            ax2.axvspan(next_left, current_right, alpha=0.15, color='gold', zorder=0)
    
    for cluster in cluster_info:
        if len(cluster['points']) >= 3:
            ellipse = get_cluster_ellipse(cluster['points'])
            if ellipse:
                ellipse.set_facecolor(cluster['color'])
                ellipse.set_alpha(0.15)
                ellipse.set_edgecolor(cluster['color'])
                ellipse.set_linewidth(2)
                ax2.add_patch(ellipse)
                ax2.text(cluster['center'][0], cluster['center'][1], f'C{cluster["id"]+1}', 
                        ha='center', va='center', fontsize=8, fontweight='bold', color='black',
                        bbox=dict(facecolor=cluster['color'], alpha=0.7, boxstyle='circle,pad=0.3'))
    
    point_to_clusters = defaultdict(list)
    for cluster in cluster_info:
        for point_idx in cluster['indices']:
            point_to_clusters[point_idx].append(cluster)
    
    drawn_edges = set()
    for point_idx, clusters in point_to_clusters.items():
        if len(clusters) > 1:
            ax2.scatter(radii[point_idx], feature2[point_idx], c='red', s=60, alpha=0.9,
                       edgecolors='black', linewidth=1, marker='*', zorder=10)
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    edge_key = tuple(sorted([clusters[i]['id'], clusters[j]['id']]))
                    if edge_key not in drawn_edges:
                        center1 = clusters[i]['center']
                        center2 = clusters[j]['center']
                        ax2.plot([center1[0], center2[0]], [center1[1], center2[1]], 
                                color='red', linewidth=2, alpha=0.7, linestyle='-')
                        drawn_edges.add(edge_key)
    
    for bin_idx, center in enumerate(bin_centers):
        ax2.text(center, feature2.max() * 0.98, f'Bin {bin_idx+1}',
                ha='center', fontsize=9, color='darkblue', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Data points'),
        Patch(facecolor='blue', alpha=0.2, edgecolor='blue', label='DBSCAN cluster (Node)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='Point in multiple clusters (Overlap)'),
        Line2D([0], [0], color='red', linewidth=2, label='Edge (between clusters)'),
        Patch(facecolor='gold', alpha=0.3, label='Bin overlap region')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    fig.suptitle(f'Mapper Process Visualization\n'
                 f'{len(cluster_info)} Clusters (Nodes) | {len(drawn_edges)} Edges\n'
                 f'{n_cubes} bins, {overlap*100:.0f}% overlap | DBSCAN: eps={eps}, min_samples={min_samples}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Found {len(cluster_info)} clusters (nodes) and {len(drawn_edges)} edges")
    print(f"Visualization saved to: {output_path}")

def visualize_overlap_diagram(lens, n_cubes, overlap, output_path):

    fig, ax = plt.subplots(figsize=(12, 4))
    
    lens_min, lens_max = lens.min(), lens.max()
    bin_edges, bin_centers, step, bin_width = calculate_bins(lens_min, lens_max, n_cubes, overlap)
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_cubes))
    
    y_base = 0
    for i in range(n_cubes):
        left, right = bin_edges[i]
        
        ax.barh(y_base, right - left, left=left, height=0.8, color=colors[i], alpha=0.7, edgecolor='black', linewidth=1)
        ax.text((left + right) / 2, y_base, f'Bin {i+1}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        
        if i < n_cubes - 1:
            next_left = bin_edges[i + 1][0]
            if right > next_left:
                ax.annotate('', xy=(right, y_base + 0.5), xytext=(next_left, y_base + 0.5),
                           arrowprops=dict(arrowstyle='<->', color='red', lw=2))
                ax.text((next_left + right) / 2, y_base + 1.0, 'Overlap', 
                       ha='center', va='bottom', fontsize=8, color='red')
    
    ax.set_xlim(lens_min - 5, lens_max + 5)
    ax.set_ylim(-0.5, n_cubes)
    ax.set_xlabel('Lens Value (Radius in pixels)', fontsize=12)
    ax.set_yticks([])
    ax.set_title(f'Mapper Cover: {n_cubes} bins with {overlap*100:.0f}% overlap', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Overlap diagram saved to: {output_path}")

def run_mapper_analysis(data, lens, feature_names, output_path, 
                        n_cubes=10, overlap=0.5, eps=0.5, min_samples=3,
                        title="Mapper Graph"):

    if data is None or len(data) < 10:
        print(f"    Insufficient data points: {len(data) if data is not None else 0}")
        return None
    
    mapper = km.KeplerMapper(verbose=0)
    lens_projection = mapper.fit_transform(data, projection=lens)
    cover = km.Cover(n_cubes=n_cubes, perc_overlap=overlap)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    
    graph = mapper.map(lens_projection, data, cover=cover, clusterer=clusterer)
    
    if not graph['nodes']:
        print(f"No nodes created - try different parameters")
        return None
    
    html_path = output_path.replace('.html', '') + '.html'
    tooltips = [f"{feature_names[0]}: {d[0]:.2f}, {feature_names[1]}: {d[1]:.2f}" for d in data[:len(lens)]]
    
    try:
        mapper.visualize(graph, path_html=html_path, title=title,
                        color_function=lens, color_function_name='Radius (pixels)',
                        custom_tooltips=tooltips if tooltips else None)
        print(f"Saved to: {os.path.basename(html_path)}")
    except Exception as e:
        print(f"Visualization error: {e}")
        try:
            mapper.visualize(graph, path_html=html_path, title=title,
                            color_function=lens, color_function_name='Radius (pixels)')
            print(f"Saved to: {os.path.basename(html_path)} (without tooltips)")
        except Exception as e2:
            print(f"Visualization failed: {e2}")
            return None
    
    return graph

def analyze_mapper_results(graph, output_dir, prefix=""):
    if graph is None or not graph['nodes']:
        return None
    
    nodes = graph['nodes']
    edges = graph['links']
    
    stats = {
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'avg_degree': 2 * len(edges) / len(nodes) if len(nodes) > 0 else 0,
        'node_sizes': [len(nodes[node]) for node in nodes]
    }
    
    print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}, Avg degree: {stats['avg_degree']:.2f}")
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(output_dir, f"{prefix}_mapper_stats.csv"), index=False)
    
    if stats['node_sizes']:
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(stats['node_sizes'], bins=min(20, len(stats['node_sizes'])), edgecolor='black')
            plt.xlabel('Number of cells per node')
            plt.ylabel('Frequency')
            plt.title(f'Node Size Distribution - {prefix}')
            plt.savefig(os.path.join(output_dir, f"{prefix}_node_distribution.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Could not create histogram: {e}")
    
    return stats

def process_single_image(csv_path, output_base_dir="mapper_analysis"):
    base_name = os.path.splitext(os.path.basename(csv_path))[0].replace("_measurements", "")
    safe_base_name = base_name.replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and")
    output_dir = os.path.join(output_base_dir, safe_base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing: {base_name}")
    df = load_cell_data(csv_path)
    print(f"Cells: {len(df)}")
    print(f"Local density column present: {'local_density' in df.columns}")
    
    results = {}
    
    if len(df) < 10:
        print(f"Skipping: Not enough cells ({len(df)} < 10)")
        return results
    
    try:
        data_norm, lens, feature_names, scaler, data_raw, radii, values = prepare_mapper_data_radius_area(df)
        
        if data_norm is not None and len(data_norm) >= 10:
            output_path = os.path.join(output_dir, "radius_area.html")
            graph = run_mapper_analysis(
                data_norm, lens, feature_names, output_path,
                n_cubes=10, overlap=0.5, eps=0.5, min_samples=3,
                title=f"Mapper: Cell Area vs Radius - {base_name}"
            )
            
            stats = analyze_mapper_results(graph, output_dir, prefix="radius_area")
            results['radius_area'] = {'graph': graph, 'stats': stats}
        else:
            print(f"Not enough valid data points for Radius+Area Mapper")
            results['radius_area'] = None
    except Exception as e:
        print(f"Error in Radius+Area Mapper: {e}")
        results['radius_area'] = None
    

    try:
        data_norm, lens, feature_names, scaler, data_raw, radii, densities = prepare_mapper_data_radius_density(df)
        
        if data_norm is not None and len(data_norm) >= 10:
            # Create overlap diagram with standard settings
            overlap_path = os.path.join(output_dir, "radius_density_overlap_diagram.png")
            visualize_overlap_diagram(lens, n_cubes=15, overlap=0.6, output_path=overlap_path)
            
            # Create clustering visualization with sensitive settings
            clustering_path = os.path.join(output_dir, "radius_density_clustering_sensitive.png")
            visualize_mapper_clustering(data_raw, radii, densities, lens,
                n_cubes=15, overlap=0.6, eps=0.35, min_samples=2, 
                output_path=clustering_path,
                title="Local Density vs Radius")
            
            # Run Mapper with sensitive parameters
            output_path = os.path.join(output_dir, "radius_density_sensitive.html")
            graph = run_mapper_analysis(
                data_norm, lens, feature_names, output_path,
                n_cubes=15,   
                overlap=0.6,     
                eps=0.35,        
                min_samples=2,  
                title=f"Mapper: Local Density vs Radius - {base_name}"
            )
            
            stats = analyze_mapper_results(graph, output_dir, prefix="radius_density_sensitive")
            results['radius_density_sensitive'] = {'graph': graph, 'stats': stats}
            
            # Also run a standard version for comparison
            output_path_std = os.path.join(output_dir, "radius_density_standard.html")
            graph_std = run_mapper_analysis(
                data_norm, lens, feature_names, output_path_std,
                n_cubes=10, overlap=0.5, eps=0.5, min_samples=3,
                title=f"Mapper: Local Density vs Radius (STANDARD) - {base_name}"
            )
            stats_std = analyze_mapper_results(graph_std, output_dir, prefix="radius_density_standard")
            results['radius_density_standard'] = {'graph': graph_std, 'stats': stats_std}
            
        else:
            print(f"Not enough valid data points for Radius+Density Mapper")
            results['radius_density'] = None
    except Exception as e:
        print(f"Error in Radius+Density Mapper: {e}")
        results['radius_density'] = None

    
    try:
        data_norm, lens, feature_names, scaler, data_raw, radii, ratios = prepare_mapper_data_radius_area_weighted_by_density(df)
        
        if data_norm is not None and len(data_norm) >= 10:
            output_path = os.path.join(output_dir, "area_density_ratio_sensitive.html")
            graph = run_mapper_analysis(
                data_norm, lens, feature_names, output_path,
                n_cubes=15,     
                overlap=0.6,     
                eps=0.35,        
                min_samples=2,   
                title=f"Mapper: Area/Density Ratio vs Radius - {base_name}"
            )
            
            stats = analyze_mapper_results(graph, output_dir, prefix="area_density_ratio_sensitive")
            results['area_density_ratio_sensitive'] = {'graph': graph, 'stats': stats}
            
            # Standard version for comparison
            output_path_std = os.path.join(output_dir, "area_density_ratio_standard.html")
            graph_std = run_mapper_analysis(
                data_norm, lens, feature_names, output_path_std,
                n_cubes=10, overlap=0.5, eps=0.5, min_samples=3,
                title=f"Mapper: Area/Density Ratio vs Radius (STANDARD) - {base_name}"
            )
            stats_std = analyze_mapper_results(graph_std, output_dir, prefix="area_density_ratio_standard")
            results['area_density_ratio_standard'] = {'graph': graph_std, 'stats': stats_std}
            
        else:
            print(f"Not enough valid data points for Area/Density Ratio Mapper")
            results['area_density_ratio'] = None
    except Exception as e:
        print(f"Error in Area/Density Ratio Mapper: {e}")
        results['area_density_ratio'] = None
    
    return results

def run_all_mapper_analyses(measurements_dir="measurements", output_base_dir="mapper_analysis"):
    
    csv_files = find_measurement_files(measurements_dir)
    if not csv_files:
        print(f"No measurement files found in {measurements_dir}")
        return None
    
    print(f"Found {len(csv_files)} measurement files")
    
    all_results = {}
    
    for csv_path in csv_files:
        results = process_single_image(csv_path, output_base_dir)
        base_name = os.path.splitext(os.path.basename(csv_path))[0].replace("_measurements", "")
        all_results[base_name] = results
    
    # Create summary report
    
    summary_data = []
    for img_name, results in all_results.items():
        row = {'image': img_name}
        
        for metric in ['radius_area', 'radius_density', 'area_density_ratio']:
            if results.get(metric) and results[metric].get('stats'):
                stats = results[metric]['stats']
                row[f'{metric}_nodes'] = stats['num_nodes']
                row[f'{metric}_edges'] = stats['num_edges']
                row[f'{metric}_avg_degree'] = stats['avg_degree']
            else:
                row[f'{metric}_nodes'] = 0
                row[f'{metric}_edges'] = 0
                row[f'{metric}_avg_degree'] = 0
        
        summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_base_dir, "mapper_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        print(summary_df.to_string(index=False))
    
    return all_results

def run_single_image_analysis(csv_path, output_base_dir="mapper_analysis"):

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None
    
    results = process_single_image(csv_path, output_base_dir)
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            sys.exit(1)
        
        results = run_single_image_analysis(csv_path, "mapper_analysis")
    else:
        run_all_mapper_analyses("measurements", "mapper_analysis")