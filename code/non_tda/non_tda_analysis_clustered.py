# The same profiles, split by cluster rather than pooled.
#
# Clusters roots by shape first, then plots each cluster separately, so a
# bimodal population is not averaged into a single misleading curve.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy import stats
from sklearn import manifold, cluster
from sklearn.preprocessing import StandardScaler
from code.config import RESULTS_FOLDER

def perform_mds_clustering(wasserstein_csv_path, n_clusters=None, method='dbscan'):
    # Load Wasserstein distances
    df = pd.read_csv(wasserstein_csv_path, index_col=0)
    distance_matrix = df.values
    sample_names = df.index.tolist()
    
    print(f"Loaded Wasserstein distances for {len(sample_names)} images")
    
    # Perform MDS
    mds_kw = {'n_components': 2, 'dissimilarity': 'precomputed', 'random_state': 0}
    X_mds = manifold.MDS(**mds_kw).fit_transform(distance_matrix)
    
    print(f"MDS complete: {X_mds.shape[0]} points in 2D")
    
    # Perform clustering
    if method == 'dbscan':
        # Automatically finds number of clusters
        clustering = cluster.DBSCAN(eps=200, min_samples=3).fit(X_mds)
        labels = clustering.labels_
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"DBSCAN found {n_clusters_found} clusters (eps=200, min_samples=3)")
        
    elif method == 'kmeans':
        if n_clusters is None:
            n_clusters = 3  # Default
        clustering = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(X_mds)
        labels = clustering.labels_
        print(f"K-Means clustering with {n_clusters} clusters")
        
    elif method == 'agglomerative':
        if n_clusters is None:
            n_clusters = 3  # Default
        clustering = cluster.AgglomerativeClustering(n_clusters=n_clusters).fit(X_mds)
        labels = clustering.labels_
        print(f"Agglomerative clustering with {n_clusters} clusters")
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Count samples per cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Cluster sizes:")
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"  Noise/Outliers: {count} images")
        else:
            print(f"  Cluster {label}: {count} images")
    
    return {
        'mds_coords': X_mds,
        'labels': labels,
        'sample_names': sample_names,
        'method': method
    }

def find_normal_cluster(cluster_result, strategy='largest'):

    labels = cluster_result['labels']
    sample_names = cluster_result['sample_names']
    mds_coords = cluster_result['mds_coords']
    
    if strategy == 'largest':
        # Find the largest cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_clusters = [(label, count) for label, count in zip(unique_labels, counts) if label != -1]
        
        if not valid_clusters:
            print("No valid clusters found")
            return []
        
        normal_cluster_label = max(valid_clusters, key=lambda x: x[1])[0]
        
    elif strategy == 'centroid':
        # Find cluster closest to the centroid of all points
        centroid = np.mean(mds_coords, axis=0)
        unique_labels = [l for l in np.unique(labels) if l != -1]
        
        min_dist = float('inf')
        normal_cluster_label = None
        
        for label in unique_labels:
            cluster_points = mds_coords[labels == label]
            cluster_centroid = np.mean(cluster_points, axis=0)
            dist = np.linalg.norm(cluster_centroid - centroid)
            
            if dist < min_dist:
                min_dist = dist
                normal_cluster_label = label
        
    else:
        # Assume strategy is a cluster number
        normal_cluster_label = int(strategy)
    
    # Get images in the normal cluster
    normal_images_mask = labels == normal_cluster_label
    normal_images = [name for name, is_normal in zip(sample_names, normal_images_mask) if is_normal]
    
    print(f"Normal cluster contains {len(normal_images)} images")
    
    return normal_images

def plot_mds_clusters(cluster_result, normal_images, output_path):

    mds_coords = cluster_result['mds_coords']
    labels = cluster_result['labels']
    sample_names = cluster_result['sample_names']
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create color map
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each cluster
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points (DBSCAN)
            mask = labels == label
            ax.scatter(mds_coords[mask, 0], mds_coords[mask, 1], 
                      c='gray', marker='x', s=100, alpha=0.5, label='Noise')
        else:
            mask = labels == label
            cluster_coords = mds_coords[mask]
            
            # Highlight normal cluster
            is_normal = label == labels[[s in normal_images for s in sample_names]][0] if normal_images else False
            
            if is_normal:
                ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                          c=[color], s=200, alpha=0.8, edgecolors='black', 
                          linewidth=3, label=f'Cluster {label} (NORMAL)', zorder=10)
            else:
                ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                          c=[color], s=100, alpha=0.6, label=f'Cluster {label}')
    
    ax.set_xlabel('MDS Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('MDS Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title('MDS Clustering of Images (Based on TDA Wasserstein Distances)', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved MDS cluster plot: {output_path}")
    plt.close()

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

def process_measurements_with_filter(measurements_folder, normal_images, angle_tolerance=5):

    # Get all measurement files
    csv_files = glob.glob(str(Path(measurements_folder) / "*_measurements.csv"))
    
    # Filter to only normal images
    normal_csv_files = []
    for csv_path in csv_files:
        base_name = Path(csv_path).stem.replace('_measurements', '')
        if base_name in normal_images:
            normal_csv_files.append(csv_path)
    
    if not normal_csv_files:
        return None
    
    all_data = []
    files_with_data = 0
    
    for csv_path in normal_csv_files:
        filtered_data = load_and_filter_cells(csv_path, angle_tolerance)
        if filtered_data is not None and not filtered_data.empty:
            all_data.append(filtered_data)
            files_with_data += 1
    
    if not all_data:
        print("No data found")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df

def create_visualizations(data, output_folder=RESULTS_FOLDER / 'non_tda_results_clustered', cluster_info=""):

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    images = data['filename'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(images)))
    
    # Average cell size per image
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for idx, filename in enumerate(images):
        image_data = data[data['filename'] == filename].copy()
        # Normalize radius to start at 0
        min_radius = image_data['radius_pixels'].min()
        image_data['radius_normalized'] = image_data['radius_pixels'] - min_radius
        image_data['radius_binned'] = (image_data['radius_normalized'] // 20) * 20
        averaged = image_data.groupby('radius_binned')['area_pixels'].mean().reset_index()
        averaged = averaged.sort_values('radius_binned')
        
        ax.plot(averaged['radius_binned'], averaged['area_pixels'], 
                alpha=0.7, linewidth=1.5, marker='o', markersize=3,
                color=colors[idx], label=filename if len(images) <= 15 else '')
    
    ax.set_xlabel('Normalized Radius from Center (pixels)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Cell Size (pixels²)', fontsize=14, fontweight='bold')
    ax.set_title(f'Cell Size vs Normalized Radius', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
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
    
    # Smoothed version with extra smoothing
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for idx, filename in enumerate(images):
        image_data = data[data['filename'] == filename].copy()
        # Normalize radius to start at 0
        min_radius = image_data['radius_pixels'].min()
        image_data['radius_normalized'] = image_data['radius_pixels'] - min_radius
        # Larger bins for smoother appearance
        image_data['radius_binned'] = (image_data['radius_normalized'] // 50) * 50
        averaged = image_data.groupby('radius_binned')['area_pixels'].mean().reset_index()
        averaged = averaged.sort_values('radius_binned')
        
        # Apply rolling average for additional smoothing
        if len(averaged) >= 3:
            averaged['area_smoothed'] = averaged['area_pixels'].rolling(window=3, center=True, min_periods=1).mean()
        else:
            averaged['area_smoothed'] = averaged['area_pixels']
        
        ax.plot(averaged['radius_binned'], averaged['area_smoothed'], 
                alpha=0.6, linewidth=2, 
                color=colors[idx], label=filename if len(images) <= 15 else '')
    
    ax.set_xlabel('Normalized Radius from Center (pixels)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Cell Size (pixels²)', fontsize=14, fontweight='bold')
    ax.set_title(f'Cell Size vs Normalized Radius', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ''
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
    
    # Normalized and averaged
    fig, ax = plt.subplots(figsize=(14, 9))
    
    all_normalized_data = []
    
    for filename in images:
        image_data = data[data['filename'] == filename].copy()
        
        # Normalize BOTH radius and cell size to 0-1 scale for this image
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
        ax.set_title(f'Normalized Average Cell Size: Normal Cluster{cluster_info}\n(5-angle sampling, both radius and size normalized per-image)', 
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
    
    # All images (normal cluster) with 0-1 normalized scales
    fig, ax = plt.subplots(figsize=(14, 9))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(images)))
    
    for idx, filename in enumerate(images):
        image_data = data[data['filename'] == filename].copy()
        
        # Normalize BOTH radius and cell size to 0-1 scale for this image
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
    ax.set_title(f'Cell Size vs Radius: Normal Cluster (Normalized Scales){cluster_info}\n(5-angle sampling, both axes 0-1 normalized per-image)', 
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
    
    # Scatter plot with trend (with normalized radius)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize radius for each image separately
    data_normalized = data.copy()
    for filename in images:
        mask = data_normalized['filename'] == filename
        min_radius = data_normalized.loc[mask, 'radius_pixels'].min()
        data_normalized.loc[mask, 'radius_normalized'] = data_normalized.loc[mask, 'radius_pixels'] - min_radius
    
    scatter = ax.scatter(data_normalized['radius_normalized'], data_normalized['area_pixels'], 
                        alpha=0.5, s=30, c=data_normalized['radius_normalized'], 
                        cmap='viridis', edgecolors='black', linewidth=0.5)
    
    z = np.polyfit(data_normalized['radius_normalized'], data_normalized['area_pixels'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(data_normalized['radius_normalized'].min(), data_normalized['radius_normalized'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Polynomial fit (degree 2)')
    
    ax.set_xlabel('Normalized Radius from Center (pixels)', fontsize=12)
    ax.set_ylabel('Cell Area (pixels²)', fontsize=12)
    ax.set_title(f'Cell Size vs Normalized Distance: Normal Cluster (5-angle sampling){cluster_info}', fontsize=14, fontweight='bold')
    ax.legend()
    plt.colorbar(scatter, label='Normalized Radius (pixels)')
    plt.tight_layout()
    plt.savefig(output_path / 'cell_size_vs_radius_scatter.png', dpi=300)
    print(f"Saved: cell_size_vs_radius_scatter.png")
    plt.close()
    
    print(f"All visualizations saved to {output_folder}/")


def main():
    
    # Configuration
    wasserstein_csv = RESULTS_FOLDER / 'chosen_tda_results' / 'wasserstein_distances_H1.csv'
    measurements_folder = "measurements"
    output_folder=RESULTS_FOLDER / 'non_tda_results_clustered'
    angle_tolerance = 5
    
    # Clustering parameters
    clustering_method = 'dbscan'  # 'dbscan', 'kmeans', or 'agglomerative'
    n_clusters = None  # For kmeans/agglomerative
    cluster_strategy = 'largest'  # 'largest', 'centroid', or specific cluster number

    # Step 1: Perform MDS and clustering
    cluster_result = perform_mds_clustering(wasserstein_csv, n_clusters, clustering_method)
    
    # Step 2: Identify normal cluster
    normal_images = find_normal_cluster(cluster_result, cluster_strategy)
    
    # Step 3: Plot MDS with clusters
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    plot_mds_clusters(cluster_result, normal_images, output_path / 'mds_clusters.png')
    
    # Step 4: Process measurements from normal cluster only
    combined_data = process_measurements_with_filter(measurements_folder, normal_images, angle_tolerance)
    
    # Step 5: Create visualizations
    cluster_info = f" (Cluster: {cluster_strategy}, {len(normal_images)} images)"
    create_visualizations(combined_data, output_folder, cluster_info)
    
    # Save processed data and image list
    data_file = output_path / 'processed_data.csv'
    combined_data.to_csv(data_file, index=False)
    

if __name__ == "__main__":
    main()
