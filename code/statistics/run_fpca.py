# Functional PCA over the fitted profiles.
#
# Lets the data choose its own shape descriptors instead of imposing the six
# features, as a check that the imposed ones are not inventing structure.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob
import re
import warnings
from code.util.file_utils import parse_image_name, load_measurement_csv
from code.config import MASTER_SUMMARY_PATH, MEASUREMENTS_FOLDER, RESULTS_FOLDER
warnings.filterwarnings('ignore')

MASTER_SUMMARY_PATH = MASTER_SUMMARY_PATH
MEASUREMENTS_FOLDER = MEASUREMENTS_FOLDER
OUTPUT_FOLDER = RESULTS_FOLDER / 'fpca_results'
SPLINE_SMOOTHING = 0.1
N_POINTS = 200
MIN_CELLS_FOR_SPLINE = 5


def fit_spline_and_get_curve(x, y, smoothing=SPLINE_SMOOTHING, n_points=N_POINTS):

    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < MIN_CELLS_FOR_SPLINE:
        return None
    
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    
    try:
        spline = UnivariateSpline(x, y, s=smoothing, ext='extrapolate')
    except Exception:
        return None
    
    x_smooth = np.linspace(0, 1, n_points)
    y_smooth = spline(x_smooth)
    
    return y_smooth


def reconstruct_curves_from_measurements(master_summary_path, measurements_folder):
    
    # Load master summary
    master_df = pd.read_csv(master_summary_path)
    print(f"Loaded {len(master_df)} entries from master summary")
    
    # Filter to entries with cells and file_count > 0
    master_df = master_df[master_df['n_cells'] > 0]
    master_df = master_df[master_df['file_count'] > 0]
    print(f"Filtered to {len(master_df)} entries with cells and file_count > 0")
    
    # Store curves per root
    root_curves = {}
    root_metadata = {}
    
    for idx, row in master_df.iterrows():
        image_name = row['image_name']
        
        # Parse metadata
        parsed = parse_image_name(image_name)
        root_id = parsed.get('root_identifier')
        
        if root_id is None:
            continue
        
        # Find measurement CSV
        csv_path = os.path.join(measurements_folder, f"{image_name}_measurements.csv")
        if not os.path.exists(csv_path):
            alt_path = os.path.join(measurements_folder, f"{image_name}_centers_measurements.csv")
            if os.path.exists(alt_path):
                csv_path = alt_path
            else:
                continue
        
        # Load data
        df = load_measurement_csv(csv_path)
        if df is None or df.empty:
            continue
        
        # Get area column
        if 'area_um2' in df.columns:
            area_col = 'area_um2'
        else:
            area_col = 'area_pixels'
        
        # Filter
        df = df[df[area_col] > 0]
        df = df[df['radius_pixels'] > 0]
        
        if len(df) < MIN_CELLS_FOR_SPLINE:
            continue
        
        # Normalize radius to 0-1
        min_radius = df['radius_pixels'].min()
        max_radius = df['radius_pixels'].max()
        if max_radius <= min_radius:
            continue
        df['radius_normalized'] = (df['radius_pixels'] - min_radius) / (max_radius - min_radius)
        
        # Normalize cell area to 0-1
        min_area = df[area_col].min()
        max_area = df[area_col].max()
        if max_area <= min_area:
            continue
        df['area_normalized'] = (df[area_col] - min_area) / (max_area - min_area)
        
        # Bin by normalized radius
        n_bins = min(20, len(df) // 3)
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
        
        # Fit spline and get curve
        curve = fit_spline_and_get_curve(x, y)
        
        if curve is None:
            continue
        
        # Store curve
        if root_id not in root_curves:
            root_curves[root_id] = []
            root_metadata[root_id] = {
                'root_identifier': root_id,
                'plant_number': parsed.get('plant_number'),
                'root_type': parsed.get('root_type', 'unknown'),
                'treatment': parsed.get('treatment', 'unknown'),
                'n_files': row.get('file_count', 0),
                'root_radius_um': row.get('root_radius_um', 0),
                'stele_area_um2': row.get('stele_area_um2', 0),
                'avg_cell_area_um2': row.get('average_cell_area_um2', 0),
                'n_cells': row.get('n_cells', 0),
                'quadrants': set()
            }
        
        root_curves[root_id].append(curve)
        root_metadata[root_id]['quadrants'].add(parsed.get('quadrant'))
    
    # Average curves across quadrants for each root
    averaged_curves = {}
    averaged_metadata = {}
    
    for root_id, curves in root_curves.items():
        if len(curves) > 0:
            # Average across quadrants
            avg_curve = np.mean(curves, axis=0)
            averaged_curves[root_id] = avg_curve
            averaged_metadata[root_id] = root_metadata[root_id]
            averaged_metadata[root_id]['n_quadrants'] = len(curves)
            averaged_metadata[root_id]['quadrants'] = ';'.join(sorted(root_metadata[root_id]['quadrants']))
    
    print(f"\nReconstructed curves for {len(averaged_curves)} roots")
    
    # Check quadrant distribution
    full_roots = sum(1 for meta in averaged_metadata.values() if meta.get('n_quadrants', 0) >= 4)
    partial_roots = len(averaged_curves) - full_roots
    print(f"Full roots (4 quadrants): {full_roots}")
    print(f"Partial roots (<4 quadrants): {partial_roots}")
    
    return averaged_curves, averaged_metadata


def run_fpca(curves_dict, metadata_dict):
    
    # Convert to matrix
    curve_list = []
    root_ids = []
    metadata_list = []
    
    for root_id, curve in curves_dict.items():
        curve_list.append(curve)
        root_ids.append(root_id)
        metadata_list.append(metadata_dict[root_id])
    
    curve_matrix = np.array(curve_list)
    
    print(f"Curve matrix shape: {curve_matrix.shape}")
    
    # Center the data (subtract the mean curve)
    mean_curve = np.mean(curve_matrix, axis=0)
    centered_curves = curve_matrix - mean_curve
    
    # Run PCA
    pca = PCA()
    scores = pca.fit_transform(centered_curves)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    
    print(f"\nExplained variance:")
    for i, var in enumerate(explained_variance[:5]):
        print(f"  PC{i+1}: {var*100:.1f}%")
    
    # Store results
    results = {
        'pca': pca,
        'scores': scores,
        'mean_curve': mean_curve,
        'explained_variance': explained_variance,
        'root_ids': root_ids,
        'metadata': metadata_list,
        'curve_matrix': curve_matrix
    }
    
    return results


def visualize_fpca(results, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    
    pca = results['pca']
    scores = results['scores']
    mean_curve = results['mean_curve']
    explained_variance = results['explained_variance']
    metadata = results['metadata']
    
    x_axis = np.linspace(0, 1, len(mean_curve))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pc1_loadings = pca.components_[0]
    
    ax.plot(x_axis, mean_curve, 'k-', linewidth=3, label='Mean curve')
    ax.plot(x_axis, mean_curve + pc1_loadings * 0.5, 'r--', linewidth=2, label='+PC1 (50%)')
    ax.plot(x_axis, mean_curve - pc1_loadings * 0.5, 'b--', linewidth=2, label='-PC1 (50%)')
    
    ax.set_xlabel('Normalized Radius (0=center, 1=edge)', fontsize=12)
    ax.set_ylabel('Normalized Cell Size', fontsize=12)
    ax.set_title(f'PC1\n({explained_variance[0]*100:.1f}% of variation)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '01_pc1_shared_hump.png'), dpi=150)
    plt.close()
    print("Saved: 01_pc1_shared_hump.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pc2_loadings = pca.components_[1]
    
    ax.plot(x_axis, mean_curve, 'k-', linewidth=3, label='Mean curve')
    ax.plot(x_axis, mean_curve + pc2_loadings * 0.5, 'r--', linewidth=2, label='+PC2 (50%)')
    ax.plot(x_axis, mean_curve - pc2_loadings * 0.5, 'b--', linewidth=2, label='-PC2 (50%)')
    
    ax.set_xlabel('Normalized Radius (0=center, 1=edge)', fontsize=12)
    ax.set_ylabel('Normalized Cell Size', fontsize=12)
    ax.set_title(f'PC2\n({explained_variance[1]*100:.1f}% of variation)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '02_pc2_deviation.png'), dpi=150)
    plt.close()
    print("Saved: 02_pc2_deviation.png")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract root types
    root_types = [meta.get('root_type', 'unknown') for meta in metadata]
    unique_types = list(set(root_types))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
    type_to_color = {t: colors[i] for i, t in enumerate(unique_types)}
    
    for root_type in unique_types:
        mask = [t == root_type for t in root_types]
        ax.scatter(scores[mask, 0], scores[mask, 1], 
                   label=root_type, alpha=0.7, s=60,
                   color=type_to_color[root_type])
    
    ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=12)
    ax.set_title('fPCA: PC1 vs PC2 Scores\n(Colored by Root Type)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '03_pc1_vs_pc2_scores.png'), dpi=150)
    plt.close()
    print("Saved: 03_pc1_vs_pc2_scores.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_components = min(10, len(explained_variance))
    ax.bar(range(1, n_components + 1), explained_variance[:n_components] * 100)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_title('Scree Plot: Variance Explained by Each PC', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, n_components + 1))
    
    # Add cumulative variance line
    cumulative = np.cumsum(explained_variance[:n_components]) * 100
    ax2 = ax.twinx()
    ax2.plot(range(1, n_components + 1), cumulative, 'r-o', linewidth=2, markersize=6)
    ax2.set_ylabel('Cumulative Variance (%)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '04_scree_plot.png'), dpi=150)
    plt.close()
    print("Saved: 04_scree_plot.png")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    curve_matrix = results['curve_matrix']
    for i, curve in enumerate(curve_matrix):
        ax.plot(x_axis, curve, alpha=0.3, linewidth=0.5, color='gray')
    
    # Overlay mean curve
    ax.plot(x_axis, mean_curve, 'k-', linewidth=3, label='Mean curve')
    
    ax.set_xlabel('Normalized Radius (0=center, 1=edge)', fontsize=12)
    ax.set_ylabel('Normalized Cell Size', fontsize=12)
    ax.set_title('All Smoothed Curves (Overlay)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '05_all_curves_overlay.png'), dpi=150)
    plt.close()
    print("Saved: 05_all_curves_overlay.png")
    
    score_df = pd.DataFrame({
        'root_id': results['root_ids'],
        'root_type': [meta.get('root_type', 'unknown') for meta in metadata],
        'n_files': [meta.get('n_files', 0) for meta in metadata],
        'root_radius_um': [meta.get('root_radius_um', 0) for meta in metadata],
        'n_quadrants': [meta.get('n_quadrants', 0) for meta in metadata],
        'PC1': scores[:, 0],
        'PC2': scores[:, 1],
        'PC3': scores[:, 2] if scores.shape[1] > 2 else np.nan
    })
    
    score_df.to_csv(os.path.join(output_folder, 'fpca_scores.csv'), index=False)
    print("Saved: fpca_scores.csv")
    
    print(f"\nResults saved to: {output_folder}/")
    
    return score_df


def main():
    # Reconstruct curves
    curves_dict, metadata_dict = reconstruct_curves_from_measurements(
        MASTER_SUMMARY_PATH, 
        MEASUREMENTS_FOLDER
    )
    
    if not curves_dict:
        print("No curves reconstructed!")
        return
    
    # Run fPCA
    results = run_fpca(curves_dict, metadata_dict)
    
    # Visualize
    score_df = visualize_fpca(results, OUTPUT_FOLDER)
    
    # Print summary
    print(f"Total roots analyzed: {len(curves_dict)}")
    print(f"PC1 (Shared Hump): {results['explained_variance'][0]*100:.1f}%")
    if len(results['explained_variance']) > 1:
        print(f"PC2 (Deviation): {results['explained_variance'][1]*100:.1f}%")


if __name__ == "__main__":
    main()