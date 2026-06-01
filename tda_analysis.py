import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from ripser import ripser
from persim import plot_diagrams
from wasserstein_distance import wasserstein_distance
from mds import mds_from_wasserstein_csv

def compute_persistence_diagram(coordinates, max_dimension=1, max_edge_length=None):

    if len(coordinates) < 2:
        print("Need at least 2 points for TDA analysis")
        return None
    
    # Ripser doesn't accept None for thresh, use np.inf instead
    thresh = max_edge_length if max_edge_length is not None else np.inf
    result = ripser(coordinates, maxdim=max_dimension, thresh=thresh)
    return result

def plot_persistence_diagram(result, output_path, title="Persistence Diagram", show=False):

    if result is None:
        print("No persistence diagram to plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_diagrams(result['dgms'], show=False, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"   Persistence diagram saved to {output_path}")

def analyze_single_image(
    csv_path,
    output_dir="tda_results",
    max_dimension=1,
    max_edge_length=None,
    show_plots=False
):

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None
    
    # Read measurements
    df = pd.read_csv(csv_path)
    
    if 'x_pixels' not in df.columns or 'y_pixels' not in df.columns:
        print(f"CSV must contain 'x_pixels' and 'y_pixels' columns")
        return None
    
    coordinates = df[['x_pixels', 'y_pixels']].values
    
    print(f"Analyzing {os.path.basename(csv_path)}: {len(coordinates)} cells")
    
    # Compute persistence diagram
    result = compute_persistence_diagram(coordinates, max_dimension, max_edge_length)
    
    if result is None:
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot persistence diagram
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    diagram_path = os.path.join(output_dir, f"{base_name}_persistence.png")
    plot_persistence_diagram(result, diagram_path, 
                            title=f"Persistence Diagram - {base_name}", 
                            show=show_plots)
    
    # Extract statistics from persistence diagrams
    stats = compute_persistence_statistics(result)
    
    return {
        'coordinates': coordinates,
        'result': result,
        'statistics': stats,
        'diagram_path': diagram_path
    }

def compute_persistence_statistics(result):

    stats = {}
    
    for dim, dgm in enumerate(result['dgms']):
        # Filter out points at infinity
        finite_dgm = dgm[dgm[:, 1] != np.inf]
        
        if len(finite_dgm) > 0:
            lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
            stats[f'H{dim}_num_features'] = len(finite_dgm)
            stats[f'H{dim}_total_persistence'] = np.sum(lifetimes)
            stats[f'H{dim}_max_persistence'] = np.max(lifetimes)
            stats[f'H{dim}_mean_persistence'] = np.mean(lifetimes)
            stats[f'H{dim}_std_persistence'] = np.std(lifetimes)
        else:
            stats[f'H{dim}_num_features'] = 0
            stats[f'H{dim}_total_persistence'] = 0
            stats[f'H{dim}_max_persistence'] = 0
            stats[f'H{dim}_mean_persistence'] = 0
            stats[f'H{dim}_std_persistence'] = 0
    
    return stats

def batch_tda_analysis(
    measurements_dir="measurements",
    output_dir="tda_results",
    max_dimension=1,
    max_edge_length=None,
    show_plots=False,
    save_summary=True
):
    if not os.path.exists(measurements_dir):
        print(f"Measurements directory not found: {measurements_dir}")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(measurements_dir) if f.endswith('_measurements.csv')]
    
    if not csv_files:
        print(f"No measurement CSV files found in {measurements_dir}")
        return None
    
    all_results = {}
    summary_data = []
    
    for csv_file in sorted(csv_files):
        csv_path = os.path.join(measurements_dir, csv_file)
        
        result = analyze_single_image(
            csv_path,
            output_dir=output_dir,
            max_dimension=max_dimension,
            max_edge_length=max_edge_length,
            show_plots=show_plots
        )
        
        if result is not None:
            image_name = os.path.splitext(csv_file)[0].replace('_measurements', '')
            all_results[image_name] = result
            
            # Add to summary
            summary_row = {'image': image_name, 'num_cells': len(result['coordinates'])}
            summary_row.update(result['statistics'])
            summary_data.append(summary_row)
    
    # Save summary statistics
    if save_summary and summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "tda_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Summary saved to {summary_path}")
    
    # Compute Wasserstein distances between all persistence diagrams
    if all_results:
        
        # Extract persistence diagrams into a list
        image_names = sorted(all_results.keys())
        diagram_list = [all_results[name]['result']['dgms'] for name in image_names]
        
        # Compute Wasserstein distances for each homology dimension
        for dimension in range(max_dimension + 1):
            
            # Save heatmap image
            heatmap_path = os.path.join(output_dir, f"wasserstein_heatmap_H{dimension}.png")
            wds = wasserstein_distance(diagram_list, dimension=dimension, 
                                      save_path=heatmap_path, show=show_plots)
            
            # Save Wasserstein distance matrix to CSV
            wds_df = pd.DataFrame(wds, index=image_names, columns=image_names)
            wds_path = os.path.join(output_dir, f"wasserstein_distances_H{dimension}.csv")
            wds_df.to_csv(wds_path)
            print(f"Wasserstein distance matrix saved to {wds_path}")
        
        for dimension in range(max_dimension + 1):
            wds_path = os.path.join(output_dir, f"wasserstein_distances_H{dimension}.csv")
            mds_output_path = os.path.join(output_dir, f"mds_wasserstein_H{dimension}.png")
            title = f"MDS: Wasserstein Distances (H_{dimension})"
            
            mds_from_wasserstein_csv(wds_path, output_path=mds_output_path, title=title, show=show_plots)
            print(f"MDS plot saved to {mds_output_path}")
    
    return all_results
