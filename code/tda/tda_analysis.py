# Persistence diagrams for each image, and the statistics over them.
#
# The main entry to the topological side: builds the diagrams, summarises
# them and runs the comparisons.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from ripser import ripser
from persim import plot_diagrams
from code.tda.wasserstein_distance import wasserstein_distance
from code.tda.mds import mds_from_wasserstein_csv
from code.tda.persistence_image import persistence_image
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from code.config import RESULTS_FOLDER, TDA_FOLDER

def compute_persistence_diagram(coordinates, max_dimension=1, max_edge_length=None):

    if len(coordinates) < 2:
        print("Need at least 2 points for TDA analysis")
        return None
    
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
    
    print(f"Persistence diagram saved to {output_path}")

def analyze_single_image(
    csv_path,
    output_dir=RESULTS_FOLDER / 'tda_results',
    max_dimension=1,
    max_edge_length=None,
    show_plots=False
):

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    if 'x_pixels' not in df.columns or 'y_pixels' not in df.columns:
        print(f"CSV must contain 'x_pixels' and 'y_pixels' columns")
        return None
    
    coordinates = df[['x_pixels', 'y_pixels']].values
    
    print(f"Analyzing {os.path.basename(csv_path)}: {len(coordinates)} cells")
    
    result = compute_persistence_diagram(coordinates, max_dimension, max_edge_length)
    
    if result is None:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    diagram_path = os.path.join(output_dir, f"{base_name}_persistence.png")
    plot_persistence_diagram(result, diagram_path, 
                            title=f"Persistence Diagram - {base_name}", 
                            show=show_plots)
    
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

def resize_h0_left_aligned(h0_pi, target_h, target_w, profile_width=3):

    # Flip horizontally so birth=0 is on the LEFT
    h0_flipped = np.fliplr(h0_pi)
    
    # Extract the profile (first columns after flipping = birth=0)
    profile = h0_flipped[:, :profile_width]
    
    from scipy.ndimage import zoom
    
    # Zoom factor for persistence dimension (rows)
    zoom_pers = target_h / profile.shape[0]
    
    # Zoom only the persistence dimension (axis 0)
    profile_resized = zoom(profile, (zoom_pers, 1), order=1)
    
    # Create the final image - profile at LEFT edge
    final_image = np.zeros((target_h, target_w))
    final_image[:, :profile_width] = profile_resized
    
    return final_image

def resize_h1_centered(image, target_h, target_w):

    h, w = image.shape
    
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w))
    
    zoom_h = target_h / h
    zoom_w = target_w / w
    zoom_factor = min(zoom_h, zoom_w)
    
    if zoom_factor <= 0:
        return np.zeros((target_h, target_w))
    
    resized = ndimage.zoom(image, (zoom_factor, zoom_factor), order=1)
    new_h, new_w = resized.shape
    
    padded = np.zeros((target_h, target_w))
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded

def batch_tda_analysis(
    measurements_dir=RESULTS_FOLDER / 'measurements',
    output_dir=TDA_FOLDER,
    max_dimension=1,
    max_edge_length=None,
    show_plots=False,
    save_summary=True,
    image_height=64,
    image_width=64,
    pixsz=0.3,
    h0_profile_width=3,
    skip_wasserstein=False,
    legend_threshold=20  # New: hide legend when number of images exceeds this
):

    if not os.path.exists(measurements_dir):
        print(f"Measurements directory not found: {measurements_dir}")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    h0_pi_dir = os.path.join(output_dir, "H0_PIs")
    h1_pi_dir = os.path.join(output_dir, "H1_PIs")
    pca_dir = os.path.join(output_dir, "PCA")
    diagrams_dir = os.path.join(output_dir, "diagrams")
    concatenated_dir = os.path.join(output_dir, "concatenated_images")
    wasserstein_dir = os.path.join(output_dir, "wasserstein")
    
    os.makedirs(h0_pi_dir, exist_ok=True)
    os.makedirs(h1_pi_dir, exist_ok=True)
    os.makedirs(pca_dir, exist_ok=True)
    os.makedirs(diagrams_dir, exist_ok=True)
    os.makedirs(concatenated_dir, exist_ok=True)
    if not skip_wasserstein:
        os.makedirs(wasserstein_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(measurements_dir) if f.endswith('_measurements.csv')]
    
    if not csv_files:
        print(f"No measurement CSV files found in {measurements_dir}")
        return None
    
    all_results = {}
    summary_data = []
    h0_images_list = []
    h1_images_list = []
    image_names_list = []
    
    
    for csv_file in sorted(csv_files):
        csv_path = os.path.join(measurements_dir, csv_file)
        
        result = analyze_single_image(
            csv_path,
            output_dir=diagrams_dir,
            max_dimension=max_dimension,
            max_edge_length=max_edge_length,
            show_plots=show_plots
        )
        
        if result is not None:
            image_name = os.path.splitext(csv_file)[0].replace('_measurements', '')
            all_results[image_name] = result
            
            summary_row = {'image': image_name, 'num_cells': len(result['coordinates'])}
            summary_row.update(result['statistics'])
            summary_data.append(summary_row)
            
            dgms = result['result']['dgms']
            
            for dim, dgm in enumerate(dgms):
                finite_dgm = dgm[dgm[:, 1] != np.inf]
                dgm_df = pd.DataFrame(finite_dgm, columns=['Birth', 'Death'])
                dgm_df.to_csv(os.path.join(diagrams_dir, f"{image_name}_H{dim}_dgm.csv"), index=False)
            
            h0_pi = persistence_image(dgms, dimension=0, sigma=0.25, pixsz=pixsz)
            h1_pi = persistence_image(dgms, dimension=1, sigma=0.25, pixsz=pixsz)
            
            if isinstance(h0_pi, tuple):
                h0_pi = h0_pi[0]
            if isinstance(h1_pi, tuple):
                h1_pi = h1_pi[0]
            
            if h0_pi is None or h1_pi is None:
                print(f"Warning: Skipping {image_name} - invalid persistence image")
                continue
            
            if not hasattr(h0_pi, 'shape') or not hasattr(h1_pi, 'shape'):
                print(f"Warning: Skipping {image_name} - no shape attribute")
                continue
            
            if len(h0_pi.shape) != 2 or len(h1_pi.shape) != 2:
                print(f"Warning: Skipping {image_name} - invalid shape")
                continue
            
            print(f"DEBUG {image_name} H0 original shape: {h0_pi.shape}")
            col0_sum = np.sum(h0_pi[:, 0])
            col_last_sum = np.sum(h0_pi[:, -1])
            print(f"  H0: Column 0 sum: {col0_sum:.4f}")
            print(f"  H0: Column -1 sum: {col_last_sum:.4f}")
            if col_last_sum > col0_sum:
                print(f"  H0: Profile is on the RIGHT side - will flip to LEFT")
            
            h0_resized = resize_h0_left_aligned(h0_pi, image_height, image_width, h0_profile_width)
            h1_resized = resize_h1_centered(h1_pi, image_height, image_width)
            
            col0_sum_resized = np.sum(h0_resized[:, 0])
            nonzero_cols = np.sum(np.any(h0_resized != 0, axis=0))
            print(f"  H0 resized: Column 0 sum: {col0_sum_resized:.4f}")
            print(f"  H0 resized: Non-zero columns: {nonzero_cols} (should be {h0_profile_width})")
            
            h0_images_list.append(h0_resized)
            h1_images_list.append(h1_resized)
            image_names_list.append(image_name)
            
            np.save(os.path.join(h0_pi_dir, f"{image_name}_H0_resized.npy"), h0_resized)
            np.save(os.path.join(h1_pi_dir, f"{image_name}_H1_resized.npy"), h1_resized)
            
            # Plot H0
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(h0_resized, origin='lower', cmap='viridis', aspect='auto')
            ax.axvline(x=0, color='red', linestyle='-', alpha=0.5, linewidth=2, label='Birth=0 (left edge)')
            ax.legend()
            ax.set_xlabel('Birth (left-aligned)', fontsize=12)
            ax.set_ylabel('Lifetime', fontsize=12)
            ax.set_title(f'H0 Left-Aligned - {image_name}', fontsize=14)
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(h0_pi_dir, f"{image_name}_H0_resized.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot H1
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(h1_resized, origin='lower', cmap='viridis', aspect='auto')
            ax.set_xlabel('Birth', fontsize=12)
            ax.set_ylabel('Lifetime', fontsize=12)
            ax.set_title(f'H1 Centered - {image_name}', fontsize=14)
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(h1_pi_dir, f"{image_name}_H1_resized.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Combined visualization
            combined_vis = np.hstack([h0_resized, h1_resized])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(combined_vis, origin='lower', cmap='viridis', aspect='auto')
            ax.axvline(x=image_width - 0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.text(image_width/2, -2, 'H0 (vertical profile)', ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax.text(image_width + image_width/2, -2, 'H1 (2D image)', ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax.set_title(f'Combined H0 (left) + H1 (right) - {image_name}', fontsize=14)
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(concatenated_dir, f"{image_name}_combined.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"DEBUG {image_name}:")
            print(f"  H0 resized shape: {h0_resized.shape}, max: {np.max(h0_resized):.4f}")
            print(f"  H1 resized shape: {h1_resized.shape}, max: {np.max(h1_resized):.4f}")
    
    if not all_results:
        print("No valid persistence diagrams generated")
        return None
    
    n_images = len(all_results)
    print(f"Total images processed: {n_images}")
    show_legend = n_images <= legend_threshold
    print(f"Showing legend on scatter plots: {show_legend} (threshold={legend_threshold})")
    
    if save_summary and summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "tda_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
    
    # Create combined feature vectors for PCA
    if h0_images_list and h1_images_list:
        pi_list = []
        for h0_img, h1_img in zip(h0_images_list, h1_images_list):
            combined_vec = np.hstack([h0_img.flatten(), h1_img.flatten()])
            pi_list.append(combined_vec)
        
        pis_array = np.array(pi_list)
        print(f"DEBUG Final PI array shape: {pis_array.shape}")
        
        # Check variance of H0 and H1 separately
        h0_size = image_height * image_width
        h0_features = pis_array[:, :h0_size]
        h1_features = pis_array[:, h0_size:]
        
        h0_variance = np.var(h0_features, axis=0)
        h1_variance = np.var(h1_features, axis=0)
        
        print(f"H0 variance: min={np.min(h0_variance):.6f}, max={np.max(h0_variance):.6f}, mean={np.mean(h0_variance):.6f}")
        print(f"H1 variance: min={np.min(h1_variance):.6f}, max={np.max(h1_variance):.6f}, mean={np.mean(h1_variance):.6f}")
        print(f"H1 non-zero variance features: {np.sum(h1_variance > 1e-10)} out of {len(h1_variance)}")
        
        if np.max(h1_variance) < 1e-10:
            print("WARNING: H1 features have NO variance! H1 will not appear in PCA.")
        elif np.mean(h1_variance) < 0.01 * np.mean(h0_variance):
            print(f"WARNING: H1 variance is {np.mean(h1_variance)/np.mean(h0_variance):.2%} of H0 variance. H1 may be invisible in PC1.")
        
        pca_result = run_pca(
            pis_array, 
            image_names_list, 
            image_height,
            image_width,
            pixsz=pixsz,
            output_dir=pca_dir,
            show_legend=show_legend
        )
        
        np.save(os.path.join(pca_dir, "pca_result.npy"), pca_result)
    
    # Compute Wasserstein distances and MDS - ONLY if skip_wasserstein is False
    if not skip_wasserstein and all_results:
        print("Computing Wasserstein distances and MDS... (this may take a while)")
        image_names_sorted = sorted(all_results.keys())
        diagram_list = [all_results[name]['result']['dgms'] for name in image_names_sorted]
        
        for dimension in range(max_dimension + 1):
            heatmap_path = os.path.join(wasserstein_dir, f"wasserstein_heatmap_H{dimension}.png")
            wds = wasserstein_distance(diagram_list, dimension=dimension, 
                                      save_path=heatmap_path, show=show_plots)
            
            wds_df = pd.DataFrame(wds, index=image_names_sorted, columns=image_names_sorted)
            wds_path = os.path.join(wasserstein_dir, f"wasserstein_distances_H{dimension}.csv")
            wds_df.to_csv(wds_path)
            print(f"Wasserstein distance matrix saved to {wds_path}")
        
        for dimension in range(max_dimension + 1):
            wds_path = os.path.join(wasserstein_dir, f"wasserstein_distances_H{dimension}.csv")
            mds_output_path = os.path.join(pca_dir, f"mds_wasserstein_H{dimension}.png")
            title = f"MDS: Wasserstein Distances (H_{dimension})"
            
            mds_from_wasserstein_csv(wds_path, output_path=mds_output_path, title=title, show=show_plots)
            print(f"MDS plot saved to {mds_output_path}")
    else:
        print("Skipping Wasserstein distances and MDS (skip_wasserstein=True)")
    
    return all_results

def run_pca(pis, labels, image_height, image_width, pixsz=0.15, output_dir=None, show_legend=True):

    fs = 14
    
    extent = [0, 2 * image_width * pixsz, 0, image_height * pixsz]
    xticks = [0, image_width * pixsz, 2 * image_width * pixsz]
    xlabs = ['H0', 'H1', '']
    
    print(f"PCA Debug: pis.shape = {pis.shape}")
    print(f"PCA Debug: image_height = {image_height}, image_width = {image_width}")
    print(f"PCA Debug: pixsz = {pixsz}")
    print(f"PCA Debug: show_legend = {show_legend}")
    
    # Standardize the data (center only, no scaling)
    scaler = StandardScaler(copy=True, with_std=False, with_mean=True).fit(pis)
    data = scaler.transform(pis)
    
    # PCA
    PCA = decomposition.PCA(n_components=6, random_state=42, svd_solver='full').fit(data)
    print('Considering the first', PCA.n_components,'PCs')
    
    pca = PCA.transform(data).astype('float32')
    explained_ratio = 100 * PCA.explained_variance_ratio_
    print(f"Explained ratio: {explained_ratio}")
    print(f"Total explained var: {np.sum(explained_ratio)}")
    
    # Scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    for i, ulab in enumerate(unique_labels):
        idx = [j for j, lab in enumerate(labels) if lab == ulab]
        ax.scatter(pca[idx, 0], pca[idx, 1], fc=colors[i], ec='k', marker='o', s=100, alpha=0.8, label=ulab)
    
    ax.set_aspect('equal', 'datalim')
    ax.set_title('PCA: Persistence Images (H0 + H1)', fontsize=fs)
    ax.set_xlabel('PC 01 [{:.1f}%]'.format(explained_ratio[0]), fontsize=fs)
    ax.set_ylabel('PC 02 [{:.1f}%]'.format(explained_ratio[1]), fontsize=fs)
    
    # Only add legend if show_legend is True AND we have fewer than some reasonable number of labels
    if show_legend and len(unique_labels) <= 30:
        ax.legend(fontsize=0.85*fs, loc='best')
    else:
        print(f"  Legend hidden (show_legend={show_legend}, n_labels={len(unique_labels)})")
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Loadings
    loadings = PCA.components_.T * np.sqrt(PCA.explained_variance_)
    
    print(f"PCA Debug: loadings shape: {loadings.shape}")
    
    fig_raw, ax_raw = plt.subplots(2, 3, figsize=(10, 4), sharex=True, sharey=True)
    ax_raw = np.atleast_1d(ax_raw).ravel()
    
    for i in range(loadings.shape[1]):
        # Reshape: H0 on LEFT, H1 on RIGHT
        ll = loadings[:, i].reshape(2 * image_width, image_height, order='C').T
        
        # Debug: Check H0 and H1 magnitudes
        h0_part = ll[:, :image_width]
        h1_part = ll[:, image_width:]
        h0_max = np.max(np.abs(h0_part))
        h1_max = np.max(np.abs(h1_part))
        print(f"PC {i+1} RAW: H0 max={h0_max:.6f}, H1 max={h1_max:.6f}")
        
        # Percentile-based scaling for raw loadings
        vmin_percentile = np.percentile(ll, 2)
        vmax_percentile = np.percentile(ll, 98)
        vmax_sym = max(abs(vmin_percentile), abs(vmax_percentile))
        
        if vmax_sym < 1e-10:
            vmax_sym = 1.0
        
        im = ax_raw[i].imshow(ll, cmap='coolwarm', vmin=-vmax_sym, vmax=vmax_sym, 
                             origin='lower', extent=extent, interpolation='nearest')
        ax_raw[i].set_xlabel('PC {:02d} ({:.1f}%)'.format(i+1, explained_ratio[i]), fontsize=fs)
        ax_raw[i].axvline(image_width * pixsz, c='k', lw=0.5)
        ax_raw[i].set_xticks(xticks)
        ax_raw[i].set_xticklabels(xlabs)
        ax_raw[i].tick_params(labelsize=.85*fs)
        plt.colorbar(im, ax=ax_raw[i], fraction=0.046, pad=0.04)
        
        for hdim in [0, 1]:
            ax_raw[i].text(.99*(hdim+1)*image_width*pixsz, .99*extent[-1], 
                      '$H_{}$'.format(hdim), c='k', ha='right', va='top', fontsize=fs)
    
    ax_raw[1].set_title('PCA Loadings (RAW - shows true magnitudes)', fontsize=1.15*fs)
    fig_raw.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_loadings_raw.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    fig_scaled, ax_scaled = plt.subplots(2, 3, figsize=(10, 4), sharex=True, sharey=True)
    ax_scaled = np.atleast_1d(ax_scaled).ravel()
    
    for i in range(loadings.shape[1]):
        ll = loadings[:, i].reshape(2 * image_width, image_height, order='C').T
        
        # Split into H0 and H1
        h0_part = ll[:, :image_width]
        h1_part = ll[:, image_width:]
        
        # Separate scaling for visualization only
        h0_vmax = max(np.percentile(np.abs(h0_part), 95), 1e-6)
        h1_vmax = max(np.percentile(np.abs(h1_part), 95), 1e-6)
        
        h0_scaled = np.clip(h0_part / h0_vmax, -1, 1)
        h1_scaled = np.clip(h1_part / h1_vmax, -1, 1)
        
        ll_scaled = np.hstack([h0_scaled, h1_scaled])
        
        print(f"PC {i+1} SCALED: H0_vmax={h0_vmax:.6f}, H1_vmax={h1_vmax:.6f}")
        print(f"PC {i+1} SCALED: H0/H1 ratio = {h0_vmax/h1_vmax:.2f}x")
        
        im = ax_scaled[i].imshow(ll_scaled, cmap='coolwarm', vmin=-1, vmax=1, 
                                origin='lower', extent=extent, interpolation='nearest')
        ax_scaled[i].set_xlabel('PC {:02d} ({:.1f}%)'.format(i+1, explained_ratio[i]), fontsize=fs)
        ax_scaled[i].axvline(image_width * pixsz, c='k', lw=0.5)
        ax_scaled[i].set_xticks(xticks)
        ax_scaled[i].set_xticklabels(xlabs)
        ax_scaled[i].tick_params(labelsize=.85*fs)
        plt.colorbar(im, ax=ax_scaled[i], fraction=0.046, pad=0.04)
        
        for hdim in [0, 1]:
            ax_scaled[i].text(.99*(hdim+1)*image_width*pixsz, .99*extent[-1], 
                      '$H_{}$'.format(hdim), c='k', ha='right', va='top', fontsize=fs)
    
    ax_scaled[1].set_title('PCA Loadings (SEPARATE SCALING - shows H0 & H1 structure)', fontsize=1.15*fs)
    fig_scaled.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_loadings_scaled.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return pca