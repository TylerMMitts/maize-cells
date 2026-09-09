# PCA over the persistence-image vectors.
#
# A linear alternative to MDS on the same data, kept so the two can be
# compared.

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import decomposition, preprocessing

def pca(pis, labels, height, width, h0_width=None, h1_width=None, pixsz=0.1, output_dir=None):

    fs = 14
    
    # If h0_width and h1_width not provided, assume equal split
    if h0_width is None or h1_width is None:
        h0_width = width // 2
        h1_width = width - h0_width
    
    # Extent for visualization
    extent = [0, width * pixsz, 0, height * pixsz]

    print(f"PCA Debug: pis.shape = {pis.shape}")
    print(f"PCA Debug: height = {height}, width = {width}")
    print(f"PCA Debug: h0_width = {h0_width}, h1_width = {h1_width}")
    print(f"PCA Debug: Expected combined image shape: ({height}, {width})")
    
    # Standardize the data
    scaler = preprocessing.StandardScaler(copy=True, with_std=False, with_mean=True).fit(pis)
    data = scaler.transform(pis)
    
    # PCA
    PCA = decomposition.PCA(n_components=6, random_state=42, svd_solver='full').fit(data)
    pca_result = PCA.transform(data).astype('float32')
    explained_ratio = 100 * PCA.explained_variance_ratio_

    # Scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    for i, ulab in enumerate(unique_labels):
        idx = [j for j, lab in enumerate(labels) if lab == ulab]
        ax.scatter(pca_result[idx, 0], pca_result[idx, 1], 
                  fc=colors[i], ec='k', marker='o', s=100, alpha=0.8, label=ulab)

    ax.set_aspect('equal', 'datalim')
    ax.set_title('PCA: Persistence Images (Padded)', fontsize=fs)
    ax.set_xlabel('PC 01 [{:.1f}%]'.format(explained_ratio[0]), fontsize=fs)
    ax.set_ylabel('PC 02 [{:.1f}%]'.format(explained_ratio[1]), fontsize=fs)
    ax.legend(fontsize=0.85*fs, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Loadings plot
    loadings = PCA.components_.T * np.sqrt(PCA.explained_variance_)

    fig, ax = plt.subplots(2, 3, figsize=(12, 5), sharex=True, sharey=True)
    ax = np.atleast_1d(ax).ravel()

    for i in range(loadings.shape[1]):
        loading_vec = loadings[:, i]
        
        # Split into H0 and H1 parts
        h0_size = height * h0_width
        h1_size = height * h1_width
        
        print(f"PCA Debug PC{i+1}: loading_vec.shape = {loading_vec.shape}")
        print(f"PCA Debug PC{i+1}: h0_size = {h0_size}, h1_size = {h1_size}")
        
        # Extract H0 and H1 loadings
        h0_loading = loading_vec[:h0_size].reshape(height, h0_width, order='C')
        h1_loading = loading_vec[h0_size:h0_size+h1_size].reshape(height, h1_width, order='C')
        
        # Combine side by side
        ll = np.hstack([h0_loading, h1_loading])
        
        # Determine color limits
        vmin_percentile = np.percentile(ll, 2)
        vmax_percentile = np.percentile(ll, 98)
        vmax_sym = max(abs(vmin_percentile), abs(vmax_percentile))
        
        # Plot
        im = ax[i].imshow(ll, cmap='coolwarm', vmin=-vmax_sym, vmax=vmax_sym, 
                         origin='lower', extent=extent, interpolation='nearest')
        ax[i].set_xlabel('PC {:02d} ({:.1f}%)'.format(i+1, explained_ratio[i]), fontsize=fs)
        
        # Add vertical line separating H0 and H1
        ax[i].axvline(h0_width * pixsz, c='k', lw=0.5)
        
        # Set xticks
        xticks = [0, h0_width * pixsz, width * pixsz]
        xlabs = ['H0', 'H1', '']
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xlabs)
        ax[i].tick_params(labelsize=.85*fs)
        
        # Add colorbar
        plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        
        # Add dimension labels
        for hdim, w in [(0, h0_width), (1, h1_width)]:
            x_pos = .99 * (w * pixsz if hdim == 0 else (h0_width + h1_width) * pixsz)
            ax[i].text(x_pos, .99 * extent[-1], 
                      '$H_{}$'.format(hdim), c='k', ha='right', va='top', fontsize=fs)

    ax[1].set_title('PCA Loadings (Padded Images)', fontsize=1.15*fs)
    fig.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_loadings.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return pca_result