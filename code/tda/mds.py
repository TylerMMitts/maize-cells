import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition, manifold, metrics, preprocessing
import os

def load_wasserstein_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    distance_matrix = df.values
    sample_names = df.index.tolist()
    return distance_matrix, sample_names

def extract_folder_names(sample_names):
    folder_names = []
    for name in sample_names:
        parts = name.split('_')
        if len(parts) >= 2:
            folder_names.append(parts[0])
        else:
            folder_names.append(name)
    return folder_names

def mds_from_wasserstein_csv(csv_path, output_path=None, title=None, show=False):
    distance_matrix, sample_names = load_wasserstein_csv(csv_path)
    
    if output_path is None:
        csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f'results/tda_results/mds_{csv_basename}.png'
    
    if title is None:
        csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
        title = f'MDS: {csv_basename}'
    
    X_mds = mds_scaling(distance_matrix, sample_names=sample_names, output_path=output_path, title=title, show=show)
    return X_mds, sample_names

def mds_scaling(bns, labels=None, folder_names=None, sample_names=None, output_path='results/tda_results/mds_plot.png', title='MDS: Piecewise distances', show=False):
    # MDS parameters
    mds_kw = {'n_components': 2, 'dissimilarity': 'precomputed', 'random_state': 0}
    
    # Takes the distance matrix D, where Dij is the distance between
    # diagrams i and j. MDS then translates these distances into 2D coordinates by doing
    # a series of equations to preserve these distances as well as possible in 2D.
    X_mds = manifold.MDS(**mds_kw).fit_transform(bns)
    
    fs = 14
    legend_kw = {'frameon': True}
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.scatter(X_mds[:, 0], X_mds[:, 1], s=100, alpha=0.7, edgecolors='k')
    
    if sample_names is not None:
        for i, name in enumerate(sample_names):
            ax.text(X_mds[i, 0], X_mds[i, 1], f' {name}', fontsize=6, va='center', ha='left')
    
    ax.set_xlabel('MDS Dimension 1', fontsize=fs)
    ax.set_ylabel('MDS Dimension 2', fontsize=fs)
    ax.set_aspect('equal', 'datalim')
    ax.set_title(title, fontsize=fs)
    ax.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return X_mds
