import numpy as np
import matplotlib.pyplot as plt
import persim


def wasserstein_distance(diagram_list, dimension=1, save_path=None, show=False):
    # Creates Wasserstein distance matrix by finding the best matching points between
    # persistence diagrams, then sums the distances for all the matched points to get the Wasserstein distance
    
    # Filter out points with infinite death times to avoid warnings
    filtered_diagrams = []
    for dgms in diagram_list:
        dgm = dgms[dimension]
        # Keep only points with finite death times
        finite_dgm = dgm[np.isfinite(dgm[:, 1])]
        filtered_diagrams.append(finite_dgm)
    
    wds = np.zeros((len(filtered_diagrams), len(filtered_diagrams)))
    for i in range(len(filtered_diagrams)-1):
        for j in range(i+1, len(filtered_diagrams)):
            wd = persim.wasserstein(filtered_diagrams[i], filtered_diagrams[j], matching=False)
            wds[i,j] = wd
            wds[j,i] = wd

    # Plot the persistence image for Wasserstein distances
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(wds, cmap='viridis')
    ax.set_title(f'Wasserstein Distance Persistence Image (H_{dimension})', fontsize=14)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Wasserstein Distance', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Wasserstein distance heatmap saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return wds