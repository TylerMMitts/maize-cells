import persim
import numpy as np
import matplotlib.pyplot as plt

def persistence_image(diagram, dimension=1, sigma=0.25, pixsz=0.15):

    births = diagram[dimension][:, 0]
    deaths = diagram[dimension][:, 1]

    lifetimes = deaths - births

    finite_mask = np.isfinite(births) & np.isfinite(lifetimes)

    finite_births = births[finite_mask]
    finite_lifetimes = lifetimes[finite_mask]

    # If there are no finite points, return a zero image
    if len(finite_births) == 0 or len(finite_lifetimes) == 0:
        print(f"Warning: No finite points for H{dimension}")
        # Return a small zero image
        return np.zeros((10, 10))
    
    lt_h1 = [np.column_stack((finite_births, finite_lifetimes))]
    max_birth = np.max(finite_births)
    max_lt = np.max(finite_lifetimes)

    # For H0, births are at 0, use symmetric range
    if dimension == 0:
        birth_min = -pixsz * np.ceil((3 * sigma) / pixsz)
        birth_max = pixsz * np.ceil((3 * sigma) / pixsz)
    else:
        birth_min = -pixsz * np.ceil((3 * sigma) / pixsz)
        birth_max = pixsz * np.ceil((max_birth + sigma) / pixsz)
    
    pers_max = pixsz * np.ceil((max_lt + sigma) / pixsz)

    pi_params = {
        'birth_range': (birth_min, birth_max),
        'pers_range': (0, pers_max),
        'pixel_size': pixsz,
        'weight': 'persistence',
        'weight_params': {'n': 1},
        'kernel': 'gaussian',
        'kernel_params': {'sigma': [[sigma, 0.0], [0.0, sigma]]}
    }

    pimgr = persim.PersistenceImager(**pi_params)
    extent = np.array([pimgr.birth_range[0], pimgr.birth_range[1], 
                       pimgr.pers_range[0], pimgr.pers_range[1]])

    pimgs = np.asarray(pimgr.transform(lt_h1, skew=False))
    
    # Extract the first (and only) image
    if len(pimgs.shape) == 3:
        pimg = pimgs[0]
    else:
        pimg = pimgs
    
    # Debug output
    if dimension == 0:
        print(f"H0 Debug: {len(finite_births)} points at birth≈0 with lifetimes up to {max_lt:.2f}")
        print(f"  Image shape: {pimg.shape} (birth_pixels={pimg.shape[0]}, persistence_pixels={pimg.shape[1]})")
        print(f"  Birth range: {extent[0]:.2f} to {extent[1]:.2f}")
        print(f"  Persistence range: {extent[2]:.2f} to {extent[3]:.2f}")
        print(f"  Pixel size: {pixsz}")
    elif dimension == 1:
        print(f"H1 Debug: {len(finite_births)} points with births up to {max_birth:.2f} and lifetimes up to {max_lt:.2f}")
        print(f"  Image shape: {pimg.shape} (birth_pixels={pimg.shape[0]}, persistence_pixels={pimg.shape[1]})")
        print(f"  Birth range: {extent[0]:.2f} to {extent[1]:.2f}")
        print(f"  Persistence range: {extent[2]:.2f} to {extent[3]:.2f}")

    # IMPORTANT: Return ONLY the image array, not a tuple
    return pimg