# Fits one curve to the normalised pattern.
#
# Takes an already-processed CSV and fits the pooled shape, so the fit can be
# redone without re-reading every measurement file.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from code.config import RESULTS_FOLDER

def fit_normalized_pattern_from_csv(csv_path, output_folder=RESULTS_FOLDER / 'non_tda_results'):
    
    data = pd.read_csv(csv_path)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Recreate the normalized average pattern from the raw data
    images = data['filename'].unique()
    all_normalized_data = []
    
    for filename in images:
        image_data = data[data['filename'] == filename].copy()
        
        # Normalize radius
        min_radius = image_data['radius_pixels'].min()
        max_radius = image_data['radius_pixels'].max()
        if max_radius > min_radius:
            image_data['radius_normalized'] = (image_data['radius_pixels'] - min_radius) / (max_radius - min_radius)
        else:
            image_data['radius_normalized'] = 0.5
        
        # Bin normalized radius
        image_data['radius_binned'] = pd.cut(image_data['radius_normalized'], bins=20, labels=False) / 20.0
        
        # Average cell size per bin
        averaged = image_data.groupby('radius_binned')['area_pixels'].mean().reset_index()
        
        if len(averaged) > 0:
            # Normalize cell size
            min_area = averaged['area_pixels'].min()
            max_area = averaged['area_pixels'].max()
            if max_area > min_area:
                averaged['normalized_area'] = (averaged['area_pixels'] - min_area) / (max_area - min_area)
            else:
                averaged['normalized_area'] = 0.5
            
            all_normalized_data.append(averaged[['radius_binned', 'normalized_area']])
    
    if not all_normalized_data:
        print("No normalized data")
        return None
    
    combined_normalized = pd.concat(all_normalized_data, ignore_index=True)
    
    # Average normalized values by radius bin
    avg_pattern = combined_normalized.groupby('radius_binned').agg({
        'normalized_area': ['mean', 'std', 'count']
    }).reset_index()
    avg_pattern.columns = ['radius_binned', 'mean_normalized', 'std_normalized', 'count']
    avg_pattern = avg_pattern.sort_values('radius_binned').reset_index(drop=True)
    
    # Smooth
    avg_pattern['mean_smoothed'] = avg_pattern['mean_normalized'].rolling(window=3, center=True, min_periods=1).mean()
    
    # Prepare for fitting
    x_data = avg_pattern['radius_binned'].values
    y_data = avg_pattern['mean_smoothed'].values
    
    # Remove NaN
    valid_idx = ~np.isnan(y_data)
    x_data = x_data[valid_idx]
    y_data = y_data[valid_idx]
    
    # Define models
    def linear(x, a, b):
        return a * x + b
    
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    def cubic(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d
    
    def sigmoid(x, a, b, c, d):
        return a / (1 + np.exp(-b * (x - c))) + d
    
    def exponential_rise(x, a, b, c):
        return a * (1 - np.exp(-b * x)) + c
    
    models = {
        'Linear': linear,
        'Quadratic': quadratic,
        'Cubic': cubic,
        'Sigmoid': sigmoid,
        'Exponential': exponential_rise
    }
    
    results = {}
    for name, func in models.items():
        try:
            p0 = {
                'Linear': [1, 0],
                'Quadratic': [1, 1, 0],
                'Cubic': [1, 1, 1, 0],
                'Sigmoid': [1, 5, 0.5, 0],
                'Exponential': [1, 5, 0]
            }[name]
            
            popt, pcov = curve_fit(func, x_data, y_data, p0=p0, maxfev=10000)
            
            y_pred = func(x_data, *popt)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            results[name] = {'params': popt, 'r2': r2, 'func': func}
        except Exception as e:
            print(f"{name} failed: {e}")
    
    best_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best = results[best_name]
    
    # Print results
    print("NORMALIZED PATTERN FIT RESULTS")
    for name, res in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
        print(f"  {name}: R² = {res['r2']:.4f}, params = {res['params']}")
    print(f"BEST: {best_name} (R² = {best['r2']:.4f})")
    param_names = ['a', 'b', 'c', 'd'][:len(best['params'])]
    for pn, pv in zip(param_names, best['params']):
        print(f"  {pn} = {pv:.6f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_data, y_data, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Data')
    
    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
    y_smooth = best['func'](x_smooth, *best['params'])
    ax.plot(x_smooth, y_smooth, 'r--', linewidth=2.5, label=f'{best_name} fit (R² = {best["r2"]:.3f})')
    
    ax.set_xlabel('Normalized Radius', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Cell Size', fontsize=12, fontweight='bold')
    ax.set_title('Normalized Average Pattern with Best Fit', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path / 'normalized_pattern_with_fit.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'normalized_pattern_with_fit.png'}")
    plt.close()
    
    return best

if __name__ == "__main__":
    csv_path = RESULTS_FOLDER / 'non_tda' / 'non_tda_results_huge' / 'processed_data.csv'
    fit_normalized_pattern_from_csv(csv_path, RESULTS_FOLDER / 'non_tda' / 'non_tda_results_huge')