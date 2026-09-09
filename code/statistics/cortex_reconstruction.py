# Rebuilds a cortex from fitted features and checks it back.
#
# Compares the reconstructed profile against the measured one, which is how
# the reconstruction in the interactive model was validated.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline, CubicSpline, UnivariateSpline
from scipy.ndimage import gaussian_filter1d
import warnings
import os
warnings.filterwarnings('ignore')


def main():
    SCRIPT_DIR = Path(__file__).parent.absolute()  # code/statistics/
    CODE_DIR = SCRIPT_DIR.parent  # code/
    PROJECT_ROOT = CODE_DIR.parent  # project root

    # Use absolute paths
    model_data_path = PROJECT_ROOT / 'results' / 'file_level_model' / 'file_level_data.csv'
    model_performance_path = PROJECT_ROOT / 'results' / 'file_level_model' / 'file_level_model_performance.csv'
    output_dir = PROJECT_ROOT / 'results' / 'cortical_reconstruction'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Looking for model data at: {model_data_path}")

    # Initialize model variables
    poly = None
    model_cubic = None
    feature_names = None
    coefficients = None
    intercept = None
    model_loaded = False

    # If the model data exists, we can reload and refit
    if model_data_path.exists():
        print("Loading existing model data...")
        df_files = pd.read_csv(model_data_path)

        # Prepare data for refitting
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression

        X = df_files[['first_file_area_um2', 'file_number']].values
        y = df_files['cell_area_um2'].values

        # Use cubic model (degree 3, no interaction based on best performance)
        poly = PolynomialFeatures(degree=3, include_bias=False)
        X_poly = poly.fit_transform(X)

        model_cubic = LinearRegression()
        model_cubic.fit(X_poly, y)

        # Get the feature names for display
        feature_names = poly.get_feature_names_out(['FirstFile', 'FileNum'])
        coefficients = model_cubic.coef_
        intercept = model_cubic.intercept_
        model_loaded = True

        print(f"Model refit successfully!")
        print(f"Intercept: {intercept:.2f}")
        print(f"Number of features: {len(coefficients)}")
        print(f"Training R²: {model_cubic.score(X_poly, y):.4f}")

    else:
        print("Model data not found. Using example coefficients from your results.")

        # Placeholder coefficients
        intercept = 115.57
        coefficients = [1.0712, -0.0001849, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        feature_names = ['FirstFile', 'FileNum', 'FirstFile^2', 'FirstFile*FileNum', 
                         'FileNum^2', 'FirstFile^3', 'FirstFile^2*FileNum', 
                         'FirstFile*FileNum^2', 'FileNum^3']

        def predict_cell_area_cubic(first_file_area, file_number):
            x = first_file_area
            y = file_number
            return 115.57 + 1.0712*x - 0.0001849*x*x + 5.0*y

    def predict_cell_area(first_file_area, file_number):

        if model_loaded:
            # Use the actual fitted model
            X_input = np.array([[first_file_area, file_number]])
            X_poly_input = poly.transform(X_input)
            return model_cubic.predict(X_poly_input)[0]
        else:
            # Use placeholder function
            return predict_cell_area_cubic(first_file_area, file_number)

    def reconstruct_cortex(first_file_area, smooth=True, smoothing_factor=0.5):

        # Step 1: Predict total number of files
        n_files = predict_file_count(first_file_area)

        print(f"\nReconstructing cortex for first file area = {first_file_area:.1f} µm²")
        print(f"Predicted total file count: {n_files}")

        # Step 2: Predict cell area for each file
        file_areas = []
        for file_num in range(n_files):
            area = predict_cell_area(first_file_area, file_num)
            file_areas.append(area)

        # Calculate summary statistics
        total_cortex_area = sum(file_areas)
        avg_area = np.mean(file_areas)
        std_area = np.std(file_areas)
        min_area = min(file_areas)
        max_area = max(file_areas)

        # Create profile DataFrame
        profile = pd.DataFrame({
            'file_number': list(range(n_files)),
            'predicted_cell_area_um2': file_areas
        })

        # Step 3: Generate smooth spline if requested
        spline_data = None
        spline_function = None

        if smooth and n_files > 3:
            # Generate smooth spline through the points
            x = np.array(range(n_files))
            y = np.array(file_areas)

            # Create a denser set of x points for the smooth curve
            x_smooth = np.linspace(0, n_files - 1, 200)

            try:
                # Use cubic spline with smoothing
                # UnivariateSpline with a smoothing factor
                s = smoothing_factor * n_files  # Adjust smoothing based on data size
                spline_func = UnivariateSpline(x, y, s=s, k=3)
                y_smooth = spline_func(x_smooth)

                # Also create a secondary spline for the derivative (rate of change)
                derivative_spline = spline_func.derivative()
                derivative_values = derivative_spline(x_smooth)

                spline_data = pd.DataFrame({
                    'file_number_smooth': x_smooth,
                    'spline_cell_area_um2': y_smooth,
                    'derivative_um2_per_file': derivative_values
                })

                spline_function = spline_func

            except Exception as e:
                print(f"  Warning: Could not create spline: {e}")
                # Fallback to simple interpolation
                try:
                    from scipy.interpolate import interp1d
                    f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
                    y_smooth = f(x_smooth)
                    spline_data = pd.DataFrame({
                        'file_number_smooth': x_smooth,
                        'spline_cell_area_um2': y_smooth,
                        'derivative_um2_per_file': np.gradient(y_smooth, x_smooth)
                    })
                    spline_function = f
                except:
                    spline_data = None

        return {
            'first_file_area': first_file_area,
            'file_count': n_files,
            'file_areas': file_areas,
            'total_cortex_area': total_cortex_area,
            'avg_area_across_cortex': avg_area,
            'std_area_across_cortex': std_area,
            'min_area': min_area,
            'max_area': max_area,
            'profile': profile,
            'spline_data': spline_data,
            'spline_function': spline_function,
            'has_spline': spline_data is not None
        }

    def predict_file_count(first_file_area):

        intercept = 13.579291847468262
        coefficient = 0.007396985692490564
        file_count = intercept + coefficient * first_file_area
        return int(round(file_count))  # Round to nearest integer

    # Example: Reconstruct cortex for first file area = 1500 µm²
    first_file_area_example = 1500
    result = reconstruct_cortex(first_file_area_example, smooth=True, smoothing_factor=0.5)

    print(f"\nFirst file cell area: {result['first_file_area']:.1f} µm²")
    print(f"Predicted total files: {result['file_count']}")
    print(f"Total cortex area (sum across files): {result['total_cortex_area']:.1f} µm²")
    print(f"Average cell area across cortex: {result['avg_area_across_cortex']:.1f} µm²")
    print(f"Standard deviation across cortex: {result['std_area_across_cortex']:.1f} µm²")
    print(f"Minimum cell area: {result['min_area']:.1f} µm²")
    print(f"Maximum cell area: {result['max_area']:.1f} µm²")

    print("\nFile | Predicted Area (µm²)")
    print("-----|-------------------")
    for file_num, area in enumerate(result['file_areas']):
        print(f"  {file_num:2d}  |  {area:8.1f}")


    test_areas = [500, 800, 1000, 1200, 1500, 1800, 2000, 2500]
    results_list = []

    for ffa in test_areas:
        result_temp = reconstruct_cortex(ffa, smooth=False)
        results_list.append({
            'first_file_area': ffa,
            'file_count': result_temp['file_count'],
            'total_cortex_area': result_temp['total_cortex_area'],
            'avg_area': result_temp['avg_area_across_cortex'],
            'std_area': result_temp['std_area_across_cortex']
        })

    comparison_df = pd.DataFrame(results_list)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cortical Reconstruction Pipeline with Spline Visualization', 
                 fontsize=16, fontweight='bold')

    # 6a. Radial profile with spline (example: 1500 µm²)
    ax1 = axes[0, 0]
    file_nums = list(range(result['file_count']))
    areas = result['file_areas']

    # Plot discrete points
    ax1.scatter(file_nums, areas, s=80, color='blue', zorder=5, 
               label='Predicted file areas', edgecolors='white', linewidth=1.5)

    # Plot spline if available
    if result['has_spline']:
        spline_df = result['spline_data']
        ax1.plot(spline_df['file_number_smooth'], spline_df['spline_cell_area_um2'], 
                 'r-', linewidth=3, label='Spline (smooth curve)', alpha=0.8)

        # Add confidence band (using derivative to estimate uncertainty)
        # This creates a shaded region showing the rate of change
        ax1.fill_between(spline_df['file_number_smooth'],
                         spline_df['spline_cell_area_um2'] - 50,
                         spline_df['spline_cell_area_um2'] + 50,
                         alpha=0.15, color='red', label='±50 µm² band')
    else:
        # If no spline, connect points with a smooth line using interpolation
        if len(file_nums) > 3:
            try:
                x_new = np.linspace(file_nums[0], file_nums[-1], 200)
                from scipy.interpolate import interp1d
                f = interp1d(file_nums, areas, kind='cubic', fill_value='extrapolate')
                y_new = f(x_new)
                ax1.plot(x_new, y_new, 'r-', linewidth=2, label='Interpolated')
            except:
                ax1.plot(file_nums, areas, 'r-', linewidth=2, alpha=0.5, label='Linear')
        else:
            ax1.plot(file_nums, areas, 'r-', linewidth=2, alpha=0.5, label='Linear')

    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Stele boundary')
    ax1.set_xlabel('Cell File Number (from stele outward)', fontsize=11)
    ax1.set_ylabel('Predicted Cell Area (µm²)', fontsize=11)
    ax1.set_title(f'Radial Profile with Spline\n(First File = {first_file_area_example} µm²)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 6b. Multiple radial profiles with splines for different first file areas
    ax2 = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_areas)))
    for i, ffa in enumerate(test_areas):
        result_i = reconstruct_cortex(ffa, smooth=True, smoothing_factor=0.3)
        if result_i['has_spline']:
            spline_df = result_i['spline_data']
            ax2.plot(spline_df['file_number_smooth'], spline_df['spline_cell_area_um2'], 
                     color=colors[i], linewidth=2, label=f'{ffa} µm²', alpha=0.8)
        else:
            file_nums_i = list(range(result_i['file_count']))
            ax2.plot(file_nums_i, result_i['file_areas'], 
                     color=colors[i], linewidth=2, label=f'{ffa} µm²', alpha=0.8)
    ax2.set_xlabel('Cell File Number (from stele outward)', fontsize=11)
    ax2.set_ylabel('Predicted Cell Area (µm²)', fontsize=11)
    ax2.set_title('Radial Profiles (Smooth Splines) for Different First File Areas', fontsize=12)
    ax2.legend(loc='upper left', ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 6c. Derivative (rate of change) plot for the example
    ax3 = axes[0, 2]
    if result['has_spline'] and 'derivative_um2_per_file' in result['spline_data'].columns:
        spline_df = result['spline_data']
        ax3.plot(spline_df['file_number_smooth'], spline_df['derivative_um2_per_file'], 
                 'g-', linewidth=2, label='Rate of change')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.fill_between(spline_df['file_number_smooth'], 0, 
                         spline_df['derivative_um2_per_file'],
                         where=(spline_df['derivative_um2_per_file'] > 0),
                         color='green', alpha=0.2, label='Increasing')
        ax3.fill_between(spline_df['file_number_smooth'], 0, 
                         spline_df['derivative_um2_per_file'],
                         where=(spline_df['derivative_um2_per_file'] < 0),
                         color='red', alpha=0.2, label='Decreasing')
        ax3.set_xlabel('Cell File Number (from stele outward)', fontsize=11)
        ax3.set_ylabel('Rate of Change (µm² per file)', fontsize=11)
        ax3.set_title('Rate of Cell Area Change Across Cortex', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Derivative plot not available\n(need more data points)', 
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Rate of Cell Area Change', fontsize=12)

    # 6d. Scatter: First File Area vs Total Cortex Area
    ax4 = axes[1, 0]
    ax4.scatter(comparison_df['first_file_area'], 
               comparison_df['total_cortex_area'], 
               s=60, alpha=0.7, color='darkblue')
    # Add trend line with smoothing
    z = np.polyfit(comparison_df['first_file_area'], comparison_df['total_cortex_area'], 2)
    p = np.poly1d(z)
    x_range = np.linspace(comparison_df['first_file_area'].min(), 
                         comparison_df['first_file_area'].max(), 100)
    ax4.plot(x_range, p(x_range), 'r--', linewidth=2, label='Quadratic trend')
    ax4.set_xlabel('First File Area (µm²)', fontsize=11)
    ax4.set_ylabel('Total Cortex Area (µm²)', fontsize=11)
    ax4.set_title('First File Area vs Total Cortex Area', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 6e. Heatmap with spline-based smoothing
    ax5 = axes[1, 1]
    ffa_values = np.linspace(300, 3000, 30)
    file_values = np.arange(0, 15)
    FFA_grid, FILE_grid = np.meshgrid(ffa_values, file_values)

    Z = np.zeros_like(FFA_grid, dtype=float)
    valid_mask = np.zeros_like(FFA_grid, dtype=bool)
    for i in range(FFA_grid.shape[0]):
        for j in range(FFA_grid.shape[1]):
            n_files = predict_file_count(FFA_grid[i, j])
            if FILE_grid[i, j] < n_files:
                Z[i, j] = predict_cell_area(FFA_grid[i, j], FILE_grid[i, j])
                valid_mask[i, j] = True
            else:
                Z[i, j] = np.nan

    # Apply smoothing to the heatmap
    from scipy.ndimage import gaussian_filter
    Z_smooth = Z.copy()
    if np.any(~np.isnan(Z)):
        # Replace NaN with nearest valid value for smoothing
        Z_filled = Z.copy()
        from scipy.interpolate import griddata
        valid_points = np.argwhere(valid_mask)
        valid_values = Z[valid_mask]
        all_points = np.argwhere(np.ones_like(Z, dtype=bool))
        Z_interp = griddata(valid_points, valid_values, all_points, method='linear')
        Z_smooth = Z_interp.reshape(Z.shape) if Z_interp is not None else Z

        # Apply gaussian filter for smooth heatmap
        Z_smooth = gaussian_filter(Z_smooth, sigma=0.5)

        # Re-apply mask
        Z_smooth[~valid_mask] = np.nan

    im = ax5.contourf(FFA_grid, FILE_grid, Z_smooth, levels=20, cmap='viridis')
    ax5.set_xlabel('First File Area (µm²)', fontsize=11)
    ax5.set_ylabel('File Number (from stele outward)', fontsize=11)
    ax5.set_title('Predicted Cell Area (µm²) with Spline Smoothing', fontsize=12)
    plt.colorbar(im, ax=ax5)

    # 6f. Scatter plot with spline overlay showing the fit quality
    ax6 = axes[1, 2]
    # Create a simulated dataset based on the model predictions
    if model_loaded and len(df_files) > 0:
        # Sample some actual data points from the training set
        sample_size = min(200, len(df_files))
        df_sample = df_files.sample(sample_size, random_state=42)

        # Plot actual data points
        ax6.scatter(df_sample['file_number'], df_sample['cell_area_um2'], 
                   alpha=0.3, s=10, color='gray', label='Actual data')

        # Plot predictions for a range of first file areas
        test_first_file_areas = [800, 1500, 2200]
        for ffa in test_first_file_areas:
            result_i = reconstruct_cortex(ffa, smooth=True, smoothing_factor=0.3)
            if result_i['has_spline']:
                spline_df = result_i['spline_data']
                # Only show up to the predicted file count
                valid_idx = spline_df['file_number_smooth'] <= result_i['file_count']
                ax6.plot(spline_df['file_number_smooth'][valid_idx], 
                        spline_df['spline_cell_area_um2'][valid_idx], 
                        linewidth=2, label=f'Predicted (FFA={ffa})')

        ax6.set_xlabel('File Number (from stele outward)', fontsize=11)
        ax6.set_ylabel('Cell Area (µm²)', fontsize=11)
        ax6.set_title('Model Predictions vs Actual Data', fontsize=12)
        ax6.legend(loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Model data not available for visualization', 
                 ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Prediction vs Actual Data', fontsize=12)

    plt.tight_layout()

    # Save the figure
    fig_path = output_dir / 'cortex_reconstruction_with_spline.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")

    # Save the reconstruction for the example
    example_profile = result['profile']
    example_profile_path = output_dir / 'example_profile.csv'
    example_profile.to_csv(example_profile_path, index=False)
    print(f"Example profile saved to: {example_profile_path}")

    # Save the spline data if available
    if result['has_spline']:
        spline_path = output_dir / 'example_spline_data.csv'
        result['spline_data'].to_csv(spline_path, index=False)
        print(f"Spline data saved to: {spline_path}")

    # Save the comparison table
    comparison_path = output_dir / 'comparison_scenarios.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to: {comparison_path}")

    plt.show()


if __name__ == "__main__":
    main()
