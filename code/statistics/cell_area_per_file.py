# Mean cell area per cell file, as a table.
#
# Flattens the per-image assignments into one row per file index so the
# outward size trend can be modelled directly.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import warnings
import os
warnings.filterwarnings('ignore')


def main():
    SCRIPT_DIR = Path(__file__).parent.absolute()  # code/statistics/
    CODE_DIR = SCRIPT_DIR.parent  # code/
    PROJECT_ROOT = CODE_DIR.parent  # project root

    # Use absolute paths
    cell_file_dir = PROJECT_ROOT / 'results' / 'cell_file' / 'cell_file_counting'
    output_dir = PROJECT_ROOT / 'results' / 'file_level_model'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Looking for cell files in: {cell_file_dir}")

    # Find all cell_assignments.csv files
    cell_files = list(cell_file_dir.rglob('cell_assignments.csv'))
    print(f"Found {len(cell_files)} cell_assignments.csv files")

    # Store per-file data
    all_file_data = []

    for cell_file in cell_files:
        root_name = cell_file.parent.name

        try:
            df_cells = pd.read_csv(cell_file)

            # Get first file area (cell_file_derivative == 0)
            first_file_cells = df_cells[df_cells['cell_file_derivative'] == 0]
            if len(first_file_cells) == 0:
                continue
            first_file_avg_area = first_file_cells['area_um2'].mean()

            # For each file, calculate the average area
            for file_num in sorted(df_cells['cell_file_derivative'].unique()):
                file_cells = df_cells[df_cells['cell_file_derivative'] == file_num]
                if len(file_cells) < 2:  # Skip files with too few cells
                    continue

                all_file_data.append({
                    'image_name': root_name,
                    'file_number': file_num,
                    'cell_area_um2': file_cells['area_um2'].mean(),
                    'cell_area_std': file_cells['area_um2'].std(),
                    'n_cells': len(file_cells),
                    'first_file_area_um2': first_file_avg_area
                })

            print(f"{root_name}: {len(df_cells['cell_file_derivative'].unique())} files extracted")

        except Exception as e:
            print(f"ERROR reading {cell_file}: {e}")

    # Convert to DataFrame
    df_files = pd.DataFrame(all_file_data)
    print(f"\nTotal file-level observations: {len(df_files)}")
    print(f"Total unique roots: {df_files['image_name'].nunique()}")
    print(f"File numbers range: {df_files['file_number'].min()} to {df_files['file_number'].max()}")

    # Drop any rows with missing values
    df_files = df_files.dropna()

    # Show the relationship between file number and cell area for a few sample roots
    print("\nSample radial profiles (area vs file number):")
    sample_roots = df_files.groupby('image_name').apply(
        lambda x: x['cell_area_um2'].std()
    ).sort_values().index[:5]

    for root in sample_roots:
        subset = df_files[df_files['image_name'] == root]
        print(f"\n  {root}:")
        for _, row in subset.sort_values('file_number').iterrows():
            print(f"    File {row['file_number']}: {row['cell_area_um2']:.1f} µm² (n={row['n_cells']})")


    # Features: First file area AND file number
    X = df_files[['first_file_area_um2', 'file_number']].values

    # Response: Cell area at that file
    y = df_files['cell_area_um2'].values

    # Split into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print(f"\nTraining set: {len(X_train)} observations")
    print(f"Testing set: {len(X_test)} observations")


    def build_model(degree, with_interaction=False):
        if degree == 1 and not with_interaction:
            # Simple linear: first_file_area + file_number
            model = LinearRegression()
            model.fit(X_train, y_train)
        else:
            # Polynomial with interaction
            # We need to create interaction terms manually
            from sklearn.preprocessing import PolynomialFeatures

            if with_interaction:
                # Include interaction between first_file_area and file_number
                poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                return model, poly
            else:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                return model, poly

        return model, None

    # Test different models
    models = {
        'Linear (No Interaction)': (1, False),
        'Linear + Interaction': (1, True),
        'Quadratic (No Interaction)': (2, False),
        'Quadratic + Interaction': (2, True),
        'Cubic (No Interaction)': (3, False),
        'Cubic + Interaction': (3, True),
    }

    results = {}

    for name, (degree, interaction) in models.items():
        if degree == 1 and not interaction:
            # Simple linear regression with 2 predictors
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            poly = None

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Cross-validation
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

            results[name] = {
                'model': model,
                'poly': poly,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'degree': degree,
                'interaction': interaction
            }
        else:
            # Polynomial with or without interaction
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_poly, y_train)

            y_train_pred = model.predict(X_train_poly)
            y_test_pred = model.predict(X_test_poly)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Cross-validation
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='r2')

            results[name] = {
                'model': model,
                'poly': poly,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'degree': degree,
                'interaction': interaction
            }

    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Training R²: {res['train_r2']:.4f}")
        print(f"  Testing R²:  {res['test_r2']:.4f}")
        print(f"  CV R²: {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
        print(f"  Test RMSE: {res['test_rmse']:.1f} µm²")

    # Select best model (highest test R²)
    best_name = max(results, key=lambda x: results[x]['test_r2'])
    best_result = results[best_name]

    print(f"Test R²: {best_result['test_r2']:.4f}")
    print(f"Test RMSE: {best_result['test_rmse']:.1f} µm²")


    if best_result['poly'] is None:
        # Simple linear model
        coefs = best_result['model'].coef_
        intercept = best_result['model'].intercept_

        print(f"\nEquation:")
        print(f"Cell Area = {intercept:.2f} + {coefs[0]:.4f} × (First File Area) + {coefs[1]:.4f} × (File Number)")

        # For a given first file area and file number, predict
        def predict_area(first_file_area, file_number):
            return intercept + coefs[0] * first_file_area + coefs[1] * file_number

    else:
        # Polynomial model
        coefs = best_result['model'].coef_
        intercept = best_result['model'].intercept_
        feature_names = best_result['poly'].get_feature_names_out(['FirstFile', 'FileNum'])

        print(f"\nEquation:")
        equation_parts = [f"{intercept:.2f}"]
        for name, coef in zip(feature_names, coefs):
            if abs(coef) > 1e-10:
                # Format the term nicely
                if '^' in name:
                    parts = name.split('^')
                    base = parts[0]
                    power = parts[1]
                    term = f"{coef:.6f} × ({base})^{power}"
                elif ' ' in name:
                    # Interaction term
                    terms = name.split(' ')
                    term = f"{coef:.6f} × ({terms[0]} × {terms[1]})"
                else:
                    term = f"{coef:.6f} × {name}"
                equation_parts.append(term)

        print("Cell Area = " + " + ".join(equation_parts))

        # For a given first file area and file number, predict
        def predict_area(first_file_area, file_number):
            X_input = np.array([[first_file_area, file_number]])
            X_poly = best_result['poly'].transform(X_input)
            return best_result['model'].predict(X_poly)[0]


    first_file_area = 1500  # µm²
    target_file = 4

    predicted_area = predict_area(first_file_area, target_file)

    print(f"\nGiven a root with first file cell area = {first_file_area} µm²,")
    print(f"the predicted average cell area at file {target_file} is:")
    print(f"  {predicted_area:.1f} µm²")

    # Show predictions for all files
    print(f"\nFull radial profile for this root (first file area = {first_file_area} µm²):")
    print("  File | Predicted Area")
    print("  -----|---------------")
    for file_num in range(0, 15):  # Files 0-14
        pred = predict_area(first_file_area, file_num)
        print(f"  {file_num:4d} | {pred:8.1f} µm²")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Predicting Cell Area at Any File Position\n(Best Model: {best_name})', 
                 fontsize=14, fontweight='bold')

    # 7a. Observed vs Predicted (Testing)
    ax1 = axes[0, 0]
    y_test_pred = []
    for i in range(len(X_test)):
        if best_result['poly'] is None:
            pred = best_result['model'].predict([X_test[i]])[0]
        else:
            X_poly = best_result['poly'].transform([X_test[i]])
            pred = best_result['model'].predict(X_poly)[0]
        y_test_pred.append(pred)

    ax1.scatter(y_test, y_test_pred, alpha=0.4, s=15)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    ax1.set_xlabel('Observed Cell Area (µm²)')
    ax1.set_ylabel('Predicted Cell Area (µm²)')
    ax1.set_title(f'Testing Set: R² = {best_result["test_r2"]:.3f}')
    ax1.grid(True, alpha=0.3)

    # 7b. Residuals
    ax2 = axes[0, 1]
    residuals = y_test - np.array(y_test_pred)
    ax2.scatter(y_test_pred, residuals, alpha=0.4, s=15)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Cell Area (µm²)')
    ax2.set_ylabel('Residuals (µm²)')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)

    # 7c. Radial profiles for different first file areas
    ax3 = axes[1, 0]
    first_file_areas = [800, 1200, 1500, 2000, 2500]
    file_range = range(0, 15)

    colors = plt.cm.viridis(np.linspace(0, 1, len(first_file_areas)))
    for i, ffa in enumerate(first_file_areas):
        preds = [predict_area(ffa, f) for f in file_range]
        ax3.plot(file_range, preds, label=f'First File = {ffa} µm²', color=colors[i], linewidth=2)

    ax3.set_xlabel('File Number (from stele outward)')
    ax3.set_ylabel('Predicted Cell Area (µm²)')
    ax3.set_title('Predicted Radial Profiles for Different First File Areas')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 7d. Heatmap of predictions
    ax4 = axes[1, 1]
    ffa_values = np.linspace(300, 3000, 30)
    file_values = np.arange(0, 15)
    FFA_grid, FILE_grid = np.meshgrid(ffa_values, file_values)

    Z = np.zeros_like(FFA_grid, dtype=float)
    for i in range(FFA_grid.shape[0]):
        for j in range(FFA_grid.shape[1]):
            Z[i, j] = predict_area(FFA_grid[i, j], FILE_grid[i, j])

    im = ax4.contourf(FFA_grid, FILE_grid, Z, levels=20, cmap='viridis')
    ax4.set_xlabel('First File Area (µm²)')
    ax4.set_ylabel('File Number (from stele outward)')
    ax4.set_title('Predicted Cell Area (µm²)')
    plt.colorbar(im, ax=ax4)

    plt.tight_layout()
    fig_path = output_dir / 'file_level_predictions.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")

    # Save the model performance summary
    results_df = pd.DataFrame([
        {
            'Model': name,
            'Degree': res['degree'],
            'Interaction': res['interaction'],
            'Train_R2': res['train_r2'],
            'Test_R2': res['test_r2'],
            'Train_RMSE': res['train_rmse'],
            'Test_RMSE': res['test_rmse'],
            'CV_R2_mean': res['cv_mean'],
            'CV_R2_std': res['cv_std']
        }
        for name, res in results.items()
    ])
    results_df.to_csv(output_dir / 'file_level_model_performance.csv', index=False)

    # Save the clean data
    df_files.to_csv(output_dir / 'file_level_data.csv', index=False)

    # Save the prediction function for reuse
    print(f"\nResults saved to: {output_dir}")

    plt.show()


if __name__ == "__main__":
    main()
