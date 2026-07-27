import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Get project root
SCRIPT_DIR = Path(__file__).parent.absolute()  # code/statistics/
CODE_DIR = SCRIPT_DIR.parent  # code/
PROJECT_ROOT = CODE_DIR.parent  # project root


def main():
    # Use absolute paths (relative paths broke when the pipeline's cwd wasn't the project root)
    master_path = PROJECT_ROOT / 'results' / 'master_summary.csv'
    cell_file_dir = PROJECT_ROOT / 'results' / 'cell_file' / 'cell_file_counting'
    output_dir = PROJECT_ROOT / 'results' / 'predictive_model_stele'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load master summary
    df_master = pd.read_csv(master_path)
    print(f"\nMaster summary loaded: {len(df_master)} rows")


    # Find all cell_assignments.csv files
    cell_files = list(cell_file_dir.rglob('cell_assignments.csv'))
    print(f"Found {len(cell_files)} cell_assignments.csv files")

    # Dictionary to store first file average areas by image_name
    first_file_data = {}

    for cell_file in cell_files:
        # Extract image_name from the path
        root_folder = cell_file.parent.name
        image_name = root_folder

        try:
            df_cells = pd.read_csv(cell_file)

            # Check if cell_file_derivative column exists
            if 'cell_file_derivative' not in df_cells.columns:
                print(f"WARNING: cell_file_derivative not found in {image_name}")
                continue

            # Filter for cells in the first cortical file (cell_file_derivative == 0)
            first_file_cells = df_cells[df_cells['cell_file_derivative'] == 0]

            if len(first_file_cells) > 0:
                # Calculate average area of first file cells
                avg_area = first_file_cells['area_um2'].mean()
                n_cells = len(first_file_cells)
                std_area = first_file_cells['area_um2'].std()
                min_area = first_file_cells['area_um2'].min()
                max_area = first_file_cells['area_um2'].max()

                first_file_data[image_name] = {
                    'first_file_avg_area_um2': avg_area,
                    'first_file_n_cells': n_cells,
                    'first_file_std_area_um2': std_area,
                    'first_file_min_area_um2': min_area,
                    'first_file_max_area_um2': max_area,
                    'first_file_cell_ids': first_file_cells['cell_id'].tolist()
                }
                print(f"{image_name}: n={n_cells}, avg area={avg_area:.1f} µm², std={std_area:.1f}")
            else:
                print(f"WARNING: No cells with cell_file_derivative == 0 in {image_name}")
                if len(df_cells) > 0:
                    available_files = sorted(df_cells['cell_file_derivative'].unique())
                    print(f"Available derivative values: {available_files}")

        except Exception as e:
            print(f"ERROR reading {cell_file}: {e}")

    print(f"\nExtracted first file areas for {len(first_file_data)} roots")

    # Convert to DataFrame
    df_first_file = pd.DataFrame.from_dict(first_file_data, orient='index')
    df_first_file.index.name = 'image_name'
    df_first_file = df_first_file.reset_index()

    # Merge the first file areas with the master summary
    df_merged = df_master.merge(df_first_file, on='image_name', how='inner')

    print(f"Merged data: {len(df_merged)} rows")

    # Check for any missing data
    if len(df_merged) == 0:
        print("\nERROR: No data matched between master_summary and cell_assignments files.")
        exit()

    # Aggregate by image_name to get per-root measurements
    df_agg = df_merged.groupby('image_name').agg({
        'first_file_avg_area_um2': 'mean',  # Our predictor (X)
        'stele_area_um2': 'mean',           # REPLACES root_radius_um
        'root_radius_um': 'mean',           # Keep for reference/backup
        'file_count': 'mean',
        'average_cell_area_um2': 'mean',    # Overall avg cell area (Y3)
        'first_file_n_cells': 'sum',
        'plant_number': 'first',
        'root_number': 'first',
        'treatment': 'first',
        'species': 'first',
        'population': 'first'
    }).reset_index()

    # Round file_count to nearest integer
    df_agg['file_count'] = df_agg['file_count'].round().astype(int)

    print(f"\nAggregated data: {len(df_agg)} unique roots")

    # Drop rows with missing values
    df_clean = df_agg.dropna(subset=['first_file_avg_area_um2', 'stele_area_um2', 'file_count'])

    # Remove any rows with unrealistic values (zero or negative)
    df_clean = df_clean[df_clean['first_file_avg_area_um2'] > 0]
    df_clean = df_clean[df_clean['stele_area_um2'] > 0]
    df_clean = df_clean[df_clean['file_count'] > 0]
    df_clean = df_clean[df_clean['average_cell_area_um2'] > 0]

    print(f"\nAfter cleaning: {len(df_clean)} roots")
    print(f"\nVariable ranges:")
    print(f"  First file cell area (µm²): {df_clean['first_file_avg_area_um2'].min():.0f} - {df_clean['first_file_avg_area_um2'].max():.0f}")
    print(f"  Stele area (µm²): {df_clean['stele_area_um2'].min():.0f} - {df_clean['stele_area_um2'].max():.0f}")
    print(f"  File count: {df_clean['file_count'].min()} - {df_clean['file_count'].max()}")
    print(f"  Avg cell area (µm²): {df_clean['average_cell_area_um2'].min():.0f} - {df_clean['average_cell_area_um2'].max():.0f}")

    X = df_clean[['first_file_avg_area_um2']].values

    # Split data into training (75%) and testing (25%)
    if 'treatment' in df_clean.columns and df_clean['treatment'].notna().all():
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(sss.split(X, df_clean['treatment']))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(df_clean)), 
            test_size=0.25, 
            random_state=42
        )

    X_train = X[train_idx]
    X_test = X[test_idx]

    # Response variables
    y_train_stele = df_clean['stele_area_um2'].values[train_idx]
    y_test_stele = df_clean['stele_area_um2'].values[test_idx]

    y_train_files = df_clean['file_count'].values[train_idx]
    y_test_files = df_clean['file_count'].values[test_idx]

    y_train_area = df_clean['average_cell_area_um2'].values[train_idx]
    y_test_area = df_clean['average_cell_area_um2'].values[test_idx]

    print(f"\nTrain-test split:")
    print(f"  Training set: {len(X_train)} roots")
    print(f"  Testing set: {len(X_test)} roots")

    # 6a. Model for Stele Area (Linear) — REPLACES root radius

    model_stele = LinearRegression()
    model_stele.fit(X_train, y_train_stele)

    # Predictions
    y_train_pred_stele = model_stele.predict(X_train)
    y_test_pred_stele = model_stele.predict(X_test)

    # Performance metrics
    train_r2_stele = r2_score(y_train_stele, y_train_pred_stele)
    test_r2_stele = r2_score(y_test_stele, y_test_pred_stele)
    train_rmse_stele = np.sqrt(mean_squared_error(y_train_stele, y_train_pred_stele))
    test_rmse_stele = np.sqrt(mean_squared_error(y_test_stele, y_test_pred_stele))

    print(f"Training R²: {train_r2_stele:.4f}")
    print(f"Testing R²:  {test_r2_stele:.4f}")
    print(f"Training RMSE: {train_rmse_stele:.1f} µm²")
    print(f"Testing RMSE:  {test_rmse_stele:.1f} µm²")
    print(f"Coefficient: {model_stele.coef_[0]:.4f}")
    print(f"Intercept: {model_stele.intercept_:.1f}")

    # 6b. Model for File Count (Linear)

    model_files = LinearRegression()
    model_files.fit(X_train, y_train_files)

    # Predictions
    y_train_pred_files = model_files.predict(X_train)
    y_test_pred_files = model_files.predict(X_test)

    # Performance metrics
    train_r2_files = r2_score(y_train_files, y_train_pred_files)
    test_r2_files = r2_score(y_test_files, y_test_pred_files)
    train_rmse_files = np.sqrt(mean_squared_error(y_train_files, y_train_pred_files))
    test_rmse_files = np.sqrt(mean_squared_error(y_test_files, y_test_pred_files))

    print(f"Training R²: {train_r2_files:.4f}")
    print(f"Testing R²:  {test_r2_files:.4f}")
    print(f"Training RMSE: {train_rmse_files:.2f} files")
    print(f"Testing RMSE:  {test_rmse_files:.2f} files")
    print(f"Coefficient: {model_files.coef_[0]:.6f}")
    print(f"Intercept: {model_files.intercept_:.2f}")

    # 6c. Model for Average Cell Area (Polynomial - testing linear, quadratic, cubic)

    # Test degrees 1, 2, and 3
    degrees = [1, 2, 3]
    models_area = {}
    results_area = {}

    for degree in degrees:
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', LinearRegression())
        ])

        pipeline.fit(X_train, y_train_area)

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_r2 = r2_score(y_train_area, y_train_pred)
        test_r2 = r2_score(y_test_area, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_area, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_area, y_test_pred))

        models_area[degree] = pipeline
        results_area[degree] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }

        print(f"\nDegree {degree}:")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Testing R²:  {test_r2:.4f}")
        print(f"  Training RMSE: {train_rmse:.1f} µm²")
        print(f"  Testing RMSE:  {test_rmse:.1f} µm²")

    # Select the best model based on test R²
    best_degree = 1
    best_test_r2 = -np.inf
    for degree, results in results_area.items():
        if results['test_r2'] > best_test_r2:
            best_test_r2 = results['test_r2']
            best_degree = degree

    # But if cubic is only slightly better, recommend quadratic for parsimony
    if best_degree == 3 and results_area[3]['test_r2'] - results_area[2]['test_r2'] < 0.02:
        best_degree = 2
        print(f"\nNOTE: Cubic only improved R² by {results_area[3]['test_r2'] - results_area[2]['test_r2']:.4f}")
        print("Using quadratic for parsimony.")

    print(f"\nBest model: Degree {best_degree}")
    model_area_best = models_area[best_degree]
    results_area_best = results_area[best_degree]

    # Get coefficients for the best model
    coefs = model_area_best.named_steps['linear'].coef_
    intercept = model_area_best.named_steps['linear'].intercept_

    if best_degree == 1:
        print(f"Equation: y = {intercept:.2f} + {coefs[0]:.4f} * x")
    elif best_degree == 2:
        print(f"Equation: y = {intercept:.2f} + {coefs[0]:.4f} * x + {coefs[1]:.6f} * x²")
    else:
        print(f"Equation: y = {intercept:.2f} + {coefs[0]:.4f} * x + {coefs[1]:.6f} * x² + {coefs[2]:.6f} * x³")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Stele area model
    cv_scores_stele = cross_val_score(model_stele, X_train, y_train_stele, cv=kf, scoring='r2')
    print(f"\nStele Area Model:")
    print(f"  Cross-validation R²: {cv_scores_stele.mean():.4f} ± {cv_scores_stele.std():.4f}")

    # File count model
    cv_scores_files = cross_val_score(model_files, X_train, y_train_files, cv=kf, scoring='r2')
    print(f"\nFile Count Model:")
    print(f"  Cross-validation R²: {cv_scores_files.mean():.4f} ± {cv_scores_files.std():.4f}")

    # Cell area model (best degree)
    cv_scores_area = cross_val_score(model_area_best, X_train, y_train_area, cv=kf, scoring='r2')
    print(f"\nAverage Cell Area Model (Degree {best_degree}):")
    print(f"  Cross-validation R²: {cv_scores_area.mean():.4f} ± {cv_scores_area.std():.4f}")


    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multivariate Predictive Model: Cortical Cell Anatomy (Stele Area)\n(Predictor: First File Cell Area via cell_file_derivative == 0)', 
                 fontsize=16, fontweight='bold')

    # 8a. Stele Area: Observed vs Predicted (Testing)
    ax1 = axes[0, 0]
    ax1.scatter(y_test_stele, y_test_pred_stele, alpha=0.5, s=20, label='Testing', color='green')
    ax1.plot([y_test_stele.min(), y_test_stele.max()], 
             [y_test_stele.min(), y_test_stele.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Observed Stele Area (µm²)', fontsize=11)
    ax1.set_ylabel('Predicted Stele Area (µm²)', fontsize=11)
    ax1.set_title(f'Stele Area: Testing (R² = {test_r2_stele:.3f})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 8b. File Count: Observed vs Predicted (Testing)
    ax2 = axes[0, 1]
    ax2.scatter(y_test_files, y_test_pred_files, alpha=0.5, s=20, label='Testing', color='orange')
    ax2.plot([y_test_files.min(), y_test_files.max()], 
             [y_test_files.min(), y_test_files.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Observed File Count', fontsize=11)
    ax2.set_ylabel('Predicted File Count', fontsize=11)
    ax2.set_title(f'File Count: Testing (R² = {test_r2_files:.3f})', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 8c. Cell Area: Observed vs Predicted (Testing)
    ax3 = axes[0, 2]
    ax3.scatter(y_test_area, results_area_best['y_test_pred'], alpha=0.5, s=20, label='Testing', color='purple')
    ax3.plot([y_test_area.min(), y_test_area.max()], 
             [y_test_area.min(), y_test_area.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Observed Cell Area (µm²)', fontsize=11)
    ax3.set_ylabel('Predicted Cell Area (µm²)', fontsize=11)
    ax3.set_title(f'Cell Area (Degree {best_degree}): Testing (R² = {results_area_best["test_r2"]:.3f})', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 8d. Polynomial comparison for cell area
    ax4 = axes[1, 0]
    X_sorted = np.sort(X_train.flatten())
    for degree in degrees:
        y_pred_sorted = models_area[degree].predict(X_sorted.reshape(-1, 1))
        ax4.plot(X_sorted, y_pred_sorted, label=f'Degree {degree}', linewidth=2)
    ax4.scatter(X_train, y_train_area, alpha=0.3, s=10, color='gray', label='Training data')
    ax4.set_xlabel('First File Cell Area (µm²)', fontsize=11)
    ax4.set_ylabel('Avg Cell Area per File (µm²)', fontsize=11)
    ax4.set_title('Polynomial Comparison for Cell Area', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 8e. Residuals plot for the best cell area model
    ax5 = axes[1, 1]
    residuals = y_test_area - results_area_best['y_test_pred']
    ax5.scatter(results_area_best['y_test_pred'], residuals, alpha=0.5, s=20)
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Predicted Cell Area (µm²)', fontsize=11)
    ax5.set_ylabel('Residuals (µm²)', fontsize=11)
    ax5.set_title(f'Residuals: Cell Area Model (Degree {best_degree})', fontsize=12)
    ax5.grid(True, alpha=0.3)

    # 8f. Scatter plot of first file area vs stele area
    ax6 = axes[1, 2]
    ax6.scatter(X_train, y_train_stele, alpha=0.4, s=15, label='Training', color='blue')
    ax6.scatter(X_test, y_test_stele, alpha=0.4, s=15, label='Testing', color='green')
    # Add regression line
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model_stele.predict(x_range.reshape(-1, 1))
    ax6.plot(x_range, y_range, 'r-', linewidth=2, label='Regression')
    ax6.set_xlabel('First File Cell Area (µm²)', fontsize=11)
    ax6.set_ylabel('Stele Area (µm²)', fontsize=11)
    ax6.set_title('First File Area vs Stele Area', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = output_dir / 'predictive_model_results_stele.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")


    # Create summary dataframe
    results_summary = pd.DataFrame({
        'Model': ['Stele Area', 'File Count', 'Cell Area'],
        'Relationship': ['Linear', 'Linear', f'Degree {best_degree}'],
        'Train_R2': [train_r2_stele, train_r2_files, results_area_best['train_r2']],
        'Test_R2': [test_r2_stele, test_r2_files, results_area_best['test_r2']],
        'Train_RMSE': [train_rmse_stele, train_rmse_files, results_area_best['train_rmse']],
        'Test_RMSE': [test_rmse_stele, test_rmse_files, results_area_best['test_rmse']],
        'CV_R2_mean': [cv_scores_stele.mean(), cv_scores_files.mean(), cv_scores_area.mean()],
        'CV_R2_std': [cv_scores_stele.std(), cv_scores_files.std(), cv_scores_area.std()]
    })

    results_path = output_dir / 'model_performance_summary.csv'
    results_summary.to_csv(results_path, index=False)
    print(f"Performance summary saved to: {results_path}")

    # Save the clean data
    clean_data_path = output_dir / 'model_data_clean.csv'
    df_clean.to_csv(clean_data_path, index=False)
    print(f"Clean data saved to: {clean_data_path}")

    # Save the model coefficients
    coefs_df = pd.DataFrame({
        'Model': ['Stele Area', 'File Count'],
        'Coefficient': [model_stele.coef_[0], model_files.coef_[0]],
        'Intercept': [model_stele.intercept_, model_files.intercept_]
    })

    # Add cell area coefficients
    if best_degree == 1:
        coefs_df = pd.concat([coefs_df, pd.DataFrame({
            'Model': ['Cell Area'],
            'Coefficient': [coefs[0]],
            'Intercept': [intercept]
        })])
    elif best_degree == 2:
        coefs_df = pd.concat([coefs_df, pd.DataFrame({
            'Model': ['Cell Area (x)', 'Cell Area (x²)'],
            'Coefficient': coefs,
            'Intercept': [intercept, np.nan]
        })])
    else:
        coefs_df = pd.concat([coefs_df, pd.DataFrame({
            'Model': ['Cell Area (x)', 'Cell Area (x²)', 'Cell Area (x³)'],
            'Coefficient': coefs,
            'Intercept': [intercept, np.nan, np.nan]
        })])

    coefs_path = output_dir / 'model_coefficients.csv'
    coefs_df.to_csv(coefs_path, index=False)
    print(f"Model coefficients saved to: {coefs_path}")


    print(f"""
    Data Summary:
    ------------
    • Total roots analyzed: {len(df_clean)}
    • Training set: {len(X_train)} roots
    • Testing set: {len(X_test)} roots
    • Predictor: First cortical file cell area (cell_file_derivative == 0)

    Model Performance:
    -----------------
    1. Stele Area (Linear):
       • Testing R²: {test_r2_stele:.4f}
       • Testing RMSE: {test_rmse_stele:.1f} µm²
       • CV R²: {cv_scores_stele.mean():.4f} ± {cv_scores_stele.std():.4f}

    2. File Count (Linear):
       • Testing R²: {test_r2_files:.4f}
       • Testing RMSE: {test_rmse_files:.2f} files
       • CV R²: {cv_scores_files.mean():.4f} ± {cv_scores_files.std():.4f}

    3. Average Cell Area (Degree {best_degree}):
       • Testing R²: {results_area_best["test_r2"]:.4f}
       • Testing RMSE: {results_area_best["test_rmse"]:.1f} µm²
       • CV R²: {cv_scores_area.mean():.4f} ± {cv_scores_area.std():.4f}

    Output Files:
    ------------
    • Figure: {fig_path}
    • Performance summary: {results_path}
    • Clean data: {clean_data_path}
    • Model coefficients: {coefs_path}
    """)


    # Check if we have root_radius data for comparison
    if 'root_radius_um' in df_clean.columns:
        # Fit a quick root radius model for comparison
        X_radius = df_clean[['first_file_avg_area_um2']].values
        y_radius = df_clean['root_radius_um'].values

        # Use the same train-test split
        X_train_radius = X_radius[train_idx]
        X_test_radius = X_radius[test_idx]
        y_train_radius = y_radius[train_idx]
        y_test_radius = y_radius[test_idx]

        model_radius_compare = LinearRegression()
        model_radius_compare.fit(X_train_radius, y_train_radius)

        y_pred_radius = model_radius_compare.predict(X_test_radius)
        r2_radius = r2_score(y_test_radius, y_pred_radius)

        print(f"\nRoot Radius Model (for comparison):")
        print(f"  Testing R²: {r2_radius:.4f}")
        print(f"  Coefficient: {model_radius_compare.coef_[0]:.4f}")
        print(f"  Intercept: {model_radius_compare.intercept_:.1f}")

        print(f"\nStele Area Model (this analysis):")
        print(f"  Testing R²: {test_r2_stele:.4f}")
        print(f"  Coefficient: {model_stele.coef_[0]:.4f}")
        print(f"  Intercept: {model_stele.intercept_:.1f}")

        # Save comparison
        comparison_df = pd.DataFrame({
            'Model': ['Root Radius', 'Stele Area'],
            'Test_R2': [r2_radius, test_r2_stele],
            'Coefficient': [model_radius_compare.coef_[0], model_stele.coef_[0]],
            'Intercept': [model_radius_compare.intercept_, model_stele.intercept_]
        })
        comparison_path = output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison saved to: {comparison_path}")

    plt.show()


if __name__ == "__main__":
    main()
