# Does the pattern hold inside every subgroup?
#
# Repeats the main analysis within each treatment and root type separately,
# so a relationship that only exists in the pooled data would show up as
# absent here.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
from code.config import CELL_FILE_COUNTS_FOLDER, MASTER_SUMMARY_PATH, RESULTS_FOLDER
warnings.filterwarnings('ignore')


def load_data(master_path=MASTER_SUMMARY_PATH, cell_file_dir=CELL_FILE_COUNTS_FOLDER):

    # Load master summary
    df_master = pd.read_csv(master_path)
    
    # Extract first file areas from cell assignments
    first_file_data = {}
    cell_files = list(Path(cell_file_dir).rglob('cell_assignments.csv'))
    
    for cell_file in cell_files:
        image_name = cell_file.parent.name
        try:
            df_cells = pd.read_csv(cell_file)
            if 'cell_file_derivative' in df_cells.columns:
                first_file_cells = df_cells[df_cells['cell_file_derivative'] == 0]
                if len(first_file_cells) > 0:
                    first_file_data[image_name] = {
                        'first_file_avg_area_um2': first_file_cells['area_um2'].mean(),
                        'first_file_n_cells': len(first_file_cells)
                    }
        except:
            continue
    
    df_first_file = pd.DataFrame.from_dict(first_file_data, orient='index')
    df_first_file.index.name = 'image_name'
    df_first_file = df_first_file.reset_index()
    
    # Merge with master summary
    df_merged = df_master.merge(df_first_file, on='image_name', how='inner')
    
    # Aggregate by image_name to get per-root measurements
    df_agg = df_merged.groupby('image_name').agg({
        'first_file_avg_area_um2': 'mean',
        'stele_area_um2': 'mean',
        'root_radius_um': 'mean',
        'file_count': 'mean',
        'average_cell_area_um2': 'mean',
        'treatment': 'first',
        'root_type': 'first',
        'species': 'first',
        'population': 'first',
        'plant_number': 'first',
        'root_number': 'first'
    }).reset_index()
    
    # Clean data
    df_clean = df_agg.dropna(subset=['first_file_avg_area_um2', 'stele_area_um2', 'file_count', 'average_cell_area_um2'])
    df_clean = df_clean[df_clean['first_file_avg_area_um2'] > 0]
    df_clean = df_clean[df_clean['stele_area_um2'] > 0]
    df_clean = df_clean[df_clean['file_count'] > 0]
    df_clean = df_clean[df_clean['average_cell_area_um2'] > 0]
    
    # Round file_count to integer
    df_clean['file_count'] = df_clean['file_count'].round().astype(int)
    
    return df_clean


def run_stratified_analysis(df, group_col, target_cols=None):

    if target_cols is None:
        target_cols = {
            'stele_area_um2': 'Stele Area',
            'file_count': 'File Count',
            'average_cell_area_um2': 'Avg Cell Area'
        }
    
    groups = df[group_col].unique()
    groups = [g for g in groups if pd.notna(g) and g != '']
    
    all_results = {}
    
    # For each target variable
    for target_col, target_name in target_cols.items():
        print(f"\n  {target_name}:")
        print(f"  {'Group':<15} {'Slope':>10} {'Intercept':>12} {'R²':>10} {'RMSE':>10} {'n':>8}")
        
        target_results = {}
        
        # Fit overall model first
        X_all = df[['first_file_avg_area_um2']].values
        y_all = df[target_col].values
        model_all = LinearRegression()
        model_all.fit(X_all, y_all)
        y_pred_all = model_all.predict(X_all)
        
        overall_r2 = r2_score(y_all, y_pred_all)
        overall_rmse = np.sqrt(mean_squared_error(y_all, y_pred_all))
        
        target_results['overall'] = {
            'slope': model_all.coef_[0],
            'intercept': model_all.intercept_,
            'r2': overall_r2,
            'rmse': overall_rmse,
            'n': len(df),
            'model': model_all
        }
        
        # Fit group-specific models
        for group in groups:
            subset = df[df[group_col] == group]
            
            if len(subset) < 5:
                continue
            
            X = subset[['first_file_avg_area_um2']].values
            y = subset[target_col].values
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            target_results[group] = {
                'n': len(subset),
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'r2': r2,
                'rmse': rmse,
                'model': model,
                'data': subset
            }
            
            print(f"  {group:<15} {model.coef_[0]:>10.4f} {model.intercept_:>12.1f} {r2:>10.3f} {rmse:>10.1f} {len(subset):>8}")
        
        all_results[target_col] = target_results
    
    return all_results


def create_stratified_plots(stratified_results, df, output_dir, factor_name):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    target_cols = {
        'stele_area_um2': 'Stele Area (µm²)',
        'file_count': 'File Count',
        'average_cell_area_um2': 'Avg Cell Area (µm²)'
    }
    
    # For each target, create plots
    for target_col, target_label in target_cols.items():
        if target_col not in stratified_results:
            continue
        
        results = stratified_results[target_col]
        groups = [g for g in results.keys() if g != 'overall']
        
        if not groups:
            continue
        
        # 1. Scatter plots with regression lines
        n_groups = len(groups)
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_groups == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
        
        for idx, group in enumerate(groups):
            ax = axes[idx]
            subset = results[group]['data']
            model = results[group]['model']
            
            ax.scatter(subset['first_file_avg_area_um2'], subset[target_col], 
                      alpha=0.5, s=30, color=colors[idx], label='Data')
            
            x_range = np.linspace(subset['first_file_avg_area_um2'].min(), 
                                 subset['first_file_avg_area_um2'].max(), 100)
            y_range = model.predict(x_range.reshape(-1, 1))
            ax.plot(x_range, y_range, 'r-', linewidth=2, 
                   label=f'Slope = {results[group]["slope"]:.3f}')
            
            ax.set_xlabel('First File Area (µm²)', fontsize=11)
            ax.set_ylabel(target_label, fontsize=11)
            ax.set_title(f'{group}\nR² = {results[group]["r2"]:.3f}, n = {results[group]["n"]}', fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(groups), len(axes)):
            axes[idx].set_visible(False)
        
        target_name = target_label.replace(' (µm²)', '').replace(' (µm²)', '')
        plt.suptitle(f'{factor_name.upper()}: {target_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / f'{factor_name}_{target_col}_stratified_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Parameter comparison (slopes and intercepts)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{factor_name.upper()}: {target_name} - Model Parameters', fontsize=14, fontweight='bold')
        
        slopes = [results[g]['slope'] for g in groups]
        intercepts = [results[g]['intercept'] for g in groups]
        
        # Slope comparison
        ax1 = axes[0]
        bars1 = ax1.bar(groups, slopes, color=plt.cm.tab10(np.linspace(0, 1, len(groups))))
        ax1.set_xlabel(factor_name.capitalize(), fontsize=12)
        ax1.set_ylabel('Slope', fontsize=12)
        ax1.set_title('Slope Comparison', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, slope in zip(bars1, slopes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{slope:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Intercept comparison
        ax2 = axes[1]
        bars2 = ax2.bar(groups, intercepts, color=plt.cm.tab10(np.linspace(0, 1, len(groups))))
        ax2.set_xlabel(factor_name.capitalize(), fontsize=12)
        ax2.set_ylabel('Intercept', fontsize=12)
        ax2.set_title('Intercept Comparison', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, intercept in zip(bars2, intercepts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{intercept:.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / f'{factor_name}_{target_col}_parameters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. R² comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        r2_values = [results[g]['r2'] for g in groups]
        n_values = [results[g]['n'] for g in groups]
        
        bars = ax.bar(groups, r2_values, color=plt.cm.tab10(np.linspace(0, 1, len(groups))))
        ax.set_xlabel(factor_name.capitalize(), fontsize=12)
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title(f'{target_name}: Model Fit by {factor_name.capitalize()}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, r2, n in zip(bars, r2_values, n_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{r2:.3f}\n(n={n})', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / f'{factor_name}_{target_col}_r2_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_stratified_predictions(stratified_results, first_file_values=None):

    if first_file_values is None:
        first_file_values = [500, 1000, 1500, 2000, 2500]
    
    all_predictions = {}
    
    for target_col, results in stratified_results.items():
        groups = [g for g in results.keys() if g != 'overall']
        
        prediction_data = []
        for group in groups:
            row = {'Group': group}
            for ffa in first_file_values:
                pred = results[group]['slope'] * ffa + results[group]['intercept']
                row[f'FFA_{ffa}'] = pred
            prediction_data.append(row)
        
        all_predictions[target_col] = pd.DataFrame(prediction_data)
    
    return all_predictions


def run_stratified_analysis_full(
    master_path=MASTER_SUMMARY_PATH,
    cell_file_dir=CELL_FILE_COUNTS_FOLDER,
    output_dir=RESULTS_FOLDER / 'stratified_analysis',
    factors=['treatment', 'root_type', 'population']
):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(master_path, cell_file_dir)
    print(f"\nLoaded {len(df)} roots for analysis")
    
    target_cols = {
        'stele_area_um2': 'Stele Area (µm²)',
        'file_count': 'File Count',
        'average_cell_area_um2': 'Avg Cell Area (µm²)'
    }
    
    all_results = {}
    
    for factor in factors:
        if factor not in df.columns:
            print(f"Warning: {factor} not found in data")
            continue
        
        # Run stratified analysis for all targets
        results = run_stratified_analysis(df, factor, target_cols)
        all_results[factor] = results
        
        # Print overall summary
        print(f"\nOverall Models (all data combined):")
        for target_col, target_name in target_cols.items():
            if target_col in results:
                overall = results[target_col]['overall']
                print(f"  {target_name}:")
                print(f"    Equation: {target_name} = {overall['slope']:.4f} × FirstFile + {overall['intercept']:.1f}")
                print(f"    R² = {overall['r2']:.3f}, RMSE = {overall['rmse']:.1f}")
        
        # Save summary for each target
        for target_col, target_results in results.items():
            target_name = target_cols.get(target_col, target_col)
            summary_rows = []
            
            for group, result in target_results.items():
                if group == 'overall':
                    summary_rows.append({
                        'group': 'Overall',
                        'n': result['n'],
                        'slope': result['slope'],
                        'intercept': result['intercept'],
                        'r2': result['r2'],
                        'rmse': result['rmse']
                    })
                else:
                    summary_rows.append({
                        'group': group,
                        'n': result['n'],
                        'slope': result['slope'],
                        'intercept': result['intercept'],
                        'r2': result['r2'],
                        'rmse': result['rmse']
                    })
            
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(output_path / f'{factor}_{target_col}_summary.csv', index=False)
        
        # Create plots
        create_stratified_plots(results, df, output_path, factor)
        
        # Generate predictions
        predictions = generate_stratified_predictions(results)
        for target_col, pred_df in predictions.items():
            pred_df.to_csv(output_path / f'{factor}_{target_col}_predictions.csv', index=False)
        
        print(f"\nSaved results for {factor} to: {output_path}")
    
    return all_results


def main(
    master_path=MASTER_SUMMARY_PATH,
    cell_file_dir=CELL_FILE_COUNTS_FOLDER,
    output_dir=RESULTS_FOLDER / 'stratified_analysis',
    factors=['treatment', 'root_type', 'population']
):
    
    results = run_stratified_analysis_full(
        master_path=master_path,
        cell_file_dir=cell_file_dir,
        output_dir=output_dir,
        factors=factors
    )

    print(f"\nResults saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    main()