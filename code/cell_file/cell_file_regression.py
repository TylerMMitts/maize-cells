import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import json
import statsmodels.api as sm
from scipy.interpolate import interp1d

MEASUREMENTS_FOLDER = "results/measurements"
CELL_FILE_COUNTS_FOLDER = "results/cell_file/cell_file_counting"
OUTPUT_FOLDER = "results/cell_file/cell_file_regression"

# Options: 'derivative', 'cumulative', 'consensus'
CELL_FILE_METHOD = 'derivative'

# Options: 'area_pixels' or 'area_um2'
AREA_COLUMN = 'area_um2'

# Visualization options
ALPHA = 0.5
LINE_WIDTH = 2.0
MARKER_SIZE = 5

# Color map for lines
COLOR_MAP = 'tab10'

# Y-axis label
if AREA_COLUMN == 'area_um2':
    Y_LABEL = 'Cell Area (µm²)'
else:
    Y_LABEL = 'Cell Area (pixels²)'


def load_cell_file_counts_from_assignments(counts_folder, method='consensus'):

    cell_file_counts = {}
    
    if not os.path.exists(counts_folder):
        print(f"Folder {counts_folder} does not exist")
        return cell_file_counts
    
    image_folders = [f for f in glob.glob(str(Path(counts_folder) / "*")) if os.path.isdir(f)]
    
    for folder in image_folders:
        image_name = os.path.basename(folder)
        
        # Look for the cell assignments CSV file
        assignment_file = os.path.join(folder, 'cell_assignments.csv')
        if not os.path.exists(assignment_file):
            # Try alternative names
            alt_files = [
                os.path.join(folder, f'{image_name}_method_consensus.csv'),
                os.path.join(folder, f'{image_name}_method_derivative.csv'),
                os.path.join(folder, f'{image_name}_method_cumulative.csv'),
                os.path.join(folder, 'cell_assignments_all_methods.csv'),
            ]
            for alt_file in alt_files:
                if os.path.exists(alt_file):
                    assignment_file = alt_file
                    break
        
        if os.path.exists(assignment_file):
            try:
                df = pd.read_csv(assignment_file)
                
                # Find the column with cell file assignments
                file_col = None
                if method == 'consensus' and 'cell_file_consensus' in df.columns:
                    file_col = 'cell_file_consensus'
                elif method == 'derivative' and 'cell_file_derivative' in df.columns:
                    file_col = 'cell_file_derivative'
                elif method == 'cumulative' and 'cell_file_cumulative' in df.columns:
                    file_col = 'cell_file_cumulative'
                else:
                    # Try to find any cell_file column
                    for col in df.columns:
                        if 'cell_file' in col:
                            file_col = col
                            break
                
                if file_col:
                    # Count unique file numbers (excluding -1)
                    valid_files = df[df[file_col] >= 0][file_col]
                    if len(valid_files) > 0:
                        n_files = valid_files.nunique()
                        cell_file_counts[image_name] = n_files
                        print(f"Loaded {image_name}: {n_files} files")
                    else:
                        print(f"No valid assignments for {image_name}")
                else:
                    print(f"No cell_file column found in {assignment_file}")
                    
            except Exception as e:
                print(f"Could not read {assignment_file}: {e}")
        else:
            print(f"No assignment file found for {image_name}")
    
    return cell_file_counts


def load_cell_assignments(counts_folder, image_name, assignment_method='consensus'):

    # Try different possible file names
    possible_files = [
        os.path.join(counts_folder, image_name, 'cell_assignments.csv'),
        os.path.join(counts_folder, image_name, f'{image_name}_method_{assignment_method}.csv'),
        os.path.join(counts_folder, image_name, 'cell_assignments_all_methods.csv'),
    ]
    
    for assignment_file in possible_files:
        if os.path.exists(assignment_file):
            df = pd.read_csv(assignment_file)
            
            # Find the column with cell file assignments
            col_name = f'cell_file_{assignment_method}'
            if col_name in df.columns:
                if AREA_COLUMN in df.columns:
                    return df[['cell_id', col_name, AREA_COLUMN]].copy()
                elif 'area_um2' in df.columns:
                    return df[['cell_id', col_name, 'area_um2']].copy()
                elif 'area_pixels' in df.columns:
                    return df[['cell_id', col_name, 'area_pixels']].copy()
                else:
                    print(f"No area column found")
                    return None
            else:
                # Try to find any cell_file column
                file_cols = [c for c in df.columns if 'cell_file' in c]
                if file_cols:
                    area_col = AREA_COLUMN if AREA_COLUMN in df.columns else 'area_pixels'
                    if area_col in df.columns:
                        return df[['cell_id', file_cols[0], area_col]].copy()
    
    return None


def load_individual_cell_data(counts_folder, image_name, assignment_method='consensus'):

    df_assignments = load_cell_assignments(counts_folder, image_name, assignment_method)
    
    if df_assignments is None or df_assignments.empty:
        return None
    
    # Find the file column and area column
    file_col = [c for c in df_assignments.columns if 'cell_file' in c][0]
    area_col = [c for c in df_assignments.columns if 'area' in c.lower()][0]
    
    # Filter out cells with negative file numbers
    df_valid = df_assignments[df_assignments[file_col] >= 0].copy()
    
    if df_valid.empty:
        return None
    
    return df_valid[['cell_id', file_col, area_col]].copy()


def perform_regression_analysis(image_profiles, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    
    all_aggregated_data = []
    for image_name, (file_numbers, avg_areas, n_files) in image_profiles.items():
        for i, (file_num, area) in enumerate(zip(file_numbers, avg_areas)):
            all_aggregated_data.append({
                'image_name': image_name,
                'file_number': file_num,
                'cell_area': area,
                'n_files': n_files
            })
    
    aggregated_df = pd.DataFrame(all_aggregated_data)
    
    all_individual_data = []
    for image_name, (file_numbers, avg_areas, n_files) in image_profiles.items():
        # Load individual cell assignments
        df_cells = load_individual_cell_data(CELL_FILE_COUNTS_FOLDER, image_name, CELL_FILE_METHOD)
        if df_cells is not None:
            file_col = [c for c in df_cells.columns if 'cell_file' in c][0]
            area_col = [c for c in df_cells.columns if 'area' in c.lower()][0]
            
            for _, row in df_cells.iterrows():
                all_individual_data.append({
                    'image_name': image_name,
                    'file_number': row[file_col],
                    'cell_area': row[area_col],
                    'n_files': n_files
                })
    
    individual_df = pd.DataFrame(all_individual_data)
    
    # MODEL 1: Simple linear regression (aggregated data)
    # area = b0 + b1 * file_number
    
    X1 = sm.add_constant(aggregated_df['file_number'])
    y1 = aggregated_df['cell_area']
    
    model1 = sm.OLS(y1, X1).fit()
    
    # MODEL 2: Simple linear regression (individual cell data)
    # area = b0 + b1 * file_number
    
    X2 = sm.add_constant(individual_df['file_number'])
    y2 = individual_df['cell_area']
    
    model2 = sm.OLS(y2, X2).fit()
    
    # MODEL 3: Quadratic regression (aggregated data)
    # area = b0 + b1 * file_number + b2 * file_number^2
    
    aggregated_df['file_number_sq'] = aggregated_df['file_number'] ** 2
    X3 = sm.add_constant(aggregated_df[['file_number', 'file_number_sq']])
    y3 = aggregated_df['cell_area']
    
    model3 = sm.OLS(y3, X3).fit()
    
    # MODEL 4: Quadratic regression (individual cell data)
    # area = b0 + b1 * file_number + b2 * file_number^2
    
    individual_df['file_number_sq'] = individual_df['file_number'] ** 2
    X4 = sm.add_constant(individual_df[['file_number', 'file_number_sq']])
    y4 = individual_df['cell_area']
    
    model4 = sm.OLS(y4, X4).fit()
    
    # MODEL 5: Linear with interaction (aggregated data)
    # area = b0 + b1 * file_number + b2 * n_files + b3 * file_number * n_files
    
    X5 = sm.add_constant(aggregated_df[['file_number', 'n_files']])
    X5['file_nfiles_interaction'] = X5['file_number'] * X5['n_files']
    y5 = aggregated_df['cell_area']
    
    model5 = sm.OLS(y5, X5).fit()
    
    # SAVE RESULTS
    
    with open(os.path.join(output_folder, 'regression_results.txt'), 'w', encoding='utf-8') as f:
        
        f.write("=" * 80 + "\n")
        f.write("MODEL 1: Linear Regression (Aggregated Data)\n")
        f.write(model1.summary().as_text())
        
        coeffs = model1.params
        pvals = model1.pvalues
        
        f.write("\nINTERPRETATION:\n")
        f.write(f"  Intercept (b0): {coeffs['const']:.4f} (p={pvals['const']:.4e})\n")
        f.write(f"    -> Predicted cell area at file number 0 (center): {coeffs['const']:.2f} {Y_LABEL.split()[-1]}\n")
        f.write(f"  Slope (b1): {coeffs['file_number']:.4f} (p={pvals['file_number']:.4e})\n")
        if pvals['file_number'] < 0.05:
            f.write(f"    -> SIGNIFICANT: For every 1 file number increase (moving outward), cell area changes by {coeffs['file_number']:.4f} {Y_LABEL.split()[-1]}\n")
        else:
            f.write(f"    -> NOT SIGNIFICANT: No linear relationship between file number and cell area\n")
        
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("MODEL 2: Linear Regression (Individual Cell Data)\n")
        f.write("area = b0 + b1 * file_number\n")
        f.write("=" * 80 + "\n")
        f.write(model2.summary().as_text())
        
        coeffs = model2.params
        pvals = model2.pvalues
        
        f.write("\nINTERPRETATION:\n")
        f.write(f"  Intercept (b0): {coeffs['const']:.4f} (p={pvals['const']:.4e})\n")
        f.write(f"    -> Predicted cell area at file number 0 (center): {coeffs['const']:.2f} {Y_LABEL.split()[-1]}\n")
        f.write(f"  Slope (b1): {coeffs['file_number']:.4f} (p={pvals['file_number']:.4e})\n")
        if pvals['file_number'] < 0.05:
            f.write(f"    -> SIGNIFICANT: For every 1 file number increase (moving outward), cell area changes by {coeffs['file_number']:.4f} {Y_LABEL.split()[-1]}\n")
        else:
            f.write(f"    -> NOT SIGNIFICANT: No linear relationship between file number and cell area\n")
        
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("MODEL 3: Quadratic Regression (Aggregated Data)\n")
        f.write("area = b0 + b1 * file_number + b2 * file_number²\n")
        f.write("=" * 80 + "\n")
        f.write(model3.summary().as_text())
        
        coeffs = model3.params
        pvals = model3.pvalues
        
        f.write("\nINTERPRETATION:\n")
        f.write(f"  Intercept (b0): {coeffs['const']:.4f} (p={pvals['const']:.4e})\n")
        f.write(f"  Linear (b1): {coeffs['file_number']:.4f} (p={pvals['file_number']:.4e})\n")
        f.write(f"  Quadratic (b2): {coeffs['file_number_sq']:.4f} (p={pvals['file_number_sq']:.4e})\n")
        if pvals['file_number_sq'] < 0.05:
            f.write(f"    -> SIGNIFICANT quadratic term: Cell size changes non-linearly with file number (curvature is significant)\n")
            if coeffs['file_number_sq'] > 0:
                f.write(f"    -> Positive curvature: Cell size accelerates as you move outward (convex)\n")
            else:
                f.write(f"    -> Negative curvature: Cell size decelerates as you move outward (concave)\n")
        else:
            f.write(f"    -> No significant quadratic term: Linear model is sufficient\n")
        
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("MODEL 4: Quadratic Regression (Individual Cell Data)\n")
        f.write("area = b0 + b1 * file_number + b2 * file_number²\n")
        f.write("=" * 80 + "\n")
        f.write(model4.summary().as_text())
        
        coeffs = model4.params
        pvals = model4.pvalues
        
        f.write("\nINTERPRETATION:\n")
        f.write(f"  Intercept (b0): {coeffs['const']:.4f} (p={pvals['const']:.4e})\n")
        f.write(f"  Linear (b1): {coeffs['file_number']:.4f} (p={pvals['file_number']:.4e})\n")
        f.write(f"  Quadratic (b2): {coeffs['file_number_sq']:.4f} (p={pvals['file_number_sq']:.4e})\n")
        
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("MODEL 5: Linear with Interaction (Aggregated Data)\n")
        f.write("area = b0 + b1 * file_number + b2 * n_files + b3 * file_number * n_files\n")
        f.write("=" * 80 + "\n")
        f.write(model5.summary().as_text())
        
        coeffs = model5.params
        pvals = model5.pvalues
        
        f.write("\nINTERPRETATION:\n")
        f.write(f"  Intercept (b0): {coeffs['const']:.4f} (p={pvals['const']:.4e})\n")
        f.write(f"  File Number (b1): {coeffs['file_number']:.4f} (p={pvals['file_number']:.4e})\n")
        f.write(f"    -> Effect of file number when n_files = 0\n")
        f.write(f"  n_files (b2): {coeffs['n_files']:.4f} (p={pvals['n_files']:.4e})\n")
        f.write(f"    -> Effect of having more files (more cell layers)\n")
        f.write(f"  Interaction (b3): {coeffs['file_nfiles_interaction']:.4e} (p={pvals['file_nfiles_interaction']:.4e})\n")
        if pvals['file_nfiles_interaction'] < 0.05:
            f.write(f"    -> SIGNIFICANT interaction: The relationship between file number and area DEPENDS on how many files (cell layers) the image has\n")
        else:
            f.write(f"    -> NOT SIGNIFICANT: The file number-area relationship does not depend on number of files\n")
    
    
    models = [model1, model2, model3, model4, model5]
    model_names = [
        'Linear (Aggregated)',
        'Linear (Individual)',
        'Quadratic (Aggregated)',
        'Quadratic (Individual)',
        'Linear + Interaction'
    ]
    
    model_comparison = []
    for name, model in zip(model_names, models):
        model_comparison.append({
            'Model': name,
            'R²': model.rsquared,
            'Adj. R²': model.rsquared_adj,
            'AIC': model.aic,
            'BIC': model.bic,
            'F-statistic': model.fvalue,
            'p-value': model.f_pvalue
        })
    
    comparison_df = pd.DataFrame(model_comparison)
    
    with open(os.path.join(output_folder, 'model_comparison.txt'), 'w', encoding='utf-8') as f:
        f.write("MODEL COMPARISON\n")
        f.write(comparison_df.to_string(index=False))
        
        # Find best model by Adj. R²
        best_adj_r2 = comparison_df.loc[comparison_df['Adj. R²'].idxmax()]
        f.write(f"\nBest by Adjusted R²: {best_adj_r2['Model']} (Adj. R² = {best_adj_r2['Adj. R²']:.4f})\n")
        
        # Find best model by AIC (lower is better)
        best_aic = comparison_df.loc[comparison_df['AIC'].idxmin()]
        f.write(f"Best by AIC: {best_aic['Model']} (AIC = {best_aic['AIC']:.2f})\n")
        
        # Find best model by BIC (lower is better)
        best_bic = comparison_df.loc[comparison_df['BIC'].idxmin()]
        f.write(f"Best by BIC: {best_bic['Model']} (BIC = {best_bic['BIC']:.2f})\n")
    
    # VISUALIZATIONS
    
    create_regression_plots(aggregated_df, individual_df, image_profiles, 
                           model1, model2, model3, model4, model5, output_folder)
    
    print(f"Saved: regression_results.txt")
    print(f"Saved: model_comparison.txt")
    print(f"Saved: regression_plots.png")
    
    return model1, model2, model3, model4, model5


def create_regression_plots(aggregated_df, individual_df, image_profiles, 
                           model1, model2, model3, model4, model5, output_folder):

    
    # Plot 1: Aggregated data with linear and quadratic fits
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Linear fit
    ax1 = axes[0]
    
    # Plot individual image profiles
    cmap = plt.get_cmap(COLOR_MAP)
    colors = cmap(np.linspace(0, 1, len(image_profiles)))
    
    for idx, (image_name, (file_numbers, avg_areas, n_files)) in enumerate(image_profiles.items()):
        color = colors[idx]
        short_name = image_name[:20] + '...' if len(image_name) > 20 else image_name
        ax1.plot(file_numbers, avg_areas, 'o-', linewidth=1.5, 
                markersize=4, color=color, alpha=ALPHA, label=short_name)
    
    # Add regression line
    x_range = np.linspace(aggregated_df['file_number'].min(), aggregated_df['file_number'].max(), 100)
    y_pred = model1.params['const'] + model1.params['file_number'] * x_range
    ax1.plot(x_range, y_pred, 'r-', linewidth=2.5, 
            label=f'Linear: R² = {model1.rsquared:.3f}')
    
    ax1.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(Y_LABEL, fontsize=12, fontweight='bold')
    ax1.set_title('Linear Regression: Cell Area vs File Number\n(Aggregated data)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8, ncol=2)
    
    # Right: Quadratic fit
    ax2 = axes[1]
    
    # Plot individual image profiles
    for idx, (image_name, (file_numbers, avg_areas, n_files)) in enumerate(image_profiles.items()):
        color = colors[idx]
        ax2.plot(file_numbers, avg_areas, 'o-', linewidth=1.5, 
                markersize=4, color=color, alpha=ALPHA)
    
    # Add quadratic regression line
    y_pred3 = (model3.params['const'] + 
               model3.params['file_number'] * x_range + 
               model3.params['file_number_sq'] * x_range**2)
    ax2.plot(x_range, y_pred3, 'r-', linewidth=2.5, 
            label=f'Quadratic: R² = {model3.rsquared:.3f}')
    
    ax2.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(Y_LABEL, fontsize=12, fontweight='bold')
    ax2.set_title('Quadratic Regression: Cell Area vs File Number\n(Aggregated data)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '01_aggregated_regression_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Individual cell data with regression fits
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Linear fit
    ax1 = axes[0]
    
    # Sample individual cells (plot with low alpha to avoid overplotting)
    sample = individual_df.sample(min(10000, len(individual_df)))
    ax1.scatter(sample['file_number'], sample['cell_area'], 
               alpha=0.15, s=3, color='steelblue')
    
    # Add regression line
    x_range = np.linspace(individual_df['file_number'].min(), individual_df['file_number'].max(), 100)
    y_pred = model2.params['const'] + model2.params['file_number'] * x_range
    ax1.plot(x_range, y_pred, 'r-', linewidth=2.5, 
            label=f'Linear: R² = {model2.rsquared:.3f}')
    
    ax1.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(Y_LABEL, fontsize=12, fontweight='bold')
    ax1.set_title('Linear Regression: Individual Cell Data', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # Right: Quadratic fit
    ax2 = axes[1]
    
    ax2.scatter(sample['file_number'], sample['cell_area'], 
               alpha=0.15, s=3, color='steelblue')
    
    # Add quadratic regression line
    y_pred4 = (model4.params['const'] + 
               model4.params['file_number'] * x_range + 
               model4.params['file_number_sq'] * x_range**2)
    ax2.plot(x_range, y_pred4, 'r-', linewidth=2.5, 
            label=f'Quadratic: R² = {model4.rsquared:.3f}')
    
    ax2.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(Y_LABEL, fontsize=12, fontweight='bold')
    ax2.set_title('Quadratic Regression: Individual Cell Data', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '02_individual_regression_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Model comparison - R² bar chart
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = ['Linear\n(Aggregated)', 'Linear\n(Individual)', 
                   'Quadratic\n(Aggregated)', 'Quadratic\n(Individual)',
                   'Linear +\nInteraction']
    r2_values = [model1.rsquared, model2.rsquared, model3.rsquared, 
                 model4.rsquared, model5.rsquared]
    
    bars = ax.bar(model_names, r2_values, color='steelblue', edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('R-squared', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: R² Values', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '03_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Interaction effect visualization
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by number of files
    groups = aggregated_df.groupby('n_files')
    
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(groups)))
    
    for idx, (n_files, group) in enumerate(groups):
        color = colors[idx]
        ax.plot(group['file_number'], group['cell_area'], 'o-', 
               linewidth=1.5, markersize=4, color=color, alpha=0.8,
               label=f'{n_files} files')
        
        # Add regression line for this group
        if len(group) > 1:
            X_group = sm.add_constant(group['file_number'])
            y_group = group['cell_area']
            model_group = sm.OLS(y_group, X_group).fit()
            x_range_group = np.linspace(group['file_number'].min(), group['file_number'].max(), 50)
            y_pred_group = model_group.params['const'] + model_group.params['file_number'] * x_range_group
            ax.plot(x_range_group, y_pred_group, '--', linewidth=1.5, color=color, alpha=0.5)
    
    ax.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
    ax.set_ylabel(Y_LABEL, fontsize=12, fontweight='bold')
    ax.set_title('Interaction Effect: File Number vs Cell Area by Number of Files', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', title='Total Files')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '04_interaction_effect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Residual plots for best model
    
    # Find best model by Adj. R²
    models = [model1, model2, model3, model4, model5]
    r2_adj = [m.rsquared_adj for m in models]
    best_idx = np.argmax(r2_adj)
    best_model = models[best_idx]
    best_name = ['Linear (Aggregated)', 'Linear (Individual)', 
                 'Quadratic (Aggregated)', 'Quadratic (Individual)',
                 'Linear + Interaction'][best_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs fitted
    ax1 = axes[0]
    ax1.scatter(best_model.fittedvalues, best_model.resid, alpha=0.3, s=5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Fitted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title(f'Residuals vs Fitted\n{best_name}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax2 = axes[1]
    from scipy import stats
    stats.probplot(best_model.resid, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot\n{best_name}', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '05_best_model_diagnostics.png'), dpi=150, bbox_inches='tight')
    plt.close()


def compute_average_cell_size_per_file(df_assignments):

    if df_assignments is None or df_assignments.empty:
        return None, None
    
    # Find the file column and area column
    file_col = [c for c in df_assignments.columns if 'cell_file' in c][0]
    area_col = [c for c in df_assignments.columns if 'area' in c.lower()][0]
    
    # Filter out cells with negative file numbers
    df_valid = df_assignments[df_assignments[file_col] >= 0].copy()
    
    if df_valid.empty:
        return None, None
    
    # Group by file number and calculate mean area
    avg_sizes = df_valid.groupby(file_col)[area_col].mean().reset_index()
    avg_sizes = avg_sizes.sort_values(file_col)
    
    file_numbers = avg_sizes[file_col].values
    avg_areas = avg_sizes[area_col].values
    
    return file_numbers, avg_areas


def main():
    
    # Create output folder
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load cell file counts
    cell_file_counts = load_cell_file_counts_from_assignments(CELL_FILE_COUNTS_FOLDER, CELL_FILE_METHOD)
    
    if not cell_file_counts:
        print("\nNo cell file counts found!")
        print("\nCheck that the cell assignment CSV files exist in:")
        print(f"  {CELL_FILE_COUNTS_FOLDER}")
        return
    
    print(f"\nSuccessfully loaded {len(cell_file_counts)} images")
    
    # Process each image to get cell size per file number
    
    image_profiles = {}
    
    for image_name, n_files in cell_file_counts.items():
        print(f"\nProcessing: {image_name}")
        
        # Load the cell assignments for this image
        df_assignments = load_cell_assignments(CELL_FILE_COUNTS_FOLDER, image_name, CELL_FILE_METHOD)
        
        if df_assignments is None:
            print(f"No assignment file found for {image_name}")
            continue
        
        # Compute average cell size per file number
        file_numbers, avg_areas = compute_average_cell_size_per_file(df_assignments)
        
        if file_numbers is None or len(file_numbers) == 0:
            print(f"No valid assignments for {image_name}")
            continue
        
        image_profiles[image_name] = (file_numbers, avg_areas, n_files)
        print(f"SUCCESS: {len(file_numbers)} files, {len(df_assignments)} cells")
    
    if not image_profiles:
        print("\nNo profiles could be created!")
        return
    
    # Perform regression analysis
    perform_regression_analysis(image_profiles, OUTPUT_FOLDER)
    

if __name__ == "__main__":
    main()