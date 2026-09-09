# Compares the two bases the shape features can be computed in.
#
# The same six features can be fitted against radius or against cell file
# index. This quantifies how much they disagree, which matters because the
# file basis is fitted through far fewer points and comes out noisier.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway, ttest_ind
import warnings
from code.config import FEATURE_TABLE_PATH, RESULTS_FOLDER
warnings.filterwarnings('ignore')

FEATURE_TABLE_PATH = FEATURE_TABLE_PATH
OUTPUT_FOLDER = RESULTS_FOLDER / 'feature_variations'

# Features to analyze (both radius and file-based)
FEATURES = [
    'peak_height',
    'peak_position', 
    'rise_slope',
    'decay_slope',
    'outer_rise_magnitude'
]

FEATURE_LABELS = {
    'peak_height': 'Peak Height',
    'peak_position': 'Peak Position',
    'rise_slope': 'Rise Slope',
    'decay_slope': 'Decay Slope',
    'outer_rise_magnitude': 'Outer Rise Magnitude'
}

# Color palettes
COLORS = {
    'radius': '#2E86AB',  # Blue
    'file': '#F18F01',    # Orange
    'combined': '#6A994E' # Green
}

def load_feature_table(path):

    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return None
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def extract_features_by_approach(df, features, approach='radius'):

    prefix = f"{approach}_"
    
    # Find which features are available
    available_features = []
    for feature in features:
        col = f"{prefix}{feature}"
        if col in df.columns:
            available_features.append(col)
    
    if not available_features:
        print(f"No {approach} features found")
        return None
    
    # Extract just the feature columns
    feature_df = df[available_features].copy()
    
    # Rename columns to remove prefix
    feature_df.columns = [col.replace(prefix, '') for col in feature_df.columns]
    
    # Add metadata
    if 'treatment' in df.columns:
        feature_df['treatment'] = df['treatment'].values
    if 'root_type' in df.columns:
        feature_df['root_type'] = df['root_type'].values
    if 'n_files' in df.columns:
        feature_df['n_files'] = df['n_files'].values
    
    return feature_df

# STATISTICAL ANALYSIS
def compute_feature_statistics(df, features, approach_name):

    stats_data = []
    
    for feature in features:
        if feature not in df.columns:
            continue
            
        values = df[feature].dropna()
        
        if len(values) == 0:
            continue
        
        # Basic statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        var_val = np.var(values)
        min_val = np.min(values)
        max_val = np.max(values)
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        iqr = q75 - q25
        
        # Coefficient of variation
        cv = std_val / mean_val if mean_val != 0 else np.nan
        
        # Skewness and kurtosis
        skew = stats.skew(values)
        kurtosis = stats.kurtosis(values)
        
        # Standard error
        se = std_val / np.sqrt(len(values))
        
        # 95% confidence interval
        ci_lower = mean_val - 1.96 * se
        ci_upper = mean_val + 1.96 * se
        
        # Shapiro-Wilk test for normality
        if len(values) < 5000 and len(values) > 3:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(values)
            except:
                shapiro_stat, shapiro_p = np.nan, np.nan
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        stats_data.append({
            'approach': approach_name,
            'feature': feature,
            'feature_label': FEATURE_LABELS.get(feature, feature),
            'n': len(values),
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'variance': var_val,
            'cv': cv,
            'min': min_val,
            'max': max_val,
            'q25': q25,
            'q75': q75,
            'iqr': iqr,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'skew': skew,
            'kurtosis': kurtosis,
            'shapiro_p': shapiro_p
        })
    
    return pd.DataFrame(stats_data)


def compute_group_statistics(df, features, group_col='treatment'):

    group_stats = []
    
    if group_col not in df.columns:
        return pd.DataFrame()
    
    groups = df[group_col].unique()
    groups = [g for g in groups if pd.notna(g)]
    
    for feature in features:
        if feature not in df.columns:
            continue
        
        for group in groups:
            values = df[df[group_col] == group][feature].dropna()
            
            if len(values) == 0:
                continue
            
            group_stats.append({
                'feature': feature,
                'feature_label': FEATURE_LABELS.get(feature, feature),
                'group': group,
                'n': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.nan,
                'min': np.min(values),
                'max': np.max(values)
            })
    
    return pd.DataFrame(group_stats)


def compare_approaches(radius_df, file_df, features):

    comparison_data = []
    
    for feature in features:
        if feature not in radius_df.columns or feature not in file_df.columns:
            continue
        
        # Get paired data (only rows where both exist)
        combined = pd.DataFrame({
            'radius': radius_df[feature],
            'file': file_df[feature]
        }).dropna()
        
        if len(combined) < 3:
            continue
        
        # Correlation
        try:
            pearson_r, pearson_p = pearsonr(combined['radius'], combined['file'])
            spearman_r, spearman_p = spearmanr(combined['radius'], combined['file'])
        except:
            pearson_r, pearson_p = np.nan, np.nan
            spearman_r, spearman_p = np.nan, np.nan
        
        # Paired t-test
        try:
            t_stat, t_p = ttest_ind(combined['radius'], combined['file'])
        except:
            t_stat, t_p = np.nan, np.nan
        
        # Mean difference
        mean_diff = np.mean(combined['radius'] - combined['file'])
        
        comparison_data.append({
            'feature': feature,
            'feature_label': FEATURE_LABELS.get(feature, feature),
            'n_pairs': len(combined),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            't_stat': t_stat,
            't_p': t_p,
            'mean_diff': mean_diff,
            'radius_mean': np.mean(combined['radius']),
            'file_mean': np.mean(combined['file'])
        })
    
    return pd.DataFrame(comparison_data)


# VISUALIZATIONS
def create_distribution_plots(stats_df, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    
    # 1. Box plots of CV (Coefficient of Variation) by approach
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cv_data = stats_df[stats_df['cv'].notna()].copy()
    cv_data['label'] = cv_data['approach'] + ': ' + cv_data['feature_label']
    
    # Sort by CV
    cv_data = cv_data.sort_values('cv', ascending=True)
    
    colors = ['#2E86AB' if a == 'radius' else '#F18F01' for a in cv_data['approach']]
    
    bars = ax.barh(cv_data['label'], cv_data['cv'], color=colors, alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='CV = 0.5')
    ax.axvline(x=0.2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='CV = 0.2')
    
    ax.set_xlabel('Coefficient of Variation (CV = Std/Mean)', fontsize=12)
    ax.set_title('Feature Variability by Approach', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cv_comparison.png'), dpi=150)
    plt.close()
    print("Saved: cv_comparison.png")
    
    # 2. Mean comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    radius_mean = stats_df[stats_df['approach'] == 'radius']
    file_mean = stats_df[stats_df['approach'] == 'file']
    
    merged = radius_mean.merge(file_mean, on='feature', suffixes=('_radius', '_file'))
    
    x = np.arange(len(merged))
    width = 0.35
    
    ax.bar(x - width/2, merged['mean_radius'], width, label='Radius-based', 
           color=COLORS['radius'], alpha=0.7)
    ax.bar(x + width/2, merged['mean_file'], width, label='File #-based', 
           color=COLORS['file'], alpha=0.7)
    
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Mean Value', fontsize=12)
    ax.set_title('Feature Means: Radius vs File #', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in merged['feature']], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mean_comparison.png'), dpi=150)
    plt.close()
    print("Saved: mean_comparison.png")
    
    # 3. Summary statistics table as heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap data
    stats_pivot = stats_df.pivot(index='feature_label', columns='approach', 
                                 values=['mean', 'std', 'cv'])
    
    if not stats_pivot.empty:
        # Flatten multi-index columns
        stats_pivot.columns = [f"{col[0]}_{col[1]}" for col in stats_pivot.columns]
        
        im = ax.imshow(stats_pivot.values, cmap='RdBu_r', aspect='auto')
        
        ax.set_xticks(range(len(stats_pivot.columns)))
        ax.set_xticklabels(stats_pivot.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(stats_pivot.index)))
        ax.set_yticklabels(stats_pivot.index)
        
        # Add value labels
        for i in range(len(stats_pivot.index)):
            for j in range(len(stats_pivot.columns)):
                val = stats_pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                           color='white' if abs(val) > 0.5 else 'black', fontsize=8)
        
        plt.colorbar(im, label='Value')
        ax.set_title('Feature Statistics Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'statistics_heatmap.png'), dpi=150)
        plt.close()
        print("Saved: statistics_heatmap.png")


def create_correlation_plots(comparison_df, radius_df, file_df, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Scatter plots for each feature
    n_features = len([f for f in FEATURES if f in radius_df.columns and f in file_df.columns])
    
    if n_features > 0:
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        for feature in FEATURES:
            if feature not in radius_df.columns or feature not in file_df.columns:
                continue
            
            ax = axes[plot_idx]
            plot_idx += 1
            
            # Get paired data
            combined = pd.DataFrame({
                'radius': radius_df[feature],
                'file': file_df[feature]
            }).dropna()
            
            if len(combined) > 1:
                ax.scatter(combined['radius'], combined['file'], alpha=0.5, s=30)
                
                # Add regression line
                if len(combined) > 2:
                    try:
                        z = np.polyfit(combined['radius'], combined['file'], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(combined['radius'].min(), combined['radius'].max(), 100)
                        ax.plot(x_range, p(x_range), 'r-', linewidth=2)
                    except:
                        pass
                
                # Add diagonal line
                max_val = max(combined['radius'].max(), combined['file'].max())
                min_val = min(combined['radius'].min(), combined['file'].min())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1)
                
                # Add correlation coefficient
                try:
                    corr, _ = pearsonr(combined['radius'], combined['file'])
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                           fontsize=10, verticalalignment='top')
                except:
                    pass
            
            ax.set_xlabel('Radius-based', fontsize=10)
            ax.set_ylabel('File #-based', fontsize=10)
            ax.set_title(FEATURE_LABELS.get(feature, feature), fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Radius vs File # Feature Correlation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'feature_correlations.png'), dpi=150)
        plt.close()
        print("Saved: feature_correlations.png")
    
    # 2. Correlation bar chart
    if not comparison_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        comparison_df = comparison_df.sort_values('pearson_r', ascending=False)
        
        bars = ax.barh(comparison_df['feature_label'], comparison_df['pearson_r'], 
                      color='#2E86AB', alpha=0.7)
        
        # Color bars by correlation strength
        for i, bar in enumerate(bars):
            r = comparison_df.iloc[i]['pearson_r']
            if r > 0.7:
                bar.set_color('#6A994E')  # Green - strong
            elif r > 0.5:
                bar.set_color('#F18F01')  # Orange - moderate
            else:
                bar.set_color('#D62828')  # Red - weak
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Pearson Correlation (r)', fontsize=12)
        ax.set_title('Radius vs File # Feature Correlation Strength', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'correlation_strength.png'), dpi=150)
        plt.close()
        print("Saved: correlation_strength.png")


def create_group_comparison_plots(group_stats_df, output_folder):

    if group_stats_df.empty:
        print("No group statistics available")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Group by feature and create grouped bar charts
    features = group_stats_df['feature'].unique()
    
    n_cols = min(2, len(features))
    n_rows = (len(features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if len(features) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        subset = group_stats_df[group_stats_df['feature'] == feature]
        groups = sorted(subset['group'].unique())
        
        x = np.arange(len(groups))
        width = 0.25
        
        # Plot mean with error bars (std as error)
        means = [subset[subset['group'] == g]['mean'].values[0] for g in groups]
        stds = [subset[subset['group'] == g]['std'].values[0] for g in groups]
        
        ax.bar(x, means, width, color='#2E86AB', alpha=0.7, 
               yerr=stds, capsize=3, label='Mean ± Std')
        
        ax.set_xlabel('Group', fontsize=10)
        ax.set_ylabel('Mean Value', fontsize=10)
        ax.set_title(FEATURE_LABELS.get(feature, feature), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Feature Variation by Group (Treatment)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'group_comparison.png'), dpi=150)
    plt.close()
    print("Saved: group_comparison.png")


def create_violin_plots(radius_df, file_df, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # Prepare data for radius approach
    if radius_df is not None:
        radius_melted = radius_df[FEATURES].melt(var_name='feature', value_name='value')
        radius_melted['approach'] = 'Radius'
        radius_melted = radius_melted.dropna()
    
    # Prepare data for file approach
    if file_df is not None:
        file_melted = file_df[FEATURES].melt(var_name='feature', value_name='value')
        file_melted['approach'] = 'File #'
        file_melted = file_melted.dropna()
    
    # Combine
    if radius_df is not None and file_df is not None:
        combined_melted = pd.concat([radius_melted, file_melted], ignore_index=True)
        
        # Map feature labels
        combined_melted['feature_label'] = combined_melted['feature'].map(FEATURE_LABELS)
        
        # Create violin plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.violinplot(data=combined_melted, x='feature_label', y='value', 
                      hue='approach', split=True, inner='quartile',
                      palette=[COLORS['radius'], COLORS['file']], ax=ax)
        
        ax.set_xlabel('Feature', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Feature Distributions: Radius vs File #', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'violin_plots.png'), dpi=150)
        plt.close()
        print("Saved: violin_plots.png")


def create_correlation_matrix(radius_df, file_df, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 8)
    
    # Radius correlation matrix
    if radius_df is not None:
        radius_corr = radius_df[FEATURES].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(radius_corr, dtype=bool))
        sns.heatmap(radius_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   fmt='.2f', square=True, linewidths=0.5, ax=ax,
                   xticklabels=[FEATURE_LABELS.get(f, f) for f in radius_corr.index],
                   yticklabels=[FEATURE_LABELS.get(f, f) for f in radius_corr.index])
        ax.set_title('Radius-based Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'radius_correlation_matrix.png'), dpi=150)
        plt.close()
        print("Saved: radius_correlation_matrix.png")
    
    # File correlation matrix
    if file_df is not None:
        file_corr = file_df[FEATURES].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(file_corr, dtype=bool))
        sns.heatmap(file_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   fmt='.2f', square=True, linewidths=0.5, ax=ax,
                   xticklabels=[FEATURE_LABELS.get(f, f) for f in file_corr.index],
                   yticklabels=[FEATURE_LABELS.get(f, f) for f in file_corr.index])
        ax.set_title('File #-based Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'file_correlation_matrix.png'), dpi=150)
        plt.close()
        print("Saved: file_correlation_matrix.png")


def main():
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Load data
    feature_table = load_feature_table(FEATURE_TABLE_PATH)
    
    if feature_table is None or feature_table.empty:
        print("Error: No data loaded")
        return
    
    print(f"\nAvailable columns in feature_table ({len(feature_table.columns)} total):")
    for col in feature_table.columns:
        print(f"  - {col}")
    
    # Extract radius and file-based features
    radius_df = extract_features_by_approach(feature_table, FEATURES, 'radius')
    file_df = extract_features_by_approach(feature_table, FEATURES, 'file')
    
    print(f"\nRadius features: {radius_df.shape if radius_df is not None else 'None'}")
    print(f"File features: {file_df.shape if file_df is not None else 'None'}")
    
    all_stats = []
    
    # Compute statistics for radius approach
    if radius_df is not None:
        radius_stats = compute_feature_statistics(radius_df, FEATURES, 'Radius')
        if not radius_stats.empty:
            all_stats.append(radius_stats)
            print(f"\nRadius statistics computed: {len(radius_stats)} rows")
    
    # Compute statistics for file approach
    if file_df is not None:
        file_stats = compute_feature_statistics(file_df, FEATURES, 'File')
        if not file_stats.empty:
            all_stats.append(file_stats)
            print(f"File statistics computed: {len(file_stats)} rows")
    
    # Combine statistics
    if all_stats:
        combined_stats = pd.concat(all_stats, ignore_index=True)
        combined_stats.to_csv(os.path.join(OUTPUT_FOLDER, 'feature_statistics.csv'), index=False)
        print(f"\nStatistics saved to: {os.path.join(OUTPUT_FOLDER, 'feature_statistics.csv')}")
        
        # Print summary
        print(combined_stats[['approach', 'feature_label', 'n', 'mean', 'std', 'cv']].to_string(index=False))
    else:
        combined_stats = pd.DataFrame()
        print("No statistics computed")
    
    # Compare approaches
    comparison_df = pd.DataFrame()
    if radius_df is not None and file_df is not None:
        comparison_df = compare_approaches(radius_df, file_df, FEATURES)
        if not comparison_df.empty:
            comparison_df.to_csv(os.path.join(OUTPUT_FOLDER, 'approach_comparison.csv'), index=False)
            print(f"\nApproach comparison saved to: {os.path.join(OUTPUT_FOLDER, 'approach_comparison.csv')}")
    
    # Create visualizations

    # 1. Distribution plots
    if not combined_stats.empty:
        create_distribution_plots(combined_stats, OUTPUT_FOLDER)
    
    # 2. Correlation plots
    if radius_df is not None and file_df is not None:
        create_correlation_plots(comparison_df, radius_df, file_df, OUTPUT_FOLDER)
    
    # 3. Group comparison plots
    if radius_df is not None:
        group_stats = compute_group_statistics(radius_df, FEATURES, 'treatment')
        create_group_comparison_plots(group_stats, OUTPUT_FOLDER)
    
    # 4. Violin plots
    create_violin_plots(radius_df, file_df, OUTPUT_FOLDER)
    
    # 5. Correlation matrices
    create_correlation_matrix(radius_df, file_df, OUTPUT_FOLDER)
    
    # Final summary
    print(f"\nAll results saved to: {OUTPUT_FOLDER}/")
    print("\nOutput files:")
    for f in sorted(os.listdir(OUTPUT_FOLDER)):
        if f.endswith('.csv'):
            print(f"  - {f} (CSV)")
        elif f.endswith('.png'):
            print(f"  - {f} (PNG)")


if __name__ == "__main__":
    main()