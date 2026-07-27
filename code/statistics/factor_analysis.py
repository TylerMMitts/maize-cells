import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway, kruskal
import warnings
warnings.filterwarnings('ignore')

# Try to import statsmodels
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.anova import anova_lm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("statsmodels not available. Some analyses will be limited.")


def load_data(master_path="results/master_summary.csv", cell_file_dir="results/cell_file/cell_file_counting"):

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
    
    # Aggregate
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


def run_oneway_anova(data, factor, target_cols=None):

    # Force convert data to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        print(f"Converting data from {type(data)} to DataFrame")
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            print(f"ERROR: Could not convert data to DataFrame: {e}")
            return None
    
    # Verify data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        print(f"ERROR: data is a {type(data)}, not a DataFrame!")
        return None
    
    # Create a copy to work with
    df = data.copy()
    
    # SAFETY CHECK: Ensure target_cols is a dictionary
    if target_cols is None:
        target_cols = {
            'stele_area_um2': 'Stele Area',
            'file_count': 'File Count',
            'average_cell_area_um2': 'Avg Cell Area'
        }
    elif not isinstance(target_cols, dict):
        # If it's a list or array, convert to dictionary
        if isinstance(target_cols, (list, tuple, np.ndarray)):
            print(f"Warning: target_cols is a {type(target_cols).__name__}, converting to dictionary")
            # If it's a list of strings, use them as keys
            if all(isinstance(x, str) for x in target_cols):
                target_cols = {col: col.replace('_', ' ').title() for col in target_cols}
            else:
                # Default targets
                target_cols = {
                    'stele_area_um2': 'Stele Area',
                    'file_count': 'File Count',
                    'average_cell_area_um2': 'Avg Cell Area'
                }
    
    if factor not in df.columns:
        print(f"Factor '{factor}' not in data columns")
        return None
    
    groups = df[factor].unique()
    groups = [g for g in groups if pd.notna(g)]
    
    if len(groups) < 2:
        print(f"Only {len(groups)} groups found for factor '{factor}'")
        return None
    
    all_results = {}
    
    # Get the column names as a list of strings
    target_names = list(target_cols.keys())
    print(f"[DEBUG] target_names: {target_names}")
    
    # Use a simple for loop with index
    for i in range(len(target_names)):
        target_col = target_names[i]
        
        # Skip if target_col is not a string
        if not isinstance(target_col, str):
            print(f"Skipping non-string target_col: {target_col} (type: {type(target_col)})")
            continue
        
        # Skip if column doesn't exist in data
        if target_col not in df.columns:
            print(f"Skipping {target_col} - not in data columns")
            continue
        
        target_name = target_cols[target_col]
        print(f"[DEBUG] Processing: {target_col} -> {target_name}")
        
        # Extract data for each group
        group_data = []
        group_names = []
        for group in groups:
            subset = df[df[factor] == group][target_col]
            if len(subset) > 0:
                group_data.append(subset.values)
                group_names.append(str(group))
        
        if len(group_data) < 2:
            continue
        
        # ANOVA
        f_stat, p_value = f_oneway(*group_data)
        
        # Kruskal-Wallis (non-parametric)
        h_stat, kw_p_value = kruskal(*group_data)
        
        # Effect size (eta-squared)
        all_data = np.concatenate(group_data)
        grand_mean = np.mean(all_data)
        
        ss_between = 0
        ss_total = np.sum((all_data - grand_mean) ** 2)
        
        for group, data in zip(groups, group_data):
            ss_between += len(data) * (np.mean(data) - grand_mean) ** 2
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Summary statistics per group
        group_stats = {}
        for group, data in zip(groups, group_data):
            group_stats[str(group)] = {
                'n': len(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'se': np.std(data) / np.sqrt(len(data))
            }
        
        all_results[target_col] = {
            'target_name': target_name,
            'factor': factor,
            'groups': group_stats,
            'f_stat': f_stat,
            'p_value': p_value,
            'kw_stat': h_stat,
            'kw_p_value': kw_p_value,
            'eta_squared': eta_squared,
            'n_total': len(all_data),
            'significant': p_value < 0.05
        }
    
    return all_results


def run_tukey_hsd(data, factor, target_cols=None):

    if not STATSMODELS_AVAILABLE:
        return None
    
    # Force convert data to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        print(f"Converting data from {type(data)} to DataFrame in Tukey HSD")
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            print(f"ERROR: Could not convert data to DataFrame: {e}")
            return None
    
    # Verify data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        print(f"ERROR: data is a {type(data)}, not a DataFrame!")
        return None
    
    # Create a copy to work with
    df = data.copy()
    
    # SAFETY CHECK: Ensure target_cols is a dictionary
    if target_cols is None:
        target_cols = {
            'stele_area_um2': 'Stele Area',
            'file_count': 'File Count',
            'average_cell_area_um2': 'Avg Cell Area'
        }
    elif not isinstance(target_cols, dict):
        if isinstance(target_cols, (list, tuple, np.ndarray)):
            print(f"Warning: target_cols is a {type(target_cols).__name__}, converting to dictionary")
            if all(isinstance(x, str) for x in target_cols):
                target_cols = {col: col.replace('_', ' ').title() for col in target_cols}
            else:
                target_cols = {
                    'stele_area_um2': 'Stele Area',
                    'file_count': 'File Count',
                    'average_cell_area_um2': 'Avg Cell Area'
                }
    
    if factor not in df.columns:
        return None
    
    all_tukey = {}
    
    target_names = list(target_cols.keys())
    
    for i in range(len(target_names)):
        target_col = target_names[i]
        
        if not isinstance(target_col, str):
            continue
        if target_col not in df.columns:
            continue
        
        try:
            tukey = pairwise_tukeyhsd(df[target_col], df[factor], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], 
                                   columns=tukey.summary().data[0])
            all_tukey[target_col] = tukey_df
        except Exception as e:
            print(f"Error in Tukey HSD for {target_col}: {e}")
    
    return all_tukey


def run_twoway_anova(data, factor1, factor2, target_cols=None):

    if not STATSMODELS_AVAILABLE:
        return None
    
    # Force convert data to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        print(f"Converting data from {type(data)} to DataFrame in Two-way ANOVA")
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            print(f"ERROR: Could not convert data to DataFrame: {e}")
            return None
    
    # Verify data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        print(f"ERROR: data is a {type(data)}, not a DataFrame!")
        return None
    
    # Create a copy to work with
    df = data.copy()
    
    # SAFETY CHECK: Ensure target_cols is a dictionary
    if target_cols is None:
        target_cols = {
            'stele_area_um2': 'Stele Area',
            'file_count': 'File Count',
            'average_cell_area_um2': 'Avg Cell Area'
        }
    elif not isinstance(target_cols, dict):
        if isinstance(target_cols, (list, tuple, np.ndarray)):
            print(f"Warning: target_cols is a {type(target_cols).__name__}, converting to dictionary")
            if all(isinstance(x, str) for x in target_cols):
                target_cols = {col: col.replace('_', ' ').title() for col in target_cols}
            else:
                target_cols = {
                    'stele_area_um2': 'Stele Area',
                    'file_count': 'File Count',
                    'average_cell_area_um2': 'Avg Cell Area'
                }
    
    if factor1 not in df.columns or factor2 not in df.columns:
        return None
    
    # Create a copy with categorical factors
    df_anova = df.dropna(subset=[factor1, factor2]).copy()
    df_anova[factor1] = df_anova[factor1].astype(str)
    df_anova[factor2] = df_anova[factor2].astype(str)
    
    all_anova = {}
    
    target_names = list(target_cols.keys())
    
    for i in range(len(target_names)):
        target_col = target_names[i]
        
        if not isinstance(target_col, str):
            continue
        if target_col not in df.columns:
            continue
        
        try:
            model = ols(f'{target_col} ~ C({factor1}) * C({factor2})', data=df_anova).fit()
            anova_table = anova_lm(model, typ=2)
            all_anova[target_col] = anova_table
        except Exception as e:
            print(f"Error in two-way ANOVA for {target_col}: {e}")
    
    return all_anova


def create_factor_plots(anova_results, data, output_dir, factor):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    target_labels = {
        'stele_area_um2': 'Stele Area (µm²)',
        'file_count': 'File Count',
        'average_cell_area_um2': 'Avg Cell Area (µm²)'
    }
    
    for target_col, result in anova_results.items():
        target_name = result['target_name']
        target_label = target_labels.get(target_col, target_name)
        
        # 1. Boxplot by group
        fig, ax = plt.subplots(figsize=(10, 6))
        
        groups = list(result['groups'].keys())
        group_data = [data[data[factor] == g][target_col].values for g in groups]
        
        bp = ax.boxplot(group_data, labels=groups, patch_artist=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel(factor.capitalize(), fontsize=12)
        ax.set_ylabel(target_label, fontsize=12)
        
        significance = 'SIGNIFICANT' if result['significant'] else 'NOT significant'
        ax.set_title(f'{target_name} by {factor.capitalize()}\nANOVA: p = {result["p_value"]:.4f} ({significance})', 
                    fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add group statistics
        stats_text = ""
        for group, stats_dict in result['groups'].items():
            stats_text += f"{group}: n={stats_dict['n']}, mean={stats_dict['mean']:.1f}±{stats_dict['std']:.1f}\n"
        
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / f'{factor}_{target_col}_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Violin plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.violinplot(data=data, x=factor, y=target_col, ax=ax, palette='Set2')
        ax.set_xlabel(factor.capitalize(), fontsize=12)
        ax.set_ylabel(target_label, fontsize=12)
        ax.set_title(f'{target_name} Distribution by {factor.capitalize()}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path / f'{factor}_{target_col}_violinplot.png', dpi=300, bbox_inches='tight')
        plt.close()


def run_factor_analysis_full(
    master_path="../results/master_summary.csv",
    cell_file_dir="../results/cell_file/cell_file_counting",
    output_dir="../results/factor_analysis",
    factors=['treatment', 'root_type', 'population']
):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(master_path, cell_file_dir)
    print(f"\nLoaded {len(df)} roots for analysis")
    print(f"Data columns: {list(df.columns)}")
    
    target_cols = {
        'stele_area_um2': 'Stele Area',
        'file_count': 'File Count',
        'average_cell_area_um2': 'Avg Cell Area'
    }
    
    all_anova_results = {}
    all_tukey_results = {}
    
    # 1. One-way ANOVA for each factor
    print("ONE-WAY ANOVA RESULTS")
    
    for factor in factors:
        if factor not in df.columns:
            print(f"Warning: {factor} not found in data")
            continue
        
        print(f"\n{factor.upper()}:")
        
        # Pass the DataFrame and target columns
        results = run_oneway_anova(df, factor, target_cols.copy())
        
        if results:
            all_anova_results[factor] = results
            
            for target_col, result in results.items():
                print(f"\n  {result['target_name']}:")
                print(f"    ANOVA: F = {result['f_stat']:.3f}, p = {result['p_value']:.4f}")
                print(f"    Kruskal-Wallis: H = {result['kw_stat']:.3f}, p = {result['kw_p_value']:.4f}")
                print(f"    Eta-squared: {result['eta_squared']:.3f}")
                print(f"    Significant: {result['significant']}")
                
                print("    Group means:")
                for group, stats_dict in result['groups'].items():
                    print(f"      {group}: {stats_dict['mean']:.1f} ± {stats_dict['std']:.1f} (n={stats_dict['n']})")
                
                # Save ANOVA results
                anova_summary = pd.DataFrame([{
                    'factor': factor,
                    'target': result['target_name'],
                    'f_stat': result['f_stat'],
                    'p_value': result['p_value'],
                    'kw_stat': result['kw_stat'],
                    'kw_p_value': result['kw_p_value'],
                    'eta_squared': result['eta_squared'],
                    'n_total': result['n_total'],
                    'significant': result['significant']
                }])
                anova_summary.to_csv(output_path / f'{factor}_{target_col}_anova_summary.csv', index=False)
    
    # 2. Tukey HSD for each factor
    print("TUKEY HSD POST-HOC RESULTS")
    
    for factor in factors:
        if factor not in df.columns:
            continue
        
        print(f"\n{factor.upper()}:")
        tukey_results = run_tukey_hsd(df, factor, target_cols.copy())
        
        if tukey_results:
            all_tukey_results[factor] = tukey_results
            
            for target_col, tukey_df in tukey_results.items():
                if tukey_df is not None and not tukey_df.empty:
                    target_name = target_cols.get(target_col, target_col)
                    print(f"\n  {target_name}:")
                    print(tukey_df.to_string(index=False))
                    tukey_df.to_csv(output_path / f'{factor}_{target_col}_tukey_results.csv', index=False)
                else:
                    print(f"  {target_name}: No significant differences found")
    
    # 3. Two-way ANOVA (Treatment × Root Type)
    print("TWO-WAY ANOVA: Treatment × Root Type")
    
    if 'treatment' in df.columns and 'root_type' in df.columns:
        twoway_results = run_twoway_anova(df, 'treatment', 'root_type', target_cols.copy())
        
        if twoway_results:
            for target_col, anova_table in twoway_results.items():
                if anova_table is not None:
                    target_name = target_cols.get(target_col, target_col)
                    print(f"\n{target_name}:")
                    print(anova_table)
                    anova_table.to_csv(output_path / f'twoway_anova_{target_col}.csv')
    
    # 4. Create visualizations
    for factor, results in all_anova_results.items():
        create_factor_plots(results, df, output_path, factor)
    
    # 5. Save summary statistics for each factor
    for factor in factors:
        if factor not in df.columns:
            continue
        
        for target_col, target_name in target_cols.items():
            if target_col not in df.columns:
                continue
            
            summary = df.groupby(factor)[target_col].agg(['mean', 'std', 'count', 'min', 'max']).round(2)
            summary.to_csv(output_path / f'{factor}_{target_col}_summary_stats.csv')
    
    return {
        'anova_results': all_anova_results,
        'tukey_results': all_tukey_results,
        'data': df
    }

def main(
    master_path="../results/master_summary.csv",
    cell_file_dir="../results/cell_file/cell_file_counting",
    output_dir="../results/factor_analysis",
    factors=['treatment', 'root_type', 'population']
):
    
    results = run_factor_analysis_full(
        master_path=master_path,
        cell_file_dir=cell_file_dir,
        output_dir=output_dir,
        factors=factors
    )

    print(f"\nResults saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    main()