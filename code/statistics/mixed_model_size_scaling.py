# Is the pattern just a side effect of root size?
#
# Regresses each shape feature against two independent size proxies, number
# of cell files and stele diameter. A feature is only called size-driven if
# it clears a marginal R-squared of 0.1, because with this many roots a
# negligible effect is still statistically significant.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Get project root
SCRIPT_DIR = Path(__file__).parent.absolute()  # code/statistics/
CODE_DIR = SCRIPT_DIR.parent  # code/
PROJECT_ROOT = CODE_DIR.parent  # project root

FEATURE_TABLE_PATH = str(PROJECT_ROOT / "results" / "feature_table.csv")
OUTPUT_FOLDER = str(PROJECT_ROOT / "results" / "mixed_model_size_scaling")

GROUP_COLUMN = 'plant_group'
COVARIATES = ['n_files', 'stele_diameter_um']

# Reconstructed guess - see module docstring
SIZE_DRIVEN_R2_THRESHOLD = 0.1

FEATURES = [
    'radius_peak_height',
    'radius_peak_position',
    'radius_rise_slope',
    'radius_decay_slope',
    'radius_outer_rise_magnitude',
]


def marginal_r_squared(model_result, df, formula_rhs_columns):

    fitted = model_result.fittedvalues
    observed = df[formula_rhs_columns['y']]
    if fitted.std() == 0 or observed.std() == 0:
        return 0.0
    r = np.corrcoef(fitted, observed)[0, 1]
    return float(r ** 2)


def fit_one_feature(df, feature):

    needed_cols = [feature, GROUP_COLUMN] + COVARIATES
    sub = df[needed_cols].dropna()

    if len(sub) < 5:
        return None, f"Insufficient data: {len(sub)} rows"

    if sub[GROUP_COLUMN].nunique() < 2:
        return None, f"Insufficient groups for random effect: {sub[GROUP_COLUMN].nunique()} unique {GROUP_COLUMN}"

    formula = f"{feature} ~ n_files + stele_diameter_um"

    model_type = 'MixedLM'
    try:
        model = smf.mixedlm(formula, data=sub, groups=sub[GROUP_COLUMN])
        result = model.fit(reml=False)
        r2 = marginal_r_squared(result, sub, {'y': feature})
    except Exception as e:
        try:
            model_type = 'OLS_fallback'
            result = smf.ols(formula, data=sub).fit()
            r2 = result.rsquared
        except Exception as e2:
            return None, f"Mixed model and OLS fallback both failed: {e2}"

    size_driven = r2 >= SIZE_DRIVEN_R2_THRESHOLD

    rows = []
    for covariate in COVARIATES:
        if covariate not in result.params.index:
            continue
        rows.append({
            'feature': feature,
            'covariate': covariate,
            'coefficient': result.params[covariate],
            'p_value': result.pvalues[covariate],
            'size_driven': size_driven,
            'model_type': model_type,
        })

    summary_row = {
        'feature': feature,
        'n_observations': len(sub),
        'n_files_coef': result.params.get('n_files', np.nan),
        'stele_diameter_coef': result.params.get('stele_diameter_um', np.nan),
        'size_driven': size_driven,
        'r2': r2,
        'model_type': model_type,
    }

    return {'coef_rows': rows, 'summary_row': summary_row}, None


def create_coefficient_plot(coefficients_df, output_folder):
    if coefficients_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, covariate in zip(axes, COVARIATES):
        sub = coefficients_df[coefficients_df['covariate'] == covariate]
        if sub.empty:
            continue
        colors = ['steelblue' if p < 0.05 else 'lightgray' for p in sub['p_value']]
        ax.barh(sub['feature'], sub['coefficient'], color=colors)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel(f'Coefficient ({covariate})', fontsize=11, fontweight='bold')
        ax.set_title(f'Effect of {covariate}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle('Size-Scaling Mixed Model Coefficients\n(dark bars: p < 0.05)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'coefficient_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: coefficient_plot.png")


def create_coefficient_heatmap(coefficients_df, output_folder):
    if coefficients_df.empty:
        return

    pivot = coefficients_df.pivot(index='feature', columns='covariate', values='coefficient')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.4g', cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'label': 'Coefficient'})
    ax.set_title('Size-Scaling Mixed Model Coefficients', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'coefficient_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: coefficient_heatmap.png")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.exists(FEATURE_TABLE_PATH):
        print("Feature table not found - nothing to do.")
        return

    df = pd.read_csv(FEATURE_TABLE_PATH)
    print(f"Loaded {len(df)} roots from feature table")

    df[GROUP_COLUMN] = df['population'].astype(str) + '_' + df['plant_number'].astype(str)

    all_coef_rows = []
    all_summary_rows = []
    error_rows = []

    for feature in FEATURES:
        if feature not in df.columns:
            error_rows.append({'feature': feature, 'status': 'error', 'message': 'Column not found in feature table'})
            continue

        print(f"\nFitting: {feature} ~ n_files + stele_diameter_um + (1|{GROUP_COLUMN})")
        result, error = fit_one_feature(df, feature)

        if error:
            print(f"  ERROR: {error}")
            error_rows.append({'feature': feature, 'status': 'error', 'message': error})
            continue

        all_coef_rows.extend(result['coef_rows'])
        all_summary_rows.append(result['summary_row'])
        print(f"  n={result['summary_row']['n_observations']}, "
              f"R²={result['summary_row']['r2']:.4f}, "
              f"size_driven={result['summary_row']['size_driven']}, "
              f"model={result['summary_row']['model_type']}")

    coefficients_df = pd.DataFrame(all_coef_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    errors_df = pd.DataFrame(error_rows, columns=['feature', 'status', 'message'])

    coefficients_df.to_csv(os.path.join(OUTPUT_FOLDER, 'coefficients.csv'), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_FOLDER, 'summary.csv'), index=False)
    errors_df.to_csv(os.path.join(OUTPUT_FOLDER, 'errors.csv'), index=False)
    print(f"\nSaved: coefficients.csv, summary.csv, errors.csv")

    create_coefficient_plot(coefficients_df, OUTPUT_FOLDER)
    create_coefficient_heatmap(coefficients_df, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
