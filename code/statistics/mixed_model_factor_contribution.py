# How much of each feature does each factor explain?
#
# Mixed-effects models with a random intercept per plant group, so repeated
# roots from one plant are not treated as independent samples.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Get project root
SCRIPT_DIR = Path(__file__).parent.absolute()  # code/statistics/
CODE_DIR = SCRIPT_DIR.parent  # code/
PROJECT_ROOT = CODE_DIR.parent  # project root

FEATURE_TABLE_PATH = str(PROJECT_ROOT / "results" / "feature_table.csv")
OUTPUT_FOLDER = str(PROJECT_ROOT / "results" / "mixed_model_factor_contribution")

GROUP_COLUMN = 'plant_group'
FACTORS = ['treatment', 'root_type', 'population']

FEATURES = [
    'radius_peak_height',
    'radius_peak_position',
    'radius_rise_slope',
    'radius_decay_slope',
    'radius_outer_rise_magnitude',
]


def usable_factors(df, feature):

    base_cols = [feature, GROUP_COLUMN] + FACTORS
    sub = df[base_cols].dropna(subset=[feature, GROUP_COLUMN])

    skipped = {}
    factors_to_use = []
    for factor in FACTORS:
        n_levels = sub[factor].dropna().nunique()
        if n_levels < 2:
            skipped[factor] = f"only {n_levels} non-null level(s) in current data"
        else:
            factors_to_use.append(factor)

    # Now drop rows missing any of the factors we're actually going to use
    sub = sub.dropna(subset=factors_to_use)

    return sub, factors_to_use, skipped


def fit_model(df, feature, factors, group_col, use_mixed):
    formula = feature + " ~ " + " + ".join(f"C({f})" for f in factors) if factors else f"{feature} ~ 1"
    if use_mixed:
        model = smf.mixedlm(formula, data=df, groups=df[group_col])
        result = model.fit(reml=False)
        _ = result.fittedvalues
        return result
    return smf.ols(formula, data=df).fit()


def model_df(result, use_mixed):
    return result.df_modelwc if use_mixed else result.df_model


def fit_one_feature(df, feature):

    sub, factors_to_use, skipped = usable_factors(df, feature)

    if len(sub) < 5:
        raise ValueError(f"Insufficient data: {len(sub)} rows")
    if not factors_to_use:
        raise ValueError("No usable factors (all have < 2 levels in current data)")
    if sub[GROUP_COLUMN].nunique() < 2:
        raise ValueError(f"Insufficient groups for random effect: {sub[GROUP_COLUMN].nunique()} unique {GROUP_COLUMN}")

    model_type = 'MixedLM'
    try:
        full_result = fit_model(sub, feature, factors_to_use, GROUP_COLUMN, use_mixed=True)
    except Exception:
        model_type = 'OLS_fallback'
        full_result = fit_model(sub, feature, factors_to_use, GROUP_COLUMN, use_mixed=False)

    use_mixed = (model_type == 'MixedLM')

    coef_rows = []
    for name, coef in full_result.params.items():
        if name in ('Intercept', 'Group Var') or not name.startswith('C('):
            continue
        # statsmodels formats dummy names like "C(treatment)[T.WS]"
        factor = name.split('[')[0][2:-1]
        level = name.split('[T.')[-1].rstrip(']')
        coef_rows.append({
            'feature': feature,
            'factor': factor,
            'level': level,
            'coefficient': coef,
            'p_value': full_result.pvalues[name],
            'model_type': model_type,
        })

    contribution_rows = []
    for factor in factors_to_use:
        remaining = [f for f in factors_to_use if f != factor]
        try:
            reduced_result = fit_model(sub, feature, remaining, GROUP_COLUMN, use_mixed=use_mixed)

            chi2_stat = 2 * (full_result.llf - reduced_result.llf)
            chi2_stat = max(chi2_stat, 0.0)  # guard against tiny negative values from optimizer noise
            df_diff = model_df(full_result, use_mixed) - model_df(reduced_result, use_mixed)
            df_diff = max(int(round(df_diff)), 1)
            p_value = scipy_stats.chi2.sf(chi2_stat, df_diff)

            contribution_rows.append({
                'feature': feature,
                'factor': factor,
                'chi2': chi2_stat,
                'df': df_diff,
                'p_value': p_value,
                'model_type': model_type,
            })
        except Exception as e:
            contribution_rows.append({
                'feature': feature,
                'factor': factor,
                'chi2': np.nan,
                'df': np.nan,
                'p_value': np.nan,
                'model_type': model_type,
            })

    fitted = full_result.fittedvalues
    observed = sub[feature]
    r2 = float(np.corrcoef(fitted, observed)[0, 1] ** 2) if fitted.std() > 0 and observed.std() > 0 else 0.0

    summary_row = {
        'feature': feature,
        'n_observations': len(sub),
        'factors_used': ','.join(factors_to_use),
        'r2': r2,
        'model_type': model_type,
    }

    return coef_rows, contribution_rows, summary_row, skipped


def create_coefficient_heatmap(coefficients_df, output_folder):
    if coefficients_df.empty:
        return

    coefficients_df = coefficients_df.copy()
    coefficients_df['factor_level'] = coefficients_df['factor'] + ': ' + coefficients_df['level']
    pivot = coefficients_df.pivot(index='factor_level', columns='feature', values='coefficient')

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), max(6, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt='.3g', cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'label': 'Coefficient'})
    ax.set_title('Factor Contribution Model: Coefficients by Level', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'coefficient_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: coefficient_heatmap.png")


def create_contribution_plot(contribution_df, output_folder):
    if contribution_df.empty:
        return

    pivot = contribution_df.pivot(index='factor', columns='feature', values='chi2')

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', ax=ax,
                cbar_kws={'label': 'LRT chi-square (higher = more contribution)'})
    ax.set_title('How Much Each Factor Contributes to Each Feature\n(Likelihood-ratio test chi-square, dropping one factor at a time)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'contribution_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: contribution_plot.png")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.exists(FEATURE_TABLE_PATH):
        print("Feature table not found - nothing to do.")
        return

    df = pd.read_csv(FEATURE_TABLE_PATH)
    print(f"Loaded {len(df)} roots from feature table")

    df[GROUP_COLUMN] = df['population'].astype(str) + '_' + df['plant_number'].astype(str)

    all_coef_rows = []
    all_contribution_rows = []
    all_summary_rows = []
    error_rows = []

    for feature in FEATURES:
        if feature not in df.columns:
            error_rows.append({'feature': feature, 'status': 'error', 'message': 'Column not found in feature table'})
            continue

        print(f"\nFitting: {feature} ~ {' + '.join(FACTORS)} + (1|{GROUP_COLUMN})")
        try:
            coef_rows, contribution_rows, summary_row, skipped = fit_one_feature(df, feature)
        except Exception as e:
            print(f"  ERROR: {e}")
            error_rows.append({'feature': feature, 'status': 'error', 'message': str(e)})
            continue

        for factor, reason in skipped.items():
            print(f"  Skipped factor '{factor}': {reason}")
            error_rows.append({'feature': feature, 'status': 'skipped_factor',
                               'message': f"{factor}: {reason}"})

        all_coef_rows.extend(coef_rows)
        all_contribution_rows.extend(contribution_rows)
        all_summary_rows.append(summary_row)
        print(f"  n={summary_row['n_observations']}, factors={summary_row['factors_used']}, "
              f"R²={summary_row['r2']:.4f}, model={summary_row['model_type']}")

    coefficients_df = pd.DataFrame(all_coef_rows)
    contribution_df = pd.DataFrame(all_contribution_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    errors_df = pd.DataFrame(error_rows, columns=['feature', 'status', 'message'])

    coefficients_df.to_csv(os.path.join(OUTPUT_FOLDER, 'coefficients.csv'), index=False)
    contribution_df.to_csv(os.path.join(OUTPUT_FOLDER, 'contribution.csv'), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_FOLDER, 'summary.csv'), index=False)
    errors_df.to_csv(os.path.join(OUTPUT_FOLDER, 'errors.csv'), index=False)
    print(f"\nSaved: coefficients.csv, contribution.csv, summary.csv, errors.csv")

    create_coefficient_heatmap(coefficients_df, OUTPUT_FOLDER)
    create_contribution_plot(contribution_df, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
