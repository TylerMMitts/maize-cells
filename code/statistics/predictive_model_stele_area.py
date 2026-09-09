# Predicts a root's full anatomy from a single cheap measurement.
#
# Stele area, root radius, cell-file count and the six shape features, all
# from the mean cell area of the innermost file -- the one number that is
# quick to obtain without measuring the whole cross-section.
# of the innermost cortical file (cell_file_derivative == 0).
#
# Unlike the earlier version of this script, everything here operates at the
# ROOT level (one row per root_id, matching results/feature_table.csv), not the
# quadrant level -- first_file_avg_area_um2 is computed per quadrant from
# cell_assignments.csv, then averaged across a root's quadrants the same way
# build_feature_table.py averages every other feature. This is what lets the
# fitted model be used to predict a full root_model.html reconstruction (which
# is keyed by root_id) from just that one value.

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def extract_root_identifier(image_name):
    # Strip the quadrant prefix (BL_/BR_/TL_/TR_) from an image name to get the
    # root-level key -- identical logic to build_feature_table.py, duplicated here
    # so this script doesn't need that module's heavier dependencies (seaborn, tqdm).
    if not image_name:
        return image_name
    for prefix in ('BL_', 'BR_', 'TL_', 'TR_'):
        if image_name.startswith(prefix):
            return image_name[len(prefix):]
    return image_name


SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

FEATURE_TABLE_PATH = PROJECT_ROOT / 'results' / 'feature_table.csv'
CELL_FILE_DIR = PROJECT_ROOT / 'results' / 'cell_file' / 'cell_file_counting'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'predictive_model_stele'

# The 9 fields generate_root_model.py needs to reconstruct a full root: the six
# spline shape features (SHAPE_FIELDS there) plus the three geometry fields
# (GEOMETRY_FIELDS there, minus stele_diameter_um which is derived below).
TARGETS = [
    'stele_area_um2',
    'root_radius_um',
    'n_files',
    'file_peak_height',
    'file_peak_position',
    'file_rise_slope',
    'file_decay_slope',
    'file_minima_position',
    'file_outer_rise_magnitude',
]
POSITIVE_TARGETS = {'stele_area_um2', 'root_radius_um', 'n_files'}
PREDICTOR = 'first_file_avg_area_um2'


def compute_first_file_area_per_root():
    # Mean first-cortical-file cell area per root, averaged across quadrants.
    cell_files = list(CELL_FILE_DIR.rglob('cell_assignments.csv'))
    print(f"Found {len(cell_files)} cell_assignments.csv files")

    per_quadrant = {}
    for cell_file in cell_files:
        image_name = cell_file.parent.name
        try:
            df = pd.read_csv(cell_file)
        except Exception as e:
            print(f"  ERROR reading {cell_file}: {e}")
            continue

        if 'cell_file_derivative' not in df.columns or 'area_um2' not in df.columns:
            continue

        first_file = df[df['cell_file_derivative'] == 0]
        if len(first_file) == 0:
            continue

        per_quadrant[image_name] = first_file['area_um2'].mean()

    print(f"Computed first-file area for {len(per_quadrant)} quadrant images")

    # Aggregate quadrants belonging to the same root (same key build_feature_table.py uses).
    by_root = {}
    for image_name, area in per_quadrant.items():
        root_id = extract_root_identifier(image_name)
        by_root.setdefault(root_id, []).append(area)

    root_rows = [{'root_id': root_id, PREDICTOR: float(np.mean(vals)), 'n_quadrants_first_file': len(vals)}
                 for root_id, vals in by_root.items()]
    df_root = pd.DataFrame(root_rows)
    print(f"Aggregated to {len(df_root)} roots")
    return df_root


def fit_best_polynomial(X_train, X_test, y_train, y_test, degrees=(1, 2, 3)):
    # Fit degrees 1-3, pick the best by test R^2 (quadratic wins over cubic
    # unless cubic improves test R^2 by more than 0.02, for parsimony).
    fits = {}
    for degree in degrees:
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', LinearRegression())
        ])
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        fits[degree] = dict(
            pipeline=pipeline,
            train_r2=r2_score(y_train, y_train_pred),
            test_r2=r2_score(y_test, y_test_pred),
            train_rmse=np.sqrt(mean_squared_error(y_train, y_train_pred)),
            test_rmse=np.sqrt(mean_squared_error(y_test, y_test_pred)),
        )

    best_degree = max(fits, key=lambda d: fits[d]['test_r2'])
    if best_degree == 3 and fits[3]['test_r2'] - fits[2]['test_r2'] < 0.02:
        best_degree = 2
    return best_degree, fits


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_first_file = compute_first_file_area_per_root()

    df_features = pd.read_csv(FEATURE_TABLE_PATH)
    print(f"\nFeature table loaded: {len(df_features)} roots")

    df = df_features.merge(df_first_file, on='root_id', how='inner')
    print(f"Merged: {len(df)} roots have both feature-table data and a first-file area")

    # Clean: predictor and every target must be present; geometry/count targets must be positive.
    required_cols = [PREDICTOR] + TARGETS
    df_clean = df.dropna(subset=required_cols).copy()
    for col in POSITIVE_TARGETS:
        df_clean = df_clean[df_clean[col] > 0]
    df_clean = df_clean[df_clean[PREDICTOR] > 0]
    print(f"After cleaning (no NaNs, positive geometry): {len(df_clean)} roots")

    if len(df_clean) < 30:
        print("ERROR: too few roots with complete data to fit a meaningful model.")
        return 1

    print(f"\nPredictor range: {df_clean[PREDICTOR].min():.1f} - {df_clean[PREDICTOR].max():.1f} um2")

    X = df_clean[[PREDICTOR]].values

    if 'treatment' in df_clean.columns and df_clean['treatment'].notna().all() and df_clean['treatment'].nunique() > 1:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(sss.split(X, df_clean['treatment']))
    else:
        train_idx, test_idx = train_test_split(np.arange(len(df_clean)), test_size=0.25, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    print(f"Train: {len(X_train)} roots, Test: {len(X_test)} roots")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for target in TARGETS:
        y = df_clean[target].values
        y_train, y_test = y[train_idx], y[test_idx]

        best_degree, fits = fit_best_polynomial(X_train, X_test, y_train, y_test)
        best = fits[best_degree]

        cv_scores = cross_val_score(best['pipeline'], X_train, y_train, cv=kf, scoring='r2')

        coefs = best['pipeline'].named_steps['linear'].coef_
        intercept = best['pipeline'].named_steps['linear'].intercept_

        results[target] = dict(
            degree=best_degree,
            coefs=coefs.tolist(),
            intercept=float(intercept),
            train_r2=best['train_r2'],
            test_r2=best['test_r2'],
            train_rmse=best['train_rmse'],
            test_rmse=best['test_rmse'],
            cv_r2_mean=float(cv_scores.mean()),
            cv_r2_std=float(cv_scores.std()),
            y_test=y_test,
            y_test_pred=best['pipeline'].predict(X_test),
            all_degrees=fits,
        )

        print(f"\n{target} (degree {best_degree}):")
        print(f"  Test R^2:  {best['test_r2']:.4f}   Test RMSE: {best['test_rmse']:.4g}")
        print(f"  CV R^2:    {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # ---- performance summary CSV ----
    summary_rows = []
    for target in TARGETS:
        r = results[target]
        summary_rows.append(dict(
            Target=target, Degree=r['degree'],
            Train_R2=r['train_r2'], Test_R2=r['test_r2'],
            Train_RMSE=r['train_rmse'], Test_RMSE=r['test_rmse'],
            CV_R2_mean=r['cv_r2_mean'], CV_R2_std=r['cv_r2_std'],
            N_train=len(X_train), N_test=len(X_test),
        ))
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / 'model_performance_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nPerformance summary saved to: {summary_path}")

    # ---- coefficients CSV (human-readable) ----
    coef_rows = []
    for target in TARGETS:
        r = results[target]
        row = dict(Target=target, Degree=r['degree'], Intercept=r['intercept'])
        for i, c in enumerate(r['coefs']):
            row[f'Coef_x{i+1}'] = c
        coef_rows.append(row)
    coef_df = pd.DataFrame(coef_rows)
    coef_path = OUTPUT_DIR / 'model_coefficients.csv'
    coef_df.to_csv(coef_path, index=False)
    print(f"Coefficients saved to: {coef_path}")

    # ---- JSON for embedding in the HTML root model ----
    html_model = dict(
        predictor=PREDICTOR,
        predictor_label='Mean cell area of the innermost cortical file (first_file_avg_area_um2)',
        predictor_range=[float(df_clean[PREDICTOR].min()), float(df_clean[PREDICTOR].max())],
        n_roots_used=len(df_clean),
        targets={t: dict(degree=results[t]['degree'], coefs=results[t]['coefs'],
                          intercept=results[t]['intercept'],
                          test_r2=round(results[t]['test_r2'], 4),
                          test_rmse=round(results[t]['test_rmse'], 4))
                 for t in TARGETS}
    )
    json_path = OUTPUT_DIR / 'model_for_html.json'
    json_path.write_text(json.dumps(html_model, indent=2), encoding='utf-8')
    print(f"HTML model JSON saved to: {json_path}")

    # ---- clean data used (for generate_root_model.py to attach firstFileArea per root) ----
    keep_cols = ['root_id', 'root_identifier', PREDICTOR, 'n_quadrants_first_file',
                 'species', 'population', 'treatment'] + TARGETS
    keep_cols = [c for c in keep_cols if c in df_clean.columns]
    clean_path = OUTPUT_DIR / 'model_data_clean.csv'
    df_clean[keep_cols].to_csv(clean_path, index=False)
    print(f"Clean per-root data saved to: {clean_path}")

    # ---- figure: observed vs predicted for all 9 targets ----
    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    fig.suptitle('Predicting Full Root Anatomy from First-Cortical-File Cell Area\n'
                 f'(n={len(df_clean)} roots, predictor: mean area of cell_file_derivative == 0)',
                 fontsize=15, fontweight='bold')
    for ax, target in zip(axes.flat, TARGETS):
        r = results[target]
        ax.scatter(r['y_test'], r['y_test_pred'], alpha=0.4, s=14, color='steelblue')
        lo = min(r['y_test'].min(), r['y_test_pred'].min())
        hi = max(r['y_test'].max(), r['y_test_pred'].max())
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
        ax.set_title(f"{target}\nDegree {r['degree']}, Test R^2={r['test_r2']:.3f}", fontsize=11)
        ax.set_xlabel('Observed', fontsize=9)
        ax.set_ylabel('Predicted', fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / 'predictive_model_results_stele.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")

    print(f"\n\nSUMMARY\n")
    print(summary_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
