# Fits each shape feature from its own best-scoring predictor combination.
#
# Where predictive_model_stele_area uses one fixed predictor for everything,
# this gives every target the combination that actually suits it. Exports the
# fitted models for the reconstruction page to use.
# reconstruction mode in root_model.html.
#
# find_best_shape_predictor.py already identified which combination best
# predicts each target -- but that search only recorded test R^2 scores, not
# fitted coefficients. This script re-runs the same combination search and
# refits the winning combination for each target on the full cleaned dataset,
# exporting the coefficients as JSON.
#
# The candidate INPUT pool is deliberately restricted to 6 "cheap" per-root
# measurements -- first/last/middle cortical-file area, their ratio, overall
# mean cell area, and total cell count -- none of which require the expensive
# per-file spline-fitting step. Geometry (stele_area_um2, root_radius_um) is
# never used as an input, only ever as a target, so no target's "best combo"
# can lean on another geometry measurement instead of a genuine relationship.
#
# Shape targets are the radius_* features (cell area as a function of
# normalized radial position), matching find_best_shape_predictor.py.
#
# Polynomial terms are exported as (powers, coef) pairs -- one exponent per
# input variable, from sklearn's PolynomialFeatures.powers_ -- so the HTML can
# evaluate any degree/dimensionality generically: y = intercept +
# sum(coef * prod(input_j ** power_j)).

import itertools
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FEATURE_TABLE_PATH = PROJECT_ROOT / 'results' / 'feature_table.csv'
PER_ROOT_CACHE_PATH = PROJECT_ROOT / 'results' / 'predictor_search' / 'per_root_candidates_cache.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'predictor_search'

CANDIDATE_VARS = [
    'first_file_avg_area_um2',
    'last_file_avg_area_um2',
    'middle_file_avg_area_um2',
    'first_to_last_ratio',
    'overall_avg_cell_area_um2',
    'n_cells',
]
MAX_COMBO_SIZE = 3

SHAPE_TARGETS = ['radius_peak_height', 'radius_peak_position', 'radius_rise_slope',
                  'radius_decay_slope', 'radius_minima_position', 'radius_outer_rise_magnitude']
GEOMETRY_TARGETS = ['stele_area_um2', 'root_radius_um', 'n_files']
ALL_TARGETS = GEOMETRY_TARGETS + SHAPE_TARGETS
POSITIVE_TARGETS = {'stele_area_um2', 'root_radius_um', 'n_files'}


def evaluate_combo(X_train, X_test, y_train, y_test, max_degree):
    # Best-of-degrees test R^2 for one predictor set against one target.
    best = None
    for d in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=d, include_bias=False)
        Xp_train = poly.fit_transform(X_train)
        Xp_test = poly.transform(X_test)
        lin = LinearRegression().fit(Xp_train, y_train)
        r2 = r2_score(y_test, lin.predict(Xp_test))
        if best is None or r2 > best['r2']:
            best = dict(degree=d, r2=r2)
    return best


def main():
    if not PER_ROOT_CACHE_PATH.exists():
        print(f"Missing {PER_ROOT_CACHE_PATH} -- run find_best_shape_predictor.py first "
              "to build the per-root candidate cache.")
        return 1

    per_root = pd.read_csv(PER_ROOT_CACHE_PATH)
    feat = pd.read_csv(FEATURE_TABLE_PATH)
    df = feat.merge(per_root, on='root_id', how='inner')
    print(f"Merged: {len(df)} roots")

    required = set(ALL_TARGETS) | set(CANDIDATE_VARS)
    df_clean = df.dropna(subset=list(required)).copy()
    for col in POSITIVE_TARGETS:
        df_clean = df_clean[df_clean[col] > 0]
    print(f"Clean rows: {len(df_clean)}")

    train_idx, test_idx = train_test_split(np.arange(len(df_clean)), test_size=0.25, random_state=42)

    combos = []
    for size in range(1, MAX_COMBO_SIZE + 1):
        combos.extend(itertools.combinations(CANDIDATE_VARS, size))
    print(f"Searching {len(combos)} combinations per target ({len(ALL_TARGETS)} targets)")

    models = {}
    for target in ALL_TARGETS:
        y = df_clean[target].values
        y_train, y_test = y[train_idx], y[test_idx]

        best_r2 = -np.inf
        best_cols, best_degree = None, None
        for combo in combos:
            if target in combo:
                continue
            cols = list(combo)
            X = df_clean[cols].values
            X_train, X_test = X[train_idx], X[test_idx]
            max_degree = 3 if len(cols) == 1 else 2
            result = evaluate_combo(X_train, X_test, y_train, y_test, max_degree)
            if result['r2'] > best_r2:
                best_r2 = result['r2']
                best_cols = cols
                best_degree = result['degree']

        # Refit the winning combo/degree on ALL clean data for production coefficients.
        X_all = df_clean[best_cols].values
        poly = PolynomialFeatures(degree=best_degree, include_bias=False)
        Xp_all = poly.fit_transform(X_all)
        lin = LinearRegression().fit(Xp_all, y)
        y_pred_all = lin.predict(Xp_all)
        rmse_all = float(np.sqrt(mean_squared_error(y, y_pred_all)))

        terms = [dict(powers=[int(p) for p in powers], coef=float(c))
                 for powers, c in zip(poly.powers_, lin.coef_)]

        models[target] = dict(
            inputs=best_cols,
            degree=best_degree,
            intercept=float(lin.intercept_),
            terms=terms,
            test_r2=round(float(best_r2), 4),
            train_rmse=round(rmse_all, 4),
        )
        print(f"{target}: inputs={best_cols}  degree={best_degree}  test_r2={best_r2:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'best_combo_model_for_html.json'
    out_path.write_text(json.dumps(models, indent=2), encoding='utf-8')
    print(f"Saved: {out_path}")

    keep_cols = ['root_id'] + [c for c in CANDIDATE_VARS if c in df_clean.columns]
    data_path = OUTPUT_DIR / 'best_combo_data_clean.csv'
    df_clean[keep_cols].to_csv(data_path, index=False)
    print(f"Saved: {data_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
