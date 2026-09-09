# Searches for the smallest set of measurements that predicts root structure.
#
# Exhaustive over combinations of the cheap measurements, as a follow-up to
# predictive_model_stele_area. Geometry columns are excluded from the
# candidates: predicting stele area from stele diameter is circular.
# whose single predictor (first-cortical-file mean cell area) explained stele
# area and root radius reasonably (R^2 ~0.6-0.7) but explained essentially none
# of the six spline shape features or the cell-file count (R^2 ~0-0.08).
#
# This script computes several other "cheap" per-root candidate measurements
# (last-file area, middle-file area, first-to-last ratio, overall mean cell
# area, total cell count) from the same cell_assignments.csv files, adds a few
# already-in-feature_table.csv candidates (stele area, root radius, cell
# count), and fits the identical polynomial-regression pipeline against all
# 9 targets for every candidate (alone and in small combinations), to see
# which one actually carries shape information.

import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def extract_root_identifier(image_name):
    if not image_name:
        return image_name
    for prefix in ('BL_', 'BR_', 'TL_', 'TR_'):
        if image_name.startswith(prefix):
            return image_name[len(prefix):]
    return image_name


SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FEATURE_TABLE_PATH = PROJECT_ROOT / 'results' / 'feature_table.csv'
CELL_FILE_DIR = PROJECT_ROOT / 'results' / 'cell_file' / 'cell_file_counting'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'predictor_search'
PER_ROOT_CACHE_PATH = OUTPUT_DIR / 'per_root_candidates_cache.csv'

# The 6 candidate variables the exhaustive search draws combinations from,
# all computed from cell_assignments.csv (per-quadrant file-area stats) or
# feature_table.csv's cell count -- deliberately excludes geometry
# (stele_area_um2, root_radius_um, stele_diameter_um) so a target's "best
# combo" can never lean on another geometry measurement instead of a genuine
# relationship; geometry is only ever a target here, never an input.
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
COUNT_TARGET = ['n_files']
SIZE_TARGETS = ['stele_area_um2', 'root_radius_um']
ALL_TARGETS = SIZE_TARGETS + COUNT_TARGET + SHAPE_TARGETS
POSITIVE_TARGETS = {'stele_area_um2', 'root_radius_um', 'n_files'}


def compute_per_root_candidates(force_rescan=False):
    # One pass over cell_assignments.csv computing several candidate
    # per-quadrant measurements, averaged to root level (same aggregation
    # build_feature_table.py uses for every other feature). Cached to disk
    # since the exhaustive combination search reuses this unchanged.
    if not force_rescan and PER_ROOT_CACHE_PATH.exists():
        print(f"Loading cached per-root candidates from {PER_ROOT_CACHE_PATH}")
        return pd.read_csv(PER_ROOT_CACHE_PATH)

    cell_files = list(CELL_FILE_DIR.rglob('cell_assignments.csv'))
    print(f"Found {len(cell_files)} cell_assignments.csv files")

    rows = []
    for cf in cell_files:
        image_name = cf.parent.name
        try:
            df = pd.read_csv(cf)
        except Exception:
            continue
        if 'cell_file_derivative' not in df.columns or 'area_um2' not in df.columns:
            continue
        df = df[df['area_um2'] > 0]
        df = df[df['cell_file_derivative'] >= 0]
        if len(df) < 5:
            continue

        file_means = df.groupby('cell_file_derivative')['area_um2'].mean().sort_index()
        if len(file_means) < 3:
            continue

        first = float(file_means.iloc[0])
        last = float(file_means.iloc[-1])
        mid_idx = len(file_means) // 2
        middle = float(file_means.iloc[mid_idx])

        rows.append(dict(
            image_name=image_name,
            first_file_avg_area_um2=first,
            last_file_avg_area_um2=last,
            middle_file_avg_area_um2=middle,
            first_to_last_ratio=(first / last) if last > 0 else np.nan,
            overall_avg_cell_area_um2=float(df['area_um2'].mean()),
            total_cells_quadrant=int(len(df)),
        ))

    per_quad = pd.DataFrame(rows)
    print(f"Computed candidates for {len(per_quad)} quadrant images")
    per_quad['root_id'] = per_quad['image_name'].apply(extract_root_identifier)
    numeric_cols = [c for c in per_quad.columns if c not in ('image_name', 'root_id')]
    per_root = per_quad.groupby('root_id')[numeric_cols].mean().reset_index()
    print(f"Aggregated to {len(per_root)} roots")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_root.to_csv(PER_ROOT_CACHE_PATH, index=False)
    print(f"Cached per-root candidates to {PER_ROOT_CACHE_PATH}")
    return per_root


def evaluate_predictor(df, predictor_cols, targets, train_idx, test_idx, max_degree):
    # Best test R^2 (over degrees 1..max_degree) per target for this predictor set.
    X = df[predictor_cols].values
    X_train, X_test = X[train_idx], X[test_idx]
    out = {}
    for target in targets:
        if target in predictor_cols:
            out[target] = np.nan
            continue
        y = df[target].values
        y_train, y_test = y[train_idx], y[test_idx]
        best_r2 = -np.inf
        for d in range(1, max_degree + 1):
            pipe = Pipeline([('poly', PolynomialFeatures(degree=d, include_bias=False)),
                              ('lin', LinearRegression())])
            pipe.fit(X_train, y_train)
            r2 = r2_score(y_test, pipe.predict(X_test))
            best_r2 = max(best_r2, r2)
        out[target] = best_r2
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_root = compute_per_root_candidates()
    feat = pd.read_csv(FEATURE_TABLE_PATH)
    print(f"Feature table: {len(feat)} roots")

    df = feat.merge(per_root, on='root_id', how='inner')
    print(f"Merged: {len(df)} roots")

    # Exhaustive: every combination of size 1..MAX_COMBO_SIZE drawn from the
    # 6 candidate variables (6 singles + 15 pairs + 20 triples = 41 sets),
    # rather than a hand-picked subset.
    candidate_sets = {}
    for size in range(1, MAX_COMBO_SIZE + 1):
        for combo in itertools.combinations(CANDIDATE_VARS, size):
            candidate_sets[' + '.join(combo)] = list(combo)
    print(f"Testing {len(candidate_sets)} candidate predictor sets "
          f"(sizes 1-{MAX_COMBO_SIZE} from {len(CANDIDATE_VARS)} variables)")

    required = set(ALL_TARGETS) | set(CANDIDATE_VARS)
    df_clean = df.dropna(subset=list(required)).copy()
    for col in POSITIVE_TARGETS:
        df_clean = df_clean[df_clean[col] > 0]
    print(f"Clean rows for search: {len(df_clean)}")

    train_idx, test_idx = train_test_split(np.arange(len(df_clean)), test_size=0.25, random_state=42)
    print(f"Train: {len(train_idx)}  Test: {len(test_idx)}")

    summary = []
    for name, cols in candidate_sets.items():
        max_degree = 3 if len(cols) == 1 else 2
        res = evaluate_predictor(df_clean, cols, ALL_TARGETS, train_idx, test_idx, max_degree)
        row = {'predictor_set': name, 'n_vars': len(cols)}
        row.update({f'{t}_R2': round(res[t], 4) if not np.isnan(res[t]) else np.nan for t in ALL_TARGETS})
        row['shape_mean_R2'] = round(np.nanmean([res[t] for t in SHAPE_TARGETS]), 4)
        row['count_R2'] = round(res['n_files'], 4) if not np.isnan(res['n_files']) else np.nan
        row['size_mean_R2'] = round(np.nanmean([res[t] for t in SIZE_TARGETS]), 4)
        summary.append(row)

    summary_df = pd.DataFrame(summary).sort_values('shape_mean_R2', ascending=False)
    out_path = OUTPUT_DIR / 'predictor_search_results.csv'
    summary_df.to_csv(out_path, index=False)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    cols = ['predictor_set', 'n_vars', 'shape_mean_R2', 'count_R2', 'size_mean_R2']

    print(f"\n\nTOP 25 BY MEAN SHAPE-FEATURE R^2 (test set)\n")
    print(summary_df[cols].head(25).to_string(index=False))

    print(f"\n\nTOP 10 BY n_files R^2 (count target)\n")
    print(summary_df.sort_values('count_R2', ascending=False)[cols].head(10).to_string(index=False))

    print(f"\n\nTOP 10 BY SIZE (stele area + root radius) R^2\n")
    print(summary_df.sort_values('size_mean_R2', ascending=False)[cols].head(10).to_string(index=False))

    print(f"\nAll {len(summary_df)} combinations, full per-target breakdown saved to: {out_path}")


if __name__ == "__main__":
    main()
