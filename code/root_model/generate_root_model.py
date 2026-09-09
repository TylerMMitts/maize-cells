# Builds the interactive HTML model of a root cross-section.
#
# Driven by the spline shape features in feature_table.csv: the six features
# are inverted back into a curve, and that curve is laid out as concentric
# cell files. Writes results/root_model/root_model.html.
#
# The page reverse-engineers the feature-extraction pipeline: the six shape
# features define a normalized cell-area curve, which is converted back into an
# absolute area per cell file, then into a radial thickness and a cell count for
# each concentric ring between the stele and the root edge.

import json
import numpy as np
import pandas as pd
from pathlib import Path
from code.config import DATA_FOLDER

SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

FEATURE_TABLE_PATH = PROJECT_ROOT / "results" / "feature_table.csv"
MASTER_SUMMARY_PATH = PROJECT_ROOT / "results" / "master_summary.csv"
DETECTIONS_FOLDER = PROJECT_ROOT / "data" / "chosen_results" / "detections"
TEMPLATE_PATH = SCRIPT_DIR / "template.html"
OUTPUT_PATH = PROJECT_ROOT / "results" / "root_model" / "root_model.html"

# Output of code/statistics/predictive_model_stele_area.py: predicts a full
# root reconstruction from just the first-cortical-file mean cell area.
PREDICTIVE_MODEL_DIR = PROJECT_ROOT / "results" / "predictive_model_stele"
PREDICTIVE_MODEL_JSON = PREDICTIVE_MODEL_DIR / "model_for_html.json"
PREDICTIVE_MODEL_DATA = PREDICTIVE_MODEL_DIR / "model_data_clean.csv"

# Output of code/statistics/best_combo_model.py: predicts each target from its
# own individually best-scoring combination of 10 candidate measurements
# (rather than the single first-file area used by the model above).
BEST_COMBO_DIR = PROJECT_ROOT / "results" / "predictor_search"
BEST_COMBO_MODEL_JSON = BEST_COMBO_DIR / "best_combo_model_for_html.json"
BEST_COMBO_DATA = BEST_COMBO_DIR / "best_combo_data_clean.csv"

# Path from the generated HTML (results/root_model/root_model.html) to the
# folder holding each root's best-detection full image ({root_id}_full.jpg).
#
# Deliberately a relative URL string and not a config path: the browser
# resolves it against the page's own location, and it is embedded in the JSON
# payload. An absolute filesystem path would neither serialise nor load.
IMAGE_DIR_RELATIVE = '../../data/chosen_results/detections/'

# The six shape features, plus the geometry columns that set absolute scale.
SHAPE_FIELDS = [
    'peak_height',
    'peak_position',
    'rise_slope',
    'decay_slope',
    'minima_position',
    'outer_rise_magnitude',
]
GEOMETRY_FIELDS = ['n_files', 'stele_diameter_um', 'root_radius_um']

# Slider tracks span the full observed range of each feature, so no preset value
# can fall outside its own slider (an out-of-range value would be silently
# pinned to the bound and render a root that isn't the one selected). Bounds are
# snapped onto the step grid so a preset lands within half a step of the track.
SLIDER_META = {
    'peak_height':          dict(label='Peak height',      step=0.001, unit='',    group='shape'),
    'peak_position':        dict(label='Peak position',    step=0.001, unit='',    group='shape'),
    'rise_slope':           dict(label='Rise slope',       step=0.001, unit='',    group='shape'),
    'decay_slope':          dict(label='Decay slope',      step=0.001, unit='',    group='shape'),
    'minima_position':      dict(label='Minima position',  step=0.001, unit='',    group='shape'),
    'outer_rise_magnitude': dict(label='Outer rise',       step=0.001, unit='',    group='shape'),
    'n_files':              dict(label='Cell files',       step=1,     unit='',    group='geometry'),
    'stele_diameter_um':    dict(label='Stele diameter',   step=1,     unit=' um', group='geometry'),
    'root_radius_um':       dict(label='Root radius',      step=1,     unit=' um', group='geometry'),
    'min_file_area_um2':    dict(label='Smallest cell',    step=1,     unit=' um2', group='scale'),
    'max_file_area_um2':    dict(label='Largest cell',     step=1,     unit=' um2', group='scale'),
}


def snap_range(lo, hi, step):
    # Widen [lo, hi] onto the step grid so slider values land predictably.
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step
    decimals = max(0, int(round(-np.log10(step)))) if step < 1 else 0
    return round(float(lo), decimals + 1), round(float(hi), decimals + 1)


def per_file_area_bounds(master_summary_path):
    # Median smallest / largest per-file mean cell area across the dataset.
    #
    # The feature table only stores the *normalized* curve, so these supply the
    # absolute um^2 scale the normalization divided out.
    fallback = dict(min_lo=150.0, min_default=537.0, min_hi=900.0,
                    max_lo=400.0, max_default=1142.0, max_hi=2600.0)
    if not master_summary_path.exists():
        return fallback

    master = pd.read_csv(master_summary_path)
    if 'per_file_avg_areas_um2' not in master.columns:
        return fallback

    mins, maxs = [], []
    for raw in master['per_file_avg_areas_um2'].dropna().astype(str):
        vals = [float(v) for v in raw.split(',') if v.strip()]
        if len(vals) >= 5:
            mins.append(min(vals))
            maxs.append(max(vals))

    if not mins:
        return fallback

    mins, maxs = np.array(mins), np.array(maxs)
    return dict(
        min_lo=float(np.percentile(mins, 1)),
        min_default=float(np.median(mins)),
        min_hi=float(np.percentile(mins, 99)),
        max_lo=float(np.percentile(maxs, 1)),
        max_default=float(np.median(maxs)),
        max_hi=float(np.percentile(maxs, 99)),
    )


def build_ranges(df, area_bounds):
    ranges = {}
    for field in SHAPE_FIELDS + GEOMETRY_FIELDS:
        col = f'file_{field}' if field in SHAPE_FIELDS else field
        series = df[col].dropna()
        series = series[series > 0] if field == 'root_radius_um' else series
        step = SLIDER_META[field]['step']
        lo, hi = snap_range(float(series.min()), float(series.max()), step)
        if field in ('peak_position', 'minima_position'):
            lo, hi = 0.0, 1.0
        elif field in ('outer_rise_magnitude', 'rise_slope'):
            lo = min(lo, 0.0)
        elif field == 'decay_slope':
            hi = max(hi, 0.0)
        elif field == 'n_files':
            lo, hi = 2, max(30, int(hi))
        ranges[field] = dict(min=lo, max=hi, **SLIDER_META[field])

    for field, lo_key, hi_key in [('min_file_area_um2', 'min_lo', 'min_hi'),
                                  ('max_file_area_um2', 'max_lo', 'max_hi')]:
        step = SLIDER_META[field]['step']
        lo, hi = snap_range(area_bounds[lo_key], area_bounds[hi_key], step)
        ranges[field] = dict(min=lo, max=hi, **SLIDER_META[field])
    return ranges


def preset_from_frame(sub, label, group, area_bounds):
    # Median feature vector over a set of roots.
    values = {}
    for field in SHAPE_FIELDS:
        series = sub[f'file_{field}'].dropna()
        if series.empty:
            return None
        values[field] = round(float(series.median()), 4)
    for field in GEOMETRY_FIELDS:
        series = sub[field].dropna()
        series = series[series > 0] if field == 'root_radius_um' else series
        if series.empty:
            return None
        values[field] = round(float(series.median()), 1)
    values['n_files'] = int(round(values['n_files']))
    values['min_file_area_um2'] = round(area_bounds['min_default'], 1)
    values['max_file_area_um2'] = round(area_bounds['max_default'], 1)
    return dict(label=label, group=group, n=int(len(sub)), values=values)


def find_root_image(root_id, detections_folder):
    # The best-detection full image for a root (root1 = highest-confidence detection).
    if not detections_folder.exists():
        return None
    candidate = detections_folder / f"{root_id}_full.jpg"
    return candidate.name if candidate.exists() else None


def load_first_file_areas(predictive_model_data_path):
    # root_id -> first_file_avg_area_um2, from predictive_model_stele_area.py's
    # output. Lets a preset offer "predict this root from just its first-file
    # area" using the model trained there.
    if not predictive_model_data_path.exists():
        return {}
    df = pd.read_csv(predictive_model_data_path)
    if 'root_id' not in df.columns or 'first_file_avg_area_um2' not in df.columns:
        return {}
    return dict(zip(df['root_id'], df['first_file_avg_area_um2']))


# predictive_model_stele_area.py fits against feature_table.csv's file_*
# columns; best_combo_model.py fits against the radius_* columns instead. The
# HTML sliders use the short names in SHAPE_FIELDS (peak_height, ...), so both
# prefixes map to the same short names here. stele_area_um2 is intentionally
# left unmapped -- the HTML derives stele_diameter_um from the predicted area
# itself (a deterministic transform, same as build_feature_table.py), rather
# than fitting diameter separately.
PREDICTION_FIELD_MAP = {
    'file_peak_height': 'peak_height',
    'file_peak_position': 'peak_position',
    'file_rise_slope': 'rise_slope',
    'file_decay_slope': 'decay_slope',
    'file_minima_position': 'minima_position',
    'file_outer_rise_magnitude': 'outer_rise_magnitude',
    'radius_peak_height': 'peak_height',
    'radius_peak_position': 'peak_position',
    'radius_rise_slope': 'rise_slope',
    'radius_decay_slope': 'decay_slope',
    'radius_minima_position': 'minima_position',
    'radius_outer_rise_magnitude': 'outer_rise_magnitude',
}


def load_prediction_model(predictive_model_json_path):
    # The fitted regression (first-file area -> full root) as JSON, remapped
    # to the slider field names, or None if predictive_model_stele_area.py
    # hasn't been run yet.
    if not predictive_model_json_path.exists():
        return None
    raw = json.loads(predictive_model_json_path.read_text(encoding='utf-8'))
    raw['targets'] = {PREDICTION_FIELD_MAP.get(k, k): v for k, v in raw['targets'].items()}
    return raw


def load_best_combo_model(best_combo_model_json_path):
    # Per-target regression models (each fit on its own best-scoring
    # combination of the 10 candidate measurements) from best_combo_model.py,
    # remapped to the slider field names, or None if that script hasn't been
    # run yet.
    if not best_combo_model_json_path.exists():
        return None
    raw = json.loads(best_combo_model_json_path.read_text(encoding='utf-8'))
    return {PREDICTION_FIELD_MAP.get(k, k): v for k, v in raw.items()}


def load_best_combo_inputs(best_combo_data_path):
    # root_id -> {candidate_var: value} for every one of the 10 candidate
    # measurements best_combo_model.py drew its combinations from. Lets a
    # preset offer "predict from each feature's best combination" using the
    # models loaded above.
    if not best_combo_data_path.exists():
        return {}
    df = pd.read_csv(best_combo_data_path)
    if 'root_id' not in df.columns:
        return {}
    value_cols = [c for c in df.columns if c != 'root_id']
    return {
        row['root_id']: {c: float(row[c]) for c in value_cols if pd.notna(row[c])}
        for _, row in df.iterrows()
    }


def build_presets(df, area_bounds, detections_folder, first_file_areas=None, best_combo_inputs=None):
    first_file_areas = first_file_areas or {}
    best_combo_inputs = best_combo_inputs or {}
    presets = []

    whole = preset_from_frame(df, 'Dataset median', 'Summary', area_bounds)
    if whole:
        presets.append(whole)

    for column, group_label in [('treatment', 'By treatment'),
                                ('root_type', 'By root type'),
                                ('population', 'By population')]:
        if column not in df.columns:
            continue
        for value, sub in df.groupby(column):
            if len(sub) < 5 or str(value) in ('unknown', 'nan'):
                continue
            preset = preset_from_frame(sub, f'{value} (n={len(sub)})', group_label, area_bounds)
            if preset:
                presets.append(preset)

    # Individual roots: exact per-root feature vectors.
    needed = [f'file_{f}' for f in SHAPE_FIELDS]
    roots = df.dropna(subset=needed)
    n_with_prediction = 0
    n_with_best_combo = 0
    for _, row in roots.iterrows():
        values = {f: round(float(row[f'file_{f}']), 4) for f in SHAPE_FIELDS}
        n_files = row.get('n_files')
        stele = row.get('stele_diameter_um')
        radius = row.get('root_radius_um')
        values['n_files'] = int(round(n_files)) if pd.notna(n_files) and n_files >= 2 else 15
        values['stele_diameter_um'] = round(float(stele), 1) if pd.notna(stele) and stele > 0 else 1403.0
        values['root_radius_um'] = round(float(radius), 1) if pd.notna(radius) and radius > 0 else 1283.0
        values['min_file_area_um2'] = round(area_bounds['min_default'], 1)
        values['max_file_area_um2'] = round(area_bounds['max_default'], 1)

        preset = dict(label=str(row['root_id']), group='Individual roots', n=1, values=values)
        image = find_root_image(row['root_id'], detections_folder)
        if image:
            preset['image'] = image

        first_file_area = first_file_areas.get(row['root_id'])
        if first_file_area is not None and pd.notna(first_file_area):
            preset['firstFileArea'] = round(float(first_file_area), 1)
            n_with_prediction += 1

        combo_inputs = best_combo_inputs.get(row['root_id'])
        if combo_inputs:
            preset['bestCombo'] = {k: round(v, 4) for k, v in combo_inputs.items()}
            n_with_best_combo += 1

        presets.append(preset)

    print(f"  {n_with_prediction} individual-root presets carry a first-file area "
          f"(enables 'predict from first-file area' for those roots)")
    print(f"  {n_with_best_combo} individual-root presets carry best-combo inputs "
          f"(enables 'predict from best combination' for those roots)")
    return presets


def main():
    # python -m code.root_model.generate_root_model
    class cfg:
        feature_table = str(FEATURE_TABLE_PATH)
        master_summary = str(MASTER_SUMMARY_PATH)
        detections_folder = str(DETECTIONS_FOLDER)
        out = str(OUTPUT_PATH)
        predictive_model_json = str(PREDICTIVE_MODEL_JSON)
        predictive_model_data = str(PREDICTIVE_MODEL_DATA)
        best_combo_model_json = str(BEST_COMBO_MODEL_JSON)
        best_combo_data = str(BEST_COMBO_DATA)


    feature_table = Path(cfg.feature_table)
    if not feature_table.exists():
        print(f"Feature table not found: {feature_table}")
        return 1

    df = pd.read_csv(feature_table)
    print(f"Loaded {len(df)} roots from {feature_table}")

    prediction_model = load_prediction_model(Path(cfg.predictive_model_json))
    first_file_areas = load_first_file_areas(Path(cfg.predictive_model_data))
    if prediction_model:
        print(f"Loaded predictive model ({prediction_model['n_roots_used']} roots used to fit it) "
              f"from {cfg.predictive_model_json}")
    else:
        print(f"No predictive model found at {cfg.predictive_model_json} "
              "-- run code/statistics/predictive_model_stele_area.py first if you want the "
              "'predict from first-file area' feature. Continuing without it.")

    best_combo_model = load_best_combo_model(Path(cfg.best_combo_model_json))
    best_combo_inputs = load_best_combo_inputs(Path(cfg.best_combo_data))
    if best_combo_model:
        print(f"Loaded best-combo model from {cfg.best_combo_model_json}")
    else:
        print(f"No best-combo model found at {cfg.best_combo_model_json} "
              "-- run code/statistics/best_combo_model.py first if you want the "
              "'predict from best combination' feature. Continuing without it.")

    area_bounds = per_file_area_bounds(Path(cfg.master_summary))
    ranges = build_ranges(df, area_bounds)
    presets = build_presets(df, area_bounds, Path(cfg.detections_folder),
                             first_file_areas, best_combo_inputs)
    n_with_image = sum(1 for p in presets if p.get('image'))
    print(f"Built {len(presets)} presets "
          f"({sum(1 for p in presets if p['group'] == 'Individual roots')} individual roots, "
          f"{n_with_image} with a linked best-detection image)")

    payload = {
        'ranges': ranges,
        'presets': presets,
        'shapeFields': SHAPE_FIELDS,
        'predictionModel': prediction_model,
        'bestComboModel': best_combo_model,
        'sourceRoots': int(len(df)),
        'imageDir': IMAGE_DIR_RELATIVE,
    }

    template = TEMPLATE_PATH.read_text(encoding='utf-8')
    html = template.replace('"__MODEL_DATA__"', json.dumps(payload, separators=(',', ':')))

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding='utf-8')
    print(f"Wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
