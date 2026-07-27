# Root Cell Image Analysis Pipeline

This project extracts cell-level anatomical measurements from cross-sectional root images (maize) and runs a suite of statistical and modeling analyses on top of them. Raw microscope images go in; per-cell measurements, per-root feature tables, and dozens of plots/statistical models come out.

The pipeline is designed to run **incrementally**: every time you add new images, it only processes what's new and appends to the existing dataset rather than reprocessing everything from scratch (except for Step 7's global analyses, which always run over the complete dataset since they produce dataset-wide models/plots).

## Quick Start

Edit the configuration block at the top of `code/run_pipeline.py` (species, population, input folder, microscope scale), then run it from the `code/` directory:

```bash
python run_pipeline.py
```

This runs all 7 steps below in order, appends new data to `results/master_summary.csv` and `results/feature_table.csv`, and writes a timestamped full debug log to `results/full_debug_log_<timestamp>.txt`.

## Pipeline Architecture

The pipeline is orchestrated by `RootAnalysisPipeline` in `code/pipeline.py`. `run_pipeline.py` is the configuration entry point that instantiates it and calls `run_full_pipeline()`.

### Step 1 — Split Images into Quadrants
Runs a YOLO root-detection model (`code/yolo/test_root_model.py`) on each raw input image to locate the root, then crops it into four quadrants (`BL`, `BR`, `TL`, `TR` — bottom-left, bottom-right, top-left, top-right). Quadrant images are saved to `data/chosen_results/`. Only newly-created quadrants are tracked for the rest of the run.

### Step 2 — Extract Cell Measurements
Runs cell segmentation (YOLO) on each new quadrant image (`code/measurements/cell_measurements.py`), detecting individual cells and computing per-cell area, radius, position, and angle. Also identifies the stele (root core) via a separate exclusion model, computing stele area and root radius. Applies the micrometers-per-pixel scale conversion (see **Per-Image Scale Detection** below). Outputs one `_measurements.csv`, one `_centers.jpg`, and one `_image_summary.json` per quadrant image into `results/measurements_all/`.

### Step 2.5 — Quality Control (new images only)
Opens an interactive web page (`code/quality_control/qc_manager.py` + `qc_server.py`) showing only the newly-processed images, so bad segmentations can be manually deleted before they enter the permanent dataset. Can be run non-interactively with cell-count thresholds (`QC_AUTO_DELETE`), or skipped entirely (`SKIP_QC`).

### Step 3 — Update Master Summary
Appends new image entries to `results/master_summary.csv` (one row per quadrant). This is **append-only**: existing rows keep their original `population`/`species` values forever; only genuinely new images get the population/species passed into the pipeline for that run.

### Step 4 — Analyze Cell Files (incremental)
For each new quadrant, identifies concentric "cell files" (radial rings of cells from stele to cortex) using a derivative-based method (`code/cell_file/cell_file_analyzer.py`). Writes per-image `cell_assignments.csv` files to `results/cell_file/cell_file_counting/<image_name>/`, then updates `master_summary.csv`'s `file_count` column for those images.

### Step 5 — Calculate Neighbors
Computes each cell's spatial neighbors (adjacency) via mask expansion (`code/measurements/mask_neighbors.py`), used later for degree/density-based analyses. Only new images are processed unless `force_neighbor_recalc=True`.

### Step 6 — Update Feature Table (incremental)
Combines each root's quadrants into a single row in `results/feature_table.csv` (`code/measurements/build_feature_table.py`). For each root, fits a smoothing spline to cell size vs. normalized radius (and separately vs. cell-file number), extracting shape features: `peak_height`, `peak_position`, `rise_slope`, `decay_slope`, `outer_rise_magnitude`, `area_under_curve`, `avg_normalized_cell_size` — computed both the "radius" way and the "file number" way (columns prefixed `radius_`/`file_`). Also computes `stele_diameter_um`. Like Step 3, this is append-only and preserves existing rows.

### Step 7 — Run Analysis (always runs on the full dataset)
Runs 16 independent analyses over the complete, current dataset (not just new data) — this is the only step that isn't incremental, since these produce dataset-wide models and plots. A failure in one analysis is logged and does not stop the others from running. See below for what each one does.

## Step 7 Analyses

| # | Name | What it does | Output folder |
|---|------|---------------|----------------|
| 1 | Feature Analysis | Compares feature variability between the radius-based and file-based spline approaches | `results/feature_variations/` |
| 2 | Spline Visualization | Per-root spline plots + combined cubic fit across all roots; incrementally skips roots already plotted | `results/combined_cubic_fit/` |
| 3 | Neighbor Analysis | Traces "travel paths" from outer cortex to stele along shortest-neighbor-distance routes; plots neighbor count vs. radius/area | `results/neighbor_analysis/` |
| 4 | Root Radius Analysis | Regresses cell area against cell radius and root radius (centered interaction model) | `results/root_radius_analysis/` |
| 5 | Stele Area Analysis | Same regression, using stele area instead of root radius | `results/stele_area_analysis/` |
| 6 | Degree vs Area Analysis | Correlates neighbor count ("degree") with cell area/radius per image; incrementally skips already-plotted images | `results/degree_analysis/` |
| 7 | FPCA Analysis | Functional Principal Component Analysis on aligned root radial profiles | `results/fpca_results/` |
| 8 | File Level Model | Predicts cell area at a given cell-file position from the first-file area and file number | `results/file_level_model/` |
| 9 | Cortex Reconstruction | Reconstructs a full radial cortex profile from just the first cell-file's area, using the File Level Model's fit | `results/cortical_reconstruction/` |
| 10 | Cell File Group Analysis | Groups roots by cell-file count and fits/plots average and cubic best-fit size profiles per group; excludes images no longer in `master_summary.csv` | `results/cell_file/cell_file_profiles/` |
| 11 | Predictive Model Stele Area | Predicts stele area, file count, and average cell area from first-file cell area | `results/predictive_model_stele/` |
| 12 | Normalized Pattern Fit | Fits candidate curve shapes (cubic/sigmoid/quadratic/linear/exponential) to the normalized cell-size pattern | `results/non_tda/non_tda_results_huge/` |
| 13 | Stratified Analysis | Per-group (treatment/root_type/population) linear regressions predicting stele area, file count, average cell area | `results/stratified_analysis/` |
| 14 | Factor Analysis | ANOVA + Tukey HSD testing treatment/root_type/population effects on all outcomes | `results/factor_analysis/` |
| 15 | Interaction Plots | Treatment × Root Type interaction plots, faceted by population | `results/interaction_plots/` |
| 16 | Mixed Model Size Scaling | Linear mixed-effects model: `feature ~ n_files + stele_diameter_um + (1\|plant)`, per spline-shape feature | `results/mixed_model_size_scaling/` |
| 17 | Mixed Model Factor Contribution | Linear mixed-effects model: `feature ~ treatment + root_type + population + (1\|plant)`, with a likelihood-ratio test quantifying each factor's contribution | `results/mixed_model_factor_contribution/` |

(Table has 17 rows because "Step 7" is one pipeline step but runs 17 distinct analyses.)

## Per-Image Scale Detection

Images can be captured at different microscope zoom levels, so a single µm-per-pixel conversion factor isn't always correct for every image. `run_pipeline.py` supports an optional per-image scale lookup:

- `SCALE_IMAGE_KEY_CSV`: maps original image filename → `scale_id` (columns: `image_id`, `scale_id`)
- `SCALE_VALUES_CSV`: maps `scale_id` → pixels-per-micrometer (column: `um_px` — despite the name, this is pixels/µm, not µm/pixel; the pipeline inverts it automatically)
- Built via `code/util/file_utils.py::build_pixel_to_um_map()`, which returns both the scale map and a set of images whose scale couldn't be resolved (missing `scale_id`, or conflicting `scale_id` across duplicate rows)
- Images with an unresolvable scale are **excluded from the run entirely** in Step 1, rather than silently falling back to a possibly-wrong default
- Set `USE_SCALE_CSV = False` to disable this and use a single `UM_PER_PIXEL` value for every image

## Directory Structure

```
code/
  pipeline.py              RootAnalysisPipeline - orchestrates all 7 steps
  run_pipeline.py          Configuration entry point - edit and run this
  config.py                Paths, model weights, thresholds, default parameters
  measurements/            Cell measurement extraction, feature table, spline/neighbor/degree analyses
  cell_file/               Cell-file (radial ring) identification and profile analysis
  statistics/              FPCA, predictive models, stratified/factor/mixed-effects analyses
  non_tda/                 Normalized pattern fitting
  quality_control/         Interactive QC web UI
  yolo/                    YOLO model training/inference wrappers
  util/                    Shared helpers (see below) - the canonical implementations
data/
  <folders of raw input images>     e.g. data/01_LAT_images/
  chosen_results/                   quadrant-split images (Step 1 output)
models/                             YOLO/SAM model weights
results/
  measurements_all/                 per-quadrant _measurements.csv / _centers.jpg / _image_summary.json
  cell_file/cell_file_counting/     per-quadrant cell_assignments.csv
  master_summary.csv                one row per quadrant (Step 3)
  feature_table.csv                 one row per root (Step 6)
  <analysis_name>/                  one output folder per Step 7 analysis
  full_debug_log_<timestamp>.txt    complete stdout/stderr from a run_pipeline.py run
```

## Key Data Files

- **`master_summary.csv`** — one row per quadrant image. Columns include `image_name`, `quadrant`, `plant_number`, `root_number`, `treatment`, `root_type`, `population`, `species`, `file_count`, `stele_area_um2`, `root_radius_um`, `average_cell_area_um2`, `n_cells`. Append-only; never overwritten.
- **`feature_table.csv`** — one row per root (combining that root's quadrants). Adds `n_quadrants`, `stele_diameter_um`, and the radius-/file-based spline-shape features. Append-only.
- **`<image>_measurements.csv`** (in `measurements_all/`) — one row per detected cell: `cell_id`, `area_pixels`/`area_um2`, `radius_pixels`/`radius_um`, `x_pixels`/`y_pixels`, `angle_degrees`, `neighbors`, `degree`.
- **`cell_assignments.csv`** (in `cell_file_counting/<image>/`) — one row per cell with its assigned `cell_file_derivative` (ring number, 0 = innermost).

## Utility Modules (`code/util/`)

Shared logic used across the pipeline lives here — this is the canonical source; other files should import from here rather than redefining the same logic:

- **`file_utils.py`** — CSV/JSON loading, `parse_image_name` (the authoritative filename parser — see note below), `update_master_summary`, `build_pixel_to_um_map`, cleanup utilities
- **`data_utils.py`** — DataFrame-level helpers: binning, outlier filtering, root-ID combination
- **`image_utils.py`** — pixel-mask geometry: stele area, root radius, area/perimeter/centroid from a mask, coordinate conversions
- **`angle_utils.py`** — quadrant angle-range logic, angle normalization, circular-aware angle filtering
- **`stats_utils.py`** — spline feature extraction, coefficient of variation, regression helpers

**Note on `parse_image_name`:** filenames encode `quadrant`, `plot_number`, `plant_number`, `root_number`, `treatment` (`WW`/`WS`/`MCS`), and `root_type` (`W2`/`W3`/`W4`) in a somewhat ad-hoc format that has evolved across image batches (e.g. newer batches use a `plant.root` decimal format like `_2.1_` instead of separate `P{n}` / `root{n}` markers). `file_utils.py`'s implementation is the correct, currently-used one.

## Known Data Gaps / Caveats

- **`treatment` is not recorded for the MAGIC_MAIZE image batch** — those filenames don't encode a treatment code the way earlier batches do, so any analysis conditioning on both `treatment` and `population` together will only see the non-MAGIC_MAIZE data. This isn't a bug; the information simply isn't in those filenames.
- **Mixed-effects models group by `population + "_" + plant_number`, not raw `plant_number`** — `plant_number` values are only unique *within* a population (both populations have a "plant 1"), so the composite key prevents unrelated plants from different populations being pooled into the same random-effect group.
- **Some mixed-model fits fail with "singular covariance structure"** — a real numerical instability from small/imbalanced random-effect groups (some plants have only 1-2 roots), not a bug; these are caught and logged per-feature rather than crashing the run.
- Several Step 7 scripts cache their expensive per-image outputs (Spline Visualization, Degree vs Area Analysis, Cell File Group Analysis) — they check what's already on disk and only regenerate plots for images that don't have one yet, which is what keeps Step 7 fast as the dataset grows into the thousands of images.
