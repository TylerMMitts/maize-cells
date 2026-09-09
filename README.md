# Root cortical cell pattern

Measures the size and arrangement of every cortical cell in root cross-sections,
across thousands of images and several species, and fits a small set of shape
features to the pattern those cells form. Those features are then used to model,
reconstruct and 3D-print roots that differ only in how cell size is distributed.

## What is here

```
code/        all source, nothing else
data/        input imagery and datasets (not in the repository)
models/      trained weights and YOLO datasets (not in the repository)
results/     everything any script produces
```

Two rules the layout depends on: `code/` holds only source, and nothing is
written next to the code. Every location comes from `code/config.py`, so a run
behaves the same whatever directory it is started from.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10. `shapely` needs GEOS 3.13 or newer, because the STL export uses
`constrained_delaunay_triangles`; an older GEOS will fail there and nowhere else.
Install the `torch` build that matches your CUDA version from pytorch.org, or a
CPU-only build if you are only running inference.

### Where the weights go

Weights are not in the repository. Place them like this:

```
models/
  runs/
    detect/runs/root_detection/root_detector/weights/best.pt
    segment/runs/
      robust_segmentation/robust_cell_detector/weights/best.pt
      robust_segmentation/robust_cell_detector_lowres/weights/best.pt
      exclusion_model/inner_part_detector/weights/best.pt
      exclusion_model/inner_part_detector_lowres/weights/best.pt
  mobile_sam.pt
```

`config.py` reads exactly these paths. If you keep weights elsewhere, change the
five constants there rather than editing individual scripts.

## The models

| Model | Base | Job |
|---|---|---|
| `root_detector` | yolov8l | Finds root cross-sections in a slide image so they can be cropped and quartered |
| `robust_cell_detector` | yolov8n-seg | Instance-segments every cortical cell |
| `inner_part_detector` | yolov8l-seg | Marks which part of the image is valid cortex, so stele and background are excluded |
| `robust_cell_detector_lowres` | yolov8n-seg | The cell segmenter retrained at 0.70 scale, for the lower-resolution WIDIV imagery |
| `inner_part_detector_lowres` | yolov8l-seg | The exclusion model, same retraining |

The cell segmenter deliberately uses the smallest backbone of the three. Cells
are small, numerous and repetitive, and the nano model at high confidence gave
cleaner masks and far faster inference over thousands of quadrants than a larger
one did.

## Running the pipeline

```bash
python -m code.run_pipeline
```

That runs every stage in order. Each stage is append-only: re-running processes
only what is new, and existing rows keep the species and population already
recorded against them.

| Step | What happens |
|---|---|
| 1 | Detect roots in each slide, crop, split into four quadrants |
| 2 | Segment every cortical cell; exclude anything outside the cortex |
| 2.5 | Quality control, in a browser, over the new images only |
| 3 | Append to `results/master_summary.csv` |
| 4 | Assign cells to concentric cell files |
| 5 | Calculate neighbour relationships |
| 6 | Append to `results/feature_table.csv` |
| 7 | Run the sixteen analysis modules listed below |

Step 7 runs these automatically, so they do not normally need running by hand:
`analyze_features`, `visualize_splines`, `neighbor_analysis`,
`root_radius_analysis`, `stele_area_analysis`, `analyze_degree_vs_area`,
`run_fpca`, `cell_area_per_file`, `cortex_reconstruction`,
`cell_file_group_analysis`, `predictive_model_stele_area`,
`stratified_analysis`, `factor_analysis`, `mixed_model_size_scaling`,
`mixed_model_factor_contribution`, `interaction_plots`.

## What each script answers

### Entry points

| Script | Question |
|---|---|
| `code.run_pipeline` | Process a folder of new slides end to end |
| `code.run_cleanup` | Which master-summary rows have lost their measurement files? |
| `code.run_pipeline_widiv` | Same as run_pipeline, but with the low-resolution WIDIV models |
| `code.run_widiv_segmentation` | Segmentation only, for WIDIV, without the rest of the pipeline |
| `code.main` | Scratch entry point for running individual stages by hand |

### Statistics

| Script | Question |
|---|---|
| `code.statistics.cells_per_cell_file` | How many cells are in each cell file, and how does that change outward? |
| `code.statistics.ring_packing_analysis` | Does a longer cell-file ring hold bigger cells, or just more of them? |
| `code.statistics.spacing_vs_radius` | How does the gap between neighbouring cells change with radius? |
| `code.statistics.feature_space_limits` | Which combinations of shape features are mathematically impossible, and how many real roots land there? |
| `code.statistics.find_best_shape_predictor` | Which small set of cheap measurements best predicts each shape feature? |
| `code.statistics.best_combo_model` | Fit each shape feature from its own best-scoring combination |
| `code.statistics.predictive_model_stele_area` | Can stele area be predicted without measuring it? |
| `code.statistics.mixed_model_size_scaling` | Is the pattern just a side effect of root size? |
| `code.statistics.mixed_model_factor_contribution` | How much do treatment, genotype and root type each explain? |
| `code.statistics.stratified_analysis` | Does the pattern hold within every subgroup? |
| `code.statistics.factor_analysis` | Which factors move which features? |
| `code.statistics.interaction_plots` | Do those factors interact? |

### Modelling and printing

| Script | Question |
|---|---|
| `code.root_model.generate_root_model` | Build the interactive reconstruction page |
| `code.root_model.root_to_fea_stl` | Turn one geometry export into a watertight wall-solid STL |
| `code.root_model.build_pattern_stls` | Build five printable roots that differ only in cell-size distribution |
| `code.root_model.build_web_bundle` | Package the model plus a curated image set for a public site |

### Other

| Script | Question |
|---|---|
| `code.age_experiment.pattern_development` | At what developmental stage does the pattern appear? |
| `code.util.rename_widiv_images` | Rename WIDIV images into the naming convention the pipeline expects |
| `code.yolo.train_robust` | Retrain the cell segmenter |
| `code.yolo.train_exclusive` | Retrain the exclusion model |
| `code.yolo.train_root_model` | Retrain the root detector |
| `code.yolo.train_robust_lowres` | Retrain the cell segmenter at reduced scale |
| `code.yolo.train_exclusion_lowres` | Retrain the exclusion model at reduced scale |

## Training output

A training run produces two different kinds of thing, and they are kept apart:

- **weights** stay under `models/`, because they are an input to every later run
- **figures, curves, previews and logs** go to `results/training/<model_name>/`,
  because they are a result you look at once

`code/yolo/training_output.py` does the split at the end of each training run.
It also renames checkpoints to carry the model name — `<model>_best.pt`,
`<model>_epoch_<N>.pt` — since a loose `best.pt` is unidentifiable as soon as it
is copied anywhere.

Weights trained before that change are still plain `best.pt`, and are left
alone. `config.py` looks for the named form first and falls back to the old one,
and `find_latest_checkpoint` understands both, so nothing has to be renamed by
hand.

## What must run before what

This is the part that is invisible from the filenames and is usually the first
thing to fail. Each of these reads a folder that another script writes.

```
run_pipeline
  ├─ writes results/master_summary.csv, results/feature_table.csv,
  │         results/cell_file/cell_file_counting/
  │
  ├─ cells_per_cell_file          needs cell_file_counting/
  │     └─ ring_packing_analysis  needs results/cells_per_file/cells_per_file_long.csv
  │     └─ spacing_vs_radius      needs the same file
  │
  ├─ feature_space_limits         needs feature_table.csv
  │
  └─ find_best_shape_predictor    needs feature_table.csv
        └─ best_combo_model       needs feature_table.csv
              └─ generate_root_model    needs best_combo_model_for_html.json
                    ├─ build_pattern_stls   needs results/root_model/root_model.html
                    └─ build_web_bundle     needs the same file
```

For WIDIV specifically, `rename_widiv_images` must run before
`run_pipeline_widiv`, because the pipeline parses metadata out of the filename.

`root_to_fea_stl` takes a geometry JSON exported from the reconstruction page,
so that page has to exist and have been used first. `build_pattern_stls` skips
that by generating the geometry itself.

## Configuration

There is no argument parsing anywhere. Every runnable script keeps its settings
in a single `class cfg:` block at the top of `main()`, with a comment naming the
command that runs it:

```python
def main():
    # python -m code.statistics.spacing_vs_radius
    class cfg:
        long_csv = str(LONG_CSV)
        out_dir = str(OUT_DIR)
```

To change what a script does, edit that block. Paths in it come from
`code/config.py`, which owns every location in the project and derives them all
from `PROJECT_ROOT`. It also provides:

- `resolve_input(path, description)` — an input that must exist, raising with the
  list of paths it tried rather than failing later as an empty result
- `resolve_output(path)` — a relative path lands under the project root, not the
  working directory
- `find_latest_checkpoint(folder, model_name)` — the newest checkpoint by epoch
  number, comparing numerically so `epoch_100` beats `epoch_9`, and falling back
  to the older bare `epoch100.pt` naming so existing weights still load

## Every file in the project

The pipeline above is the main path through the project. This is the rest of
it, one line each. Every file's own header says the same thing in more detail.

### code/ — entry points and the pipeline

| File | What it is |
|---|---|
| `__init__.py` | Package marker for the project source. |
| `config.py` | Every path and tuning parameter the project uses, in one place. |
| `main.py` | Scratch entry point for running individual pipeline stages by hand. |
| `pipeline.py` | The pipeline itself: every stage, in order, as one class. |
| `run_cleanup.py` | Entry point for reconciling master_summary.csv against the files on disk. |
| `run_pipeline.py` | Entry point for the full root analysis pipeline. |
| `run_pipeline_widiv.py` | Run the full pipeline over the widiv dataset using the low-resolution models. |
| `run_widiv_segmentation.py` | Segmentation-only run over the widiv dataset. |

### code/measurements/ — turning segmented images into numbers

| File | What it is |
|---|---|
| `__init__.py` | Package marker for the measurement stages. |
| `analyze_degree_vs_area.py` | Does a cell with more neighbours tend to be larger? |
| `analyze_features.py` | Compares the two bases the shape features can be computed in. |
| `attach_tomato_metadata.py` | Fills in master_summary metadata for the hand-annotated tomato set. |
| `build_feature_table.py` | Builds feature_table.csv, one row per root. |
| `cell_measurements.py` | Measures every segmented cell. |
| `coco_to_measurements.py` | Converts hand-annotated COCO polygons into the pipeline's measurement CSVs. |
| `create_master_summary_only.py` | Rebuilds master_summary.csv from the measurement files. |
| `density_analysis.py` | Local cell density and the air pockets between cells. |
| `mask_neighbors.py` | Works out which cells touch which, from the masks. |
| `neighbor_analysis.py` | Analyses the neighbour graph across the whole dataset. |
| `root_radius_analysis.py` | Does the pattern change with overall root size? |
| `stele_area_analysis.py` | The same question, against stele area rather than root radius. |
| `visualize_splines.py` | Fits the spline to each root and extracts the six shape features. |

### code/cell_file/ — concentric cell files

| File | What it is |
|---|---|
| `cell_file_analyzer.py` | Assigns each cell to a concentric cell file. |
| `cell_file_group_analysis.py` | Cell-file profiles pooled across images and groups. |
| `cell_file_regression.py` | Regressions of cell size against cell file number. |

### code/statistics/ — statistical analyses

| File | What it is |
|---|---|
| `best_combo_model.py` | Fits each shape feature from its own best-scoring predictor combination. |
| `cell_area_per_file.py` | Mean cell area per cell file, as a table. |
| `cells_per_cell_file.py` | How many cells sit in each cell file, across the whole dataset. |
| `cortex_reconstruction.py` | Rebuilds a cortex from fitted features and checks it back. |
| `factor_analysis.py` | Which experimental factors move which shape features? |
| `feature_space_limits.py` | Where the spline feature space stops being physically realisable. |
| `find_best_shape_predictor.py` | Searches for the smallest set of measurements that predicts root structure. |
| `interaction_plots.py` | Do those factors interact, or act independently? |
| `mixed_model_factor_contribution.py` | How much of each feature does each factor explain? |
| `mixed_model_size_scaling.py` | Is the pattern just a side effect of root size? |
| `predictive_model.py` | Predicts shape features from cheap measurements. |
| `predictive_model_stele_area.py` | Predicts a root's full anatomy from a single cheap measurement. |
| `ring_packing_analysis.py` | Does a longer cell-file ring hold more cells, or bigger ones? |
| `run_fpca.py` | Functional PCA over the fitted profiles. |
| `spacing_vs_radius.py` | How circumferential cell spacing changes as you move outward through the root. |
| `stratified_analysis.py` | Does the pattern hold inside every subgroup? |

### code/non_tda/ — the pattern as a curve

| File | What it is |
|---|---|
| `__init__.py` | Package marker for the non-topological pattern analyses. |
| `fit_normalized_pattern.py` | Fits one curve to the normalised pattern. |
| `non_tda_analysis.py` | The pooled cell-size profile across all images. |
| `non_tda_analysis_clustered.py` | The same profiles, split by cluster rather than pooled. |

### code/tda/ — topological analyses

| File | What it is |
|---|---|
| `PCA.py` | PCA over the persistence-image vectors. |
| `__init__.py` | Package marker for the topological analyses. |
| `level_set_analysis.py` | Persistence of the density field as a threshold sweeps. |
| `mapper_analysis.py` | Mapper graphs of the cell population. |
| `mds.py` | Multidimensional scaling of the Wasserstein distances. |
| `persistence_image.py` | Turns a persistence diagram into a fixed-length vector. |
| `tda_analysis.py` | Persistence diagrams for each image, and the statistics over them. |
| `utils.py` | Graph helpers shared by the topological analyses. |
| `wasserstein_distance.py` | Wasserstein distance between two persistence diagrams. |

### code/yolo/ — model training and inference

| File | What it is |
|---|---|
| `__init__.py` | Package marker for model training and inference. |
| `test_innerpart.py` | Visual check of the exclusion model on one image. |
| `test_morph_postprocess.py` | Sweeps the morphological kernel sizes. |
| `test_robust.py` | Runs the cell segmenter and applies the full post-process. |
| `test_root_model.py` | Runs the root detector and splits each root into quadrants. |
| `train_exclusion_lowres.py` | Trains the exclusion model on downscaled images. |
| `train_exclusive.py` | Trains the exclusion model that marks valid cortex. |
| `train_robust.py` | Trains the cell segmentation model. |
| `train_robust_lowres.py` | Trains the cell segmenter on downscaled images. |
| `train_root_model.py` | Trains the root detector that finds roots in a slide. |
| `training_output.py` | Separates what a training run produces into weights and everything else. |

### code/util/ — shared helpers

| File | What it is |
|---|---|
| `__init__.py` | Re-exports the helpers the rest of the project reaches for most often. |
| `angle_utils.py` | Angle handling for quadrant images. |
| `convert_coco.py` | Entry point for the COCO annotation converter. |
| `data_utils.py` | Normalisation and binning shared across analyses. |
| `file_utils.py` | Filename parsing, loading, and the master summary. |
| `image_utils.py` | Geometry read off the image itself. |
| `lightweight_sam_refiner.py` | Optional mask refinement with a small SAM model. |
| `rename_widiv_images.py` | Rename the widiv slide images into the naming convention the pipeline parses. |
| `root_radius_stele_area_correlation.py` | How strongly does stele area track root radius? |
| `stats_utils.py` | Shared spline fitting and summary statistics. |
| `test_color_spaces.py` | Compares colour spaces for separating cells from walls. |
| `visualize_cell_centers.py` | Draws detected cell centres onto the original image. |

### code/quality_control/ — reviewing new segmentations

| File | What it is |
|---|---|
| `qc_manager.py` | Reviews newly segmented images and flags bad ones. |
| `qc_server.py` | The small web server behind the QC review page. |

### code/root_model/ — reconstruction and 3D printing

| File | What it is |
|---|---|
| `__init__.py` | Package marker for the reconstruction and 3D-printing tools. |
| `build_pattern_stls.py` | Five printable roots that differ only in how cell size is distributed. |
| `build_web_bundle.py` | Package the root model for the public website, with a curated image set. |
| `generate_root_model.py` | Builds the interactive HTML model of a root cross-section. |
| `root_to_fea_stl.py` | Turns a geometry export into a watertight wall-solid STL. |
| `template.html` | The reconstruction page itself, before per-root data is injected into it. |

### code/age_experiment/ — developmental stages

| File | What it is |
|---|---|
| `pattern_development.py` | How the cortical cell pattern emerges across root development. |

## Notes

Results in `results/` are all reproducible by re-running the pipeline, and are
not tracked. Neither are weights, data, or imagery — see `.gitignore`.
