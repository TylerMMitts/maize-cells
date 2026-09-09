# Segmentation-only run over the widiv dataset.
#
# Root detection, quadrant splitting and cell measurement using the
# low-resolution models, without the analysis stages that follow. Useful for
# checking segmentation quality before committing to a full pipeline run.
#
# Everything lands under results/widiv/ and nothing touches the main dataset --
# no rows are added to master_summary.csv and no existing measurements are
# overwritten, so this is purely an evaluation run.
#
#     results/widiv/
#         quadrants/            quadrant images + per-crop metadata
#             detections/       full-root crops
#         measurements/         *_measurements.csv and *_centers.jpg  <- the overlays
#         density_analysis/     per-image density output
#
# Two stages, each skippable so the run can be resumed:
#
#     stage 1  root detection + quadrant split      (data/widiv -> quadrants/)
#     stage 2  cell segmentation + exclusion +      (quadrants/ -> measurements/)
#              post-processing
#
# Stage 2 skips any quadrant that already has a measurements CSV, so an
# interrupted run continues where it stopped.
#
# SCALE CAVEAT
# widiv has no calibrated micron-per-pixel value, so config.DEFAULT_PIXEL_TO_UM
# is used as a placeholder. Pixel measurements (area_pixels, radius_pixels) are
# valid; every um-denominated column is on an arbitrary scale until a real
# conversion factor is supplied via --pixel-to-um.

import os
import sys
from pathlib import Path

# cell_measurements pins CPU at import; do the same here so both stages agree.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code import config  # noqa: E402

SOURCE_IMAGES = PROJECT_ROOT / 'data' / 'widiv'
OUT_ROOT = PROJECT_ROOT / 'results' / 'widiv'
QUADRANTS = OUT_ROOT / 'quadrants'
MEASUREMENTS = OUT_ROOT / 'measurements'
DENSITY = OUT_ROOT / 'density_analysis'

LOWRES_WEIGHTS = (PROJECT_ROOT / 'models' / 'runs' / 'segment' / 'runs' /
                  'robust_segmentation' / 'robust_cell_detector_lowres' / 'weights' / 'best.pt')

# The stock exclusion model over-excludes badly on widiv -- it marked 63-70% of
# a quadrant as stele on the two checked, swallowing whole bands of ordinary
# cortex. The low-resolution retrain brings that to 52% and 45% on the same two
# images, releasing the cortex it was wrongly discarding. Defaults to the
# retrained model; pass --exclusion-weights to compare against the original.
LOWRES_EXCLUSION_WEIGHTS = (PROJECT_ROOT / 'models' / 'runs' / 'segment' / 'runs' /
                            'exclusion_model' / 'inner_part_detector_lowres' / 'weights' / 'best.pt')

# Post-processing constants are absolute pixel sizes, so they have to be scaled
# to the imagery the way the model was. widiv cells are ~23 px across where the
# original imagery has ~46 px, and the pipeline's 21 px morphological kernel is
# nearly as wide as a widiv cell -- opening with it erases them. Measured on one
# quadrant (269 raw detections): kernel 21 keeps 38 cells, kernel 11 keeps 124,
# kernel 7 keeps 133, morphology off keeps 137. 11 recovers most of the loss
# while still cleaning up mask noise, so both constants are halved here.
MORPH_KERNEL = 11        # main pipeline: 21
EXPANSION_PIXELS = 10    # main pipeline: 20 (neighbour contact test)


def stage1_split(source=SOURCE_IMAGES, out=QUADRANTS):
    from code.yolo.test_root_model import batch_test_root_model
    out.mkdir(parents=True, exist_ok=True)
    print(f'STAGE 1  root detection + quadrant split\n  {source} -> {out}\n')
    batch_test_root_model(
        weights_path=str(config.ROOT_DETECTION_WEIGHTS),
        image_folder=str(source),
        confidence=config.ROOT_DETECTION_CONFIG['confidence'],
        output_dir=str(out),
    )
    quads = [p for p in out.glob('*.jpg')
             if not p.name.endswith('_full.jpg') and not p.name.startswith('detected_')]
    print(f'\n  {len(quads)} quadrant images written')
    return quads


def stage2_measure(weights, quadrants=QUADRANTS, out=MEASUREMENTS, pixel_to_um=None, limit=None,
                   morph_kernel=MORPH_KERNEL, expansion_pixels=EXPANSION_PIXELS,
                   exclusion_weights=None, image_list=None):
    from code.measurements.cell_measurements import batch_extract_measurements

    out.mkdir(parents=True, exist_ok=True)
    DENSITY.mkdir(parents=True, exist_ok=True)

    if pixel_to_um is None:
        pixel_to_um = config.DEFAULT_PIXEL_TO_UM

    if exclusion_weights is None:
        exclusion_weights = LOWRES_EXCLUSION_WEIGHTS

    specific = None
    if image_list:
        specific = list(image_list)
        print(f'  restricted to an explicit list of {len(specific)} quadrants')
    elif limit:
        # deterministic subset, for a quick look before committing to the full run
        names = sorted(p.name for p in quadrants.glob('*.jpg')
                       if not p.name.endswith('_full.jpg') and not p.name.startswith('detected_'))
        specific = names[:limit]
        print(f'  limiting to first {len(specific)} quadrants')

    print(f'STAGE 2  segmentation + exclusion + post-processing')
    print(f'  cell model      : {weights}')
    print(f'  exclusion model : {exclusion_weights}')
    print(f'  input           : {quadrants}')
    print(f'  output          : {out}')
    print(f'  morph kernel    : {morph_kernel}  (main pipeline uses 21)')
    print(f'  expansion px    : {expansion_pixels}  (main pipeline uses 20)')
    print(f'  pixel_to_um     : {pixel_to_um}  (placeholder -- see module docstring)\n')

    batch_extract_measurements(
        weights_path=str(weights),
        exclusion_weights=str(exclusion_weights),
        image_folder=str(quadrants),
        output_dir=str(out),
        density_output_dir=str(DENSITY),
        specific_images=specific,
        confidence=config.CELL_SEGMENTATION_CONFIG['confidence'],
        exclusion_confidence=config.EXCLUSION_CONFIG['confidence'],
        filter_by_inner_part=config.EXCLUSION_CONFIG['filter_by_inner_part'],
        inner_open_kernel=config.EXCLUSION_CONFIG['inner_open_kernel'],
        inner_close_kernel=config.EXCLUSION_CONFIG['inner_close_kernel'],
        outer_open_kernel=config.EXCLUSION_CONFIG['outer_open_kernel'],
        outer_close_kernel=config.EXCLUSION_CONFIG['outer_close_kernel'],
        exclusion_overlap_threshold=config.EXCLUSION_CONFIG['exclusion_overlap_threshold'],
        keep_largest_component=config.EXCLUSION_CONFIG['keep_largest_component'],
        overlap_threshold=config.OVERLAP_CONFIG['overlap_threshold'],
        refine=config.CELL_SEGMENTATION_CONFIG['refine'],
        color_tolerance=config.CELL_SEGMENTATION_CONFIG['color_tolerance'],
        use_darkest_seed=config.CELL_SEGMENTATION_CONFIG['use_darkest_seed'],
        morph_post_process=config.CELL_SEGMENTATION_CONFIG['morph_post_process'],
        morph_close_kernel=morph_kernel,
        morph_open_kernel=morph_kernel,
        expansion_pixels=expansion_pixels,
        max_neighbors=config.NEIGHBOR_CONFIG['max_neighbors'],
        sam_refiner=None,
        # never write into the shared master summary from an evaluation run
        create_master_summary_flag=False,
        pixel_to_um=pixel_to_um,
    )


def summarise():
    import pandas as pd
    csvs = sorted(MEASUREMENTS.glob('*_measurements.csv'))
    overlays = sorted(MEASUREMENTS.glob('*_centers.jpg'))
    print(f'\n\nRESULT')
    print(f'  measurement CSVs : {len(csvs)}')
    print(f'  centre overlays  : {len(overlays)}')
    if not csvs:
        return
    counts, empty = [], 0
    for c in csvs:
        try:
            n = len(pd.read_csv(c))
        except Exception:
            empty += 1
            continue
        counts.append(n)
    if counts:
        s = pd.Series(counts)
        print(f'  cells per quadrant: min {s.min()}  median {int(s.median())}  '
              f'mean {s.mean():.0f}  max {s.max()}')
        print(f'  quadrants with 0 cells: {int((s == 0).sum())}')
    if empty:
        print(f'  unreadable CSVs: {empty}')
    print(f'  overlays are in: {MEASUREMENTS}')


def main():
    # python -m code.run_widiv_segmentation
    class cfg:
        # cell segmentation weights (default: the low-resolution model)
        weights = str(LOWRES_WEIGHTS)
        # reuse existing quadrants
        skip_split = False
        # split only
        skip_measure = False
        # process only the first N quadrants (quick look)
        limit = None
        pixel_to_um = None
        morph_kernel = MORPH_KERNEL
        expansion_pixels = EXPANSION_PIXELS
        # exclusion model weights (default: the low-resolution retrain)
        exclusion_weights = str(LOWRES_EXCLUSION_WEIGHTS)
        # text file of quadrant filenames to process, one per line
        image_list = None


    image_list = None
    if cfg.image_list:
        image_list = [ln.strip() for ln in Path(cfg.image_list).read_text().splitlines() if ln.strip()]

    weights = Path(cfg.weights)
    if not weights.exists():
        raise SystemExit(f'weights not found: {weights}')
    if not SOURCE_IMAGES.is_dir():
        raise SystemExit(f'source images not found: {SOURCE_IMAGES}')

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not cfg.skip_split:
        stage1_split()
    else:
        print('STAGE 1 skipped (--skip-split)')

    if not cfg.skip_measure:
        stage2_measure(weights, pixel_to_um=cfg.pixel_to_um, limit=cfg.limit,
                       morph_kernel=cfg.morph_kernel, expansion_pixels=cfg.expansion_pixels,
                       exclusion_weights=Path(cfg.exclusion_weights), image_list=image_list)
        summarise()
    else:
        print('STAGE 2 skipped (--skip-measure)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
