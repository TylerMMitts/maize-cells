# Run the full pipeline over the widiv dataset using the low-resolution models.
#
# Mirrors run_pipeline.py, with the model choice (and the two post-processing
# constants that depend on it) overridden IN MEMORY for this run only. config.py
# is never written to, so every other dataset keeps using the original models.
#
# Overridden for this run
#     ROBUST_CELL_WEIGHTS   -> robust_cell_detector_lowres
#     EXCLUSION_WEIGHTS     -> inner_part_detector_lowres
#     morph open/close      21 -> 11
#     neighbour expansion   20 -> 10
#
# The two kernel values are not cosmetic. They are absolute pixel sizes tuned for
# imagery where cells span ~46 px; widiv cells span ~23 px, so a 21 px
# morphological opening is nearly as wide as a cell and erases it. Measured on a
# widiv quadrant with 269 raw detections: kernel 21 keeps 38 cells, kernel 11
# keeps 124. Running with the stock value would produce a dataset undercounted by
# roughly 3x, so the models and these constants have to move together.
#
# Deviations from run_pipeline.py's literal defaults, and why
#   * SPECIES / POPULATION are 'Zea mays' / 'Widiv'. run_pipeline.py currently
#     reads 'Tomato' for both, which was for the previous dataset.
#   * QC runs non-interactively. run_pipeline.py sets QC_INTERACTIVE=True, which
#     opens a browser and waits for a human; that cannot complete in an unattended
#     run over ~1,700 quadrants. Nothing is auto-deleted -- the QC report is still
#     written for review afterwards.
#
# UNCALIBRATED SCALE -- READ THIS
# UM_PER_PIXEL is taken from run_pipeline.py (2.3416) as instructed, but that
# value was calibrated for the tomato imaging setup, NOT for widiv, whose true
# micron-per-pixel is unknown. Pixel columns (area_pixels, radius_pixels) are
# correct; every um-denominated column for widiv rows -- cell area, stele area,
# root radius -- is on an arbitrary scale and must not be compared against maize
# values until a real conversion factor is measured.

import os
import sys
from datetime import datetime
from pathlib import Path

# Some pipeline steps print emoji (e.g. the cell-file analyser's success line).
# With stdout redirected to a file Windows picks cp1252, which cannot encode
# them, and the resulting UnicodeEncodeError kills the step -- step 4 died this
# way after completing all its work but before it could fold the cell-file
# counts back into master_summary. Force UTF-8 and reconfigure the existing
# streams, since they are created before this module runs.
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code import config

# ---- dataset ---------------------------------------------------------------
SPECIES = "Zea mays"
POPULATION = "Widiv"
UM_PER_PIXEL = 2.3416                      # from run_pipeline.py -- see caveat above
NEW_IMAGES_FOLDER = str(PROJECT_ROOT / "data" / "widiv")
OUTPUT_FOLDER = str(PROJECT_ROOT / "results")

# ---- processing options (as run_pipeline.py) -------------------------------
SKIP_EXISTING = True
VERBOSE = True
FORCE_REBUILD_MASTER_SUMMARY = False       # append-only; never rebuild
SKIP_QC = False
QC_INTERACTIVE = False                     # see deviations above
QC_AUTO_DELETE = False
QC_PORT = 8888

# ---- run-only model overrides ----------------------------------------------
LOWRES_CELL = (PROJECT_ROOT / "models" / "runs" / "segment" / "runs" /
               "robust_segmentation" / "robust_cell_detector_lowres" / "weights" / "best.pt")
LOWRES_EXCLUSION = (PROJECT_ROOT / "models" / "runs" / "segment" / "runs" /
                    "exclusion_model" / "inner_part_detector_lowres" / "weights" / "best.pt")
MORPH_KERNEL = 11
EXPANSION_PIXELS = 10


def apply_overrides():
    missing = [p for p in (LOWRES_CELL, LOWRES_EXCLUSION) if not p.exists()]
    if missing:
        raise SystemExit("missing low-resolution weights:\n  " +
                         "\n  ".join(str(p) for p in missing))

    config.ROBUST_CELL_WEIGHTS = LOWRES_CELL
    config.EXCLUSION_WEIGHTS = LOWRES_EXCLUSION
    config.CELL_SEGMENTATION_CONFIG['morph_close_kernel'] = MORPH_KERNEL
    config.CELL_SEGMENTATION_CONFIG['morph_open_kernel'] = MORPH_KERNEL
    config.NEIGHBOR_CONFIG['expansion_pixels'] = EXPANSION_PIXELS

    print("RUN-ONLY OVERRIDES (config.py on disk is unchanged)")
    print(f"  cell model      : {LOWRES_CELL.parent.parent.name}/{LOWRES_CELL.name}")
    print(f"  exclusion model : {LOWRES_EXCLUSION.parent.parent.name}/{LOWRES_EXCLUSION.name}")
    print(f"  morph kernels   : {MORPH_KERNEL} (stock 21)")
    print(f"  expansion px    : {EXPANSION_PIXELS} (stock 20)")
    print()


def main():
    apply_overrides()

    # code.pipeline rather than the bare `pipeline` run_pipeline.py uses -- that
    # form only resolves when the working directory is code/ itself.
    from code.pipeline import RootAnalysisPipeline

    pipeline = RootAnalysisPipeline(
        new_images_folder=NEW_IMAGES_FOLDER,
        output_folder=OUTPUT_FOLDER,
        skip_existing=SKIP_EXISTING,
        verbose=VERBOSE,
        species=SPECIES,
        population=POPULATION,
        config_overrides={
            'DEFAULT_PIXEL_TO_UM': UM_PER_PIXEL,
            'DEFAULT_PIXEL_TO_UM_SQUARED': UM_PER_PIXEL * UM_PER_PIXEL,
        },
        force_rebuild_master_summary=FORCE_REBUILD_MASTER_SUMMARY,
    )

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    log_path = os.path.join(
        OUTPUT_FOLDER, f"widiv_pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, text):
            for s in self.streams:
                s.write(text)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    log_file = open(log_path, 'w', encoding='utf-8')
    out, err = sys.stdout, sys.stderr
    sys.stdout = Tee(out, log_file)
    sys.stderr = Tee(err, log_file)
    try:
        print(f"species={SPECIES}  population={POPULATION}")
        print(f"images={NEW_IMAGES_FOLDER}")
        print(f"pixel_to_um={UM_PER_PIXEL}  (UNCALIBRATED for widiv -- see module docstring)\n")
        pipeline.run_full_pipeline(
            skip_qc=SKIP_QC,
            qc_interactive=QC_INTERACTIVE,
            qc_auto_delete=QC_AUTO_DELETE,
            qc_port=QC_PORT,
        )
        print(pipeline.generate_summary())
    finally:
        sys.stdout, sys.stderr = out, err
        log_file.close()

    print(f"Full log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
