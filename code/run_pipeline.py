# Entry point for the full root analysis pipeline.
#
# Runs every stage in order over the images in cfg.new_images_folder: root
# detection and quadrant splitting, cell segmentation, quality control,
# measurement, cell file counting, feature extraction and the statistics.
# Writes into results/ (master_summary.csv, feature_table.csv, and one folder
# per analysis) plus a timestamped debug log of the whole run.

import os
import sys
from datetime import datetime
from pathlib import Path

from code.config import DATA_FOLDER, RESULTS_FOLDER, resolve_output
from code.pipeline import RootAnalysisPipeline


class TeeOutput:
    # Sends everything to the console and the log file at once, so a run that
    # is being watched live still leaves a complete record behind.
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def main():
    # python -m code.run_pipeline
    class cfg:
        # Dataset metadata, written onto every row this run produces
        species = 'Maize'
        population = 'age_experiment'

        # Microscopy calibration, used when no per-image scale is available
        um_per_pixel = 2.3416

        # Per-image scale detection. Off by default; the two CSVs are only
        # read when this is True.
        use_scale_csv = False
        scale_image_key_csv = DATA_FOLDER / '01_LAT_images' / 'scale_image_key.csv'
        scale_values_csv = DATA_FOLDER / '01_LAT_images' / 'scale_values.csv'

        # Input and output
        new_images_folder = DATA_FOLDER / 'test_pipeline'
        output_folder = RESULTS_FOLDER

        skip_existing = True
        verbose = True

        # Append-only is the whole point of the master summary: rebuilding
        # discards the species and population already recorded against older
        # rows. Only set this True to deliberately start over.
        force_rebuild_master_summary = False

        # Quality control
        skip_qc = False
        qc_interactive = True
        qc_auto_delete = False
        qc_port = 8888

    config_overrides = {
        'DEFAULT_PIXEL_TO_UM': cfg.um_per_pixel,
        'DEFAULT_PIXEL_TO_UM_SQUARED': cfg.um_per_pixel * cfg.um_per_pixel,
    }

    # Images whose scale cannot be resolved are dropped from the run rather
    # than processed at a guessed scale, which would put wrong micron values
    # into the master summary with nothing to flag them later.
    pixel_to_um_map = None
    exclude_images = None
    if cfg.use_scale_csv:
        if os.path.exists(cfg.scale_image_key_csv) and os.path.exists(cfg.scale_values_csv):
            from code.util.file_utils import build_pixel_to_um_map
            pixel_to_um_map, exclude_images = build_pixel_to_um_map(
                str(cfg.scale_image_key_csv), str(cfg.scale_values_csv))
            print(f'Loaded per-image scale map: {len(pixel_to_um_map)} images')
            if exclude_images:
                print(f'Excluding {len(exclude_images)} images with unresolvable scale: '
                      f'{sorted(exclude_images)}')
        else:
            print('WARNING: use_scale_csv is True but a CSV was not found:')
            print(f'  {cfg.scale_image_key_csv} (exists: {os.path.exists(cfg.scale_image_key_csv)})')
            print(f'  {cfg.scale_values_csv} (exists: {os.path.exists(cfg.scale_values_csv)})')
            print(f'  Falling back to um_per_pixel={cfg.um_per_pixel} for every image.')

    pipeline = RootAnalysisPipeline(
        new_images_folder=str(cfg.new_images_folder),
        output_folder=str(cfg.output_folder),
        skip_existing=cfg.skip_existing,
        verbose=cfg.verbose,
        species=cfg.species,
        population=cfg.population,
        config_overrides=config_overrides,
        force_rebuild_master_summary=cfg.force_rebuild_master_summary,
        pixel_to_um_map=pixel_to_um_map,
        exclude_images=exclude_images,
    )

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = resolve_output(Path(cfg.output_folder) / f'full_debug_log_{stamp}.txt')

    log_file = open(log_path, 'w', encoding='utf-8')
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeOutput(original_stdout, log_file)
    sys.stderr = TeeOutput(original_stderr, log_file)

    try:
        if pixel_to_um_map:
            print(f'Per-image scale map: ENABLED ({len(pixel_to_um_map)} images)')
        else:
            print('Per-image scale map: disabled (single um per pixel for all images)')

        pipeline.run_full_pipeline(
            skip_qc=cfg.skip_qc,
            qc_interactive=cfg.qc_interactive,
            qc_auto_delete=cfg.qc_auto_delete,
            qc_port=cfg.qc_port,
        )
        print(pipeline.generate_summary())
    finally:
        # Restored in a finally so a crash mid-run still closes the log and
        # leaves the console usable, rather than losing both.
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    print(f'Full debug log saved to: {log_path}')


if __name__ == '__main__':
    main()
