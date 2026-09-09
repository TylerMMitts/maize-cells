# Entry point for reconciling master_summary.csv against the files on disk.
#
# Images whose measurement files have gone missing are dropped from the master
# summary, and optionally the reverse: measurement files with no row in the
# summary are deleted. Rewrites master_summary.csv in results/ and, when
# cfg.rebuild_analysis is on, re-runs the downstream analyses over what is left.

import sys

from code.config import MEASUREMENTS_FOLDER, RESULTS_FOLDER
from code.pipeline import RootAnalysisPipeline


def main():
    # python -m code.run_cleanup
    class cfg:
        output_folder = RESULTS_FOLDER
        measurements_folder = MEASUREMENTS_FOLDER

        # Deleting measurement files that the summary does not mention is the
        # destructive half of this script, so it stays off unless asked for.
        remove_orphaned_files = False

        backup_before_cleanup = False
        rebuild_analysis = True

        # Report what would change and stop, without touching anything.
        preview_only = False

    pipeline = RootAnalysisPipeline(
        output_folder=str(cfg.output_folder),
        verbose=True,
    )

    print(f'Output folder: {cfg.output_folder}')
    print(f'Measurements folder: {cfg.measurements_folder}')
    print(f'Remove orphaned files: {cfg.remove_orphaned_files}')
    print(f'Preview only: {cfg.preview_only}')

    if cfg.preview_only:
        print('\nRUNNING IN PREVIEW MODE - No changes will be made')
        pipeline.preview_cleanup(measurements_folder=str(cfg.measurements_folder))
        return

    print('\nWARNING: You are about to modify your dataset')
    print('Images missing measurement files will be REMOVED from master_summary.')

    # The preview runs first every time, so the confirmation below is answered
    # against the actual list of changes rather than in the dark.
    pipeline.preview_cleanup(measurements_folder=str(cfg.measurements_folder))

    response = input('\n\nContinue with cleanup? (yes/no): ')
    if response.lower() != 'yes':
        print('Cleanup cancelled.')
        sys.exit(0)

    results = pipeline.full_cleanup_and_rebuild(
        measurements_folder=str(cfg.measurements_folder),
        remove_orphaned=cfg.remove_orphaned_files,
        backup=cfg.backup_before_cleanup,
        run_analysis=cfg.rebuild_analysis,
        dry_run=False,
    )

    print('CLEANUP COMPLETE')
    print(f"Images remaining: {results['images_remaining']}")
    print(f"Measurements folder: {results['measurements_folder']}")
    print(pipeline.generate_summary())


if __name__ == '__main__':
    main()
