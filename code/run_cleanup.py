from pipeline import RootAnalysisPipeline
import sys
import os
from pathlib import Path
from datetime import datetime

# Folder locations
OUTPUT_FOLDER = "../results"

MEASUREMENTS_FOLDER = "../results/measurements_all"  # Set to None to use config default, or specify path

# Cleanup options
# remove_orphaned: If True, delete measurement files not in master_summary
REMOVE_ORPHANED_FILES = False  # Changed default to False

BACKUP_BEFORE_CLEANUP = False  # Create backup before making changes
REBUILD_ANALYSIS = True  # Re-run analysis after cleanup

# Set to True to preview what would be removed without actually making changes
PREVIEW_ONLY = False  # Change to False when you're ready to actually run cleanup

# Initialize pipeline
pipeline = RootAnalysisPipeline(
    output_folder=OUTPUT_FOLDER,
    verbose=True
)

# Determine measurements folder
if MEASUREMENTS_FOLDER is None:
    from code import config
    measurements_folder = str(config.MEASUREMENTS_FOLDER)
else:
    measurements_folder = str(Path(MEASUREMENTS_FOLDER).resolve())

print(f"Output folder: {OUTPUT_FOLDER}")
print(f"Measurements folder: {measurements_folder}")
print(f"Remove orphaned files: {REMOVE_ORPHANED_FILES}")
print(f"Preview only: {PREVIEW_ONLY}")

if PREVIEW_ONLY:
    # Preview what would be cleaned up
    print("\nRUNNING IN PREVIEW MODE - No changes will be made")
    
    pipeline.preview_cleanup(measurements_folder=measurements_folder)
    
else:
    # Confirm before proceeding
    print("\nWARNING: You are about to modify your dataset")
    print("Images missing measurement files will be REMOVED from master_summary.")
    
    # Show preview first
    pipeline.preview_cleanup(measurements_folder=measurements_folder)
    
    response = input("\n\nContinue with cleanup? (yes/no): ")
    if response.lower() != 'yes':
        print("Cleanup cancelled.")
        sys.exit(0)
    
    # Run full cleanup and rebuild
    results = pipeline.full_cleanup_and_rebuild(
        measurements_folder=measurements_folder,
        remove_orphaned=REMOVE_ORPHANED_FILES,
        backup=BACKUP_BEFORE_CLEANUP,
        run_analysis=REBUILD_ANALYSIS,
        dry_run=False
    )
    
    print("CLEANUP COMPLETE")
    print(f"Images remaining: {results['images_remaining']}")
    print(f"Measurements folder: {results['measurements_folder']}")
    
    # Generate final summary
    print(pipeline.generate_summary())