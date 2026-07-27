from pipeline import RootAnalysisPipeline
import sys
import os
from datetime import datetime

# Dataset metadata
SPECIES = "Zea mays" 
POPULATION = "MAGIC_MAIZE" 

# Microscopy calibration
UM_PER_PIXEL = 2.3416 

# Per-image scale detection (optional)
USE_SCALE_CSV = True
SCALE_IMAGE_KEY_CSV = "../data/01_LAT_images/scale_image_key.csv"
SCALE_VALUES_CSV = "../data/01_LAT_images/scale_values.csv"

# Input/output folders
NEW_IMAGES_FOLDER = "../data/01_LAT_images"
OUTPUT_FOLDER = "../results"

# Processing options
SKIP_EXISTING = True          # Skip images already in master summary
VERBOSE = True                # Print detailed logging information

# ONLY set to True if you want to completely reset the master summary.
FORCE_REBUILD_MASTER_SUMMARY = False  # IMPORTANT: Keep this False for append-only mode!

# Quality Control settings
SKIP_QC = False               # Set to True to skip QC entirely
QC_INTERACTIVE = True         # Open browser for manual deletion
QC_AUTO_DELETE = False        # Automatically delete bad images based on thresholds
QC_PORT = 8888                # Port for the QC web server

# QC thresholds (only used if QC_AUTO_DELETE=True)
QC_MIN_CELLS = 10             # Minimum number of cells to keep an image
QC_MAX_CELLS = 5000           # Maximum number of cells to keep an image

# Apply um_per_pixel configuration (fallback default)
config_overrides = {
    'DEFAULT_PIXEL_TO_UM': UM_PER_PIXEL,
    'DEFAULT_PIXEL_TO_UM_SQUARED': UM_PER_PIXEL * UM_PER_PIXEL
}

# Build the per-image scale map, if requested.
# Images with an unresolvable scale (missing scale_id, or conflicting scale_id
# across rows) are excluded from the run entirely rather than guessed at.
pixel_to_um_map = None
exclude_images = None
if USE_SCALE_CSV:
    if os.path.exists(SCALE_IMAGE_KEY_CSV) and os.path.exists(SCALE_VALUES_CSV):
        from code.util.file_utils import build_pixel_to_um_map
        pixel_to_um_map, exclude_images = build_pixel_to_um_map(SCALE_IMAGE_KEY_CSV, SCALE_VALUES_CSV)
        print(f"Loaded per-image scale map: {len(pixel_to_um_map)} images "
              f"(from {SCALE_IMAGE_KEY_CSV} + {SCALE_VALUES_CSV})")
        if exclude_images:
            print(f"Excluding {len(exclude_images)} images with unresolvable scale: "
                  f"{sorted(exclude_images)}")
    else:
        print(f"WARNING: USE_SCALE_CSV=True but one of the CSV files was not found:")
        print(f"  {SCALE_IMAGE_KEY_CSV} (exists: {os.path.exists(SCALE_IMAGE_KEY_CSV)})")
        print(f"  {SCALE_VALUES_CSV} (exists: {os.path.exists(SCALE_VALUES_CSV)})")
        print(f"  Falling back to the single UM_PER_PIXEL={UM_PER_PIXEL} for all images.")

# Process new images
pipeline = RootAnalysisPipeline(
    new_images_folder=NEW_IMAGES_FOLDER,
    output_folder=OUTPUT_FOLDER,
    skip_existing=SKIP_EXISTING,
    verbose=VERBOSE,
    species=SPECIES,
    population=POPULATION,
    config_overrides=config_overrides,
    force_rebuild_master_summary=FORCE_REBUILD_MASTER_SUMMARY,
    pixel_to_um_map=pixel_to_um_map,
    exclude_images=exclude_images
)

# Redirect all output to a log file as well as console
# Create a log file with timestamp
os.makedirs("../results", exist_ok=True)
log_filename = f"../results/full_debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

class TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Open log file and redirect output
log_file = open(log_filename, 'w', encoding='utf-8')
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = TeeOutput(original_stdout, log_file)
sys.stderr = TeeOutput(original_stderr, log_file)


if pixel_to_um_map:
    print(f"Per-image scale map: ENABLED ({len(pixel_to_um_map)} images)")
else:
    print(f"Per-image scale map: disabled (using single Um per pixel for all images)")

# Run full pipeline with QC enabled
pipeline.run_full_pipeline(
    skip_qc=SKIP_QC,
    qc_interactive=QC_INTERACTIVE,
    qc_auto_delete=QC_AUTO_DELETE,
    qc_port=QC_PORT
)

# Access results
print(pipeline.generate_summary())

# Restore original output and close log file
sys.stdout = original_stdout
sys.stderr = original_stderr
log_file.close()

print(f"Full debug log saved to: {log_filename}")