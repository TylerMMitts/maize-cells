# Sweeps the morphological kernel sizes.
#
# The close and open kernels strongly affect how the exclusion mask is
# cleaned up; this shows the effect of each setting side by side.

from code.yolo.test_robust import batch_test_robust_model
import os
from code.config import DATA_FOLDER, EXCLUSION_WEIGHTS, ROBUST_CELL_WEIGHTS

# Create output directories for comparison
output_no_morph = "test_results_no_morph"
output_with_morph = "test_results_with_morph"

# Test without morphological post-processing
results_no_morph = batch_test_robust_model(
    cell_weights=ROBUST_CELL_WEIGHTS,
    exclusion_weights=EXCLUSION_WEIGHTS,
    image_folder=DATA_FOLDER / 'test_folder',
    confidence=0.98,
    exclusion_confidence=0.5,
    output_dir=output_no_morph,
    refine=True,
    color_tolerance=15,
    use_darkest_seed=False,
    max_det=3000,
    filter_by_inner_part=True,
    inner_open_kernel=99,
    inner_close_kernel=155,
    inner_open_iterations=1,
    inner_close_iterations=1,
    outer_open_kernel=77,
    outer_close_kernel=99,
    outer_open_iterations=1,
    outer_close_iterations=1,
    keep_largest_component=True,
    exclusion_overlap_threshold=25.0,
    overlap_threshold=15.0,
    morph_post_process=False  # No post-processing
)

# Test with morphological post-processing
results_with_morph = batch_test_robust_model(
    cell_weights=ROBUST_CELL_WEIGHTS,
    exclusion_weights=EXCLUSION_WEIGHTS,
    image_folder=DATA_FOLDER / 'test_folder',
    confidence=0.98,
    exclusion_confidence=0.5,
    output_dir=output_with_morph,
    refine=True,
    color_tolerance=15,
    use_darkest_seed=False,
    max_det=3000,
    filter_by_inner_part=True,
    inner_open_kernel=99,
    inner_close_kernel=155,
    inner_open_iterations=1,
    inner_close_iterations=1,
    outer_open_kernel=77,
    outer_close_kernel=99,
    outer_open_iterations=1,
    outer_close_iterations=1,
    keep_largest_component=True,
    exclusion_overlap_threshold=25.0,
    overlap_threshold=15.0,
    morph_post_process=True,    # Enable post-processing
    morph_close_kernel=21,        # Closing kernel size
    morph_open_kernel=21          # Opening kernel size
)
