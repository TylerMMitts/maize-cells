from code.yolo.test_robust import batch_test_robust_model
import os

# Create output directories for comparison
output_no_morph = "test_results_no_morph"
output_with_morph = "test_results_with_morph"

# Test without morphological post-processing
results_no_morph = batch_test_robust_model(
    cell_weights="models/runs/segment/runs/robust_segmentation/robust_cell_detector/weights/best.pt",
    exclusion_weights="models/runs/segment/runs/exclusion_model/inner_part_detector/weights/best.pt",
    image_folder="data/test_folder",
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
    cell_weights="models/runs/segment/runs/robust_segmentation/robust_cell_detector/weights/best.pt",
    exclusion_weights="models/runs/segment/runs/exclusion_model/inner_part_detector/weights/best.pt",
    image_folder="data/test_folder",
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
