# Scratch entry point for running individual pipeline stages by hand.
#
# Kept separate from run_pipeline so a single stage can be re-run against an
# existing results/ folder without going through the whole sequence. Most of
# the body is commented-out calls, switched on as needed.

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from code.yolo.train_robust import train_robust_model
from code.yolo.test_robust import batch_test_robust_model
from code.measurements.cell_measurements import extract_cell_measurements, batch_extract_measurements, visualize_measurements
from code.util.visualize_cell_centers import visualize_cell_centers
# from code.tda.tda_analysis import batch_tda_analysis, analyze_single_image
# from code.tda.mds import mds_from_wasserstein_csv
from code.util.lightweight_sam_refiner import LightweightSAMRefiner
from code.config import DATA_FOLDER, EXCLUSION_WEIGHTS, MEASUREMENTS_FOLDER, MODELS_FOLDER, RESULTS_FOLDER, ROBUST_CELL_WEIGHTS

USE_SAM_REFINEMENT = False
SAM_MODEL_TYPE = 'mobilesam'
SAM_DEVICE = 'cuda'

if USE_SAM_REFINEMENT:
    sam_refiner = LightweightSAMRefiner(
        model_type=SAM_MODEL_TYPE,
        device=SAM_DEVICE,
        verbose=True,
        use_point_prompts=True,
        use_box_prompts=True,
        post_process=True,
        close_kernel_size=5,
        open_kernel_size=5,
        min_mask_area=50,
        max_mask_area_ratio=0.98
    )
else:
    sam_refiner = None

EXCLUSION_WEIGHTS = EXCLUSION_WEIGHTS

ROBUST_TRAIN_CONFIG = {
    'dataset_yaml': MODELS_FOLDER / 'cell_dataset' / 'data.yaml',
    'model_size': 'yolov8l-seg.pt',
    'epochs': 100,
    'max_det': 3000,
    'batch_size': 2,
    'image_size': 640,
    'project': MODELS_FOLDER / 'runs' / 'segment' / 'runs',
    'run_name': 'robust_segmentation',
    'device': 'cpu'
}

ROBUST_WEIGHTS = ROBUST_CELL_WEIGHTS

ROBUST_TEST_CONFIG = {
    'cell_weights': ROBUST_WEIGHTS,
    'exclusion_weights': EXCLUSION_WEIGHTS,
    'image_folder': DATA_FOLDER / 'poster_results',
    'confidence': 0.98,
    'output_dir': RESULTS_FOLDER / 'specific_tests_sam' if USE_SAM_REFINEMENT else RESULTS_FOLDER / 'chosen_results_robust',
    'refine': True,
    'color_tolerance': 15,
    'use_darkest_seed': False,
    'max_det': 3000,
    'filter_by_inner_part': True,
    'inner_open_kernel': 99,
    'inner_close_kernel': 155,
    'inner_open_iterations': 1,
    'inner_close_iterations': 1,
    'outer_open_kernel': 77,
    'outer_close_kernel': 99,
    'outer_open_iterations': 1,
    'outer_close_iterations': 1,
    'keep_largest_component': True,
    'exclusion_overlap_threshold': 25.0,
    'overlap_threshold': 15.0,
    'exclusion_confidence': 0.25,
    'morph_close_kernel': 21,
    'morph_open_kernel': 21,
    'morph_post_process': True,
    'sam_refiner': sam_refiner
}

MEASUREMENT_CONFIG = {
    'weights_path': ROBUST_WEIGHTS,
    'exclusion_weights': EXCLUSION_WEIGHTS,
    'image_folder': DATA_FOLDER / 'chosen_results_some',
    'confidence': 0.98,
    'exclusion_confidence': 0.25,
    'output_dir': RESULTS_FOLDER / 'measurements',
    'filter_by_inner_part': True,
    'inner_open_kernel': 99,
    'inner_close_kernel': 155,
    'outer_open_kernel': 77,
    'outer_close_kernel': 99,
    'exclusion_overlap_threshold': 25.0,
    'overlap_threshold': 15.0,
    'keep_largest_component': True,
    'refine': True,
    'color_tolerance': 15,
    'use_darkest_seed': False,
    'morph_close_kernel': 21,
    'morph_open_kernel': 21,
    'morph_post_process': True,
    'sam_refiner': sam_refiner,
    'density_output_dir': RESULTS_FOLDER / 'measurements' / 'density_analysis',
    'expansion_pixels': 20,
    'max_neighbors': 7
}

TDA_CONFIG = {
    'measurements_dir': MEASUREMENTS_FOLDER,
    'output_dir': RESULTS_FOLDER / 'tda' / 'VR',
    'max_dimension': 1,
    'max_edge_length': None,
    'show_plots': False,
    'save_summary': False,
    'skip_wasserstein': True
}

def main():
    # train_robust_model(**ROBUST_TRAIN_CONFIG)

    batch_test_robust_model(**ROBUST_TEST_CONFIG)

    # batch_extract_measurements(**MEASUREMENT_CONFIG)
    
    # batch_tda_analysis(**TDA_CONFIG)

    # mds_from_wasserstein_csv('results/chosen_tda_results/wasserstein_distances_H1.csv', output_path='results/chosen_tda_results/mds_results.png', show=True)
    

if __name__ == "__main__":
    main()