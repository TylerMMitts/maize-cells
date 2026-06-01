from train_robust import train_robust_model
from test_robust import batch_test_robust_model
from cell_measurements import extract_cell_measurements, batch_extract_measurements, visualize_measurements
from visualize_cell_centers import visualize_cell_centers
from tda_analysis import batch_tda_analysis, analyze_single_image
from mds import mds_from_wasserstein_csv

EXCLUSION_WEIGHTS = "runs/segment/runs/exclusion_model/inner_part_detector/weights/best.pt"

ROBUST_TRAIN_CONFIG = {
    'dataset_yaml': 'cell_dataset/data.yaml',
    'model_size': 'yolov8l-seg.pt',
    'epochs': 100,
    'max_det': 3000,
    'batch_size': 4,
    'image_size': 640,
    'project': 'runs/robust_segmentation',
    'run_name': 'robust_cell_detector'
}

ROBUST_WEIGHTS = "runs/segment/runs/robust_segmentation/robust_cell_detector/weights/best.pt"

ROBUST_TEST_CONFIG = {
    'cell_weights': ROBUST_WEIGHTS,
    'exclusion_weights': EXCLUSION_WEIGHTS,
    'image_folder': 'root_results',
    'confidence': 0.98,
    'output_dir': 'test_results_robust',
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
    'exclusion_confidence': 0.25
}

MEASUREMENT_CONFIG = {
    'weights_path': ROBUST_WEIGHTS,
    'exclusion_weights': EXCLUSION_WEIGHTS,
    'image_folder': 'root_results',
    'confidence': 0.98,
    'exclusion_confidence': 0.25,
    'output_dir': 'measurements',
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
    'use_darkest_seed': False
}

TDA_CONFIG = {
    'measurements_dir': 'measurements',
    'output_dir': 'tda_results',
    'max_dimension': 1,
    'max_edge_length': None,
    'show_plots': False,
    'save_summary': False
}

def main():
    # train_robust_model(**ROBUST_TRAIN_CONFIG)

    # batch_test_robust_model(**ROBUST_TEST_CONFIG)

    # batch_extract_measurements(**MEASUREMENT_CONFIG)
    
    # batch_tda_analysis(**TDA_CONFIG)

    mds_from_wasserstein_csv('tda_results/wasserstein_distances_H1.csv', show=True)
    

if __name__ == "__main__":
    main()