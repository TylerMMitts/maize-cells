from train_robust import train_robust_model
from test_robust import batch_test_robust_model
from cell_measurements import extract_cell_measurements, batch_extract_measurements, visualize_measurements
from visualize_cell_centers import visualize_cell_centers

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
    'image_folder': 'cropped_images',
    'confidence': 0.98,
    'output_dir': 'test_results_robust',
    'refine': True,
    'color_tolerance': 15,
    'use_darkest_seed': False,
    'max_det': 3000,
    'filter_by_inner_part': True,
    'morph_open_kernel': 99,
    'morph_close_kernel': 155,
    'morph_open_iterations': 1,
    'morph_close_iterations': 1
}

MEASUREMENT_CONFIG = {
    'weights_path': ROBUST_WEIGHTS,
    'exclusion_weights': EXCLUSION_WEIGHTS,
    'image_folder': 'cropped_images',
    'confidence': 0.99,
    'exclusion_confidence': 0.5,
    'output_dir': 'measurements',
    'filter_by_inner_part': True,
    'morph_open_kernel': 99,
    'morph_close_kernel': 155
}

def main():
    # train_robust_model(**ROBUST_TRAIN_CONFIG)

    batch_test_robust_model(**ROBUST_TEST_CONFIG)

    # batch_extract_measurements(**MEASUREMENT_CONFIG)
    

if __name__ == "__main__":
    main()