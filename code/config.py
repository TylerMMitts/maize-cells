import os
from pathlib import Path
from typing import Dict, Any, Optional

# ============================================================
# PROJECT PATHS
# ============================================================

# Get the base directory (parent of code folder)
BASE_DIR = Path(__file__).parent.parent

# Data folders
DATA_FOLDER = BASE_DIR / "data"
CHOSEN_FOLDER = DATA_FOLDER / "chosen"
CHOSEN_RESULTS_FOLDER = DATA_FOLDER / "chosen_results"
CHOSEN_SOME_FOLDER = DATA_FOLDER / "chosen_some"
ROOT_RESULTS_FOLDER = DATA_FOLDER / "root_results"

# Model folders
MODELS_FOLDER = BASE_DIR / "models"
CELL_DATASET_FOLDER = MODELS_FOLDER / "cell_dataset"
ROOT_DATASET_FOLDER = MODELS_FOLDER / "root_dataset"
EXCLUSION_DATASET_FOLDER = MODELS_FOLDER / "exclusion_dataset"

# Results folders
RESULTS_FOLDER = BASE_DIR / "results"
MEASUREMENTS_FOLDER = RESULTS_FOLDER / "measurements_all"
CELL_FILE_FOLDER = RESULTS_FOLDER / "cell_file"
CELL_FILE_COUNTS_FOLDER = CELL_FILE_FOLDER / "cell_file_counting"
TDA_FOLDER = RESULTS_FOLDER / "tda"
NON_TDA_FOLDER = RESULTS_FOLDER / "non_tda"
FEATURE_ANALYSIS_FOLDER = RESULTS_FOLDER / "feature_analysis"

# ============================================================
# MODEL WEIGHTS
# ============================================================

# YOLO model weights
ROBUST_CELL_WEIGHTS = MODELS_FOLDER / "runs" / "segment" / "runs" / "robust_segmentation" / "robust_cell_detector" / "weights" / "best.pt"
EXCLUSION_WEIGHTS = MODELS_FOLDER / "runs" / "segment" / "runs" / "exclusion_model" / "inner_part_detector" / "weights" / "best.pt"
ROOT_DETECTION_WEIGHTS = MODELS_FOLDER / "runs" / "detect" / "runs" / "root_detection" / "root_detector" / "weights" / "best.pt"

# SAM weights
SAM_WEIGHTS = MODELS_FOLDER / "mobile_sam.pt"

# ============================================================
# CONVERSION FACTORS
# ============================================================

# Default conversion: 50 µm = 77 pixels
DEFAULT_PIXEL_TO_UM = 50.0 / 77.0
DEFAULT_PIXEL_TO_UM_SQUARED = DEFAULT_PIXEL_TO_UM * DEFAULT_PIXEL_TO_UM

# ============================================================
# PROCESSING PARAMETERS
# ============================================================

# SAM refinement settings
USE_SAM_REFINEMENT = False
SAM_MODEL_TYPE = 'mobilesam'
SAM_DEVICE = 'cuda'

SAM_CONFIG = {
    'model_type': SAM_MODEL_TYPE,
    'device': SAM_DEVICE,
    'verbose': True,
    'use_point_prompts': True,
    'use_box_prompts': True,
    'post_process': True,
    'close_kernel_size': 5,
    'open_kernel_size': 5,
    'min_mask_area': 50,
    'max_mask_area_ratio': 0.98
}

# Root detection settings
ROOT_DETECTION_CONFIG = {
    'confidence': 0.50,
    'save_quadrants': True
}

# Cell segmentation settings
CELL_SEGMENTATION_CONFIG = {
    'confidence': 0.98,
    'max_det': 3000,
    'refine': True,
    'color_tolerance': 15,
    'use_darkest_seed': False,
    'morph_post_process': True,
    'morph_close_kernel': 21,
    'morph_open_kernel': 21
}

# Exclusion (inner part) detection settings
EXCLUSION_CONFIG = {
    'confidence': 0.25,
    'filter_by_inner_part': True,
    'inner_open_kernel': 99,
    'inner_close_kernel': 155,
    'inner_open_iterations': 1,
    'inner_close_iterations': 1,
    'outer_open_kernel': 77,
    'outer_close_kernel': 99,
    'outer_open_iterations': 1,
    'outer_close_iterations': 1,
    'exclusion_overlap_threshold': 25.0,
    'keep_largest_component': True
}

# Cell overlap filtering
OVERLAP_CONFIG = {
    'overlap_threshold': 15.0
}

# Neighbor analysis settings
NEIGHBOR_CONFIG = {
    'expansion_pixels': 20,
    'max_neighbors': 7,
    'density_output_dir': RESULTS_FOLDER / "measurements" / "density_analysis"
}

# ============================================================
# ANALYSIS PARAMETERS
# ============================================================

# Spline fitting parameters
SPLINE_CONFIG = {
    'smoothing': 0.1,
    'min_cells_for_spline': 5,
    'n_profile_bins': 20
}

# Feature extraction
FEATURE_CONFIG = {
    'min_quadrants_for_root': 2,
    'features': [
        'peak_height',
        'peak_position',
        'rise_slope',
        'decay_slope',
        'outer_rise_magnitude'
    ],
    'feature_labels': {
        'peak_height': 'Peak Height',
        'peak_position': 'Peak Position',
        'rise_slope': 'Rise Slope',
        'decay_slope': 'Decay Slope',
        'outer_rise_magnitude': 'Outer-Rise Magnitude'
    }
}

# Binning parameters
BINNING_CONFIG = {
    'radius_bins': 15,
    'area_bins': 15,
    'norm_points': 20,
    'n_bins': 5  # For root radius and stele area analysis
}

# Angle sampling
ANGLE_CONFIG = {
    'tolerance': 5,
    'n_angles': 5
}

# Cell file analysis
CELL_FILE_CONFIG = {
    'method': 'derivative',  # 'derivative', 'gradient', or 'radial'
    'min_cells_per_file': 5
}

# TDA parameters
TDA_CONFIG = {
    'max_dimension': 1,
    'max_edge_length': None,
    'show_plots': False,
    'save_summary': False,
    'skip_wasserstein': True
}

# ============================================================
# OUTPUT SETTINGS
# ============================================================

# File naming
MASTER_SUMMARY_FILENAME = "master_summary.csv"
FEATURE_TABLE_FILENAME = "feature_table.csv"

# Full paths
MASTER_SUMMARY_PATH = RESULTS_FOLDER / MASTER_SUMMARY_FILENAME
FEATURE_TABLE_PATH = RESULTS_FOLDER / FEATURE_TABLE_FILENAME

# Visualization settings
VISUALIZATION_CONFIG = {
    'dpi': 300,
    'figure_size': (12, 8),
    'color_map': 'tab20',
    'alpha': 0.7,
    'line_width': 2,
    'marker_size': 6,
    'legend_threshold': 20
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# ============================================================
# PIPELINE SETTINGS
# ============================================================

PIPELINE_CONFIG = {
    'skip_existing': True,
    'verbose': True,
    'save_intermediate': True,
    'min_cells_per_image': 10,
    'force_rebuild_master': False,
    'force_rebuild_features': False
}

# ============================================================
# SPECIES AND POPULATION INFO
# ============================================================

METADATA_CONFIG = {
    'species': 'Zea mays',
    'population': 'IBM'
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_config_dict() -> Dict[str, Any]:
    return {
        'paths': {
            'base_dir': str(BASE_DIR),
            'data_folder': str(DATA_FOLDER),
            'models_folder': str(MODELS_FOLDER),
            'results_folder': str(RESULTS_FOLDER),
            'measurements_folder': str(MEASUREMENTS_FOLDER),
            'master_summary_path': str(MASTER_SUMMARY_PATH),
            'feature_table_path': str(FEATURE_TABLE_PATH)
        },
        'weights': {
            'robust_cell_weights': str(ROBUST_CELL_WEIGHTS),
            'exclusion_weights': str(EXCLUSION_WEIGHTS),
            'root_detection_weights': str(ROOT_DETECTION_WEIGHTS),
            'sam_weights': str(SAM_WEIGHTS)
        },
        'conversion': {
            'pixel_to_um': DEFAULT_PIXEL_TO_UM,
            'pixel_to_um_squared': DEFAULT_PIXEL_TO_UM_SQUARED
        },
        'sam': SAM_CONFIG,
        'root_detection': ROOT_DETECTION_CONFIG,
        'cell_segmentation': CELL_SEGMENTATION_CONFIG,
        'exclusion': EXCLUSION_CONFIG,
        'overlap': OVERLAP_CONFIG,
        'neighbor': NEIGHBOR_CONFIG,
        'spline': SPLINE_CONFIG,
        'feature': FEATURE_CONFIG,
        'binning': BINNING_CONFIG,
        'angle': ANGLE_CONFIG,
        'cell_file': CELL_FILE_CONFIG,
        'tda': TDA_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'pipeline': PIPELINE_CONFIG,
        'metadata': METADATA_CONFIG
    }


def ensure_directories_exist():

    directories = [
        DATA_FOLDER,
        CHOSEN_FOLDER,
        CHOSEN_RESULTS_FOLDER,
        MODELS_FOLDER,
        RESULTS_FOLDER,
        MEASUREMENTS_FOLDER,
        CELL_FILE_FOLDER,
        CELL_FILE_COUNTS_FOLDER,
        TDA_FOLDER,
        NON_TDA_FOLDER,
        FEATURE_ANALYSIS_FOLDER
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_output_folder(analysis_type: str) -> Path:

    folder_map = {
        'measurements': MEASUREMENTS_FOLDER,
        'cell_file': CELL_FILE_FOLDER,
        'cell_file_counting': CELL_FILE_COUNTS_FOLDER,
        'tda': TDA_FOLDER,
        'non_tda': NON_TDA_FOLDER,
        'feature_analysis': FEATURE_ANALYSIS_FOLDER,
        'neighbor_analysis': RESULTS_FOLDER / 'neighbor_analysis',
        'root_radius_analysis': RESULTS_FOLDER / 'root_radius_analysis',
        'stele_area_analysis': RESULTS_FOLDER / 'stele_area_analysis',
        'spline_visualizations': RESULTS_FOLDER / 'spline_visualizations',
        'fpca_results': RESULTS_FOLDER / 'fpca_results',
        'predictive_model': RESULTS_FOLDER / 'predictive_model',
        'cortical_reconstruction': RESULTS_FOLDER / 'cortical_reconstruction'
    }
    
    folder = folder_map.get(analysis_type, RESULTS_FOLDER / analysis_type)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


if __name__ != "__main__":
    pass 
