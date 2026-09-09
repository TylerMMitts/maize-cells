# The pipeline itself: every stage, in order, as one class.
#
# RootAnalysisPipeline owns the sequence from raw slide to feature table.
# Every stage is append-only, so re-running processes only what is new and
# existing rows keep the species and population already recorded against
# them. Imports its stage modules inside the methods rather than at the top,
# so one broken analysis cannot stop the whole pipeline from loading.

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
from tqdm import tqdm
import json
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration
from code import config

# Import utilities
from code.util.file_utils import (
    load_measurement_csv,
    load_image_summary,
    save_dataframe,
    update_master_summary,
    find_all_measurement_files
)
from code.util.data_utils import create_root_id
from code.config import CHOSEN_RESULTS_FOLDER, MEASUREMENTS_FOLDER

def setup_logging(verbose: bool = True) -> logging.Logger:

    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format=config.LOGGING_CONFIG['format'],
        datefmt=config.LOGGING_CONFIG['date_format']
    )
    
    return logging.getLogger(__name__)

class RootAnalysisPipeline:
    
    def __init__(
        self,
        new_images_folder: Optional[str] = None,
        output_folder: Optional[str] = None,
        skip_existing: bool = True,
        verbose: bool = True,
        config_overrides: Optional[Dict[str, Any]] = None,
        species: str = "Zea mays",
        population: str = "IBM",
        dataset_metadata: Optional[Dict[str, Any]] = None,
        force_rebuild_master_summary: bool = False,
        pixel_to_um_map: Optional[Dict[str, float]] = None,
        exclude_images: Optional[set] = None
    ):

        self.logger = setup_logging(verbose)
        self.verbose = verbose
        self.skip_existing = skip_existing
        self.species = species
        self.population = population
        self.dataset_metadata = dataset_metadata or {}
        self.force_rebuild_master_summary = force_rebuild_master_summary
        self.pixel_to_um_map = pixel_to_um_map
        self.exclude_images = exclude_images or set()
        
        # Setup paths
        self.new_images_folder = Path(new_images_folder) if new_images_folder else None
        self.output_folder = Path(output_folder) if output_folder else config.RESULTS_FOLDER
        
        # Apply config overrides
        if config_overrides:
            self._apply_config_overrides(config_overrides)
        
        # Initialize state
        self.master_df: Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None
        self.processing_summary: Dict[str, Any] = {}
        self.run_log: List[str] = []
        
        # Track processed files
        self.processed_images: List[str] = []
        self.failed_images: List[str] = []
        self.new_quadrant_images: List[str] = []  # Track newly created quadrant images
        
        # QC state
        self.qc_manager = None
        self.deleted_images: List[str] = []
        
        if self.new_images_folder:
            self.logger.info(f"New images folder: {self.new_images_folder}")
        self.logger.info(f"Output folder: {self.output_folder}")
        self.logger.info(f"Skip existing: {self.skip_existing}")
        self.logger.info(f"Species: {self.species}")
        self.logger.info(f"Population: {self.population}")
        if self.force_rebuild_master_summary:
            self.logger.info("** FORCE REBUILD MASTER SUMMARY: Will rebuild from scratch with new population **")
        if self.dataset_metadata:
            self.logger.info(f"Additional metadata: {self.dataset_metadata}")
        if self.pixel_to_um_map:
            self.logger.info(f"Per-image scale map: {len(self.pixel_to_um_map)} images "
                              f"(others fall back to config.DEFAULT_PIXEL_TO_UM)")
        if self.exclude_images:
            self.logger.info(f"Excluding {len(self.exclude_images)} images from processing "
                              f"(unresolvable scale or other reason)")
        
    def _apply_config_overrides(self, overrides: Dict[str, Any]):
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                self.logger.debug(f"Config override: {key} = {value}")
    
    def _log_step(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.run_log.append(log_entry)
        self.logger.info(message)
    
    def step1_split_images(self) -> bool:

        self._log_step("STEP 1: Splitting images into quadrants")
        
        if not self.new_images_folder or not self.new_images_folder.exists():
            self.logger.warning("No new images folder specified or folder doesn't exist")
            return False
        
        try:
            from code.yolo.test_root_model import test_root_model
            
            # Find all images in the folder (case-insensitive)
            image_extensions = ['.jpg', '.png', '.jpeg', '.tif', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(self.new_images_folder.glob(f'*{ext}')))
                image_files.extend(list(self.new_images_folder.glob(f'*{ext.upper()}')))
            
            # Remove duplicates
            image_files = list(set(image_files))

            # Exclude any images flagged by the caller (e.g. unresolvable scale)
            if self.exclude_images:
                before_count = len(image_files)
                excluded_here = [p for p in image_files if p.stem in self.exclude_images]
                image_files = [p for p in image_files if p.stem not in self.exclude_images]
                if excluded_here:
                    self.logger.warning(f"Excluding {len(excluded_here)} of {before_count} images "
                                         f"from this run: {sorted(p.name for p in excluded_here)}")

            self.logger.info(f"Found {len(image_files)} images to process")
            
            # Use absolute path for chosen_results folder
            # Get the project root directory (where data/ folder is)
            project_root = Path(__file__).parent.parent  # Goes up from code/ to project root
            chosen_results_folder = project_root / "data" / "chosen_results"
            chosen_results_folder.mkdir(parents=True, exist_ok=True)
            
            detections_folder = chosen_results_folder / "detections"
            detections_folder.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving cropped full images to: {detections_folder}")
            self.logger.info(f"Saving quadrants to: {chosen_results_folder}")
            
            # Process each image
            for image_path in tqdm(image_files, desc="Splitting images", disable=not self.verbose):
                try:
                    base_name = image_path.stem
                    
                    # Get list of existing quadrants BEFORE processing
                    existing_quadrants = set()
                    if chosen_results_folder.exists():
                        for f in chosen_results_folder.glob(f"*{base_name}*"):
                            if f.is_file() and not f.name.endswith('_full.jpg') and not f.name.startswith('detected_') and not f.name.endswith('_metadata.json'):
                                existing_quadrants.add(f.name)
                    
                    self.logger.debug(f"Existing quadrants for {base_name}: {len(existing_quadrants)}")
                    
                    # Run root detection - output goes to data/chosen_results/
                    # Pass the absolute path to test_root_model
                    test_root_model(
                        weights_path=str(config.ROOT_DETECTION_WEIGHTS),
                        image_path=str(image_path),
                        confidence=config.ROOT_DETECTION_CONFIG['confidence'],
                        output_dir=str(chosen_results_folder),  # Now this is absolute path
                        save_quadrants=True
                    )
                    
                    # Track newly created quadrants
                    if chosen_results_folder.exists():
                        # Get all files after processing
                        all_files = set()
                        for f in chosen_results_folder.glob(f"*{base_name}*"):
                            if f.is_file() and not f.name.endswith('_full.jpg') and not f.name.startswith('detected_') and not f.name.endswith('_metadata.json'):
                                all_files.add(f.name)
                        
                        # Find newly created files
                        newly_created = all_files - existing_quadrants
                        
                        # Filter to only quadrant images (start with BL_, BR_, TL_, TR_)
                        valid_quadrants = []
                        for f_name in newly_created:
                            if f_name.startswith(('BL_', 'BR_', 'TL_', 'TR_')) and '_root' in f_name:
                                valid_quadrants.append(f_name)
                        
                        if valid_quadrants:
                            self.new_quadrant_images.extend(valid_quadrants)
                            self.logger.info(f"Found {len(valid_quadrants)} new quadrants for {base_name}")
                            self.logger.debug(f"  New quadrants: {valid_quadrants}")
                        
                        self.processed_images.append(str(image_path))
                        
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")
                    self.failed_images.append(str(image_path))
            
            self.logger.info(f"Successfully split {len(self.processed_images)} images into {len(self.new_quadrant_images)} quadrants")
            
            if self.new_quadrant_images:
                self.logger.info(f"New quadrants: {', '.join(list(self.new_quadrant_images)[:5])}{'...' if len(self.new_quadrant_images) > 5 else ''}")
            else:
                self.logger.warning("No new quadrants were created")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in step1_split_images: {e}", exc_info=True)
            return False
        
    
    def step2_extract_cells(self, quarters_folder: Optional[str] = None) -> bool:

        self._log_step("STEP 2: Extracting cell measurements")
        
        # Check if there are new images to process
        if hasattr(self, 'new_quadrant_images') and not self.new_quadrant_images:
            self.logger.info("No new quadrant images to process, skipping cell extraction")
            return True
        
        # Use absolute path for chosen_results folder
        if quarters_folder is None:
            project_root = Path(__file__).parent.parent
            quarters_folder = project_root / "data" / "chosen_results"
        else:
            quarters_folder = Path(quarters_folder)
        
        if not quarters_folder.exists():
            self.logger.warning(f"Quarters folder doesn't exist: {quarters_folder}")
            return False
        
        try:
            from code.measurements.cell_measurements import batch_extract_measurements, set_conversion_factors
            
            measurements_output = self.output_folder / "measurements_all"
            measurements_output.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving measurements to: {measurements_output}")
            self.logger.info(f"Looking for quadrant images in: {quarters_folder}")
            self.logger.info(f"Processing {len(self.new_quadrant_images) if self.new_quadrant_images else 'all'} quadrant images")
            
            # Get conversion factor from config
            pixel_to_um = getattr(config, 'DEFAULT_PIXEL_TO_UM', 50.0 / 77.0)
            pixel_to_um_squared = getattr(config, 'DEFAULT_PIXEL_TO_UM_SQUARED', pixel_to_um * pixel_to_um)
            
            self.logger.info(f"Using conversion factor: pixel_to_um={pixel_to_um} "
                              f"(default fallback{' + per-image scale map' if self.pixel_to_um_map else ''})")

            # Set global conversion factors in the measurements module
            set_conversion_factors(pixel_to_um, pixel_to_um_squared)
            
            # Configure measurement extraction
            measurement_config = {
                'weights_path': str(config.ROBUST_CELL_WEIGHTS),
                'exclusion_weights': str(config.EXCLUSION_WEIGHTS),
                'image_folder': str(quarters_folder),
                'output_dir': str(measurements_output),
                'specific_images': self.new_quadrant_images if hasattr(self, 'new_quadrant_images') else None,
                'confidence': config.CELL_SEGMENTATION_CONFIG['confidence'],
                'exclusion_confidence': config.EXCLUSION_CONFIG['confidence'],
                'filter_by_inner_part': config.EXCLUSION_CONFIG['filter_by_inner_part'],
                'inner_open_kernel': config.EXCLUSION_CONFIG['inner_open_kernel'],
                'inner_close_kernel': config.EXCLUSION_CONFIG['inner_close_kernel'],
                'outer_open_kernel': config.EXCLUSION_CONFIG['outer_open_kernel'],
                'outer_close_kernel': config.EXCLUSION_CONFIG['outer_close_kernel'],
                'exclusion_overlap_threshold': config.EXCLUSION_CONFIG['exclusion_overlap_threshold'],
                'overlap_threshold': config.OVERLAP_CONFIG['overlap_threshold'],
                'keep_largest_component': config.EXCLUSION_CONFIG['keep_largest_component'],
                'refine': config.CELL_SEGMENTATION_CONFIG['refine'],
                'color_tolerance': config.CELL_SEGMENTATION_CONFIG['color_tolerance'],
                'use_darkest_seed': config.CELL_SEGMENTATION_CONFIG['use_darkest_seed'],
                'morph_close_kernel': config.CELL_SEGMENTATION_CONFIG['morph_close_kernel'],
                'morph_open_kernel': config.CELL_SEGMENTATION_CONFIG['morph_open_kernel'],
                'morph_post_process': config.CELL_SEGMENTATION_CONFIG['morph_post_process'],
                'sam_refiner': None,
                'density_output_dir': str(config.NEIGHBOR_CONFIG['density_output_dir']),
                'expansion_pixels': config.NEIGHBOR_CONFIG['expansion_pixels'],
                'max_neighbors': config.NEIGHBOR_CONFIG['max_neighbors'],
                'create_master_summary_flag': False,
                'pixel_to_um': pixel_to_um,
                'pixel_to_um_squared': pixel_to_um_squared,
                'pixel_to_um_map': self.pixel_to_um_map,
            }
            
            batch_extract_measurements(**measurement_config)
            
            self.logger.info("Cell measurements extracted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in step2_extract_cells: {e}", exc_info=True)
            return False
    
    def step2_5_quality_control(
        self,
        measurements_folder: Optional[str] = None,
        interactive: bool = True,
        auto_delete: bool = False,
        min_cells_threshold: int = 10,
        max_cells_threshold: int = 5000,
        port: int = 8888
    ) -> bool:

        print(f"hasattr new_quadrant_images: {hasattr(self, 'new_quadrant_images')}")
        if hasattr(self, 'new_quadrant_images'):
            print(f"new_quadrant_images: {self.new_quadrant_images}")
            print(f"len: {len(self.new_quadrant_images)}")
        else:
            print(f"new_quadrant_images: NOT SET")
        print(f"interactive: {interactive}")
        print(f"auto_delete: {auto_delete}")
        print(f"port: {port}")
        
        self._log_step("STEP 2.5: Quality Control (New Images Only)")
        
        # Check if there are new images to review
        if not hasattr(self, 'new_quadrant_images') or not self.new_quadrant_images:
            self.logger.info("No new quadrant images to review, skipping QC")
            print("QC SKIPPED - No new quadrant images")
            return True
        
        if measurements_folder is None:
            measurements_folder = str(config.MEASUREMENTS_FOLDER)
        
        measurements_folder = Path(measurements_folder)
        
        if not measurements_folder.exists():
            self.logger.error(f"Measurements folder not found: {measurements_folder}")
            return False
        
        try:
            from code.quality_control.qc_manager import QCManager
            
            # DEBUG: Print what quadrants we have
            self.logger.info(f"DEBUG: new_quadrant_images = {self.new_quadrant_images}")
            print(f"\nDEBUG: new_quadrant_images = {self.new_quadrant_images}\n")
            
            # Extract base names from new quadrant images
            valid_new_images = []
            
            for quad in self.new_quadrant_images:
                # Remove the extension to get the base name
                base_name = Path(quad).stem
                
                # Check if measurement files exist
                csv_path = measurements_folder / f"{base_name}_measurements.csv"
                centers_path = measurements_folder / f"{base_name}_centers.jpg"
                
                print(f"Checking: {base_name}")
                print(f"CSV exists: {csv_path.exists()} - {csv_path}")
                print(f"Centers exists: {centers_path.exists()} - {centers_path}")
                
                if csv_path.exists() and centers_path.exists():
                    valid_new_images.append(base_name)
                    self.logger.debug(f"{base_name} has measurement files")
                else:
                    self.logger.debug(f"{base_name} - measurement files not found")
                    print(f"{base_name} - measurement files not found")
            
            if not valid_new_images:
                self.logger.warning("No new images with complete measurement files found for QC")
                print("QC SKIPPED - No valid new images with measurement files")
                return True
            
            self.logger.info(f"QC will review {len(valid_new_images)} images")
            self.logger.info(f"Images: {', '.join(sorted(valid_new_images))}")
            print(f"QC WILL REVIEW {len(valid_new_images)} IMAGES:")
            for img in sorted(valid_new_images):
                print(f"  - {img}")
            
            # Create QC Manager with only new images
            self.qc_manager = QCManager(
                measurements_folder=str(measurements_folder),
                output_folder=str(self.output_folder),
                new_quadrant_images=valid_new_images,
                min_cells_threshold=min_cells_threshold,
                max_cells_threshold=max_cells_threshold,
                port=port
            )
            
            # Run quality control
            qc_df = self.qc_manager.run_quality_control(
                auto_delete=auto_delete,
                interactive=interactive
            )
            
            # Store deleted images
            self.deleted_images = self.qc_manager.get_deleted_images()
            
            # If images were deleted, remove them from new_quadrant_images
            if self.deleted_images:
                self.logger.info(f"Deleted {len(self.deleted_images)} images during QC")
                
                # Remove deleted images from new_quadrant_images
                deleted_set = set(self.deleted_images)
                self.new_quadrant_images = [
                    img for img in self.new_quadrant_images
                    if Path(img).stem not in deleted_set
                ]
                
                # Delete the actual files
                for img_name in self.deleted_images:
                    csv_path = measurements_folder / f"{img_name}_measurements.csv"
                    if csv_path.exists():
                        csv_path.unlink()
                        self.logger.debug(f"Deleted: {csv_path}")
                    
                    centers_path = measurements_folder / f"{img_name}_centers.jpg"
                    if centers_path.exists():
                        centers_path.unlink()
                        self.logger.debug(f"Deleted: {centers_path}")
                    
                    quadrant_dir = Path(CHOSEN_RESULTS_FOLDER)
                    if quadrant_dir.exists():
                        for ext in ['.jpg', '.png', '.jpeg']:
                            quad_path = quadrant_dir / f"{img_name}{ext}"
                            if quad_path.exists():
                                quad_path.unlink()
                                self.logger.debug(f"Deleted quadrant: {quad_path}")
                
                self.logger.info(f"Removed {len(self.deleted_images)} images from dataset")
            else:
                self.logger.info("No images were deleted during quality control")
            
            # Clean up the QC folder if it's empty
            qc_folder = self.output_folder / "quality_control"
            if qc_folder.exists() and not any(qc_folder.iterdir()):
                qc_folder.rmdir()
                self.logger.debug("Removed empty QC folder")
            
            return True
            
        except ImportError as e:
            self.logger.error(f"Quality control module not available: {e}")
            self.logger.error("Make sure qc_manager.py and qc_server.py are in code/quality_control/")
            return False
        except Exception as e:
            self.logger.error(f"Error in quality control: {e}", exc_info=True)
            print(f"QC ERROR: {e}")
            return False
                        
    def step3_update_master_summary(
        self,
        measurements_folder: Optional[str] = None,
        force_rebuild: Optional[bool] = None
    ) -> bool:

        self._log_step("STEP 3: Updating master summary (append-only)")
        
        # Use instance setting if force_rebuild not explicitly provided
        if force_rebuild is None:
            force_rebuild = self.force_rebuild_master_summary
        
        if measurements_folder is None:
            measurements_folder = str(config.MEASUREMENTS_FOLDER)
        
        try:
            master_summary_path = self.output_folder / "master_summary.csv"
            
            self.logger.warning("STEP 3: UPDATE MASTER SUMMARY")
            self.logger.warning(f"Reading measurements from: {measurements_folder}")
            self.logger.warning(f"Using species: '{self.species}'")
            self.logger.warning(f"Using population: '{self.population}'")
            if self.force_rebuild_master_summary:
                self.logger.warning("** FORCE REBUILD: Will rebuild from scratch **")
            
            # Warn if rebuilding
            if force_rebuild:
                self.logger.warning("REBUILDING master summary from scratch with new population settings!")
            
            # Update (append) master summary with species, population, and metadata
            self.master_df = update_master_summary(
                measurements_folder=measurements_folder,
                cell_file_counts_folder=str(config.CELL_FILE_COUNTS_FOLDER) if config.CELL_FILE_COUNTS_FOLDER.exists() else None,
                output_path=str(master_summary_path),
                force_rebuild=force_rebuild,
                species=self.species,
                population=self.population,
                dataset_metadata=self.dataset_metadata
            )
            
            self.logger.info(f"Master summary updated: {len(self.master_df)} total rows")
            self.logger.info(f"Saved to: {master_summary_path}")
            self.logger.info(f"Species: {self.species}, Population: {self.population}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in step3_update_master_summary: {e}", exc_info=True)
            return False
    
    def step4_analyze_cell_files(
        self, 
        measurements_folder: Optional[str] = None,
        force_rebuild: bool = False
    ) -> bool:

        self._log_step("STEP 4: Analyzing cell files (incremental)")
        
        if measurements_folder is None:
            measurements_folder = str(config.MEASUREMENTS_FOLDER)
        
        master_summary_path = self.output_folder / "master_summary.csv"
        
        if not master_summary_path.exists():
            self.logger.warning("Master summary not found. Please run Step 3 first.")
            return False
        
        try:
            import pandas as pd
            df = pd.read_csv(master_summary_path)
            
            # Determine which images need cell file analysis
            images_to_process = []
            
            # Check if force_rebuild is True
            if force_rebuild:
                # Process all images
                images_to_process = df['image_name'].tolist()
                self.logger.warning(f"FORCE REBUILD: Processing all {len(images_to_process)} images for cell file analysis")
            else:
                # Find images with file_count=0 that need cell file analysis
                zero_file_images = df[df['file_count'] == 0]['image_name'].tolist()
                
                if not zero_file_images:
                    self.logger.info("All images already have cell file data (file_count > 0)")
                    return True
                
                self.logger.info(f"Found {len(zero_file_images)} images with file_count=0 that need cell file analysis")
                
                # Check which images already have cell file data in the output folder
                output_folder = self.output_folder / "cell_file" / "cell_file_counting"
                existing_data = set()
                
                if output_folder.exists():
                    for img in zero_file_images:
                        # Check for derivative method file
                        cell_file_path = output_folder / img / f"{img}_method_derivative.csv"
                        if cell_file_path.exists():
                            existing_data.add(img)
                        # Also check for legacy cell_assignments.csv
                        elif (output_folder / img / "cell_assignments.csv").exists():
                            existing_data.add(img)
                
                if existing_data:
                    self.logger.info(f"  {len(existing_data)} images already have cell file data, skipping")
                    zero_file_images = [img for img in zero_file_images if img not in existing_data]
                
                images_to_process = zero_file_images
            
            if not images_to_process:
                self.logger.info("No new images need cell file analysis")
                return True
            
            self.logger.info(f"Processing {len(images_to_process)} images for cell file analysis")
            
            # Filter to images that have _centers.jpg files
            image_folder = Path(measurements_folder)
            valid_images = []
            skipped_images = []
            
            for img_name in images_to_process:
                # Check measurement file exists
                csv_path = os.path.join(measurements_folder, f"{img_name}_measurements.csv")
                if not os.path.exists(csv_path):
                    skipped_images.append(f"{img_name} (no measurement file)")
                    continue
                
                # Check _centers.jpg exists in measurements_all
                centers_path = image_folder / f"{img_name}_centers.jpg"
                if centers_path.exists():
                    valid_images.append(img_name)
                else:
                    skipped_images.append(f"{img_name} (no _centers.jpg)")
            
            self.logger.info(f"Images to process: {len(valid_images)}")
            if skipped_images:
                self.logger.info(f"Skipped images: {len(skipped_images)}")
                for skipped in skipped_images[:5]:  # Show first 5
                    self.logger.debug(f"  - {skipped}")
                if len(skipped_images) > 5:
                    self.logger.debug(f"  ... and {len(skipped_images) - 5} more")
            
            if not valid_images:
                self.logger.warning("No images with _centers.jpg found to process")
                return True
            
            # Show which images will be processed
            self.logger.info(f"Processing {len(valid_images)} images:")
            for img in valid_images[:10]:
                self.logger.info(f"  - {img}")
            if len(valid_images) > 10:
                self.logger.info(f"  ... and {len(valid_images) - 10} more")
            
            # Run cell file analysis
            from code.cell_file.cell_file_analyzer import batch_analyze_cell_files
            
            output_folder = self.output_folder / "cell_file" / "cell_file_counting"
            output_folder.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Using image folder (for _centers.jpg): {image_folder}")
            
            # Pass force_rebuild to the analyzer
            batch_analyze_cell_files(
                measurements_folder=measurements_folder,
                output_folder=str(output_folder),
                method=config.CELL_FILE_CONFIG['method'],
                specific_images=valid_images,
                image_folder=str(image_folder),
                force_rebuild=force_rebuild
            )
            
            self.logger.info("Cell file analysis completed")
            
            # Update master summary with cell file data
            self.logger.info("Updating master summary with cell file data...")
            
            from code.util.file_utils import update_master_summary
            
            update_master_summary(
                measurements_folder=measurements_folder,
                cell_file_counts_folder=str(config.CELL_FILE_COUNTS_FOLDER) if config.CELL_FILE_COUNTS_FOLDER.exists() else None,
                output_path=str(master_summary_path),
                force_rebuild=False,
                species=self.species,
                population=self.population,
                dataset_metadata=self.dataset_metadata
            )
            
            self.logger.info(f"Master summary updated with cell file data")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in step4_analyze_cell_files: {e}", exc_info=True)
            return False

    
    def step5_calculate_neighbors(
        self, 
        measurements_folder: Optional[str] = None,
        force_recalc: bool = False
    ) -> bool:

        self._log_step("STEP 5: Calculating neighbor information")
        
        # If force_recalc is True, process all images
        # Otherwise, only process new images
        if not force_recalc and hasattr(self, 'new_quadrant_images') and not self.new_quadrant_images:
            self.logger.info("No new quadrant images to process, skipping neighbor calculation")
            return True
        
        if measurements_folder is None:
            measurements_folder = str(config.MEASUREMENTS_FOLDER)
        
        try:
            from code.measurements.mask_neighbors import batch_calculate_neighbors
            
            # Determine which images to process
            if force_recalc:
                specific_images = None
                self.logger.warning("FORCE RECALC: Calculating neighbors for ALL images")
            else:
                specific_images = self.new_quadrant_images if hasattr(self, 'new_quadrant_images') else None
            
            batch_calculate_neighbors(
                measurements_folder=measurements_folder,
                expansion_pixels=config.NEIGHBOR_CONFIG['expansion_pixels'],
                specific_images=specific_images,
                force_recalc=force_recalc
            )
            
            self.logger.info("Neighbor calculation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in step5_calculate_neighbors: {e}", exc_info=True)
            return False
    
    def step6_update_feature_table(
        self,
        master_summary_path: Optional[str] = None,
        force_rebuild: bool = False
    ) -> bool:

        self._log_step("STEP 6: Updating feature table (append-only)")
        
        if master_summary_path is None:
            master_summary_path = str(self.output_folder / "master_summary.csv")
        
        try:
            from code.measurements.build_feature_table import build_feature_table_incremental
            
            feature_table_path = self.output_folder / "feature_table.csv"
            
            # Build/update feature table
            self.feature_df = build_feature_table_incremental(
                master_summary_path=master_summary_path,
                measurements_folder=str(config.MEASUREMENTS_FOLDER),
                output_path=str(feature_table_path),
                force_rebuild=force_rebuild
            )
            
            self.logger.info(f"Feature table updated: {len(self.feature_df)} total roots")
            self.logger.info(f"Saved to: {feature_table_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in step6_update_feature_table: {e}", exc_info=True)
            return False
    
    def step7_run_analysis(self) -> bool:

        self._log_step("STEP 7: Running analysis modules")
        
        # Build list of analyses with their import paths and functions
        analyses_to_run = [
            # Measurements analyses
            ('Feature Analysis', self._run_feature_analysis),
            ('Spline Visualization', self._run_spline_visualization),
            ('Neighbor Analysis', self._run_neighbor_analysis),
            ('Root Radius Analysis', self._run_root_radius_analysis),
            ('Stele Area Analysis', self._run_stele_area_analysis),
            ('Degree vs Area Analysis', self._run_degree_vs_area_analysis),
            
            # Statistics/FPCA analyses
            ('FPCA Analysis', self._run_fpca_analysis),
            ('File Level Model', self._run_file_level_model),
            ('Cortex Reconstruction', self._run_cortex_reconstruction),
            ('Cell File Group Analysis', self._run_cell_file_group_analysis),
            ('Predictive Model Stele Area', self._run_predictive_model_stele),
            ('Normalized Pattern Fit', self._run_normalized_pattern_fit),
            
            # Statistical analyses
            ('Stratified Analysis', self._run_stratified_analysis),
            ('Factor Analysis', self._run_factor_analysis),
            ('Interaction Plots', self._run_interaction_plots),

            # Mixed-effects models (run last)
            ('Mixed Model Size Scaling', self._run_mixed_model_size_scaling),
            ('Mixed Model Factor Contribution', self._run_mixed_model_factor_contribution),
        ]
        
        successful = []
        failed = []
        
        for name, func in analyses_to_run:
            try:
                self.logger.info(f"Running {name}...")
                result = func()
                if result:
                    successful.append(name)
                    self.logger.info(f"{name} completed successfully")
                else:
                    failed.append(name)
                    self.logger.warning(f"{name} failed or was skipped")
            except Exception as e:
                self.logger.error(f"Error in {name}: {e}", exc_info=True)
                failed.append(name)
        
        self.logger.info(f"Analysis complete: {len(successful)} successful, {len(failed)} failed")
        
        # Log which analyses succeeded and failed
        if successful:
            self.logger.info(f"Successful: {', '.join(successful)}")
        if failed:
            self.logger.warning(f"Failed: {', '.join(failed)}")
        
        return len(failed) == 0
    
    def _run_feature_analysis(self) -> bool:
        try:
            # Import the module
            import importlib
            spec = importlib.util.find_spec('code.measurements.analyze_features')
            if spec is None:
                self.logger.warning("analyze_features module not available")
                return False
            
            # Use the main function from analyze_features.py
            from code.measurements.analyze_features import main as run_analyze_features
            
            # Override paths if needed
            import code.measurements.analyze_features as af
            af.FEATURE_TABLE_PATH = str(self.output_folder / "feature_table.csv")
            af.OUTPUT_FOLDER = str(self.output_folder / "feature_variations")
            
            run_analyze_features()
            return True
            
        except ImportError as e:
            self.logger.warning(f"analyze_features module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in feature analysis: {e}")
            return False
    
    def _run_spline_visualization(self) -> bool:
        try:
            from code.measurements.visualize_splines import main as run_visualize_splines
            
            # Override paths
            import code.measurements.visualize_splines as vs
            vs.MASTER_SUMMARY_PATH = str(self.output_folder / "master_summary.csv")
            vs.MEASUREMENTS_FOLDER = str(config.MEASUREMENTS_FOLDER)
            vs.OUTPUT_FOLDER = str(self.output_folder / "combined_cubic_fit")
            vs.INDIVIDUAL_PLOTS_RADIUS = str(self.output_folder / "combined_cubic_fit" / "individual_plots_radius")
            vs.INDIVIDUAL_PLOTS_FILE = str(self.output_folder / "combined_cubic_fit" / "individual_plots_file")
            
            run_visualize_splines()
            return True
            
        except ImportError as e:
            self.logger.warning(f"visualize_splines module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in spline visualization: {e}")
            return False
    
    def _run_neighbor_analysis(self) -> bool:
        try:
            from code.measurements.neighbor_analysis import main as run_neighbor_analysis
            
            # Override paths
            import code.measurements.neighbor_analysis as na
            na.MEASUREMENTS_FOLDER = str(config.MEASUREMENTS_FOLDER)
            na.CELL_FILE_COUNTS_FOLDER = str(config.CELL_FILE_COUNTS_FOLDER) if config.CELL_FILE_COUNTS_FOLDER.exists() else str(self.output_folder / "cell_file" / "cell_file_counting")
            na.OUTPUT_FOLDER = str(self.output_folder / "neighbor_analysis")
            
            run_neighbor_analysis()
            return True
            
        except ImportError as e:
            self.logger.warning(f"neighbor_analysis module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in neighbor analysis: {e}")
            return False
    
    def _run_root_radius_analysis(self) -> bool:
        try:
            from code.measurements.root_radius_analysis import main as run_root_radius_analysis
            
            # Override paths
            import code.measurements.root_radius_analysis as rr
            rr.MEASUREMENTS_FOLDER = str(config.MEASUREMENTS_FOLDER)
            rr.OUTPUT_FOLDER = str(self.output_folder / "root_radius_analysis")
            
            run_root_radius_analysis()
            return True
            
        except ImportError as e:
            self.logger.warning(f"root_radius_analysis module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in root radius analysis: {e}")
            return False
    
    def _run_stele_area_analysis(self) -> bool:
        try:
            from code.measurements.stele_area_analysis import main as run_stele_area_analysis
            
            # Override paths
            import code.measurements.stele_area_analysis as sa
            sa.MEASUREMENTS_FOLDER = str(config.MEASUREMENTS_FOLDER)
            sa.OUTPUT_FOLDER = str(self.output_folder / "stele_area_analysis")
            
            run_stele_area_analysis()
            return True
            
        except ImportError as e:
            self.logger.warning(f"stele_area_analysis module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in stele area analysis: {e}")
            return False
    
    def _run_degree_vs_area_analysis(self) -> bool:
        try:
            from code.measurements.analyze_degree_vs_area import main as run_degree_analysis
            
            # Override paths
            import code.measurements.analyze_degree_vs_area as da
            # The module uses sys.argv, so we need to set it
            import sys
            sys.argv = [sys.argv[0], str(config.MEASUREMENTS_FOLDER), str(self.output_folder / "degree_analysis")]
            
            run_degree_analysis()
            return True
            
        except ImportError as e:
            self.logger.warning(f"analyze_degree_vs_area module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in degree vs area analysis: {e}")
            return False
    
    def _run_fpca_analysis(self) -> bool:
        try:
            from code.statistics.run_fpca import main as run_fpca
            
            # Override paths
            import code.statistics.run_fpca as fpca
            fpca.MASTER_SUMMARY_PATH = str(self.output_folder / "master_summary.csv")
            fpca.MEASUREMENTS_FOLDER = str(config.MEASUREMENTS_FOLDER)
            fpca.OUTPUT_FOLDER = str(self.output_folder / "fpca_results")
            
            run_fpca()
            return True
            
        except ImportError as e:
            self.logger.warning(f"run_fpca module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in FPCA analysis: {e}")
            return False
    
    def _run_file_level_model(self) -> bool:
        try:
            from code.statistics.cell_area_per_file import main as run_cell_area_per_file
            
            # The module uses hardcoded paths, so we need to run it directly
            # It will look for results/cell_file/cell_file_counting/
            run_cell_area_per_file()
            return True
            
        except ImportError as e:
            self.logger.warning(f"cell_area_per_file module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in file level model: {e}")
            return False
    
    def _run_cortex_reconstruction(self) -> bool:
        try:
            from code.statistics.cortex_reconstruction import main as run_cortex_reconstruction
            
            # The module uses hardcoded paths
            run_cortex_reconstruction()
            return True
            
        except ImportError as e:
            self.logger.warning(f"cortex_reconstruction module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in cortex reconstruction: {e}")
            return False
    
    def _run_cell_file_group_analysis(self) -> bool:
        try:
            from code.cell_file.cell_file_group_analysis import main as run_cell_file_group_analysis

            # The module uses PROJECT_ROOT-based absolute defaults
            run_cell_file_group_analysis()
            return True

        except ImportError as e:
            self.logger.warning(f"cell_file_group_analysis module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in cell file group analysis: {e}")
            return False

    def _run_predictive_model_stele(self) -> bool:
        try:
            from code.statistics.predictive_model_stele_area import main as run_predictive_model
            
            # The module uses hardcoded paths
            run_predictive_model()
            return True
            
        except ImportError as e:
            self.logger.warning(f"predictive_model_stele_area module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in predictive model: {e}")
            return False
    
    def _run_normalized_pattern_fit(self) -> bool:
        try:
            from code.non_tda.fit_normalized_pattern import fit_normalized_pattern_from_csv
            
            csv_path = str(self.output_folder / "non_tda" / "non_tda_results_huge" / "processed_data.csv")
            output_folder = str(self.output_folder / "non_tda" / "non_tda_results_huge")
            
            # Check if CSV exists
            if not os.path.exists(csv_path):
                self.logger.warning(f"Normalized pattern CSV not found: {csv_path}")
                return False
            
            fit_normalized_pattern_from_csv(csv_path, output_folder)
            return True
            
        except ImportError as e:
            self.logger.warning(f"fit_normalized_pattern module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in normalized pattern fit: {e}")
            return False
    
    def _run_stratified_analysis(self) -> bool:
        try:
            from code.statistics.stratified_analysis import main as run_stratified
            output_dir = str(self.output_folder / "stratified_analysis")
            run_stratified(
                master_path=str(self.output_folder / "master_summary.csv"),
                cell_file_dir=str(self.output_folder / "cell_file" / "cell_file_counting"),
                output_dir=output_dir,
                factors=['treatment', 'root_type', 'population']
            )
            self.logger.info(f"Stratified analysis completed. Results saved to: {output_dir}")
            return True
        except ImportError as e:
            self.logger.warning(f"stratified_analysis module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in stratified analysis: {e}")
            return False
    
    def _run_factor_analysis(self) -> bool:
        try:
            from code.statistics.factor_analysis import main as run_factor_analysis
            output_dir = str(self.output_folder / "factor_analysis")
            run_factor_analysis(
                master_path=str(self.output_folder / "master_summary.csv"),
                cell_file_dir=str(self.output_folder / "cell_file" / "cell_file_counting"),
                output_dir=output_dir,
                factors=['treatment', 'root_type', 'population']
            )
            self.logger.info(f"Factor analysis completed. Results saved to: {output_dir}")
            return True
        except ImportError as e:
            self.logger.warning(f"factor_analysis module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in factor analysis: {e}")
            return False

    def _run_mixed_model_size_scaling(self) -> bool:
        try:
            from code.statistics.mixed_model_size_scaling import main as run_size_scaling_model

            run_size_scaling_model()
            return True
        except ImportError as e:
            self.logger.warning(f"mixed_model_size_scaling module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in mixed model size scaling: {e}")
            return False

    def _run_mixed_model_factor_contribution(self) -> bool:
        try:
            from code.statistics.mixed_model_factor_contribution import main as run_factor_contribution_model

            run_factor_contribution_model()
            return True
        except ImportError as e:
            self.logger.warning(f"mixed_model_factor_contribution module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in mixed model factor contribution: {e}")
            return False

    def _run_interaction_plots(self) -> bool:

        try:
            from code.statistics.interaction_plots import main as run_interaction_plots
            output_dir = str(self.output_folder / "interaction_plots")
            run_interaction_plots(
                master_path=str(self.output_folder / "master_summary.csv"),
                cell_file_dir=str(self.output_folder / "cell_file" / "cell_file_counting"),
                output_dir=output_dir
            )
            self.logger.info(f"Interaction plots completed. Results saved to: {output_dir}")
            return True
        except ImportError as e:
            self.logger.warning(f"interaction_plots module not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in interaction plots: {e}")
            return False
    
    def cleanup_deleted_files(
        self,
        measurements_folder: Optional[str] = None,
        remove_orphaned: bool = False,
        backup: bool = True,
        rebuild_after_cleanup: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:

        self._log_step("CLEANING UP DELETED FILES")
        
        # Determine measurements folder
        if measurements_folder is None:
            measurements_folder = str(config.MEASUREMENTS_FOLDER)
        
        measurements_folder = str(Path(measurements_folder).resolve())
        
        self.logger.info(f"Looking for measurement files in: {measurements_folder}")
        
        if not os.path.exists(measurements_folder):
            self.logger.error(f"Measurements folder not found: {measurements_folder}")
            return {'success': False, 'error': 'Measurements folder not found'}
        
        from code.util.file_utils import cleanup_missing_files
        
        # Clean up missing files
        results = cleanup_missing_files(
            measurements_folder=measurements_folder,
            master_summary_path=str(self.output_folder / "master_summary.csv"),
            backup=backup,
            remove_orphaned=remove_orphaned,
            dry_run=dry_run
        )
        
        if not results['success']:
            self.logger.error("Cleanup failed")
            return results
        
        # Rebuild master summary from existing files
        if not dry_run and rebuild_after_cleanup and results['total_images_remaining'] > 0:
            self._log_step("REBUILDING MASTER SUMMARY AFTER CLEANUP")
            self.step3_update_master_summary(
                measurements_folder=measurements_folder,
                force_rebuild=False
            )
            
            # Rebuild feature table
            self._log_step("REBUILDING FEATURE TABLE AFTER CLEANUP")
            self.step6_update_feature_table(force_rebuild=True)
        
        self.logger.info("Cleanup complete!")
        self.logger.info(f"Images remaining: {results['total_images_remaining']}")
        
        return results
    
    def full_cleanup_and_rebuild(
        self,
        measurements_folder: Optional[str] = None,
        remove_orphaned: bool = False,
        backup: bool = True,
        run_analysis: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:

        self._log_step("FULL CLEANUP AND REBUILD")
        
        if dry_run:
            self._log_step("DRY RUN MODE - No changes will be made")
        
        # Determine measurements folder
        if measurements_folder is None:
            measurements_folder = str(config.MEASUREMENTS_FOLDER)
        
        measurements_folder = str(Path(measurements_folder).resolve())
        
        self.logger.info(f"Measurements folder: {measurements_folder}")
        
        results = {
            'cleanup': None,
            'master_summary_rebuilt': False,
            'feature_table_rebuilt': False,
            'analysis_run': False,
            'images_remaining': 0,
            'measurements_folder': measurements_folder,
            'dry_run': dry_run
        }
        
        # Step 1: Clean up deleted files
        cleanup_results = self.cleanup_deleted_files(
            measurements_folder=measurements_folder,
            remove_orphaned=remove_orphaned,
            backup=backup,
            rebuild_after_cleanup=False,
            dry_run=dry_run
        )
        results['cleanup'] = cleanup_results
        results['images_remaining'] = cleanup_results.get('total_images_remaining', 0)
        
        # Step 2: Rebuild master summary
        if not dry_run and results['images_remaining'] > 0:
            self._log_step("REBUILDING MASTER SUMMARY")
            master_success = self.step3_update_master_summary(
                measurements_folder=measurements_folder,
                force_rebuild=False
            )
            results['master_summary_rebuilt'] = master_success
            
            # Step 3: Rebuild feature table
            self._log_step("REBUILDING FEATURE TABLE")
            feature_success = self.step6_update_feature_table(force_rebuild=True)
            results['feature_table_rebuilt'] = feature_success
            
            # Step 4: Re-run analysis
            if run_analysis and master_success and feature_success:
                self._log_step("RE-RUNNING ANALYSIS")
                analysis_success = self.step7_run_analysis()
                results['analysis_run'] = analysis_success
        
        # Print summary
        if dry_run:
            print("DRY RUN COMPLETE - No changes were made")
        else:
            print("CLEANUP AND REBUILD COMPLETE")
        print(f"Measurements folder: {measurements_folder}")
        print(f"Images removed from master summary: {len(cleanup_results.get('missing_images_removed', []))}")
        print(f"Orphaned files removed: {len(cleanup_results.get('orphaned_files_removed', []))}")
        print(f"Images remaining: {results['images_remaining']}")
        if not dry_run:
            print(f"Master summary rebuilt: {results['master_summary_rebuilt']}")
            print(f"Feature table rebuilt: {results['feature_table_rebuilt']}")
            print(f"Analysis re-run: {results['analysis_run']}")
        
        return results
    
    def preview_cleanup(
        self,
        measurements_folder: Optional[str] = None
    ) -> None:
        from code.util.file_utils import preview_cleanup
        
        # Determine measurements folder
        if measurements_folder is None:
            measurements_folder = str(config.MEASUREMENTS_FOLDER)
        
        measurements_folder = str(Path(measurements_folder).resolve())
        
        self.logger.info(f"Previewing cleanup for: {measurements_folder}")
        
        preview_cleanup(
            measurements_folder=measurements_folder,
            master_summary_path=str(self.output_folder / "master_summary.csv")
        )
    
    def run_full_pipeline(
        self,
        force_neighbor_recalc: bool = False,
        force_feature_rebuild: bool = False,
        force_cell_file_rebuild: bool = False,
        skip_qc: bool = False,
        qc_interactive: bool = True,
        qc_auto_delete: bool = False,
        qc_port: int = 8888
    ) -> bool:

        self._log_step("STARTING FULL PIPELINE")
        
        start_time = datetime.now()
        
        # DEBUG: Print initial state
        print(f"skip_qc: {skip_qc}")
        print(f"qc_interactive: {qc_interactive}")
        print(f"qc_auto_delete: {qc_auto_delete}")
        print(f"qc_port: {qc_port}")
        print(f"force_cell_file_rebuild: {force_cell_file_rebuild}")
        print(f"new_quadrant_images (initial): {self.new_quadrant_images if hasattr(self, 'new_quadrant_images') else 'NOT SET'}")

        # Define steps in order
        steps = [
            ('1. Split Images', self.step1_split_images),
            ('2. Extract Cells', self.step2_extract_cells),
        ]
        
        # Add QC step between Step 2 and Step 3
        if not skip_qc:
            # DEBUG: Print before adding QC step
            print("ADDING QC STEP TO PIPELINE")
            print(f"new_quadrant_images before QC: {self.new_quadrant_images if hasattr(self, 'new_quadrant_images') else 'NOT SET'}")
            
            # Define the QC function to be called
            def run_qc():
                print("EXECUTING QC STEP")
                print(f"new_quadrant_images in QC: {self.new_quadrant_images if hasattr(self, 'new_quadrant_images') else 'NOT SET'}")
                return self.step2_5_quality_control(
                    interactive=qc_interactive,
                    auto_delete=qc_auto_delete,
                    port=qc_port
                )
            
            steps.append(('2.5. Quality Control (New Images Only)', run_qc))
        else:
            print("QC STEP IS SKIPPED")
        
        # Continue with remaining steps
        steps.extend([
            ('3. Update Master Summary', self.step3_update_master_summary),
            ('4. Analyze Cell Files (incremental)', lambda: self.step4_analyze_cell_files(force_rebuild=force_cell_file_rebuild)),
            ('5. Calculate Neighbors', lambda: self.step5_calculate_neighbors(force_recalc=force_neighbor_recalc)),
            ('6. Update Feature Table (incremental)', lambda: self.step6_update_feature_table(force_rebuild=force_feature_rebuild)),
            ('7. Run Analysis (all data)', self.step7_run_analysis),
        ])
        
        results = {}
        
        for step_name, step_func in steps:
            try:
                self.logger.info(f"Starting: {step_name}")
                
                # DEBUG: Print before each step
                if step_name == '2.5. Quality Control (New Images Only)':
                    print(f"BEFORE EXECUTING: {step_name}")
                    print(f"new_quadrant_images: {self.new_quadrant_images if hasattr(self, 'new_quadrant_images') else 'NOT SET'}")
                
                success = step_func()
                results[step_name] = 'SUCCESS' if success else 'FAILED'
                
                if not success:
                    self.logger.warning(f"Step '{step_name}' failed but continuing...")
                    
            except Exception as e:
                self.logger.error(f"Critical error in step '{step_name}': {e}", exc_info=True)
                results[step_name] = 'ERROR'
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        self._log_step("PIPELINE COMPLETE")
        self._log_step(f"Duration: {duration}")
        
        # Generate summary
        self.processing_summary = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'steps': results,
            'processed_images': len(self.processed_images),
            'failed_images': len(self.failed_images),
            'deleted_images': len(self.deleted_images) if hasattr(self, 'deleted_images') else 0,
            'species': self.species,
            'population': self.population,
            'dataset_metadata': self.dataset_metadata
        }
        
        # Save run log
        self._save_run_log()
        
        # Generate and display summary
        summary = self.generate_summary()
        print("\n" + summary)
        
        # Check if all steps succeeded
        all_successful = all(v == 'SUCCESS' for v in results.values())
        return all_successful
    
    def _save_run_log(self):
        log_path = self.output_folder / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(log_path, 'w') as f:
                f.write('\n'.join(self.run_log))
            self.logger.info(f"Run log saved to: {log_path}")
        except Exception as e:
            self.logger.error(f"Could not save run log: {e}")
    
    def generate_summary(self) -> str:

        lines = []
        lines.append("PIPELINE SUMMARY REPORT")
        lines.append("")
        
        # Dataset info
        lines.append("DATASET INFORMATION")
        lines.append(f"Species: {self.species}")
        lines.append(f"Population: {self.population}")
        if self.dataset_metadata:
            for key, value in self.dataset_metadata.items():
                lines.append(f"{key}: {value}")
        lines.append("")
        
        # Processing summary
        if self.processing_summary:
            lines.append("PROCESSING SUMMARY")
            lines.append(f"Start time: {self.processing_summary.get('start_time', 'N/A')}")
            lines.append(f"End time: {self.processing_summary.get('end_time', 'N/A')}")
            lines.append(f"Duration: {self.processing_summary.get('duration_seconds', 0):.2f} seconds")
            lines.append(f"Images processed: {self.processing_summary.get('processed_images', 0)}")
            lines.append(f"Images failed: {self.processing_summary.get('failed_images', 0)}")
            lines.append(f"Images deleted during QC: {self.processing_summary.get('deleted_images', 0)}")
            lines.append("")
        
        # Master summary statistics
        if self.master_df is not None:
            lines.append("MASTER SUMMARY STATISTICS")
            lines.append(f"Total images: {len(self.master_df)}")
            lines.append(f"Total cells: {self.master_df['n_cells'].sum()}")
            lines.append(f"Average cells per image: {self.master_df['n_cells'].mean():.1f}")
            lines.append(f"Quadrants: {', '.join(self.master_df['quadrant'].unique())}")
            
            treatments = []
            for t in self.master_df['treatment'].unique():
                if t is not None:
                    treatments.append(str(t))
            if treatments:
                lines.append(f"Treatments: {', '.join(treatments)}")
            else:
                lines.append("Treatments: (none)")
            
            species_list = []
            for s in self.master_df['species'].unique():
                if s is not None:
                    species_list.append(str(s))
            population_list = []
            for p in self.master_df['population'].unique():
                if p is not None:
                    population_list.append(str(p))
            
            lines.append(f"Species in dataset: {', '.join(species_list) if species_list else '(none)'}")
            lines.append(f"Populations in dataset: {', '.join(population_list) if population_list else '(none)'}")
            lines.append(f"(Current run configured for: {self.species}, {self.population})")
            lines.append("")
        
        # Feature table statistics
        if self.feature_df is not None:
            lines.append("FEATURE TABLE STATISTICS")
            lines.append(f"Total roots: {len(self.feature_df)}")
            
            for feature in config.FEATURE_CONFIG['features']:
                if feature in self.feature_df.columns:
                    mean_val = self.feature_df[feature].mean()
                    std_val = self.feature_df[feature].std()
                    label = config.FEATURE_CONFIG['feature_labels'].get(feature, feature)
                    lines.append(f"{label}: {mean_val:.3f} ± {std_val:.3f}")
            lines.append("")
        
        # QC Summary
        if self.qc_manager is not None:
            lines.append("QUALITY CONTROL SUMMARY")
            qc_summary_path = self.output_folder / "quality_control" / "qc_summary.csv"
            if qc_summary_path.exists():
                qc_df = pd.read_csv(qc_summary_path)
                total = len(qc_df)
                bad = len(qc_df[qc_df['is_bad'] == True])
                lines.append(f"Total images reviewed: {total}")
                lines.append(f"Bad images flagged: {bad}")
                lines.append(f"Deleted images: {len(self.deleted_images)}")
            else:
                lines.append("QC summary not available")
            lines.append("")
        
        # Output files
        lines.append("OUTPUT FILES")
        lines.append(f"Master summary: {self.output_folder / 'master_summary.csv'}")
        lines.append(f"Feature table: {self.output_folder / 'feature_table.csv'}")
        lines.append(f"Measurements: {config.MEASUREMENTS_FOLDER}")
        if self.qc_manager is not None:
            lines.append(f"QC Report: {self.output_folder / 'quality_control' / 'qc_report.html'}")
        lines.append("")
        
        
        return '\n'.join(lines)


def quick_process_new_images(
    new_images_folder: str,
    output_folder: str = "results",
    skip_existing: bool = True,
    species: str = "Zea mays",
    population: str = "IBM",
    dataset_metadata: Optional[Dict[str, Any]] = None,
    force_rebuild_master_summary: bool = False,
    force_cell_file_rebuild: bool = False,
    skip_qc: bool = False,
    qc_interactive: bool = True,
    qc_auto_delete: bool = False,
    qc_port: int = 8888
) -> RootAnalysisPipeline:

    pipeline = RootAnalysisPipeline(
        new_images_folder=new_images_folder,
        output_folder=output_folder,
        skip_existing=skip_existing,
        verbose=True,
        species=species,
        population=population,
        dataset_metadata=dataset_metadata,
        force_rebuild_master_summary=force_rebuild_master_summary
    )
    
    pipeline.run_full_pipeline(
        skip_qc=skip_qc,
        qc_interactive=qc_interactive,
        qc_auto_delete=qc_auto_delete,
        qc_port=qc_port,
        force_cell_file_rebuild=force_cell_file_rebuild
    )
    
    return pipeline


def update_summaries_only(
    measurements_folder: str = MEASUREMENTS_FOLDER,
    output_folder: str = "results",
    species: str = "Zea mays",
    population: str = "IBM",
    dataset_metadata: Optional[Dict[str, Any]] = None,
    force_rebuild_master_summary: bool = False
) -> RootAnalysisPipeline:

    pipeline = RootAnalysisPipeline(
        output_folder=output_folder,
        verbose=True,
        species=species,
        population=population,
        dataset_metadata=dataset_metadata,
        force_rebuild_master_summary=force_rebuild_master_summary
    )
    
    # Run only summary update steps
    pipeline.step3_update_master_summary(measurements_folder=measurements_folder)
    pipeline.step6_update_feature_table()
    
    print(pipeline.generate_summary())
    
    return pipeline


def run_analysis_only(
    output_folder: str = "results",
    species: str = "Zea mays",
    population: str = "IBM"
) -> RootAnalysisPipeline:

    pipeline = RootAnalysisPipeline(
        output_folder=output_folder,
        verbose=True,
        species=species,
        population=population
    )
    
    # Load existing master summary and feature table
    master_path = pipeline.output_folder / "master_summary.csv"
    feature_path = pipeline.output_folder / "feature_table.csv"
    
    if master_path.exists():
        pipeline.master_df = pd.read_csv(master_path)
        print(f"Loaded master summary: {len(pipeline.master_df)} rows")
    else:
        print(f"Warning: {master_path} not found")
    
    if feature_path.exists():
        pipeline.feature_df = pd.read_csv(feature_path)
        print(f"Loaded feature table: {len(pipeline.feature_df)} roots")
    else:
        print(f"Warning: {feature_path} not found")
    
    # Run analysis only
    pipeline.step7_run_analysis()
    
    print(pipeline.generate_summary())
    
    return pipeline


if __name__ == "__main__":
    pass