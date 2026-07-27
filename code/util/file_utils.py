import os
import pandas as pd
import json
import glob
import re
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import logging
import datetime

logger = logging.getLogger(__name__)


def parse_image_name(filename):
    
    # Remove extensions and suffixes
    base_name = filename
    if base_name.endswith('_centers'):
        base_name = base_name[:-8]
    if base_name.endswith('.jpg') or base_name.endswith('.png') or base_name.endswith('.jpeg'):
        base_name = os.path.splitext(base_name)[0]
    
    # Initialize parsed dict with all fields as None (will become empty in CSV)
    parsed = {
        'image_name': base_name,
        'quadrant': None,
        'plot_number': None,
        'plant_number': None,
        'root_number': None,
        'technical_replicate': None,
        'treatment': None,
        'root_type': None,
    }
    
    # 1. Extract quadrant - look for BL, BR, TL, TR at start
    quadrant_patterns = ['BL', 'BR', 'TL', 'TR']
    for q in quadrant_patterns:
        if base_name.startswith(q + '_'):
            parsed['quadrant'] = q
            break
    
    # 2. Extract plot_number - the first number after quadrant (e.g., 478 from BL_478_)
    if parsed['quadrant'] is not None:
        quadrant_match = re.match(rf'{parsed["quadrant"]}_(\d+)_', base_name)
        if quadrant_match:
            parsed['plot_number'] = int(quadrant_match.group(1))
            print(f"[DEBUG] Found plot_number: {parsed['plot_number']}")
    
    # 3. Extract root_type (W followed by number) - e.g., W3 from _W3_
    root_type_match = re.search(r'_W(\d+)_', base_name)
    if root_type_match:
        parsed['root_type'] = f'W{root_type_match.group(1)}'
        print(f"[DEBUG] Found root_type: {parsed['root_type']}")
    
    # 4. Extract treatment (WS, WW, MCS) - ONLY if present
    for t in ['WS', 'WW', 'MCS']:
        if t in base_name:
            parsed['treatment'] = t
            print(f"[DEBUG] Found treatment: {parsed['treatment']}")
            break
    
    # 5. Extract plant_number and root_number from decimal format (e.g., 2.1)
    # This pattern matches: _(\d+)\.(\d+)_ or _(\d+)\.(\d+)_root or _(\d+)\.(\d+)_test
    decimal_match = re.search(r'_(\d+)\.(\d+)(?:_|$)', base_name)
    if decimal_match:
        parsed['plant_number'] = int(decimal_match.group(1))
        parsed['root_number'] = int(decimal_match.group(2))
        print(f"[DEBUG] Found plant_number from decimal: {parsed['plant_number']}")
        print(f"[DEBUG] Found root_number from decimal: {parsed['root_number']}")
    else:
        # If no decimal format, try P number format (e.g., P1)
        plant_match = re.search(r'P(\d+)', base_name)
        if plant_match:
            parsed['plant_number'] = int(plant_match.group(1))
            print(f"[DEBUG] Found plant_number from P: {parsed['plant_number']}")
        
        # Try root number after root_type (e.g., WW_1)
        root_after_type_match = re.search(r'_([A-Z]{2,3})_(\d+)', base_name)
        if root_after_type_match:
            possible_type = root_after_type_match.group(1)
            if possible_type in ['WS', 'WW', 'MCS']:
                # Only set root_number if not already set
                if parsed['root_number'] is None:
                    parsed['root_number'] = int(root_after_type_match.group(2))
                    print(f"[DEBUG] Found root_number from treatment: {parsed['root_number']}")
    
    # 6. Extract technical_replicate
    # First try: test number (e.g., test992)
    test_match = re.search(r'test(\d+)', base_name)
    if test_match:
        parsed['technical_replicate'] = int(test_match.group(1))
        print(f"[DEBUG] Found technical_replicate from test: {parsed['technical_replicate']}")
    else:
        # Second try: number in parentheses (e.g., (1))
        tech_match = re.search(r'\((\d+)\)', base_name)
        if tech_match:
            parsed['technical_replicate'] = int(tech_match.group(1))
            print(f"[DEBUG] Found technical_replicate from parentheses: {parsed['technical_replicate']}")
        else:
            # Third try: number after root at end (e.g., root1)
            root_end_match = re.search(r'root(\d+)$', base_name)
            if root_end_match:
                # This is often just a suffix, but we can use it if nothing else
                # Only set if no other technical_replicate found
                if parsed['technical_replicate'] is None:
                    # Don't use this as technical_replicate - it's just a suffix
                    pass
    
    return parsed


def load_measurement_csv(csv_path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logger.warning(f"Could not load {csv_path}: {e}")
        return None


def load_image_summary(image_name: str, measurements_folder: str) -> Optional[Dict[str, Any]]:
    summary_path = os.path.join(measurements_folder, f"{image_name}_image_summary.json")
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {summary_path}: {e}")
    
    return None


def find_image_file(image_name: str, search_folders: List[str], extensions: List[str] = None) -> Optional[str]:
    if extensions is None:
        extensions = ['.jpg', '.png', '.jpeg', '.tif', '.tiff']
    
    for folder in search_folders:
        for ext in extensions:
            path = os.path.join(folder, f"{image_name}{ext}")
            if os.path.exists(path):
                return path
    
    return None


def save_dataframe(df: pd.DataFrame, output_path: str, create_dirs: bool = True) -> bool:

    try:
        if create_dirs:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved DataFrame to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Could not save DataFrame to {output_path}: {e}")
        return False


def load_cell_file_data(cell_file_counts_folder, image_name):

    # Try different possible file paths
    possible_paths = [
        os.path.join(cell_file_counts_folder, image_name, "cell_assignments.csv"),
        os.path.join(cell_file_counts_folder, image_name, f"{image_name}_method_derivative.csv"),
        os.path.join(cell_file_counts_folder, image_name, "cell_assignments_all_methods.csv"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                
                # Look for the derivative column
                if 'cell_file_derivative' in df.columns:
                    # Group by file number and calculate average area
                    file_areas = []
                    file_numbers = sorted(df['cell_file_derivative'].unique())
                    
                    for file_num in file_numbers:
                        if file_num >= 0:  # Only include valid files
                            file_cells = df[df['cell_file_derivative'] == file_num]
                            avg_area = file_cells['area_um2'].mean()
                            file_areas.append(round(avg_area, 2))
                    
                    n_files = len(file_areas)
                    return n_files, file_areas, True
                
                # Try to find any cell_file column
                for col in df.columns:
                    if 'cell_file' in col and col != 'cell_file_derivative':
                        file_numbers = sorted(df[col].unique())
                        file_areas = []
                        for file_num in file_numbers:
                            if file_num >= 0:
                                file_cells = df[df[col] == file_num]
                                avg_area = file_cells['area_um2'].mean()
                                file_areas.append(round(avg_area, 2))
                        n_files = len(file_areas)
                        return n_files, file_areas, True
                        
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
                continue
    
    return 0, [], False


def _import_rebuild_master_summary():
    # Try relative import (when used as part of package)
    try:
        from ..measurements.create_master_summary_only import rebuild_master_summary
        return rebuild_master_summary
    except (ImportError, ValueError):
        pass
    
    # Try absolute import (when code folder is in PYTHONPATH)
    try:
        from code.measurements.create_master_summary_only import rebuild_master_summary
        return rebuild_master_summary
    except ImportError:
        pass
    
    # Try direct import (when running from project root)
    try:
        from measurements.create_master_summary_only import rebuild_master_summary
        return rebuild_master_summary
    except ImportError:
        pass
    
    # Try importing using importlib (most flexible)
    try:
        import importlib.util
        import sys
        # Get the project root (assuming this file is in code/utils/)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Go up to project root
        
        # Try different possible paths
        possible_paths = [
            project_root / "code" / "measurements" / "create_master_summary_only.py",
            project_root / "measurements" / "create_master_summary_only.py",
            Path.cwd() / "code" / "measurements" / "create_master_summary_only.py",
            Path.cwd() / "measurements" / "create_master_summary_only.py",
        ]
        
        for module_path in possible_paths:
            if module_path.exists():
                spec = importlib.util.spec_from_file_location("create_master_summary_only", module_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules["create_master_summary_only"] = module
                spec.loader.exec_module(module)
                return module.rebuild_master_summary
    except Exception as e:
        logger.debug(f"Importlib import failed: {e}")
    
    raise ImportError(
        "Could not import rebuild_master_summary. "
        "Please ensure create_master_summary_only.py exists in code/measurements/ "
        "and that the path is correct."
    )


def update_master_summary(
    measurements_folder: str,
    cell_file_counts_folder: Optional[str],
    output_path: str,
    force_rebuild: bool = False,
    species: str = "Zea mays",
    population: str = "IBM",
    dataset_metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    
    print(f"[DEBUG file_utils.py] update_master_summary called with:")
    print(f"[DEBUG file_utils.py]   species='{species}'")
    print(f"[DEBUG file_utils.py]   population='{population}'")
    print(f"[DEBUG file_utils.py]   force_rebuild={force_rebuild}")
    print(f"[DEBUG file_utils.py]   output_path={output_path}")
    print(f"[DEBUG file_utils.py]   file exists: {os.path.exists(output_path)}")
    
    # Import the rebuild function using the helper
    rebuild_master_summary = _import_rebuild_master_summary()
    
    # Load existing master summary if it exists and we're not forcing rebuild
    existing_df = None
    existing_images = set()
    existing_populations = {}  # Track original population for each image
    existing_species = {}  # Track original species for each image
    
    if os.path.exists(output_path) and not force_rebuild:
        print(f"[DEBUG] Attempting to load existing master_summary from {output_path}")
        try:
            existing_df = pd.read_csv(output_path)
            print(f"[DEBUG] Successfully loaded {len(existing_df)} rows")
            print(f"[DEBUG] ACTUAL populations in loaded CSV: {existing_df['population'].unique()}")
            print(f"[DEBUG] ACTUAL species in loaded CSV: {existing_df['species'].unique()}")
            
            # Validate the loaded dataframe has required columns
            required_cols = ['image_name', 'population', 'species']
            missing_cols = [col for col in required_cols if col not in existing_df.columns]
            if missing_cols:
                logger.error(f"ERROR: Existing master_summary is missing required columns: {missing_cols}")
                logger.error(f"This would normally trigger a rebuild, but we're preventing that to avoid data loss.")
                logger.error(f"Please fix the master_summary.csv manually or set force_rebuild=True explicitly.")
                raise ValueError(f"Invalid master_summary.csv: missing columns {missing_cols}")
            
            existing_images = set(existing_df['image_name'].unique())
            
            # Store original population and species for existing images
            for idx, row in existing_df.iterrows():
                img_name = row['image_name']
                existing_populations[img_name] = row.get('population', population)
                existing_species[img_name] = row.get('species', species)
            
            print(f"[DEBUG] Built existing_populations dict with {len(existing_populations)} entries")
            print(f"[DEBUG] Unique populations in dict: {set(existing_populations.values())}")
            print(f"[DEBUG] Sample entries: {list(existing_populations.items())[:3]}")
            
            # Check if we have mixed populations that would be overwritten
            unique_populations = set(existing_populations.values())
            if len(unique_populations) > 1 and force_rebuild:
                logger.warning("WARNING: FORCE REBUILD WITH MIXED POPULATIONS ⚠️")
                logger.warning(f"Your existing data has {len(unique_populations)} different populations:")
                for pop in sorted(unique_populations):
                    count = list(existing_populations.values()).count(pop)
                    logger.warning(f"  - {pop}: {count} images")
                logger.warning(f"\nForce rebuild will OVERWRITE ALL to: '{population}'")
                logger.warning("This cannot be undone! Consider setting force_rebuild=False")
                logger.warning("="*70)
            
            logger.info(f"Loaded existing master summary with {len(existing_df)} rows covering {len(existing_images)} images")
            logger.info(f"Existing populations in dataset: {set(existing_populations.values())}")
            logger.info(f"NOTE: Existing images will KEEP their original population/species values")
            logger.info(f"NOTE: ONLY NEW images will be assigned population='{population}', species='{species}'")
        except Exception as e:
            logger.error("ERROR: Could not load existing master_summary.csv")
            logger.error(f"Error: {e}")
            logger.error(f"File path: {output_path}")
            
            # CRITICAL: Don't automatically rebuild - this could destroy data
            logger.error("\nPREVENTING AUTOMATIC REBUILD")
            logger.error("Normally this would trigger a rebuild, but that could overwrite")
            logger.error("all existing populations and species in your dataset.")
            logger.error("\nTo proceed, either:")
            logger.error("  1. Fix the master_summary.csv file manually")
            logger.error("  2. Set FORCE_REBUILD_MASTER_SUMMARY=True explicitly in run_pipeline.py")
            raise RuntimeError(f"Cannot load master_summary.csv and automatic rebuild is disabled for safety. Error: {e}")
    
    # If force_rebuild or no existing file, use the rebuild function
    print(f"[DEBUG] Checking rebuild condition: force_rebuild={force_rebuild}, existing_df is None={existing_df is None}")
    if force_rebuild or existing_df is None:
        print(f"[DEBUG] ENTERING REBUILD PATH")
        logger.warning("REBUILDING master summary from scratch (force_rebuild=True)")
        logger.warning(f"All images will be assigned: Population='{population}', Species='{species}'")
        
        rebuilt_df = rebuild_master_summary(
            measurements_folder=measurements_folder,
            cell_file_counts_folder=cell_file_counts_folder,
            output_path=output_path,
            species=species,
            population=population,
            dataset_metadata=dataset_metadata
        )
        
        # ENSURE THE REBUILT DATAFRAME IS SAVED
        if rebuilt_df is not None:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                rebuilt_df.to_csv(output_path, index=False)
                logger.warning(f"★ REBUILT master summary saved to {output_path}")
                logger.warning(f"★ Total rows: {len(rebuilt_df)}")
                logger.warning(f"★ Population: {rebuilt_df['population'].unique()}")
                logger.warning(f"★ Images with per_file_avg_areas: {len(rebuilt_df[rebuilt_df['per_file_avg_areas_um2'].notna() & (rebuilt_df['per_file_avg_areas_um2'] != '')])}")
            except Exception as e:
                logger.error(f"Could not save rebuilt master summary: {e}")
        
        return rebuilt_df
    
    # Find all measurement CSV files
    print(f"[DEBUG] Taking APPEND-ONLY path")
    print(f"[DEBUG] Scanning for new measurement CSVs in {measurements_folder}")
    csv_files = glob.glob(os.path.join(measurements_folder, "*_measurements.csv"))
    
    # CRITICAL FIX: Filter out any CSV files that don't have corresponding images
    # This prevents stale CSVs from being processed
    valid_csv_files = []
    for csv_path in csv_files:
        base_name = os.path.basename(csv_path).replace('_measurements.csv', '')
        # Check if there's a corresponding image in the detections folder
        # This is a safety check to prevent processing stale CSVs
        valid_csv_files.append((csv_path, base_name))
    
    print(f"[DEBUG] Found {len(valid_csv_files)} total measurement CSVs")
    print(f"[DEBUG] Existing images in master_summary: {len(existing_images)}")
    print(f"[DEBUG] Sample existing_images: {list(existing_images)[:5]}")
    
    new_csv_files = []
    update_csv_files = []  # For existing images that need cell file data updated
    
    for csv_path, base_name in valid_csv_files:
        if base_name not in existing_images:
            print(f"[DEBUG] NEW IMAGE FOUND: {base_name}")
            new_csv_files.append((csv_path, base_name))
        else:
            # Check if this existing image has file_count=0 and might have new cell file data
            # DO NOT update species/population for existing images - only for new images
            img_row = existing_df[existing_df['image_name'] == base_name]
            if not img_row.empty and img_row.iloc[0].get('file_count', 0) == 0:
                # Check if cell file data now exists
                if cell_file_counts_folder and os.path.exists(cell_file_counts_folder):
                    file_count, _, _ = load_cell_file_data(cell_file_counts_folder, base_name)
                    if file_count > 0:
                        update_csv_files.append((csv_path, base_name))
    
    if not new_csv_files and not update_csv_files:
        logger.info("No new images to add or update in master summary")
        logger.info(f"Existing images: {len(existing_images)}")
        logger.info("Append-only mode: No changes to master summary")
        return existing_df
    
    if new_csv_files:
        logger.warning(f"Found {len(new_csv_files)} NEW images to add to master summary")
        logger.warning(f"★ NEW images will be assigned: Population='{population}', Species='{species}'")
        logger.warning(f"★ EXISTING images ({len(existing_images)}) will KEEP their original population/species values")
    if update_csv_files:
        logger.info(f"Found {len(update_csv_files)} existing images to update with cell file data (preserving population/species)")
    
    # Process new measurement CSVs
    new_data = []
    update_data = {}  # Map image_name -> row_data for updates
    
    # Process new images
    for csv_path, base_name in new_csv_files:
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                logger.debug(f"Skipping {base_name}: empty")
                continue
            
            # Load image summary
            summary = load_image_summary(base_name, measurements_folder)
            
            # Parse filename for metadata
            parsed = parse_image_name(base_name)
            
            # Load cell file data if available (now that analysis has run)
            file_count = 0
            per_file_areas = []
            has_assignments = False
            
            if cell_file_counts_folder and os.path.exists(cell_file_counts_folder):
                file_count, per_file_areas, has_assignments = load_cell_file_data(
                    cell_file_counts_folder, base_name
                )
            
            # Validate per_file_avg_areas_um2
            if file_count > 0 and len(per_file_areas) != file_count:
                logger.warning(f"⚠ WARNING for {base_name}: file_count={file_count} but got {len(per_file_areas)} per-file areas")
                if len(per_file_areas) == 0:
                    logger.warning(f"   Cell file data exists but per-file areas could not be extracted")
                    has_assignments = False
            
            # Calculate average cell area
            if 'area_um2' in df.columns:
                avg_area = round(df['area_um2'].mean(), 2)
                n_cells = len(df)
            else:
                avg_area = 0
                n_cells = len(df)
            
            # Get stele area and root radius from summary
            stele_area = round(summary.get('stele_area_um2', 0), 2) if summary else 0
            root_radius = round(summary.get('root_radius_um', 0), 2) if summary else 0
            
            # Calculate stele diameter
            stele_diameter = 2 * np.sqrt(stele_area / np.pi) if stele_area > 0 else 0
            
            # Build per_file_str BEFORE creating row_data (FIXED)
            per_file_str = ''
            if per_file_areas and len(per_file_areas) > 0:
                per_file_str = ','.join(map(str, per_file_areas))
                logger.debug(f"  Loaded {len(per_file_areas)} per-file areas for {base_name}")
            
            # Build row data for NEW image with provided species and population
            print(f"[DEBUG file_utils.py] Creating NEW row for {base_name}")
            print(f"[DEBUG file_utils.py]   species='{species}', population='{population}'")
            print(f"[DEBUG file_utils.py]   parsed: plant={parsed['plant_number']}, root={parsed['root_number']}, tech_rep={parsed['technical_replicate']}")
            
            row_data = {
                'image_name': base_name,
                'quadrant': parsed['quadrant'],
                'plot_number': parsed['plot_number'],
                'plant_number': parsed['plant_number'],
                'root_number': parsed['root_number'],
                'technical_replicate': parsed['technical_replicate'],
                'treatment': parsed['treatment'],
                'root_type': parsed['root_type'],
                'species': species,
                'population': population,
                'file_count': file_count,
                'average_cell_area_um2': avg_area,
                'n_cells': n_cells,
                'stele_area_um2': stele_area,
                'root_radius_um': root_radius,
                'stele_diameter_um': round(stele_diameter, 2),
                'per_file_avg_areas_um2': per_file_str,
                'per_file_count': len(per_file_areas) if per_file_areas else 0,
                'measurements_csv': f"{base_name}_measurements.csv",
                'has_image_summary': summary is not None,
                'has_cell_assignments': has_assignments
            }
            
            # Add any additional metadata from dataset_metadata
            if dataset_metadata:
                for key, value in dataset_metadata.items():
                    if key not in row_data:
                        row_data[key] = value
            
            new_data.append(row_data)
            logger.warning(f"✓ NEW IMAGE: {base_name}")
            logger.warning(f"  └─ Population: '{population}' (NEW) | Species: '{species}' (NEW)")
            logger.warning(f"  └─ Plant: {parsed['plant_number']} | Root: {parsed['root_number']} | Files: {file_count}")
            
        except Exception as e:
            logger.error(f"Error processing NEW image {base_name}: {e}", exc_info=True)
            continue
    
    # Process updates for existing images
    for csv_path, base_name in update_csv_files:
        try:
            # Load cell file data
            file_count = 0
            per_file_areas = []
            has_assignments = False
            
            if cell_file_counts_folder and os.path.exists(cell_file_counts_folder):
                file_count, per_file_areas, has_assignments = load_cell_file_data(
                    cell_file_counts_folder, base_name
                )
            
            # Build update data (ONLY cell file related fields, preserving species/population)
            per_file_str = ','.join(map(str, per_file_areas)) if per_file_areas else ''
            update_data[base_name] = {
                'file_count': file_count,
                'has_cell_assignments': has_assignments,
                'per_file_avg_areas_um2': per_file_str,
                'per_file_count': len(per_file_areas)
            }
            
            # Get the original population/species that will be preserved
            orig_pop = existing_populations.get(base_name, population)
            orig_species = existing_species.get(base_name, species)
            logger.info(f"Will update EXISTING image {base_name}: {file_count} files (preserving population='{orig_pop}', species='{orig_species}')")
            
        except Exception as e:
            logger.error(f"Error updating EXISTING image {base_name}: {e}", exc_info=True)
            continue
    
    # Apply updates to existing_df (ONLY cell file data, NOT population/species)
    if update_data:
        for img_name, update_fields in update_data.items():
            mask = existing_df['image_name'] == img_name
            for field, value in update_fields.items():
                existing_df.loc[mask, field] = value
        logger.info(f"Updated cell file data for {len(update_data)} existing images (preserving all other fields)")
    
    if not new_data:
        if update_data:
            # Only updates, save and return
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                existing_df.to_csv(output_path, index=False)
                logger.info(f"Updated master summary saved to {output_path}")
            except Exception as e:
                logger.error(f"Could not save updated master summary: {e}")
            return existing_df
        else:
            logger.info("No valid new data to add")
            return existing_df
    
    # Create DataFrame from new data
    new_df = pd.DataFrame(new_data)
    
    # CRITICAL FIX: Ensure the new DataFrame has the correct species and population
    print(f"[DEBUG] New DataFrame created with {len(new_df)} rows")
    print(f"[DEBUG] New DataFrame species: {new_df['species'].unique() if 'species' in new_df.columns else 'MISSING'}")
    print(f"[DEBUG] New DataFrame population: {new_df['population'].unique() if 'population' in new_df.columns else 'MISSING'}")
    
    # Log the new data summary before merging
    logger.warning("NEW DATA SUMMARY (before merging)")
    logger.warning(f"New images: {len(new_df)}")
    logger.warning(f"Populations: {new_df['population'].unique()}")
    logger.warning(f"Species: {new_df['species'].unique()}")
    logger.warning(f"Images with per_file_avg_areas: {len(new_df[new_df['per_file_avg_areas_um2'].notna() & (new_df['per_file_avg_areas_um2'] != '')])}")
    
    # Combine with existing data
    # Use concat with sort=False to preserve column order and handle new columns
    combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    
    # Ensure consistent column order
    column_order = [
        'image_name', 
        'quadrant', 
        'plot_number',          # First number after quadrant
        'plant_number', 
        'root_number',          # Number after root_type (e.g., WW_1 -> 1)
        'technical_replicate',  # Number in parentheses (e.g., (1))
        'treatment', 
        'root_type', 
        'species', 
        'population',
        'file_count', 
        'average_cell_area_um2', 
        'n_cells',
        'stele_area_um2', 
        'root_radius_um', 
        'stele_diameter_um',
        'per_file_avg_areas_um2', 
        'per_file_count',
        'measurements_csv', 
        'has_image_summary', 
        'has_cell_assignments'
    ]
    
    # Add any additional metadata columns to the order
    if dataset_metadata:
        for key in dataset_metadata.keys():
            if key not in column_order and key in combined_df.columns:
                column_order.append(key)
    
    # Reorder columns, keeping only those that exist
    existing_columns = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[existing_columns]
    
    # Sort the combined dataframe
    combined_df = combined_df.sort_values(['plant_number', 'root_number', 'quadrant']).reset_index(drop=True)
    
    # Check that existing populations are preserved
    for img_name in existing_images:
        if img_name in combined_df['image_name'].values:
            row = combined_df[combined_df['image_name'] == img_name].iloc[0]
            orig_pop = existing_populations.get(img_name)
            curr_pop = row['population']
            if orig_pop and orig_pop != curr_pop:
                logger.error(f"ERROR: {img_name} population changed from '{orig_pop}' to '{curr_pop}' (should be preserved!)")
            else:
                logger.info(f"{img_name} population preserved: '{curr_pop}'")
    
    # Verify new images have the correct population
    for img_name in new_df['image_name'].values:
        if img_name in combined_df['image_name'].values:
            row = combined_df[combined_df['image_name'] == img_name].iloc[0]
            curr_pop = row['population']
            if curr_pop != population:
                logger.error(f"ERROR: New image {img_name} has population '{curr_pop}' but should be '{population}'")
            else:
                logger.info(f"New image {img_name} population correct: '{curr_pop}'")
    
    populations_in_summary = combined_df['population'].unique()
    logger.warning(f"All populations in summary: {populations_in_summary}")
    logger.warning(f"Total rows: {len(combined_df)} (existing: {len(existing_images)}, new: {len(new_data)})")
    logger.warning(f"Images with per_file_avg_areas: {len(combined_df[combined_df['per_file_avg_areas_um2'].notna() & (combined_df['per_file_avg_areas_um2'] != '')])}")
    
    # Save updated master summary
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Updated master summary saved to {output_path}")
        logger.info(f"Added {len(new_data)} new images. Total: {len(combined_df)} rows")
        logger.info(f"Append-only mode: Existing {len(existing_images)} images preserved with original population/species")
    except Exception as e:
        logger.error(f"Could not save updated master summary: {e}")
    
    return combined_df


def find_all_measurement_files(measurements_folder: str, min_cells: int = 0) -> List[Dict[str, Any]]:

    csv_files = glob.glob(os.path.join(measurements_folder, "*_measurements.csv"))
    
    image_data = []
    for csv_path in csv_files:
        base_name = os.path.basename(csv_path).replace("_measurements.csv", "")
        
        df = load_measurement_csv(csv_path)
        if df is None or len(df) < min_cells:
            continue
        
        summary = load_image_summary(base_name, measurements_folder)
        
        image_data.append({
            'name': base_name,
            'csv_path': csv_path,
            'summary': summary,
            'df': df,
            'n_cells': len(df)
        })
    
    return image_data


def load_cell_assignments(cell_file_counts_folder: str, image_name: str) -> Optional[pd.DataFrame]:

    possible_paths = [
        os.path.join(cell_file_counts_folder, image_name, 'cell_assignments.csv'),
        os.path.join(cell_file_counts_folder, image_name, f'{image_name}_method_derivative.csv'),
        os.path.join(cell_file_counts_folder, image_name, 'cell_assignments_all_methods.csv'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                
                # Look for the derivative column
                if 'cell_file_derivative' in df.columns:
                    return df[['cell_id', 'cell_file_derivative']].copy()
                
                # Try to find any cell_file column
                for col in df.columns:
                    if 'cell_file' in col:
                        return df[['cell_id', col]].copy()
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")
                continue
    
    return None


def build_pixel_to_um_map(
    scale_image_key_csv: str,
    scale_values_csv: str
) -> "tuple[Dict[str, float], set]":

    key_df = pd.read_csv(scale_image_key_csv)
    values_df = pd.read_csv(scale_values_csv)

    scale_to_px_per_um = dict(zip(values_df['scale_id'], values_df['um_px']))

    # First pass: collect every (scale_id -> pixel_to_um) candidate per image,
    # so we can detect images with CONFLICTING scale assignments across rows
    # (same image_id, different scale_id) instead of silently letting the
    # last row win.
    candidates: Dict[str, Dict[str, float]] = {}  # base_name -> {scale_id: pixel_to_um}
    skipped = []
    for _, row in key_df.iterrows():
        image_id = row.get('image_id')
        scale_id = row.get('scale_id')
        if pd.isna(image_id):
            continue
        if pd.isna(scale_id):
            skipped.append(str(image_id))
            continue

        px_per_um = scale_to_px_per_um.get(scale_id)
        if px_per_um is None or pd.isna(px_per_um) or px_per_um == 0:
            skipped.append(str(image_id))
            continue

        base_name = os.path.splitext(str(image_id))[0]
        candidates.setdefault(base_name, {})[str(scale_id)] = 1.0 / float(px_per_um)

    pixel_to_um_map = {}
    conflicts = {}
    for base_name, scale_options in candidates.items():
        if len(scale_options) > 1:
            conflicts[base_name] = scale_options
            # Do NOT guess: leave this image out of the map so it falls back
            # to the caller's default, rather than silently picking a value
            # that could be off by 50%+.
            continue
        pixel_to_um_map[base_name] = next(iter(scale_options.values()))

    logger.info(f"Built pixel-to-um map for {len(pixel_to_um_map)} images "
                f"from {scale_image_key_csv} + {scale_values_csv}")
    if skipped:
        logger.warning(f"Could not resolve scale for {len(skipped)} images "
                        f"(missing/unknown scale_id): {skipped[:10]}"
                        f"{'...' if len(skipped) > 10 else ''}")
    if conflicts:
        logger.warning(f"{len(conflicts)} images have CONFLICTING scale_id assignments "
                        f"across multiple rows in {scale_image_key_csv} — excluded entirely "
                        f"(not just falling back to default):")
        for base_name, options in list(conflicts.items())[:10]:
            logger.warning(f"    {base_name}: {options}")
        if len(conflicts) > 10:
            logger.warning(f"    ... and {len(conflicts) - 10} more")

    excluded_images = {os.path.splitext(name)[0] for name in skipped} | set(conflicts.keys())
    if excluded_images:
        logger.warning(f"Total images to exclude from the dataset due to unresolvable scale: "
                        f"{len(excluded_images)}")

    return pixel_to_um_map, excluded_images


def resolve_original_image_name(quadrant_image_basename: str) -> str:
    name = quadrant_image_basename
    for prefix in ('BL_', 'BR_', 'TL_', 'TR_'):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = re.sub(r'_root\d+$', '', name)
    return name


def ensure_directory_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_output_path(base_folder: str, *path_parts: str) -> str:
    output_path = os.path.join(base_folder, *path_parts)
    ensure_directory_exists(os.path.dirname(output_path))
    return output_path


def detect_missing_measurement_files(
    measurements_folder: str,
    master_summary_path: str
) -> Dict[str, Any]:
    
    # Ensure paths are absolute and exist
    measurements_folder = str(Path(measurements_folder).resolve())
    master_summary_path = str(Path(master_summary_path).resolve())
    
    print(f"Measurements folder: {measurements_folder}")
    print(f"Master summary path: {master_summary_path}")
    
    if not os.path.exists(measurements_folder):
        print(f"Warning: Measurements folder not found: {measurements_folder}")
        return {'missing_images': [], 'orphaned_files': [], 'error': 'Folder not found'}
    
    if not os.path.exists(master_summary_path):
        print(f"Error: {master_summary_path} not found")
        return {'missing_images': [], 'orphaned_files': [], 'error': 'Master summary not found'}
    
    # Load master summary
    df = pd.read_csv(master_summary_path)
    print(f"Loaded {len(df)} images from master summary")
    
    # Check which images have measurement files
    missing_images = []
    existing_images_with_files = []
    
    for img_name in df['image_name']:
        csv_path = os.path.join(measurements_folder, f"{img_name}_measurements.csv")
        if os.path.exists(csv_path):
            existing_images_with_files.append(img_name)
        else:
            missing_images.append(img_name)
    
    # Find orphaned measurement files (files not in master summary)
    csv_files = glob.glob(os.path.join(measurements_folder, "*_measurements.csv"))
    existing_images_set = set(df['image_name'])
    orphaned_files = []
    
    for csv_path in csv_files:
        base_name = os.path.basename(csv_path).replace('_measurements.csv', '')
        if base_name not in existing_images_set:
            orphaned_files.append(base_name)
    
    print(f"\nResults:")
    print(f"Images with measurement files: {len(existing_images_with_files)}")
    print(f"Images MISSING measurement files: {len(missing_images)}")
    print(f"Orphaned measurement files (not in master summary): {len(orphaned_files)}")
    
    if missing_images:
        print("\nImages missing measurement files (first 20):")
        for img in missing_images[:20]:
            print(f"    - {img}")
        if len(missing_images) > 20:
            print(f"    ... and {len(missing_images) - 20} more")
    
    if orphaned_files:
        print("\nOrphaned measurement files (first 10):")
        for img in orphaned_files[:10]:
            print(f"    - {img}")
        if len(orphaned_files) > 10:
            print(f"    ... and {len(orphaned_files) - 10} more")
    
    return {
        'missing_images': missing_images,
        'orphaned_files': orphaned_files,
        'existing_images_with_files': existing_images_with_files,
        'total_in_master_summary': len(df),
        'total_with_files': len(existing_images_with_files),
        'total_missing': len(missing_images),
        'total_orphaned': len(orphaned_files)
    }


def cleanup_missing_files(
    measurements_folder: str,
    master_summary_path: str,
    backup: bool = True,
    remove_orphaned: bool = False,
    dry_run: bool = False
) -> Dict[str, Any]:
    
    if dry_run:
        print("DRY RUN MODE - No changes will be made")
    
    # Ensure paths are absolute and exist
    measurements_folder = str(Path(measurements_folder).resolve())
    master_summary_path = str(Path(master_summary_path).resolve())
    
    if not os.path.exists(measurements_folder):
        print(f"Error: Measurements folder not found: {measurements_folder}")
        return {'success': False, 'error': 'Measurements folder not found'}
    
    if not os.path.exists(master_summary_path):
        print(f"Error: Master summary not found: {master_summary_path}")
        return {'success': False, 'error': 'Master summary not found'}
    
    # Detect missing files
    detection = detect_missing_measurement_files(measurements_folder, master_summary_path)
    missing_images = detection['missing_images']
    orphaned_files = detection['orphaned_files']
    
    results = {
        'success': True,
        'missing_images_removed': [],
        'orphaned_files_removed': [],
        'kept_images': 0,
        'total_images_remaining': 0,
        'dry_run': dry_run,
        'measurements_folder': measurements_folder,
        'master_summary_path': master_summary_path
    }
    
    # Remove missing images from master summary
    if missing_images:
        print(f"\nRemoving {len(missing_images)} images from master summary...")
        
        # Load master summary
        df = pd.read_csv(master_summary_path)
        
        if not dry_run:
            # Create backup
            if backup:
                backup_path = master_summary_path.replace('.csv', f'_backup_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                df.to_csv(backup_path, index=False)
                print(f"   Backup saved to: {backup_path}")
            
            # Remove missing images
            df_cleaned = df[~df['image_name'].isin(missing_images)]
            
            # Save updated master summary
            df_cleaned.to_csv(master_summary_path, index=False)
            
            results['missing_images_removed'] = missing_images
            results['kept_images'] = len(df_cleaned)
            results['total_images_remaining'] = len(df_cleaned)
            
            print(f"Removed {len(missing_images)} images from master summary")
            print(f"Kept {len(df_cleaned)} images")
        else:
            print(f"   [DRY RUN] Would remove {len(missing_images)} images from master summary")
            print(f"   [DRY RUN] Would keep {len(df) - len(missing_images)} images")
            results['missing_images_removed'] = missing_images
            results['kept_images'] = len(df) - len(missing_images)
            results['total_images_remaining'] = len(df) - len(missing_images)
    else:
        print("\nNo missing images found in master summary!")
        results['kept_images'] = len(pd.read_csv(master_summary_path))
        results['total_images_remaining'] = results['kept_images']
    
    # Remove orphaned measurement files (optional)
    if remove_orphaned and orphaned_files:
        print(f"\nRemoving {len(orphaned_files)} orphaned measurement files...")
        
        if not dry_run:
            removed_count = 0
            for img_name in orphaned_files:
                # Remove measurement CSV
                csv_path = os.path.join(measurements_folder, f"{img_name}_measurements.csv")
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                    removed_count += 1
                    results['orphaned_files_removed'].append(csv_path)
                
                # Remove associated files
                for ext in ['_image_summary.json', '_centers.jpg']:
                    file_path = os.path.join(measurements_folder, f"{img_name}{ext}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        results['orphaned_files_removed'].append(file_path)
            
            print(f"Removed {len(results['orphaned_files_removed'])} orphaned files")
        else:
            print(f"   [DRY RUN] Would remove {len(orphaned_files)} orphaned files")
    elif orphaned_files:
        print(f"\n{len(orphaned_files)} orphaned files found (use remove_orphaned=True to delete)")
    
    # Print summary
    if dry_run:
        print("DRY RUN - No changes were made")
    print(f"Images in master summary before: {detection['total_in_master_summary']}")
    print(f"Images with measurement files: {detection['total_with_files']}")
    print(f"Images missing measurement files: {detection['total_missing']}")
    print(f"Images remaining after cleanup: {results['total_images_remaining']}")
    print(f"Images removed from master summary: {len(results['missing_images_removed'])}")
    
    if remove_orphaned:
        print(f"Orphaned files removed: {len(results['orphaned_files_removed'])}")
    
    if not dry_run and results['total_images_remaining'] > 0:
        print(f"\nCleanup complete! Updated master summary saved to: {master_summary_path}")
    
    return results


def preview_cleanup(
    measurements_folder: str,
    master_summary_path: str
) -> None:
    
    detection = detect_missing_measurement_files(measurements_folder, master_summary_path)
    
    if 'error' in detection:
        return
    
    print(f"\nSUMMARY:")
    print(f"  Images in master summary: {detection['total_in_master_summary']}")
    print(f"  Images with measurement files: {detection['total_with_files']}")
    print(f"  Images MISSING measurement files: {detection['total_missing']}")
    print(f"  Orphaned measurement files: {detection['total_orphaned']}")
    
    if detection['missing_images']:
        print(f"\nIMAGES TO BE REMOVED FROM MASTER SUMMARY ({detection['total_missing']}):")
        print("   (These images are in master_summary but their measurement files are missing)")
        for i, img in enumerate(detection['missing_images'][:30], 1):
            print(f"    {i}. {img}")
        if len(detection['missing_images']) > 30:
            print(f"    ... and {len(detection['missing_images']) - 30} more")
    
    if detection['orphaned_files']:
        print(f"\nORPHANED FILES ({detection['total_orphaned']}):")
        print("   (These measurement files exist but are not in master_summary)")
        for i, img in enumerate(detection['orphaned_files'][:20], 1):
            print(f"    {i}. {img}")
        if len(detection['orphaned_files']) > 20:
            print(f"    ... and {len(detection['orphaned_files']) - 20} more")
    
    if not detection['missing_images'] and not detection['orphaned_files']:
        print("\nNo issues found! Everything is clean.")
    else:
        print(f"\nACTION SUMMARY:")
        if detection['missing_images']:
            print(f"   • {len(detection['missing_images'])} images will be REMOVED from master_summary")
        if detection['orphaned_files']:
            print(f"   • {len(detection['orphaned_files'])} orphaned files found (they won't be deleted unless remove_orphaned=True)")
        print(f"\n   To apply changes, run cleanup_missing_files() with dry_run=False")