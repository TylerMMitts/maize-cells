import os
import pandas as pd
import json
import glob
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Import the shared parse function
from code.util.file_utils import parse_image_name

logger = logging.getLogger(__name__)


def rebuild_master_summary(
    measurements_folder: str,
    cell_file_counts_folder: Optional[str] = None,
    output_path: Optional[str] = None,
    species: str = "Zea mays",
    population: str = "IBM",
    dataset_metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:

    print("REBUILDING MASTER SUMMARY")
    print(f"Measurements folder: {measurements_folder}")
    print(f"Species: {species}")
    print(f"Population: {population}")
    
    # Find all measurement CSV files
    csv_files = glob.glob(os.path.join(measurements_folder, "*_measurements.csv"))
    
    if not csv_files:
        print(f"No measurement CSV files found in {measurements_folder}")
        return None
    
    print(f"Found {len(csv_files)} measurement files")
    
    # Load image summaries
    json_files = glob.glob(os.path.join(measurements_folder, "*_image_summary.json"))
    print(f"Found {len(json_files)} image summary files")
    
    # Load image summaries into a dictionary
    image_summaries = {}
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                summary = json.load(f)
                image_name = summary.get('image_name', '')
                if image_name:
                    base_name = os.path.splitext(image_name)[0]
                    # Remove _centers suffix if present
                    if base_name.endswith('_centers'):
                        base_name = base_name[:-8]
                    image_summaries[base_name] = summary
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
    
    # Process each measurement file
    master_data = []
    failed = []
    
    for csv_path in csv_files:
        base_name = os.path.basename(csv_path).replace('_measurements.csv', '')
        
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                print(f"  Skipping {base_name}: empty")
                continue
            
            # Use shared parse_image_name function
            parsed = parse_image_name(base_name)
            summary = image_summaries.get(base_name, {})
            
            # Calculate file count
            if 'cell_file' in df.columns:
                valid_files = df[df['cell_file'] >= 0]['cell_file']
                n_files = valid_files.nunique() if not valid_files.empty else 0
            elif 'file_number' in df.columns:
                n_files = df['file_number'].nunique()
            elif 'cell_file_derivative' in df.columns:
                valid_files = df[df['cell_file_derivative'] >= 0]['cell_file_derivative']
                n_files = valid_files.nunique() if not valid_files.empty else 0
            else:
                n_files = 0
            
            # Calculate average cell area
            if 'area_um2' in df.columns:
                avg_area = df['area_um2'].mean()
                n_cells = len(df)
            elif 'area_pixels' in df.columns:
                conversion = summary.get('conversion_factor_pixels_to_um', 0.649)
                avg_area = (df['area_pixels'].mean()) * (conversion ** 2)
                n_cells = len(df)
            else:
                avg_area = 0
                n_cells = len(df)
            
            # Get stele area and root radius
            stele_area = summary.get('stele_area_um2', 0)
            root_radius = summary.get('root_radius_um', 0)
            
            # Calculate stele diameter
            stele_diameter = 2 * np.sqrt(stele_area / np.pi) if stele_area > 0 else 0
            
            # Get per-file average areas from cell file data
            per_file_areas = []
            per_file_count = 0
            
            if cell_file_counts_folder and os.path.exists(cell_file_counts_folder):
                # Try to load cell file data
                possible_paths = [
                    os.path.join(cell_file_counts_folder, base_name, "cell_assignments.csv"),
                    os.path.join(cell_file_counts_folder, base_name, f"{base_name}_method_derivative.csv"),
                    os.path.join(cell_file_counts_folder, base_name, "cell_assignments_all_methods.csv"),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        try:
                            cell_df = pd.read_csv(path)
                            if 'cell_file_derivative' in cell_df.columns:
                                for file_num in sorted(cell_df['cell_file_derivative'].unique()):
                                    if file_num >= 0:
                                        file_cells = cell_df[cell_df['cell_file_derivative'] == file_num]
                                        if 'area_um2' in file_cells.columns:
                                            avg_area_file = file_cells['area_um2'].mean()
                                            per_file_areas.append(round(avg_area_file, 2))
                                per_file_count = len(per_file_areas)
                                break
                        except Exception as e:
                            continue
            
            per_file_str = ','.join(map(str, per_file_areas)) if per_file_areas else ''
            
            # Build row data with NEW column structure
            row_data = {
                # Identifiers
                'image_name': base_name,
                'quadrant': parsed.get('quadrant', 'unknown'),
                'plot_number': parsed.get('plot_number'),
                'plant_number': parsed.get('plant_number'),
                'root_number': parsed.get('root_number'),
                'technical_replicate': parsed.get('technical_replicate'),
                
                # Experimental design
                'treatment': parsed.get('treatment', 'unknown'),
                'root_type': parsed.get('root_type', 'unknown'),
                'species': species,
                'population': population,
                
                # Root measurements
                'file_count': n_files,
                'average_cell_area_um2': round(avg_area, 2) if avg_area > 0 else 0,
                'n_cells': n_cells,
                'stele_area_um2': round(stele_area, 2) if stele_area > 0 else 0,
                'root_radius_um': round(root_radius, 2) if root_radius > 0 else 0,
                'stele_diameter_um': round(stele_diameter, 2),
                'per_file_avg_areas_um2': per_file_str,
                'per_file_count': per_file_count,
                
                # Metadata tracking
                'measurements_csv': os.path.basename(csv_path),
                'has_image_summary': base_name in image_summaries,
                'has_cell_assignments': n_files > 0
            }
            
            # Add any additional metadata from dataset_metadata
            if dataset_metadata:
                for key, value in dataset_metadata.items():
                    if key not in row_data:
                        row_data[key] = value
            
            master_data.append(row_data)
            print(f"  Processed {base_name}: {n_files} files, {n_cells} cells, avg area {avg_area:.1f}µm²")
            
        except Exception as e:
            print(f"  Error processing {base_name}: {e}")
            failed.append(base_name)
    
    if not master_data:
        print("No data to save")
        return None
    
    # Create DataFrame with new column order
    column_order = [
        'image_name', 
        'quadrant', 
        'plot_number', 
        'plant_number', 
        'root_number',
        'technical_replicate', 
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
    
    # Add any additional metadata columns
    if dataset_metadata:
        for key in dataset_metadata.keys():
            if key not in column_order:
                column_order.append(key)
    
    # Create DataFrame and reorder columns
    df = pd.DataFrame(master_data)
    
    # Only keep columns that exist
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    # Sort the dataframe
    df = df.sort_values(['plant_number', 'root_number', 'quadrant']).reset_index(drop=True)
    
    print("REBUILD SUMMARY")
    print(f"Total images: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    if failed:
        print(f"\nFailed to process {len(failed)} images:")
        for img in failed[:10]:
            print(f"  - {img}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nMaster summary saved to: {output_path}")
        print(f"Total images summarized: {len(df)}")
        
        # Print statistics
        print("SUMMARY STATISTICS")
        print(f"Total images: {len(df)}")
        if 'plant_number' in df.columns:
            print(f"Plants (biological replicates): {df['plant_number'].nunique()}")
        if 'root_number' in df.columns:
            print(f"Roots: {df['root_number'].nunique()}")
        print(f"Images with file_count > 0: {len(df[df['file_count'] > 0])}")
        print(f"Average cells per image: {df['n_cells'].mean():.1f}")
        print(f"Average cell area: {df['average_cell_area_um2'].mean():.1f} µm²")
        print(f"Average stele area: {df['stele_area_um2'].mean():.1f} µm²")
        print(f"Average root radius: {df['root_radius_um'].mean():.1f} µm")
        
        print("\nQuadrant distribution:")
        quadrant_counts = df['quadrant'].value_counts()
        for quadrant, count in quadrant_counts.items():
            print(f"  {quadrant}: {count} images")
        
        print(f"\nSpecies: {df['species'].unique()}")
        print(f"Populations: {df['population'].unique()}")
    
    return df