import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def normalize_radius_and_area(
    df: pd.DataFrame,
    root_radius: float,
    radius_col: str = 'radius_um',
    area_col: str = 'area_um2'
) -> pd.DataFrame:

    df = df.copy()
    
    if root_radius > 0:
        df['norm_radius'] = df[radius_col] / root_radius
    else:
        df['norm_radius'] = 0
    
    # Normalize area (could use different methods)
    max_area = df[area_col].max()
    if max_area > 0:
        df['norm_area'] = df[area_col] / max_area
    else:
        df['norm_area'] = 0
    
    return df


def bin_by_radius(
    df: pd.DataFrame,
    n_bins: int = 20,
    radius_col: str = 'norm_radius',
    value_col: str = 'area_um2'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Create bins from 0 to 1
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    mean_values = []
    std_values = []
    
    for i in range(n_bins):
        in_bin = (df[radius_col] >= bins[i]) & (df[radius_col] < bins[i+1])
        values_in_bin = df.loc[in_bin, value_col]
        
        if len(values_in_bin) > 0:
            mean_values.append(values_in_bin.mean())
            std_values.append(values_in_bin.std())
        else:
            mean_values.append(np.nan)
            std_values.append(np.nan)
    
    # Convert to numpy arrays
    mean_values = np.array(mean_values)
    std_values = np.array(std_values)
    
    # Handle NaN values by forward/backward fill
    mean_values = fill_nan_values(mean_values)
    std_values = fill_nan_values(std_values)
    
    return bin_centers, mean_values, std_values


def fill_nan_values(arr: np.ndarray) -> np.ndarray:

    arr = arr.copy()
    
    # Find first and last valid indices
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return arr
    
    first_valid = np.where(valid)[0][0]
    last_valid = np.where(valid)[0][-1]
    
    # Fill NaN values at the beginning with the first valid value
    for i in range(first_valid):
        arr[i] = arr[first_valid]
    
    # Fill NaN values at the end with the last valid value
    for i in range(last_valid + 1, len(arr)):
        arr[i] = arr[last_valid]
    
    # Fill internal NaN values with linear interpolation
    for i in range(first_valid, last_valid + 1):
        if np.isnan(arr[i]):
            # Find previous and next valid values
            prev_idx = i - 1
            while prev_idx >= 0 and np.isnan(arr[prev_idx]):
                prev_idx -= 1
            
            next_idx = i + 1
            while next_idx < len(arr) and np.isnan(arr[next_idx]):
                next_idx += 1
            
            if prev_idx >= 0 and next_idx < len(arr):
                # Linear interpolation
                weight = (i - prev_idx) / (next_idx - prev_idx)
                arr[i] = arr[prev_idx] * (1 - weight) + arr[next_idx] * weight
    
    return arr


def combine_quarters_to_roots(df: pd.DataFrame, id_column: str = 'root_identifier') -> pd.DataFrame:

    if id_column not in df.columns:
        logger.warning(f"Column {id_column} not found. Cannot combine quarters.")
        return df
    
    # Group by root identifier
    grouped = df.groupby(id_column)
    
    # Aggregate functions
    agg_funcs = {}
    
    for col in df.columns:
        if col == id_column:
            continue
        
        # Choose aggregation function based on column type
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Numeric columns: use mean for most, sum for counts
            if 'count' in col.lower() or 'n_' in col.lower():
                agg_funcs[col] = 'sum'
            else:
                agg_funcs[col] = 'mean'
        else:
            # Non-numeric: take first value
            agg_funcs[col] = 'first'
    
    # Perform aggregation
    result = grouped.agg(agg_funcs).reset_index()
    
    return result


def create_root_id(row: pd.Series) -> str:

    parts = []
    
    if 'plant_number' in row and pd.notna(row['plant_number']):
        parts.append(f"P{int(row['plant_number'])}")
    
    if 'treatment' in row and pd.notna(row['treatment']):
        parts.append(str(row['treatment']))
    
    if 'root_type' in row and pd.notna(row['root_type']):
        parts.append(str(row['root_type']))
    
    if 'root_number' in row and pd.notna(row['root_number']):
        parts.append(f"root{int(row['root_number'])}")
    
    if 'technical_replicate' in row and pd.notna(row['technical_replicate']):
        parts.append(f"({int(row['technical_replicate'])})")
    
    return "_".join(parts) if parts else "unknown"


def filter_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:

    if column not in df.columns:
        logger.warning(f"Column {column} not found. Returning original DataFrame.")
        return df
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        z_scores = np.abs((df[column] - mean) / std)
        filtered_df = df[z_scores < threshold]
    
    else:
        logger.warning(f"Unknown method: {method}. Returning original DataFrame.")
        return df
    
    n_removed = len(df) - len(filtered_df)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} outliers from column {column}")
    
    return filtered_df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    
    # Circularity (4π * area / perimeter²)
    if 'area_um2' in df.columns and 'perimeter_um' in df.columns:
        df['circularity'] = 4 * np.pi * df['area_um2'] / (df['perimeter_um'] ** 2)
    
    # Equivalent diameter (diameter of circle with same area)
    if 'area_um2' in df.columns:
        df['equivalent_diameter_um'] = 2 * np.sqrt(df['area_um2'] / np.pi)
    
    return df
