# Shared spline fitting and summary statistics.
#
# Holds the spline fit used in more than one place, so the feature
# definitions cannot drift between callers.

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def fit_spline_and_extract_features(
    x: np.ndarray,
    y: np.ndarray,
    smoothing: float = 0.1,
    min_cells: int = 5,
    n_points: int = 200
) -> Optional[Dict[str, float]]:

    # Remove NaN values
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < min_cells:
        return None
    
    # Sort by x
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    
    # Fit spline
    try:
        spline = UnivariateSpline(x, y, s=smoothing, ext='extrapolate')
    except Exception as e:
        logger.warning(f"Spline fitting failed: {e}")
        return None
    
    # Evaluate spline on smooth grid
    x_smooth = np.linspace(0, 1, n_points)
    y_smooth = spline(x_smooth)
    
    # 1. Peak height and position
    peak_idx = np.argmax(y_smooth)
    peak_height = y_smooth[peak_idx]
    peak_position = x_smooth[peak_idx]
    
    # 2. Rise slope (from stele to peak)
    stele_to_peak_mask = x_smooth <= peak_position
    x_rise = x_smooth[stele_to_peak_mask]
    y_rise = y_smooth[stele_to_peak_mask]
    
    if len(x_rise) > 1:
        slope, _, _, _, _ = linregress(x_rise, y_rise)
        rise_slope = slope
    else:
        derivative = spline.derivative()
        rise_slope = derivative(peak_position / 2) if peak_position > 0 else 0
    
    # 3. Decay slope (immediately after peak)
    decay_window = 0.10  # Look 10% ahead of peak
    decay_mask = (x_smooth >= peak_position) & (x_smooth <= peak_position + decay_window)
    x_decay = x_smooth[decay_mask]
    y_decay = y_smooth[decay_mask]
    
    if len(x_decay) > 1:
        slope, _, _, _, _ = linregress(x_decay, y_decay)
        decay_slope = slope
    else:
        derivative = spline.derivative()
        decay_slope = derivative(peak_position) if peak_position < 1 else 0
    
    # 4. Post-peak minimum
    post_peak_mask = x_smooth >= peak_position
    y_post_peak = y_smooth[post_peak_mask]
    
    if len(y_post_peak) > 0:
        post_peak_min = np.min(y_post_peak)
        post_peak_min_idx = np.argmin(y_post_peak)
        post_peak_min_x = x_smooth[post_peak_mask][post_peak_min_idx]
    else:
        post_peak_min = peak_height
        post_peak_min_x = peak_position
    
    # 5. Outer-edge value
    outer_edge_value = y_smooth[-1]
    
    # 6. Outer-rise magnitude (positive = rise, negative = decline)
    outer_rise_magnitude = outer_edge_value - post_peak_min
    
    # 7. Area under curve
    area_under_curve = np.trapz(y_smooth, x_smooth)
    
    return {
        'peak_height': peak_height,
        'peak_position': peak_position,
        'rise_slope': rise_slope,
        'decay_slope': decay_slope,
        'post_peak_min': post_peak_min,
        'post_peak_min_position': post_peak_min_x,
        'outer_edge_value': outer_edge_value,
        'outer_rise_magnitude': outer_rise_magnitude,
        'area_under_curve': area_under_curve
    }


def calculate_cv(data: np.ndarray, handle_zero_mean: str = 'nan') -> float:

    data = data[~np.isnan(data)]  # Remove NaN values
    
    if len(data) == 0:
        return np.nan
    
    mean = np.mean(data)
    std = np.std(data)
    
    if mean == 0:
        if handle_zero_mean == 'nan':
            return np.nan
        elif handle_zero_mean == 'inf':
            return np.inf
        else:  # 'zero'
            return 0.0
    
    return std / mean


def feature_statistics(df: pd.DataFrame, feature_column: str, group_by: Optional[str] = None) -> pd.DataFrame:

    if feature_column not in df.columns:
        raise ValueError(f"Column {feature_column} not found in DataFrame")
    
    if group_by is not None and group_by not in df.columns:
        raise ValueError(f"Group column {group_by} not found in DataFrame")
    
    def calc_stats(series: pd.Series) -> Dict[str, float]:
        values = series.dropna().values
        
        if len(values) == 0:
            return {
                'mean': np.nan,
                'std': np.nan,
                'cv': np.nan,
                'median': np.nan,
                'min': np.nan,
                'max': np.nan,
                'q25': np.nan,
                'q75': np.nan,
                'n': 0
            }
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'cv': calculate_cv(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'n': len(values)
        }
    
    if group_by is None:
        # Calculate statistics for the entire column
        stats = calc_stats(df[feature_column])
        return pd.DataFrame([stats])
    else:
        # Calculate statistics for each group
        results = []
        for group_name, group_df in df.groupby(group_by):
            stats = calc_stats(group_df[feature_column])
            stats[group_by] = group_name
            results.append(stats)
        
        result_df = pd.DataFrame(results)
        # Move group column to front
        cols = [group_by] + [col for col in result_df.columns if col != group_by]
        return result_df[cols]


def normalize_by_size(
    df: pd.DataFrame,
    feature_column: str,
    size_column: str,
    method: str = 'divide'
) -> pd.Series:

    if method == 'divide' or method == 'ratio':
        # Simple division
        return df[feature_column] / df[size_column]
    
    elif method == 'residual':
        # Linear regression residuals
        from scipy.stats import linregress
        
        # Remove NaN values
        valid_mask = df[feature_column].notna() & df[size_column].notna()
        x = df.loc[valid_mask, size_column].values
        y = df.loc[valid_mask, feature_column].values
        
        if len(x) < 2:
            return pd.Series(np.nan, index=df.index)
        
        # Fit linear regression
        slope, intercept, _, _, _ = linregress(x, y)
        
        # Calculate residuals
        predicted = slope * df[size_column] + intercept
        residuals = df[feature_column] - predicted
        
        return residuals
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def bin_data_by_variable(
    df: pd.DataFrame,
    bin_column: str,
    n_bins: int = 5,
    method: str = 'quantile'
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:

    df = df.copy()
    
    if bin_column not in df.columns:
        raise ValueError(f"Column {bin_column} not found in DataFrame")
    
    values = df[bin_column].dropna().values
    
    if len(values) < n_bins:
        logger.warning(f"Only {len(values)} values, cannot create {n_bins} bins")
        df['bin_group'] = 0
        return df, {0: (values.min(), values.max())}
    
    if method == 'quantile':
        # Create bins based on percentiles
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(values, percentiles)
    elif method == 'equal':
        # Create equal-width bins
        bin_edges = np.linspace(values.min(), values.max(), n_bins + 1)
    else:
        raise ValueError(f"Unknown binning method: {method}")
    
    # Ensure edges are unique
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # Assign bins
    df['bin_group'] = pd.cut(df[bin_column], bins=bin_edges, labels=False, include_lowest=True)
    
    # Create bin ranges dictionary
    bin_ranges = {}
    for i in range(n_bins):
        bin_ranges[i] = (bin_edges[i], bin_edges[i + 1])
    
    return df, bin_ranges


def calculate_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'pearson'
) -> pd.DataFrame:

    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation
    corr_matrix = df[columns].corr(method=method)
    
    return corr_matrix


def perform_regression(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    log_transform: bool = False
) -> Dict[str, Any]:

    # Remove NaN values
    valid_mask = df[x_column].notna() & df[y_column].notna()
    x = df.loc[valid_mask, x_column].values
    y = df.loc[valid_mask, y_column].values
    
    if len(x) < 2:
        return {
            'slope': np.nan,
            'intercept': np.nan,
            'r_value': np.nan,
            'p_value': np.nan,
            'std_err': np.nan,
            'r_squared': np.nan,
            'n': len(x)
        }
    
    if log_transform:
        # Remove zeros and negative values
        valid = (x > 0) & (y > 0)
        x = x[valid]
        y = y[valid]
        
        if len(x) < 2:
            return {
                'slope': np.nan,
                'intercept': np.nan,
                'r_value': np.nan,
                'p_value': np.nan,
                'std_err': np.nan,
                'r_squared': np.nan,
                'n': len(x)
            }
        
        x = np.log(x)
        y = np.log(y)
    
    # Perform regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'r_squared': r_value ** 2,
        'n': len(x)
    }
