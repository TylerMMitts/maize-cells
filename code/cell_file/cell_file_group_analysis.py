# Cell-file profiles pooled across images and groups.
#
# Averages the per-file cell size profiles so a whole population can be
# plotted as one curve, and compares those curves between groups.

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import stats

# Get project root
SCRIPT_DIR = Path(__file__).parent.absolute()  # code/cell_file/
CODE_DIR = SCRIPT_DIR.parent  # code/
PROJECT_ROOT = CODE_DIR.parent  # project root

# Use absolute paths (relative paths break when the caller's cwd isn't the project root)
CELL_FILE_COUNTS_FOLDER = str(PROJECT_ROOT / "results" / "cell_file" / "cell_file_counting")
OUTPUT_FOLDER = str(PROJECT_ROOT / "results" / "cell_file" / "cell_file_profiles")

# Options: 'derivative', 'cumulative', 'consensus'
CELL_FILE_METHOD = 'derivative'

ANGLE_TOLERANCE = 5  # degrees around each sampled angle

ASSIGNMENT_METHOD = 'derivative'  # This determines which file numbers to use for each cell

# Options: 'area_pixels' or 'area_um2'
AREA_COLUMN = 'area_um2' 

# Visualization options
ALPHA = 0.7  # Transparency for lines
LINE_WIDTH = 2.0  # Width of lines
MARKER_SIZE = 5  # Size of markers

# Color map for lines
COLOR_MAP = 'tab10'

# Y-axis label (auto-updates based on AREA_COLUMN)
if AREA_COLUMN == 'area_um2':
    Y_LABEL = 'Average Cell Area (µm²)'
else:
    Y_LABEL = 'Average Cell Area (pixels²)'


def load_cell_file_counts(counts_folder, method='consensus'):

    cell_file_counts = {}
    
    if not os.path.exists(counts_folder):
        print(f"Folder {counts_folder} does not exist")
        return cell_file_counts
    
    image_folders = [f for f in glob.glob(str(Path(counts_folder) / "*")) if os.path.isdir(f)]
    
    for folder in image_folders:
        image_name = os.path.basename(folder)
        count_file = os.path.join(folder, 'cell_file_count.txt')
        
        if os.path.exists(count_file):
            with open(count_file, 'r') as f:
                content = f.read()
                
                # Try multiple patterns for each method
                if method == 'derivative':
                    patterns = [
                        r'Derivative method: (\d+) files',
                        r'Derivative \(Growth Rate Analysis\): (\d+) files',
                        r'DERIVATIVE METHOD:\s*\n.*?(\d+) files',
                        r'Method 1.*?(\d+) files',
                        r'Derivative.*?(\d+)\s+files',
                    ]
                elif method == 'cumulative':
                    patterns = [
                        r'Cumulative method: (\d+) files',
                        r'Cumulative \(Distribution Analysis\): (\d+) files',
                        r'CUMULATIVE METHOD:\s*\n.*?(\d+) files',
                        r'Method 2.*?(\d+) files',
                        r'Cumulative.*?(\d+)\s+files',
                    ]
                else:  # consensus
                    patterns = [
                        r'CONSENSUS: (\d+) cell files',
                        r'Consensus: (\d+) files',
                        r'Final consensus: (\d+) files',
                        r'Consensus number of cell files: (\d+)',
                        r'Number of cell files \(consensus\): (\d+)',
                        r'CONSENSUS.*?(\d+)\s+cell files',
                    ]
                
                count = None
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        count = int(match.group(1))
                        break
                
                # If still not found, try to find any number after "files" or "cell files"
                if count is None:
                    # Look for patterns like "X files" or "X cell files"
                    matches = re.findall(r'(\d+)\s+(?:cell\s+)?files?', content)
                    if matches:
                        # Take the most likely one (usually the consensus or first number)
                        count = int(matches[0])
                
                if count is not None:
                    cell_file_counts[image_name] = count
                    print(f"Loaded {image_name}: {count} files")
                else:
                    print(f"Could not parse count from {count_file}")
    
    return cell_file_counts


def load_cell_file_counts_from_assignments(counts_folder, method='consensus'):

    cell_file_counts = {}
    
    if not os.path.exists(counts_folder):
        print(f"Folder {counts_folder} does not exist")
        return cell_file_counts
    
    image_folders = [f for f in glob.glob(str(Path(counts_folder) / "*")) if os.path.isdir(f)]
    
    for folder in image_folders:
        image_name = os.path.basename(folder)
        
        # Look for the cell assignments CSV file
        assignment_file = os.path.join(folder, 'cell_assignments.csv')
        if not os.path.exists(assignment_file):
            # Try alternative names
            alt_files = [
                os.path.join(folder, f'{image_name}_method_consensus.csv'),
                os.path.join(folder, f'{image_name}_method_derivative.csv'),
                os.path.join(folder, f'{image_name}_method_cumulative.csv'),
                os.path.join(folder, 'cell_assignments_all_methods.csv'),
            ]
            for alt_file in alt_files:
                if os.path.exists(alt_file):
                    assignment_file = alt_file
                    break
        
        if os.path.exists(assignment_file):
            try:
                df = pd.read_csv(assignment_file)
                
                # Find the column with cell file assignments
                file_col = None
                if method == 'consensus' and 'cell_file_consensus' in df.columns:
                    file_col = 'cell_file_consensus'
                elif method == 'derivative' and 'cell_file_derivative' in df.columns:
                    file_col = 'cell_file_derivative'
                elif method == 'cumulative' and 'cell_file_cumulative' in df.columns:
                    file_col = 'cell_file_cumulative'
                else:
                    # Try to find any cell_file column
                    for col in df.columns:
                        if 'cell_file' in col:
                            file_col = col
                            break
                
                if file_col:
                    # Count unique file numbers (excluding -1)
                    valid_files = df[df[file_col] >= 0][file_col]
                    if len(valid_files) > 0:
                        n_files = valid_files.nunique()
                        cell_file_counts[image_name] = n_files
                        print(f"Loaded {image_name}: {n_files} files (from assignments)")
                    else:
                        print(f"No valid assignments for {image_name}")
                else:
                    print(f"No cell_file column found in {assignment_file}")
                    
            except Exception as e:
                print(f"Could not read {assignment_file}: {e}")
        else:
            print(f"No assignment file found for {image_name}")
    
    return cell_file_counts


def load_cell_assignments(counts_folder, image_name, assignment_method='consensus'):

    # Try different possible file names
    possible_files = [
        os.path.join(counts_folder, image_name, 'cell_assignments.csv'),
        os.path.join(counts_folder, image_name, f'{image_name}_method_{assignment_method}.csv'),
        os.path.join(counts_folder, image_name, 'cell_assignments_all_methods.csv'),
    ]
    
    for assignment_file in possible_files:
        if os.path.exists(assignment_file):
            print(f"Found assignment file: {assignment_file}")
            df = pd.read_csv(assignment_file)
            
            # Find the column with cell file assignments
            col_name = f'cell_file_{assignment_method}'
            if col_name in df.columns:
                # Check if area_um2 column exists, fall back to area_pixels if not
                if AREA_COLUMN in df.columns:
                    return df[['cell_id', col_name, AREA_COLUMN]].copy()
                elif 'area_um2' in df.columns:
                    return df[['cell_id', col_name, 'area_um2']].copy()
                elif 'area_pixels' in df.columns:
                    return df[['cell_id', col_name, 'area_pixels']].copy()
                else:
                    print(f"No area column found")
                    return None
            else:
                # Try to find any cell_file column
                file_cols = [c for c in df.columns if 'cell_file' in c]
                if file_cols:
                    area_col = AREA_COLUMN if AREA_COLUMN in df.columns else 'area_pixels'
                    if area_col in df.columns:
                        return df[['cell_id', file_cols[0], area_col]].copy()
    
    return None


def compute_average_cell_size_per_file(df_assignments):

    if df_assignments is None or df_assignments.empty:
        return None, None
    
    # Find the file column and area column
    file_col = [c for c in df_assignments.columns if 'cell_file' in c][0]
    area_col = [c for c in df_assignments.columns if 'area' in c.lower()][0]
    
    # Filter out cells with negative file numbers
    df_valid = df_assignments[df_assignments[file_col] >= 0].copy()
    
    if df_valid.empty:
        return None, None
    
    # Group by file number and calculate mean area
    avg_sizes = df_valid.groupby(file_col)[area_col].mean().reset_index()
    avg_sizes = avg_sizes.sort_values(file_col)
    
    file_numbers = avg_sizes[file_col].values
    avg_areas = avg_sizes[area_col].values
    
    return file_numbers, avg_areas


def create_individual_profiles(image_profiles, output_folder, force_rebuild=False):
\
    os.makedirs(output_folder, exist_ok=True)

    unit_suffix = 'um2' if AREA_COLUMN == 'area_um2' else 'px2'
    created = 0
    skipped = 0

    for image_name, (file_numbers, avg_areas, n_files) in image_profiles.items():
        safe_name = re.sub(r'[^\w\-_\. ]', '_', image_name)
        output_path = os.path.join(output_folder, f'{safe_name}_profile_{unit_suffix}.png')

        if not force_rebuild and os.path.exists(output_path):
            skipped += 1
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the line
        ax.plot(file_numbers, avg_areas, 'o-', linewidth=1.5,
               markersize=4, color='steelblue', alpha=ALPHA)

        # Add labels and title
        ax.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
        ax.set_ylabel(Y_LABEL, fontsize=12, fontweight='bold')
        ax.set_title(f'{image_name}\n{len(file_numbers)} cell files detected', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Set x-axis to show integer ticks
        ax.set_xticks(range(len(file_numbers)))
        ax.set_xticklabels([f'{int(f)}' for f in file_numbers])

        # Add value labels on points
        if AREA_COLUMN == 'area_um2':
            label_format = lambda x: f'{x:.1f}'
        else:
            label_format = lambda x: f'{x:.0f}'

        for i, (x, y) in enumerate(zip(file_numbers, avg_areas)):
            ax.annotate(label_format(y), (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        created += 1
        print(f"  Saved: {safe_name}_profile_{unit_suffix}.png")

    print(f"Individual profiles: created {created} new, skipped {skipped} already existing")


def create_combined_profile(image_profiles, output_folder):

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get a color map
    cmap = plt.get_cmap(COLOR_MAP)
    colors = cmap(np.linspace(0, 1, len(image_profiles)))
    
    # Plot each image's profile
    for idx, (image_name, (file_numbers, avg_areas, n_files)) in enumerate(image_profiles.items()):
        color = colors[idx]
        short_name = image_name[:30] + '...' if len(image_name) > 30 else image_name
        ax.plot(file_numbers, avg_areas, 'o-', linewidth=LINE_WIDTH, 
               markersize=MARKER_SIZE, color=color, alpha=ALPHA, 
               label=f'{short_name} ({n_files} files)')
    
    ax.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
    ax.set_ylabel(Y_LABEL, fontsize=12, fontweight='bold')
    
    unit_text = "µm²" if AREA_COLUMN == "area_um2" else "pixels²"
    ax.set_title(f'Cell Size vs Cell File Number - All Images\n(Using {CELL_FILE_METHOD.upper()} method, {unit_text})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    unit_suffix = 'um2' if AREA_COLUMN == 'area_um2' else 'px2'
    plt.savefig(os.path.join(output_folder, f'all_images_combined_profile_{unit_suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: all_images_combined_profile_{unit_suffix}.png")


def create_normalized_combined_profile(image_profiles, output_folder):

    fig, ax = plt.subplots(figsize=(14, 8))
    
    cmap = plt.get_cmap(COLOR_MAP)
    colors = cmap(np.linspace(0, 1, len(image_profiles)))
    
    for idx, (image_name, (file_numbers, avg_areas, n_files)) in enumerate(image_profiles.items()):
        color = colors[idx]
        
        # Normalize cell sizes to 0-1 scale
        min_area = np.min(avg_areas)
        max_area = np.max(avg_areas)
        if max_area > min_area:
            normalized_areas = (avg_areas - min_area) / (max_area - min_area)
        else:
            normalized_areas = np.ones_like(avg_areas) * 0.5
        
        short_name = image_name[:30] + '...' if len(image_name) > 30 else image_name
        ax.plot(file_numbers, normalized_areas, 'o-', linewidth=LINE_WIDTH, 
               markersize=MARKER_SIZE, color=color, alpha=ALPHA,
               label=f'{short_name} ({n_files} files)')
    
    ax.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Cell Size (0-1 scale)', fontsize=12, fontweight='bold')
    ax.set_title(f'Normalized Cell Size vs Cell File Number - All Images\n(Using {CELL_FILE_METHOD.upper()} method)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'all_images_normalized_profile.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: all_images_normalized_profile.png")


def create_grouped_normalized_best_fit_profiles(image_profiles, output_folder):
    
    def linear(x, a, b):
        return a * x + b
    
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    def cubic(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d
    
    models = {
        'Linear': linear,
        'Quadratic': quadratic,
        'Cubic': cubic
    }
    
    # Group images by their file count
    grouped_by_count = {}
    for image_name, (file_numbers, avg_areas, n_files) in image_profiles.items():
        if n_files not in grouped_by_count:
            grouped_by_count[n_files] = []
        grouped_by_count[n_files].append((file_numbers, avg_areas, image_name))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    cmap = plt.get_cmap('tab10')
    file_counts = sorted(grouped_by_count.keys())
    colors = cmap(np.linspace(0, 1, len(file_counts)))
    
    fit_results = []
    
    for idx, n_files in enumerate(file_counts):
        color = colors[idx]
        profiles = grouped_by_count[n_files]
        
        # Align all profiles to the same length (n_files)
        aligned_profiles = []
        
        for file_numbers, avg_areas, image_name in profiles:
            if len(avg_areas) == n_files:
                aligned_profiles.append(avg_areas)
            else:
                # Interpolate to correct length if needed
                x_old = np.linspace(0, 1, len(avg_areas))
                x_new = np.linspace(0, 1, n_files)
                interp_func = interp1d(x_old, avg_areas, kind='linear', fill_value='extrapolate')
                interp_profile = interp_func(x_new)
                aligned_profiles.append(interp_profile)
        
        if not aligned_profiles:
            continue
        
        # Calculate the average profile for this group
        aligned_array = np.array(aligned_profiles)
        mean_profile = np.mean(aligned_array, axis=0)
        
        # Normalize X axis (file positions) to 0-1
        x_values = np.arange(n_files)
        x_normalized = x_values / (n_files - 1) if n_files > 1 else np.array([0.5])
        
        # Normalize Y axis (cell areas) to 0-1
        y_min = np.min(mean_profile)
        y_max = np.max(mean_profile)
        if y_max > y_min:
            y_normalized = (mean_profile - y_min) / (y_max - y_min)
        else:
            y_normalized = np.ones_like(mean_profile) * 0.5
        
        # Find the best fit model for this group's average profile
        best_r2 = -np.inf
        best_model_name = None
        best_params = None
        best_func = None
        
        for model_name, func in models.items():
            try:
                # Initial guesses
                if model_name == 'Linear':
                    p0 = [1, np.mean(y_normalized)]
                elif model_name == 'Quadratic':
                    p0 = [0, 1, np.mean(y_normalized)]
                else:  # Cubic
                    p0 = [0, 0, 1, np.mean(y_normalized)]
                
                popt, pcov = curve_fit(func, x_normalized, y_normalized, p0=p0, maxfev=10000)
                
                # Calculate R² on normalized data
                y_pred = func(x_normalized, *popt)
                ss_res = np.sum((y_normalized - y_pred) ** 2)
                ss_tot = np.sum((y_normalized - np.mean(y_normalized)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_name = model_name
                    best_params = popt
                    best_func = func
                    
            except Exception as e:
                continue
        
        # If no fit worked, use linear as fallback
        if best_func is None:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_normalized, y_normalized)
            best_model_name = 'Linear'
            best_params = [slope, intercept]
            best_func = linear
            best_r2 = r_value ** 2
        
        # Generate smooth best-fit line
        fit_x = np.linspace(0, 1, 200)
        fit_y = best_func(fit_x, *best_params)
        
        # Plot the normalized best-fit line for this group
        label = f'{n_files} files (n={len(profiles)} images, {best_model_name})'
        ax.plot(fit_x, fit_y, '-', linewidth=3, color=color, alpha=ALPHA, label=label)
        
        fit_results.append({
            'n_files': n_files,
            'n_images': len(profiles),
            'best_model': best_model_name,
            'r_squared': best_r2,
            'params': best_params
        })
    
    ax.set_xlabel('Normalized Cell File Position (0 = inner, 1 = outer)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Cell Size (0-1 scale)', fontsize=12, fontweight='bold')
    ax.set_title(f'Grouped Normalized Best-Fit Lines\n(Averaged profiles grouped by file count, {CELL_FILE_METHOD.upper()} method)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'grouped_normalized_best_fit_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: grouped_normalized_best_fit_profiles.png")
    
    # Save fit results to CSV
    fit_df = pd.DataFrame(fit_results)
    csv_path = os.path.join(output_folder, 'grouped_normalized_best_fit_results.csv')
    fit_df.to_csv(csv_path, index=False)
    print(f"Saved: grouped_normalized_best_fit_results.csv")
    
    # Print summary
    for result in fit_results:
        print(f"\n{result['n_files']} files ({result['n_images']} images):")
        print(f"  Best model: {result['best_model']}")
        print(f"  R²: {result['r_squared']:.4f}")
        param_str = ', '.join([f'{p:.4f}' for p in result['params']])
        print(f"  Parameters: [{param_str}]")
    
    return fit_results


def create_grouped_profiles_by_file_count(image_profiles, output_folder):

    # Group images by their max file number
    grouped_by_count = {}
    for image_name, (file_numbers, avg_areas, n_files) in image_profiles.items():
        if n_files not in grouped_by_count:
            grouped_by_count[n_files] = []
        grouped_by_count[n_files].append((image_name, file_numbers, avg_areas))
    
    # Create a plot for each file count group
    for n_files, profiles in sorted(grouped_by_count.items()):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        cmap = plt.get_cmap(COLOR_MAP)
        colors = cmap(np.linspace(0, 1, len(profiles)))
        
        for idx, (image_name, file_numbers, avg_areas) in enumerate(profiles):
            color = colors[idx]
            short_name = image_name[:35] + '...' if len(image_name) > 35 else image_name
            ax.plot(file_numbers, avg_areas, 'o-', linewidth=LINE_WIDTH, 
                   markersize=MARKER_SIZE, color=color, alpha=ALPHA,
                   label=short_name)
        
        ax.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
        ax.set_ylabel(Y_LABEL, fontsize=12, fontweight='bold')
        ax.set_title(f'Images with {n_files} Cell Files\n({len(profiles)} images)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        # Set x-axis to show all integers up to n_files-1
        ax.set_xticks(range(n_files))
        
        plt.tight_layout()
        unit_suffix = 'um2' if AREA_COLUMN == 'area_um2' else 'px2'
        plt.savefig(os.path.join(output_folder, f'group_{n_files}_files_{unit_suffix}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: group_{n_files}_files_{unit_suffix}.png")


def create_comparison_plot(image_profiles, output_folder):

    # Group by file count
    grouped_by_count = {}
    for image_name, (file_numbers, avg_areas, n_files) in image_profiles.items():
        if n_files not in grouped_by_count:
            grouped_by_count[n_files] = []
        grouped_by_count[n_files].append(avg_areas)
    
    # Need to align profiles to same length for averaging
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(grouped_by_count)))
    
    for idx, (n_files, profiles) in enumerate(sorted(grouped_by_count.items())):
        color = colors[idx]
        
        # Align all profiles to the same length (n_files)
        aligned_profiles = []
        for profile in profiles:
            if len(profile) == n_files:
                aligned_profiles.append(profile)
            else:
                # Interpolate to correct length
                x_old = np.linspace(0, 1, len(profile))
                x_new = np.linspace(0, 1, n_files)
                interp_func = interp1d(x_old, profile, kind='linear', fill_value='extrapolate')
                interp_profile = interp_func(x_new)
                aligned_profiles.append(interp_profile)
        
        if aligned_profiles:
            aligned_array = np.array(aligned_profiles)
            mean_profile = np.mean(aligned_array, axis=0)
            std_profile = np.std(aligned_array, axis=0)
            
            x_values = np.arange(n_files)
            ax.plot(x_values, mean_profile, 'o-', linewidth=2, markersize=6,
                   color=color, label=f'{n_files} files (n={len(profiles)})')
    
    ax.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
    ax.set_ylabel(Y_LABEL, fontsize=12, fontweight='bold')
    
    unit_text = "µm²" if AREA_COLUMN == "area_um2" else "pixels²"
    ax.set_title(f'Average Cell Size Profile by Number of Cell Files ({unit_text})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    unit_suffix = 'um2' if AREA_COLUMN == 'area_um2' else 'px2'
    plt.savefig(os.path.join(output_folder, f'file_count_group_comparison_{unit_suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: file_count_group_comparison_{unit_suffix}.png")


def create_average_by_file_count_plot(image_profiles, output_folder):

    # Group images by their file count
    grouped_by_count = {}
    for image_name, (file_numbers, avg_areas, n_files) in image_profiles.items():
        if n_files not in grouped_by_count:
            grouped_by_count[n_files] = []
        grouped_by_count[n_files].append((file_numbers, avg_areas, image_name))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get a color map for different file counts
    cmap = plt.get_cmap('tab10')
    file_counts = sorted(grouped_by_count.keys())
    colors = cmap(np.linspace(0, 1, len(file_counts)))
    
    # For each file count group, calculate the average profile
    for idx, n_files in enumerate(file_counts):
        color = colors[idx]
        profiles = grouped_by_count[n_files]
        
        # Align all profiles to the same length (n_files)
        aligned_profiles = []
        
        for file_numbers, avg_areas, image_name in profiles:
            if len(avg_areas) == n_files:
                aligned_profiles.append(avg_areas)
            else:
                # Interpolate to correct length if needed
                x_old = np.linspace(0, 1, len(avg_areas))
                x_new = np.linspace(0, 1, n_files)
                interp_func = interp1d(x_old, avg_areas, kind='linear', fill_value='extrapolate')
                interp_profile = interp_func(x_new)
                aligned_profiles.append(interp_profile)
        
        if aligned_profiles:
            aligned_array = np.array(aligned_profiles)
            mean_profile = np.mean(aligned_array, axis=0)
            std_profile = np.std(aligned_array, axis=0)
            
            x_values = np.arange(n_files)
            
            # Plot mean profile
            ax.plot(x_values, mean_profile, '-', linewidth=2.5, 
                   color=color, label=f'{n_files} files (n={len(profiles)} images)')
            
            # Add markers at each point
            ax.plot(x_values, mean_profile, 'o', markersize=6, color=color, alpha=0.7)
    
    ax.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Average {Y_LABEL}', fontsize=12, fontweight='bold')
    ax.set_title(f'Average Cell Size Profile by Number of Cell Files\n(Using {CELL_FILE_METHOD.upper()} method)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Set x-axis ticks
    max_x = max(file_counts)
    ax.set_xticks(range(max_x))
    
    plt.tight_layout()
    unit_suffix = 'um2' if AREA_COLUMN == 'area_um2' else 'px2'
    output_path = os.path.join(output_folder, f'average_by_file_count_{unit_suffix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: average_by_file_count_{unit_suffix}.png")
    
    # Save the average values to CSV
    create_average_by_file_count_csv(grouped_by_count, output_folder, unit_suffix)


def create_average_by_file_count_csv(grouped_by_count, output_folder, unit_suffix):

    # Get all unique file counts sorted
    file_counts = sorted(grouped_by_count.keys())
    
    # Find the maximum number of files
    max_files = max(file_counts)
    
    # Initialize the data dictionary
    # First row will be the column headers (number of cell files)
    # Subsequent rows will be file_number, then values for each group
    data = {}
    
    # For each group (number of cell files)
    for n_files in file_counts:
        profiles = grouped_by_count[n_files]
        
        # Align and average profiles for this group
        aligned_profiles = []
        for file_numbers, avg_areas, image_name in profiles:
            if len(avg_areas) == n_files:
                aligned_profiles.append(avg_areas)
            else:
                x_old = np.linspace(0, 1, len(avg_areas))
                x_new = np.linspace(0, 1, n_files)
                interp_func = interp1d(x_old, avg_areas, kind='linear', fill_value='extrapolate')
                interp_profile = interp_func(x_new)
                aligned_profiles.append(interp_profile)
        
        if aligned_profiles:
            aligned_array = np.array(aligned_profiles)
            mean_profile = np.mean(aligned_array, axis=0)
            
            # Store the average profile for this group
            data[n_files] = mean_profile
    
    # Now create the CSV data structure
    # Row 0 (header): file count values
    # Rows 1..max_files-1: file number, then values for each group
    
    # Create the DataFrame
    df_dict = {'file_number': []}
    
    # Add columns for each file count group
    for n_files in file_counts:
        df_dict[str(n_files)] = []
    
    # Fill in the data
    for file_num in range(max_files):
        df_dict['file_number'].append(file_num)
        
        for n_files in file_counts:
            if n_files in data and file_num < len(data[n_files]):
                df_dict[str(n_files)].append(data[n_files][file_num])
            else:
                df_dict[str(n_files)].append(np.nan)  # No data for this file number
    
    # Create DataFrame and save
    df = pd.DataFrame(df_dict)
    
    # Reorder columns to have file_number first, then the file count columns in order
    columns = ['file_number'] + [str(n) for n in file_counts]
    df = df[columns]
    
    # Save to CSV
    csv_path = os.path.join(output_folder, f'average_by_file_count_{unit_suffix}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: average_by_file_count_{unit_suffix}.csv")
    
    # Also save a transposed version for easier reading (optional)
    # This creates a CSV where rows are the file count groups
    df_transposed = df.set_index('file_number').T
    transposed_path = os.path.join(output_folder, f'average_by_file_count_transposed_{unit_suffix}.csv')
    df_transposed.to_csv(transposed_path)
    print(f"  Saved: average_by_file_count_transposed_{unit_suffix}.csv")


def create_average_normalized_by_file_count(image_profiles, output_folder):

    # Group images by their file count
    grouped_by_count = {}
    for image_name, (file_numbers, avg_areas, n_files) in image_profiles.items():
        if n_files not in grouped_by_count:
            grouped_by_count[n_files] = []
        grouped_by_count[n_files].append((file_numbers, avg_areas, image_name))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get a color map for different file counts
    cmap = plt.get_cmap('tab10')
    file_counts = sorted(grouped_by_count.keys())
    colors = cmap(np.linspace(0, 1, len(file_counts)))
    
    # For each file count group, calculate the average normalized profile
    for idx, n_files in enumerate(file_counts):
        color = colors[idx]
        profiles = grouped_by_count[n_files]
        
        # Align all profiles to the same length (n_files)
        aligned_profiles = []
        
        for file_numbers, avg_areas, image_name in profiles:
            if len(avg_areas) == n_files:
                aligned_profiles.append(avg_areas)
            else:
                x_old = np.linspace(0, 1, len(avg_areas))
                x_new = np.linspace(0, 1, n_files)
                interp_func = interp1d(x_old, avg_areas, kind='linear', fill_value='extrapolate')
                interp_profile = interp_func(x_new)
                aligned_profiles.append(interp_profile)
        
        if aligned_profiles:
            aligned_array = np.array(aligned_profiles)
            mean_profile = np.mean(aligned_array, axis=0)
            
            # Normalize the mean profile
            min_val = np.min(mean_profile)
            max_val = np.max(mean_profile)
            if max_val > min_val:
                normalized_profile = (mean_profile - min_val) / (max_val - min_val)
            else:
                normalized_profile = np.ones_like(mean_profile) * 0.5
            
            x_values = np.arange(n_files)
            
            # Plot normalized profile
            ax.plot(x_values, normalized_profile, '-o', linewidth=2.5, 
                   markersize=6, color=color, alpha=0.8,
                   label=f'{n_files} files (n={len(profiles)} images)')
    
    ax.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Cell Area (0-1 scale)', fontsize=12, fontweight='bold')
    ax.set_title(f'Normalized Average Cell Size Profile by Number of Cell Files\n(Using {CELL_FILE_METHOD.upper()} method)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    unit_suffix = 'um2' if AREA_COLUMN == 'area_um2' else 'px2'
    output_path = os.path.join(output_folder, f'average_normalized_by_file_count_{unit_suffix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: average_normalized_by_file_count_{unit_suffix}.png")


def create_average_by_file_count_with_best_fit(image_profiles, output_folder):

    def cubic(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    # Group images by their file count
    grouped_by_count = {}
    for image_name, (file_numbers, avg_areas, n_files) in image_profiles.items():
        if n_files not in grouped_by_count:
            grouped_by_count[n_files] = []
        grouped_by_count[n_files].append((file_numbers, avg_areas, image_name))

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 9))

    # Get a color map for different file counts
    cmap = plt.get_cmap('tab10')
    file_counts = sorted(grouped_by_count.keys())
    colors = cmap(np.linspace(0, 1, len(file_counts)))

    # Store fit results for summary
    all_fit_results = []
    cubic_fit_results = []
    non_cubic_groups = []   # Groups skipped for having < 4 file positions

    # For each file count group
    for idx, n_files in enumerate(file_counts):
        color = colors[idx]
        profiles = grouped_by_count[n_files]

        # Align all profiles to the same length (n_files)
        aligned_profiles = []

        for file_numbers, avg_areas, image_name in profiles:
            if len(avg_areas) == n_files:
                aligned_profiles.append(avg_areas)
            else:
                x_old = np.linspace(0, 1, len(avg_areas))
                x_new = np.linspace(0, 1, n_files)
                interp_func = interp1d(x_old, avg_areas, kind='linear', fill_value='extrapolate')
                interp_profile = interp_func(x_new)
                aligned_profiles.append(interp_profile)

        if not aligned_profiles:
            continue

        aligned_array = np.array(aligned_profiles)
        mean_profile = np.mean(aligned_array, axis=0)

        x_values = np.arange(n_files)

        # A cubic (4 parameters) cannot be uniquely fit with fewer than 4 points
        if n_files < 4:
            non_cubic_groups.append({
                'n_files': n_files,
                'n_images': len(profiles),
                'reason': 'fewer than 4 file positions (cubic fit not mathematically possible)'
            })
            continue

        try:
            p0 = [0, 0, 1, np.mean(mean_profile)]
            popt, pcov = curve_fit(cubic, x_values, mean_profile, p0=p0, maxfev=10000)
            y_pred = cubic(x_values, *popt)
            ss_res = np.sum((mean_profile - y_pred) ** 2)
            ss_tot = np.sum((mean_profile - np.mean(mean_profile)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        except Exception as e:
            non_cubic_groups.append({
                'n_files': n_files,
                'n_images': len(profiles),
                'reason': f'cubic fit failed to converge: {e}'
            })
            continue

        best_params = popt
        best_r2 = r2

        # Store all fit results
        all_fit_results.append({
            'n_files': n_files,
            'n_images': len(profiles),
            'best_model': 'Cubic',
            'r_squared': best_r2,
            'params': best_params
        })

        # Generate best-fit line
        fit_x = np.linspace(x_values.min(), x_values.max(), 100)
        fit_y = cubic(fit_x, *best_params)

        # Format label with model type, R², and number of images
        label = f'{n_files} files (n={len(profiles)} images)\nCubic: R² = {best_r2:.3f}'
        ax.plot(fit_x, fit_y, '-', linewidth=3, color=color, label=label)

        # Store cubic fit results
        cubic_fit_results.append({
            'n_files': n_files,
            'n_images': len(profiles),
            'best_model': 'Cubic',
            'r_squared': best_r2,
            'params': best_params
        })

    # Add a note to the plot if any groups were skipped
    if non_cubic_groups:
        skipped_n_images = sum(g['n_images'] for g in non_cubic_groups)
        filtered_text = (f"Skipped (< 4 file positions, cubic not possible): "
                         f"{', '.join([str(g['n_files']) for g in non_cubic_groups])} "
                         f"files ({skipped_n_images} images)")
        ax.text(0.02, 0.02, filtered_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Average {Y_LABEL}', fontsize=12, fontweight='bold')
    ax.set_title(f'Cubic Best-Fit Lines by Number of Cell Files\n(Cubic fit forced for every group with >= 4 file positions, {CELL_FILE_METHOD.upper()} method)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    
    # Set x-axis ticks
    if cubic_fit_results:
        max_x = max([result['n_files'] for result in cubic_fit_results])
        ax.set_xticks(range(max_x))
    else:
        # Fallback if no cubic fits found
        max_x = max(file_counts)
        ax.set_xticks(range(max_x))
    
    plt.tight_layout()
    unit_suffix = 'um2' if AREA_COLUMN == 'area_um2' else 'px2'
    output_path = os.path.join(output_folder, f'average_by_file_count_cubic_only_{unit_suffix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: average_by_file_count_cubic_only_{unit_suffix}.png")
    
    # Save cubic fit results to CSV
    if cubic_fit_results:
        fit_results_list = []
        for result in cubic_fit_results:
            fit_results_list.append({
                'n_files': result['n_files'],
                'n_images': result['n_images'],
                'best_model': result['best_model'],
                'r_squared': result['r_squared'],
                'param_1': result['params'][0] if len(result['params']) > 0 else np.nan,
                'param_2': result['params'][1] if len(result['params']) > 1 else np.nan,
                'param_3': result['params'][2] if len(result['params']) > 2 else np.nan,
                'param_4': result['params'][3] if len(result['params']) > 3 else np.nan,
            })
        
        fit_df = pd.DataFrame(fit_results_list)
        fit_csv_path = os.path.join(output_folder, f'cubic_fit_results_by_file_count_{unit_suffix}.csv')
        fit_df.to_csv(fit_csv_path, index=False)
        print(f"  Saved: cubic_fit_results_by_file_count_{unit_suffix}.csv")
    
    # Print summary    
    if cubic_fit_results:
        print(f"\nSHOWING {len(cubic_fit_results)} CUBIC FITS:")
        for result in cubic_fit_results:
            print(f"\n{result['n_files']} files ({result['n_images']} images):")
            print(f"  Best model: {result['best_model']}")
            print(f"  R²: {result['r_squared']:.4f}")
            param_str = ', '.join([f'{p:.4f}' for p in result['params']])
            print(f"  Parameters: [{param_str}]")
    else:
        print("\nNo cubic fits found")
    
    if non_cubic_groups:
        skipped_n_images = sum(g['n_images'] for g in non_cubic_groups)
        print(f"\nSKIPPED ({len(non_cubic_groups)} groups, {skipped_n_images} images):")
        for result in non_cubic_groups:
            print(f"  {result['n_files']} files ({result['n_images']} images): {result['reason']}")
    
    
    return cubic_fit_results, non_cubic_groups


def create_normalized_cubic_best_fit(image_profiles, output_folder):

    def cubic(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    # STEP 1: Group images by file count, and split by whether a cubic fit
    # is even mathematically possible (needs >= 4 distinct file positions)

    grouped_by_count = {}
    for image_name, (file_numbers, avg_areas, n_files) in image_profiles.items():
        if n_files not in grouped_by_count:
            grouped_by_count[n_files] = []
        grouped_by_count[n_files].append((file_numbers, avg_areas, image_name))

    cubic_groups = []
    non_cubic_groups = []
    group_profiles = {}

    for n_files, profiles in grouped_by_count.items():
        aligned_profiles = []
        for file_numbers, avg_areas, image_name in profiles:
            if len(avg_areas) == n_files:
                aligned_profiles.append(avg_areas)
            else:
                x_old = np.linspace(0, 1, len(avg_areas))
                x_new = np.linspace(0, 1, n_files)
                interp_func = interp1d(x_old, avg_areas, kind='linear', fill_value='extrapolate')
                interp_profile = interp_func(x_new)
                aligned_profiles.append(interp_profile)

        if not aligned_profiles:
            continue

        aligned_array = np.array(aligned_profiles)
        mean_profile = np.mean(aligned_array, axis=0)
        group_profiles[n_files] = mean_profile

        if n_files >= 4:
            cubic_groups.append(n_files)
        else:
            non_cubic_groups.append(n_files)

    cubic_groups.sort()
    non_cubic_groups.sort()

    print(f"\nCubic groups (kept): {cubic_groups}")
    print(f"Groups skipped (< 4 file positions, cubic not possible): {non_cubic_groups} "
          f"({sum(len(grouped_by_count[n]) for n in non_cubic_groups)} images)")

    # STEP 2: Fit cubic to each group, then NORMALIZE THE CURVE
    
    if not cubic_groups:
        print("\nNo cubic groups found! No plot will be created.")
        return [], []
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(cubic_groups)))
    
    normalized_results = []
    failed_groups = []
    
    for idx, n_files in enumerate(cubic_groups):
        color = colors[idx]
        profiles = grouped_by_count[n_files]
        mean_profile = group_profiles[n_files]
        
        # Use RAW x values (file numbers)
        x_raw = np.arange(n_files)
        
        # Fit cubic to the raw data
        try:
            p0 = [0.1, -0.1, 1.0, np.mean(mean_profile)]
            popt, pcov = curve_fit(cubic, x_raw, mean_profile, p0=p0, maxfev=10000)
            
            # Generate the full cubic curve
            fit_x = np.linspace(x_raw.min(), x_raw.max(), 300)
            fit_y = cubic(fit_x, *popt)
            
            # Calculate R² on the raw data
            y_pred = cubic(x_raw, *popt)
            ss_res = np.sum((mean_profile - y_pred) ** 2)
            ss_tot = np.sum((mean_profile - np.mean(mean_profile)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            # NORMALIZE THE CURVE ITSELF (min -> 0, max -> 1)
            curve_min = np.min(fit_y)
            curve_max = np.max(fit_y)
            
            if curve_max > curve_min:
                fit_y_normalized = (fit_y - curve_min) / (curve_max - curve_min)
            else:
                fit_y_normalized = np.ones_like(fit_y) * 0.5
            
            # Also normalize the data points using the SAME curve min/max
            # This ensures the data points align with the curve
            y_normalized = (mean_profile - curve_min) / (curve_max - curve_min)
            
            # Plot the normalized curve (now perfectly 0-1)
            label = f'{n_files} files (n={len(profiles)} images)\nR² = {r2:.3f}'
            ax.plot(fit_x, fit_y_normalized, '-', linewidth=3, color=color, label=label)
            
            # Plot normalized data points
            ax.plot(x_raw, y_normalized, 'o', markersize=5, 
                   color=color, alpha=0.4, label='_nolegend_')
            
            normalized_results.append({
                'n_files': n_files,
                'n_images': len(profiles),
                'a': popt[0],
                'b': popt[1],
                'c': popt[2],
                'd': popt[3],
                'r_squared': r2,
                'curve_min_raw': curve_min,
                'curve_max_raw': curve_max,
                'curve_min_norm': np.min(fit_y_normalized),
                'curve_max_norm': np.max(fit_y_normalized)
            })
            
        except Exception as e:
            print(f"  Warning: Fit failed for {n_files} files: {e}")
            failed_groups.append(n_files)
            # Fallback: plot connecting line through normalized data
            x_raw = np.arange(n_files)
            min_val = np.min(mean_profile)
            max_val = np.max(mean_profile)
            if max_val > min_val:
                y_norm = (mean_profile - min_val) / (max_val - min_val)
            else:
                y_norm = np.ones_like(mean_profile) * 0.5
            ax.plot(x_raw, y_norm, 'o-', linewidth=2, color=color, alpha=0.7,
                   label=f'{n_files} files (fallback)')
    
    # Add note about excluded groups
    if non_cubic_groups:
        excluded_text = f"Excluded (non-cubic): {', '.join([str(g) for g in non_cubic_groups])}"
        ax.text(0.02, 0.02, excluded_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Cell File Number (0 = closest to center)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Cell Area (0-1 scale, curve normalized)', fontsize=12, fontweight='bold')
    ax.set_title(f'Normalized Cubic Best-Fit Lines\n(Each curve normalized to its own min=0, max=1, {CELL_FILE_METHOD.upper()} method)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(-0.5, max(cubic_groups) + 0.5)
    ax.set_ylim(-0.05, 1.05)
    
    # Set x-axis ticks
    max_files = max(cubic_groups)
    ax.set_xticks(range(max_files + 1))
    
    plt.tight_layout()
    unit_suffix = 'um2' if AREA_COLUMN == 'area_um2' else 'px2'
    output_path = os.path.join(output_folder, f'normalized_cubic_best_fit_{unit_suffix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: normalized_cubic_best_fit_{unit_suffix}.png")
    
    # Save normalized fit results to CSV
    if normalized_results:
        fit_df = pd.DataFrame(normalized_results)
        csv_path = os.path.join(output_folder, f'normalized_cubic_fit_results_{unit_suffix}.csv')
        fit_df.to_csv(csv_path, index=False)
        print(f"Saved: normalized_cubic_fit_results_{unit_suffix}.csv")
    
    # Print summary
    print("NORMALIZED CUBIC BEST-FIT RESULTS (Curve normalized to 0-1)")
    if normalized_results:
        print(f"\nSHOWING {len(normalized_results)} CUBIC FITS:")
        for result in normalized_results:
            print(f"\n{result['n_files']} files ({result['n_images']} images):")
            print(f"R²: {result['r_squared']:.4f}")
            print(f"Curve raw range: {result['curve_min_raw']:.2f} - {result['curve_max_raw']:.2f} {Y_LABEL.split()[-1]}")
            print(f"Curve normalized range: {result['curve_min_norm']:.4f} - {result['curve_max_norm']:.4f}")
            print(f"Parameters: a={result['a']:.4f}, b={result['b']:.4f}, c={result['c']:.4f}, d={result['d']:.4f}")
    if non_cubic_groups:
        print(f"\nEXCLUDED (non-cubic): {', '.join([str(g) for g in non_cubic_groups])}")
    
    return normalized_results, non_cubic_groups

def print_average_statistics(image_profiles):

    # Group by file count
    grouped_by_count = {}
    for image_name, (file_numbers, avg_areas, n_files) in image_profiles.items():
        if n_files not in grouped_by_count:
            grouped_by_count[n_files] = []
        grouped_by_count[n_files].append(avg_areas)
    
    print("AVERAGE STATISTICS BY NUMBER OF CELL FILES")
    
    for n_files in sorted(grouped_by_count.keys()):
        profiles = grouped_by_count[n_files]
        
        # Calculate mean of the last value (outermost file) for each image
        last_values = [profile[-1] for profile in profiles if len(profile) > 0]
        first_values = [profile[0] for profile in profiles if len(profile) > 0]
        
        if last_values and first_values:
            mean_first = np.mean(first_values)
            mean_last = np.mean(last_values)
            mean_change = mean_last - mean_first


def print_summary_statistics(image_profiles):
    
    all_areas = []
    for image_name, (file_numbers, avg_areas, n_files) in image_profiles.items():
        all_areas.extend(avg_areas)
        print(f"\n{image_name}:")
        print(f"Cell files: {n_files}")
        print(f"Mean cell area: {np.mean(avg_areas):.1f} ± {np.std(avg_areas):.1f} {Y_LABEL.split()[-1]}")
        print(f"Min cell area: {np.min(avg_areas):.1f} {Y_LABEL.split()[-1]}")
        print(f"Max cell area: {np.max(avg_areas):.1f} {Y_LABEL.split()[-1]}")
        print(f"Size change from file 0 to {n_files-1}: {avg_areas[-1] - avg_areas[0]:.1f} {Y_LABEL.split()[-1]}")
    
    if all_areas:
        print(f"\n")
        print(f"OVERALL STATISTICS (all images):")
        print(f"Mean across all files: {np.mean(all_areas):.1f} ± {np.std(all_areas):.1f} {Y_LABEL.split()[-1]}")
        print(f"Range: {np.min(all_areas):.1f} - {np.max(all_areas):.1f} {Y_LABEL.split()[-1]}")


def main(force_rebuild=False):

    # Create output folder
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Try loading cell file counts from assignment files first (more reliable)
    cell_file_counts = load_cell_file_counts_from_assignments(CELL_FILE_COUNTS_FOLDER, CELL_FILE_METHOD)

    # If that fails, try the text file method
    if not cell_file_counts:
        cell_file_counts = load_cell_file_counts(CELL_FILE_COUNTS_FOLDER, CELL_FILE_METHOD)

    master_summary_path = PROJECT_ROOT / "results" / "master_summary.csv"
    if cell_file_counts and master_summary_path.exists():
        valid_images = set(pd.read_csv(master_summary_path)['image_name'])
        before_count = len(cell_file_counts)
        cell_file_counts = {img: n for img, n in cell_file_counts.items() if img in valid_images}
        orphaned_count = before_count - len(cell_file_counts)
        if orphaned_count > 0:
            print(f"Excluded {orphaned_count} orphaned images no longer in master_summary.csv "
                  f"(stale cell_file_counting data from deleted images)")

    if not cell_file_counts:
        print("\nNo cell file counts found")
        print("\nDebug: Checking folder contents...")
        if os.path.exists(CELL_FILE_COUNTS_FOLDER):
            print(f"  Folder exists: {CELL_FILE_COUNTS_FOLDER}")
            subfolders = [f for f in glob.glob(str(Path(CELL_FILE_COUNTS_FOLDER) / "*")) if os.path.isdir(f)]
            print(f"  Found {len(subfolders)} subfolders")
            for subfolder in subfolders[:3]:  # Show first 3
                print(f"    - {os.path.basename(subfolder)}")
                files = os.listdir(subfolder)
                print(f"      Files: {files}")
        else:
            print(f"Folder does not exist: {CELL_FILE_COUNTS_FOLDER}")
        return
    
    print(f"\nSuccessfully loaded {len(cell_file_counts)} images")
    
    # Process each image to get cell size per file number
    
    image_profiles = {}
    
    for image_name, n_files in cell_file_counts.items():
        
        # Load the cell assignments for this image
        df_assignments = load_cell_assignments(CELL_FILE_COUNTS_FOLDER, image_name, ASSIGNMENT_METHOD)
        
        if df_assignments is None:
            print(f"No assignment file found for {image_name}")
            print(f"Looking in: {os.path.join(CELL_FILE_COUNTS_FOLDER, image_name)}")
            continue
        
        # Compute average cell size per file number
        file_numbers, avg_areas = compute_average_cell_size_per_file(df_assignments)
        
        if file_numbers is None or len(file_numbers) == 0:
            print(f"No valid assignments for {image_name}")
            continue
        
        image_profiles[image_name] = (file_numbers, avg_areas, n_files)
        print(f"SUCCESS: {len(file_numbers)} files, {len(df_assignments)} cells")
    
    if not image_profiles:
        print("\nNo profiles could be created!")
        print("\nDebug: Check that the cell assignment CSV files exist in:")
        for image_name in cell_file_counts.keys():
            expected_path = os.path.join(CELL_FILE_COUNTS_FOLDER, image_name)
            print(f"  - {expected_path}")
            if os.path.exists(expected_path):
                print(f"EXISTS - Contents: {os.listdir(expected_path)}")
            else:
                print(f"DOES NOT EXIST")
        return
    
    # Print summary statistics
    print_summary_statistics(image_profiles)
    
    # Print average statistics by file count
    print_average_statistics(image_profiles)
    
    # Individual profiles
    create_individual_profiles(image_profiles, OUTPUT_FOLDER, force_rebuild=force_rebuild)
    
    # Combined profile (all images on one plot)
    create_combined_profile(image_profiles, OUTPUT_FOLDER)
    
    # Normalized combined profile
    create_normalized_combined_profile(image_profiles, OUTPUT_FOLDER)
    
    # NEW: Grouped normalized best-fit profiles (what you want)
    create_grouped_normalized_best_fit_profiles(image_profiles, OUTPUT_FOLDER)
    
    # Grouped by file count (individual plots per group)
    create_grouped_profiles_by_file_count(image_profiles, OUTPUT_FOLDER)
    
    # Comparison of averages by file count group
    create_comparison_plot(image_profiles, OUTPUT_FOLDER)
    
    # Average by file count (one plot with averages)
    create_average_by_file_count_plot(image_profiles, OUTPUT_FOLDER)
    
    # Cubic-only best-fit lines (filters out non-cubic artifacts)
    cubic_results, filtered_out = create_average_by_file_count_with_best_fit(image_profiles, OUTPUT_FOLDER)
    
    # Normalized average by file count
    create_average_normalized_by_file_count(image_profiles, OUTPUT_FOLDER)
    
    # Normalized cubic best-fit lines (global normalization)
    normalized_results, failed_groups = create_normalized_cubic_best_fit(image_profiles, OUTPUT_FOLDER)
    

if __name__ == "__main__":
    import sys
    force_rebuild = '--force-rebuild' in sys.argv or '-f' in sys.argv
    main(force_rebuild=force_rebuild)