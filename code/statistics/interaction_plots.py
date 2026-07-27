import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_data(master_path="results/master_summary.csv", cell_file_dir="results/cell_file/cell_file_counting"):

    # Load master summary
    df_master = pd.read_csv(master_path)
    
    # Extract first file areas from cell assignments
    first_file_data = {}
    cell_files = list(Path(cell_file_dir).rglob('cell_assignments.csv'))
    
    for cell_file in cell_files:
        image_name = cell_file.parent.name
        try:
            df_cells = pd.read_csv(cell_file)
            if 'cell_file_derivative' in df_cells.columns:
                first_file_cells = df_cells[df_cells['cell_file_derivative'] == 0]
                if len(first_file_cells) > 0:
                    first_file_data[image_name] = {
                        'first_file_avg_area_um2': first_file_cells['area_um2'].mean(),
                        'first_file_n_cells': len(first_file_cells)
                    }
        except:
            continue
    
    df_first_file = pd.DataFrame.from_dict(first_file_data, orient='index')
    df_first_file.index.name = 'image_name'
    df_first_file = df_first_file.reset_index()
    
    # Merge with master summary
    df_merged = df_master.merge(df_first_file, on='image_name', how='inner')
    
    # Aggregate
    df_agg = df_merged.groupby('image_name').agg({
        'first_file_avg_area_um2': 'mean',
        'stele_area_um2': 'mean',
        'root_radius_um': 'mean',
        'file_count': 'mean',
        'average_cell_area_um2': 'mean',
        'treatment': 'first',
        'root_type': 'first',
        'species': 'first',
        'population': 'first',
        'plant_number': 'first',
        'root_number': 'first'
    }).reset_index()
    
    # Clean data
    df_clean = df_agg.dropna(subset=['first_file_avg_area_um2', 'stele_area_um2', 'file_count', 'average_cell_area_um2'])
    df_clean = df_clean[df_clean['first_file_avg_area_um2'] > 0]
    df_clean = df_clean[df_clean['stele_area_um2'] > 0]
    df_clean = df_clean[df_clean['file_count'] > 0]
    df_clean = df_clean[df_clean['average_cell_area_um2'] > 0]
    
    # Round file_count to integer
    df_clean['file_count'] = df_clean['file_count'].round().astype(int)
    
    return df_clean


def get_automatic_columns(df):

    columns = {}
    
    # Auto-detect population column
    pop_candidates = ['population', 'Population', 'genotype', 'Genotype']
    for col in pop_candidates:
        if col in df.columns:
            columns['population'] = col
            break
    if 'population' not in columns:
        columns['population'] = 'population'  # Default, will be handled gracefully
    
    # Auto-detect treatment column
    treatment_candidates = ['treatment', 'Treatment']
    for col in treatment_candidates:
        if col in df.columns:
            columns['treatment'] = col
            break
    if 'treatment' not in columns:
        columns['treatment'] = 'treatment'
    
    # Auto-detect root_type column
    root_candidates = ['root_type', 'RootType', 'roottype']
    for col in root_candidates:
        if col in df.columns:
            columns['root_type'] = col
            break
    if 'root_type' not in columns:
        columns['root_type'] = 'root_type'
    
    return columns


def create_faceted_interaction_plots(
    df,
    output_dir="results/interaction_plots",
    targets=None,
    x_var='treatment',
    hue_var='root_type',
    facet_var='population',
    figsize=(5, 4),
    dpi=150
):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if targets is None:
        targets = {
            'stele_area_um2': 'Stele Area (µm²)',
            'file_count': 'File Count',
            'average_cell_area_um2': 'Avg Cell Area (µm²)'
        }
    
    # Filter targets to only those that exist in the data
    targets = {k: v for k, v in targets.items() if k in df.columns}
    
    if not targets:
        print("  No valid targets found in data")
        return {}
    
    # Auto-detect columns
    detected = get_automatic_columns(df)
    x_var = detected.get('treatment', x_var)
    hue_var = detected.get('root_type', hue_var)
    facet_var = detected.get('population', facet_var)
    
    # Get unique populations (facet values)
    facet_values = [v for v in df[facet_var].unique() if pd.notna(v)]
    n_facets = len(facet_values)
    
    print(f"Found {n_facets} population(s): {facet_values}")
    
    # If no populations found, use a dummy value
    if n_facets == 0:
        print("  No populations found, creating dummy facet")
        df[facet_var] = 'All Data'
        facet_values = ['All Data']
        n_facets = 1
    
    figures = {}
    
    # For each target, create faceted plots
    for target_col, target_label in targets.items():
        if target_col not in df.columns:
            print(f"Warning: {target_col} not found in data, skipping")
            continue
        
        # Check if we have enough groups
        n_treatments = len([v for v in df[x_var].unique() if pd.notna(v)])
        n_root_types = len([v for v in df[hue_var].unique() if pd.notna(v)])
        
        if n_treatments < 2 or n_root_types < 2:
            print(f"Warning: Not enough groups for {target_col}")
            print(f"  - Treatments: {n_treatments}")
            print(f"  - Root types: {n_root_types}")
            continue
        
        # Determine layout
        n_cols = min(3, n_facets)
        n_rows = (n_facets + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
            squeeze=False
        )
        axes_flat = axes.flatten()
        
        # Plot each facet
        for idx, pop in enumerate(facet_values):
            if idx >= len(axes_flat):
                break
                
            ax = axes_flat[idx]
            subset = df[df[facet_var] == pop]
            
            # Skip if not enough data
            if len(subset) < 5:
                ax.text(0.5, 0.5, f'{pop}\n(n={len(subset)})', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{facet_var.capitalize()}: {pop}', fontsize=11)
                continue
            
            try:
                # Create pointplot
                sns.pointplot(
                    data=subset,
                    x=x_var,
                    y=target_col,
                    hue=hue_var,
                    capsize=0.1,
                    errwidth=1.5,
                    palette='Set2',
                    dodge=True,
                    ax=ax,
                    errorbar='sd'
                )
                
                # Add value labels
                for line in ax.lines:
                    y_data = line.get_ydata()
                    x_data = line.get_xdata()
                    for x, y in zip(x_data, y_data):
                        if not np.isnan(y):
                            ax.annotate(
                                f'{y:.0f}',
                                (x, y),
                                xytext=(0, 10),
                                textcoords='offset points',
                                ha='center',
                                va='bottom',
                                fontsize=7,
                                alpha=0.8
                            )
                
                # Remove individual legend
                ax.legend_.remove()
                
                # Add sample size info
                n_samples = len(subset)
                ax.text(
                    0.02, 0.98, f'n={n_samples}',
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
                
                ax.set_title(f'{facet_var.capitalize()}: {pop}', fontsize=11, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:40]}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{facet_var.capitalize()}: {pop}', fontsize=11)
        
        # Hide unused subplots
        for idx in range(n_facets, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        # Set common labels
        fig.text(0.5, 0.02, x_var.capitalize(), ha='center', fontsize=12, fontweight='bold')
        fig.text(0.02, 0.5, target_label, va='center', rotation=90, fontsize=12, fontweight='bold')
        
        # Add global title
        facet_label = facet_var.capitalize()
        fig.suptitle(
            f'{target_label} by {x_var.capitalize()} and {hue_var.capitalize()}\n'
            f'Faceted by {facet_label} ({n_facets} populations)',
            fontsize=14,
            fontweight='bold'
        )
        
        # Add a single legend
        try:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles, labels,
                    loc='upper right',
                    title=hue_var.capitalize(),
                    fontsize=9,
                    bbox_to_anchor=(0.98, 0.98)
                )
        except:
            pass
        
        plt.tight_layout()
        
        # Save figure
        output_filename = f'faceted_interaction_{target_col}.png'
        plt.savefig(output_path / output_filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        figures[target_col] = fig
        print(f"  Saved: {output_filename}")
    
    # Create summary grid
    if len(figures) > 0 and n_facets > 1:
        create_summary_grid(df, output_path, targets, x_var, hue_var, facet_var, facet_values)
    
    return figures


def create_summary_grid(df, output_path, targets, x_var, hue_var, facet_var, facet_values):

    n_targets = len(targets)
    n_facets = min(3, len(facet_values))
    n_rows = n_targets
    n_cols = n_facets
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False
    )
    
    for target_idx, (target_col, target_label) in enumerate(targets.items()):
        for facet_idx, pop in enumerate(facet_values[:n_cols]):
            ax = axes[target_idx, facet_idx]
            subset = df[df[facet_var] == pop]
            
            if len(subset) < 5:
                ax.text(0.5, 0.5, f'n={len(subset)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{pop}', fontsize=9)
                continue
            
            try:
                sns.pointplot(
                    data=subset,
                    x=x_var,
                    y=target_col,
                    hue=hue_var,
                    capsize=0.1,
                    errwidth=1.5,
                    palette='Set2',
                    dodge=True,
                    ax=ax,
                    errorbar='sd'
                )
                ax.legend_.remove()
                ax.set_title(f'{pop}', fontsize=10)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{pop}', fontsize=9)
    
    # Set labels
    for target_idx, (_, target_label) in enumerate(targets.items()):
        axes[target_idx, 0].set_ylabel(target_label, fontsize=11, fontweight='bold')
    
    fig.text(0.5, 0.02, x_var.capitalize(), ha='center', fontsize=12, fontweight='bold')
    fig.suptitle('Summary: All Targets by Population', fontsize=14, fontweight='bold')
    
    # Add legend
    try:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels,
                loc='upper right',
                title=hue_var.capitalize(),
                fontsize=9,
                bbox_to_anchor=(0.98, 0.98)
            )
    except:
        pass
    
    plt.tight_layout()
    plt.savefig(output_path / 'faceted_interaction_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: faceted_interaction_summary.png")


def create_simple_interaction_plots(df, output_dir="results/interaction_plots"):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect columns
    detected = get_automatic_columns(df)
    x_var = detected.get('treatment', 'treatment')
    hue_var = detected.get('root_type', 'root_type')
    
    targets = {
        'stele_area_um2': 'Stele Area (µm²)',
        'file_count': 'File Count',
        'average_cell_area_um2': 'Avg Cell Area (µm²)'
    }
    
    # Filter targets
    targets = {k: v for k, v in targets.items() if k in df.columns}
    
    for target_col, target_label in targets.items():
        fig, ax = plt.subplots(figsize=(10, 7))
        
        sns.pointplot(
            data=df,
            x=x_var,
            y=target_col,
            hue=hue_var,
            capsize=0.1,
            errwidth=2,
            palette='Set2',
            dodge=True,
            ax=ax,
            errorbar='sd'
        )
        
        # Add value labels
        for line in ax.lines:
            y_data = line.get_ydata()
            x_data = line.get_xdata()
            for x, y in zip(x_data, y_data):
                if not np.isnan(y):
                    ax.annotate(
                        f'{y:.0f}',
                        (x, y),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        color='black'
                    )
        
        ax.set_xlabel(x_var.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylabel(target_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{x_var.capitalize()} × {hue_var.capitalize()}: {target_label}', fontsize=14, fontweight='bold')
        ax.legend(title=hue_var.capitalize(), loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'interaction_{target_col}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Simple interaction plots saved to: {output_dir}")


def main(
    master_path="../results/master_summary.csv",
    cell_file_dir="../results/cell_file/cell_file_counting",
    output_dir="../results/interaction_plots",
    x_var=None,
    hue_var=None,
    facet_var=None
):
    
    # Load data
    df = load_data(master_path, cell_file_dir)
    print(f"\nLoaded {len(df)} roots for analysis")
    
    # Auto-detect columns
    detected = get_automatic_columns(df)
    
    # Use provided or auto-detected
    x_var = x_var or detected.get('treatment', 'treatment')
    hue_var = hue_var or detected.get('root_type', 'root_type')
    facet_var = facet_var or detected.get('population', 'population')
    
    # Check if we have multiple populations
    facet_values = [v for v in df[facet_var].unique() if pd.notna(v)]
    n_facets = len(facet_values)
    
    print(f"\nFound {n_facets} population(s): {facet_values}")
    
    if n_facets > 1:
        # Create faceted interaction plots (one per population)
        print("\nCreating faceted interaction plots")
        create_faceted_interaction_plots(
            df=df,
            output_dir=output_dir,
            targets=None,
            x_var=x_var,
            hue_var=hue_var,
            facet_var=facet_var
        )
    elif n_facets == 1:
        # Only one population - create simple plots
        print("\nOnly one population found, creating simple interaction plots...")
        create_simple_interaction_plots(df, output_dir)
    else:
        # No populations found - create a dummy
        print("\nNo population column found, creating simple interaction plots...")
        df['population'] = 'All Data'
        create_simple_interaction_plots(df, output_dir)

    print(f"\nResults saved to: {output_dir}")
    
    return True


if __name__ == "__main__":
    main()