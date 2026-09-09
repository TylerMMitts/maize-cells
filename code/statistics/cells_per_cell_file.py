# How many cells sit in each cell file, across the whole dataset.
#
# For every quadrant image the cells are already assigned to a concentric file by
# cell_file_analyzer.py; this counts them per file and aggregates across images,
# so the question "does a root add more cells as you move outward, or bigger
# ones?" can be answered from the data rather than assumed.
#
# The unit is the QUADRANT, because that is what an image is: a quarter of the
# cross-section, with the root centre at one corner. A file therefore appears as
# a 90-degree arc, not a full ring, and the counts here are per quarter-ring.
# Multiply by four for a whole-root estimate -- that is an approximation, since
# the four quadrants of a root do not always resolve the same number of files.
#
# Alongside the counts it records the ring geometry each file implies -- mean
# radius, arc length, mean cell diameter -- which is what the circumference /
# cell-size analysis needs, so that does not require a second pass over 6,000+
# files.
#
# Outputs (results/cells_per_file/):
#     cells_per_file_long.csv      one row per image x file
#     cells_per_file_summary.csv   per file index, pooled across images
#     cells_per_file.png           counts against file number
#     cells_per_file_normalised.png  counts against relative depth in the cortex

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CELL_FILE_DIR = PROJECT_ROOT / 'results' / 'cell_file' / 'cell_file_counting'
MASTER_SUMMARY = PROJECT_ROOT / 'results' / 'master_summary.csv'
OUT_DIR = PROJECT_ROOT / 'results' / 'cells_per_file'

# A quadrant image spans a quarter turn, so a file's arc is a quarter circle.
QUADRANT_FRACTION = 0.25


def collect(cell_file_dir=CELL_FILE_DIR, min_cells=5):
    # One row per (image, cell file) with counts and ring geometry.
    rows = []
    dirs = sorted((e for e in os.scandir(cell_file_dir) if e.is_dir()), key=lambda e: e.name)
    print(f'scanning {len(dirs)} image folders ...')

    for i, entry in enumerate(dirs, 1):
        path = os.path.join(entry.path, 'cell_assignments.csv')
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if 'cell_file_derivative' not in df.columns or len(df) < min_cells:
            continue

        df = df[(df['cell_file_derivative'] >= 0) & (df['area_um2'] > 0)]
        if df.empty:
            continue

        n_files = int(df['cell_file_derivative'].max()) + 1
        for file_idx, g in df.groupby('cell_file_derivative'):
            mean_r = float(g['radius_um'].mean())
            mean_area = float(g['area_um2'].mean())
            # equivalent circular diameter of the mean cell
            mean_dia = 2.0 * math.sqrt(mean_area / math.pi)
            # Angular geometry of the ring. The span is measured centroid to
            # centroid, so it falls short of the full quarter turn by about
            # half a cell at each end -- the observed median is 79 deg, not 90.
            # The median gap between ANGULARLY ADJACENT cells is the robust
            # quantity: it is the centre-to-centre spacing along the arc, and
            # unlike the span it is unaffected by where the quadrant was cut or
            # by a missing cell at one end.
            ang = np.sort(g['angle_degrees'].to_numpy())
            span = float(ang[-1] - ang[0]) if len(ang) > 1 else 0.0
            gaps = np.diff(ang)
            median_gap = float(np.median(gaps)) if len(gaps) else float('nan')

            rows.append({
                'image_name': entry.name,
                'cell_file': int(file_idx),
                'n_files': n_files,
                'file_position': int(file_idx) / max(n_files - 1, 1),
                'n_cells': int(len(g)),
                'mean_radius_um': mean_r,
                'mean_area_um2': mean_area,
                'mean_diameter_um': mean_dia,
                'arc_length_um': QUADRANT_FRACTION * 2 * math.pi * mean_r,
                'full_circumference_um': 2 * math.pi * mean_r,
                'angular_span_deg': span,
                'median_gap_deg': median_gap,
                # centre-to-centre spacing of neighbouring cells along the arc
                'arc_step_um': math.radians(median_gap) * mean_r if median_gap == median_gap else float('nan'),
            })

        if i % 500 == 0:
            print(f'  {i}/{len(dirs)}')

    long = pd.DataFrame(rows)
    print(f'collected {len(long)} image-file rows from {long["image_name"].nunique()} images')
    return long


def attach_metadata(long, master_summary=MASTER_SUMMARY):
    if not Path(master_summary).exists():
        return long
    ms = pd.read_csv(master_summary)
    keep = [c for c in ('image_name', 'species', 'population', 'treatment', 'root_type')
            if c in ms.columns]
    return long.merge(ms[keep].drop_duplicates('image_name'), on='image_name', how='left')


def summarise(long):
    g = long.groupby('cell_file')['n_cells']
    out = pd.DataFrame({
        'cell_file': g.mean().index,
        'n_images': g.size().values,
        'mean_cells': g.mean().values,
        'median_cells': g.median().values,
        'q25': g.quantile(0.25).values,
        'q75': g.quantile(0.75).values,
        'std': g.std().values,
    })
    geo = long.groupby('cell_file').agg(
        mean_radius_um=('mean_radius_um', 'mean'),
        mean_diameter_um=('mean_diameter_um', 'mean'),
        arc_length_um=('arc_length_um', 'mean'),
    ).reset_index()
    return out.merge(geo, on='cell_file')


def plot_counts(long, summary, out_dir, max_file=None):
    if max_file is None:
        # cut where the sample thins out, so the tail is not one noisy image
        enough = summary[summary['n_images'] >= max(20, 0.02 * summary['n_images'].max())]
        max_file = int(enough['cell_file'].max()) if len(enough) else int(summary['cell_file'].max())
    s = summary[summary['cell_file'] <= max_file]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.fill_between(s['cell_file'], s['q25'], s['q75'], alpha=0.25, color='steelblue',
                    label='IQR')
    ax.plot(s['cell_file'], s['median_cells'], 'o-', color='steelblue', lw=2, label='median')
    ax.plot(s['cell_file'], s['mean_cells'], '--', color='darkorange', lw=1.5, label='mean')
    ax.set_xlabel('Cell file number (0 = innermost)')
    ax.set_ylabel('Cells in file (per quadrant)')
    ax.set_title(f'Cells per cell file\n{long["image_name"].nunique():,} quadrant images')
    ax.grid(alpha=0.3)
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(s['cell_file'], s['n_images'], ':', color='grey', lw=1)
    ax2.set_ylabel('images contributing', color='grey', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='grey', labelsize=8)

    ax = axes[1]
    pops = [p for p in long['population'].dropna().unique()] if 'population' in long else []
    for pop in sorted(pops):
        sub = long[long['population'] == pop]
        gg = sub.groupby('cell_file')['n_cells'].agg(['median', 'size'])
        gg = gg[(gg['size'] >= 20) & (gg.index <= max_file)]
        if len(gg) > 1:
            ax.plot(gg.index, gg['median'], 'o-', ms=3, lw=1.5, label=f'{pop} (n={sub["image_name"].nunique()})')
    ax.set_xlabel('Cell file number (0 = innermost)')
    ax.set_ylabel('Median cells in file (per quadrant)')
    ax.set_title('By population')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    p = Path(out_dir) / 'cells_per_file.png'
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved {p}')


def plot_normalised(long, out_dir, bins=20):
    # Counts against relative depth, so roots with different file counts overlay.
    d = long.copy()
    d['bin'] = pd.cut(d['file_position'], bins=bins, labels=False)
    g = d.groupby('bin')['n_cells'].agg(['median', 'mean', 'size',
                                          lambda s: s.quantile(0.25),
                                          lambda s: s.quantile(0.75)])
    g.columns = ['median', 'mean', 'size', 'q25', 'q75']
    x = (g.index + 0.5) / bins

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.fill_between(x, g['q25'], g['q75'], alpha=0.25, color='seagreen', label='IQR')
    ax.plot(x, g['median'], 'o-', color='seagreen', lw=2, label='median')
    ax.plot(x, g['mean'], '--', color='darkorange', lw=1.5, label='mean')
    ax.set_xlabel('Relative position in the cortex (0 = stele, 1 = epidermis)')
    ax.set_ylabel('Cells in file (per quadrant)')
    ax.set_title('Cells per file, normalised for differing file counts')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    p = Path(out_dir) / 'cells_per_file_normalised.png'
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved {p}')


def main():
    # python -m code.statistics.cells_per_cell_file
    class cfg:
        cell_file_dir = str(CELL_FILE_DIR)
        out_dir = str(OUT_DIR)
        # rescan even if the long CSV exists
        force = False


    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / 'cells_per_file_long.csv'

    if long_path.exists() and not cfg.force:
        print(f'loading cached {long_path} (--force to rescan)')
        long = pd.read_csv(long_path)
    else:
        long = collect(Path(cfg.cell_file_dir))
        long = attach_metadata(long)
        long.to_csv(long_path, index=False)
        print(f'  saved {long_path}')

    summary = summarise(long)
    summary.to_csv(out_dir / 'cells_per_file_summary.csv', index=False)
    print(f'  saved {out_dir / "cells_per_file_summary.csv"}')

    plot_counts(long, summary, out_dir)
    plot_normalised(long, out_dir)

    print(f'\n\nCELLS PER CELL FILE')
    show = summary[summary['n_images'] >= 20].head(20)
    print(show[['cell_file', 'n_images', 'median_cells', 'mean_cells',
                'mean_radius_um', 'mean_diameter_um']].round(1).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
