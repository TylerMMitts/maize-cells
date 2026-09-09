# How circumferential cell spacing changes as you move outward through the root.
#
# cells_per_cell_file.py already records, per (image, cell file), the mean cell
# diameter and the arc-step (centre-to-centre spacing between angularly-adjacent
# cells in that file -- see that module for the derivation). This re-bins that
# same data against radius instead of file index or normalised file position, so
# the question "as the ring gets bigger, does the spacing between cells grow
# with it, or does the cell size keep up?" can be read directly off physical
# distance from the stele.
#
# Two versions of the x-axis are plotted side by side, because pooling absolute
# radius across roots of very different size is a real confound (a 400 um ring
# in a small root and a 400 um ring in a large root are not the same point in
# the root's development):
#
#     absolute radius_um        -- physical distance, pooled across all roots
#     normalised radius (0-1)   -- each root's own radius divided by that root's
#                                   own outermost measured file radius, so root
#                                   size is controlled for
#
# Requires results/cells_per_file/cells_per_file_long.csv (built by
# cells_per_cell_file.py); does not rescan the per-image CSVs.
#
# Outputs (results/ring_packing/):
#     spacing_vs_radius.png
#     spacing_vs_radius_binned.csv

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

LONG_CSV = PROJECT_ROOT / 'results' / 'cells_per_file' / 'cells_per_file_long.csv'
OUT_DIR = PROJECT_ROOT / 'results' / 'ring_packing'


def load(long_csv=LONG_CSV, min_cells=6):
    d = pd.read_csv(long_csv)
    d = d[(d.n_cells >= min_cells) & (d.arc_step_um > 0) & d.arc_step_um.notna()].copy()
    d['packing_efficiency'] = d.mean_diameter_um / d.arc_step_um

    # each root's own outer radius = its largest measured file radius, so a
    # ring at "normalised radius 0.8" means the same developmental point
    # whether the root is big or small
    outer = d.groupby('image_name')['mean_radius_um'].transform('max')
    d['norm_radius'] = d['mean_radius_um'] / outer
    return d


def bin_stat(d, xcol, edges):
    d = d.copy()
    d['bin'] = pd.cut(d[xcol], bins=edges, include_lowest=True)
    g = d.groupby('bin', observed=True).agg(
        x=(xcol, 'median'),
        n=(xcol, 'size'),
        eff_med=('packing_efficiency', 'median'),
        eff_q25=('packing_efficiency', lambda s: s.quantile(0.25)),
        eff_q75=('packing_efficiency', lambda s: s.quantile(0.75)),
        step_med=('arc_step_um', 'median'),
        dia_med=('mean_diameter_um', 'median'),
    ).dropna(subset=['x'])
    return g[g.n >= 20]


def _panel_pair(bins, xlabel, out_dir, fname, title, n_images):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(bins.x, bins.step_med, 'o-', color='crimson', ms=3, lw=1.8,
            label='arc-step (spacing claimed by each cell)')
    ax.plot(bins.x, bins.dia_med, 'o-', color='steelblue', ms=3, lw=1.8,
            label='mean cell diameter (equivalent-circle)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('um')
    ax.set_title('Spacing vs. cell size')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.fill_between(bins.x, bins.eff_q25, bins.eff_q75, alpha=0.25, color='seagreen')
    ax.plot(bins.x, bins.eff_med, 'o-', color='seagreen', ms=3, lw=2)
    ax.axhline(1.0, color='grey', ls='--', lw=1, label='edge-to-edge (ratio = 1)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('packing efficiency  (diameter / spacing)')
    ax.set_title('Circumferential packing efficiency')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(f'{title}  ({n_images:,} images)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = Path(out_dir) / fname
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved {p}')


def plot(d, out_dir):
    # absolute radius: bin every 20 um out to the 99th percentile, so the
    # long thin tail of a few very large roots doesn't stretch the axis
    r_max = np.nanpercentile(d.mean_radius_um, 99)
    abs_edges = np.arange(0, r_max + 20, 20)
    abs_bins = bin_stat(d, 'mean_radius_um', abs_edges)

    norm_edges = np.linspace(0, 1, 26)
    norm_bins = bin_stat(d, 'norm_radius', norm_edges)

    n_images = d.image_name.nunique()
    _panel_pair(abs_bins, 'radius from stele centre (um)', out_dir,
                'spacing_vs_radius_absolute.png',
                'Circumferential spacing vs. absolute radius (pools roots of all sizes)', n_images)
    _panel_pair(norm_bins, "radius, normalised to each root's own outer file (0 = stele, 1 = edge)",
                out_dir, 'spacing_vs_radius_normalised.png',
                "Circumferential spacing vs. normalised radius (each root's own size controlled for)",
                n_images)

    abs_bins.assign(axis='absolute_um').to_csv(Path(out_dir) / 'spacing_vs_radius_binned_abs.csv')
    norm_bins.assign(axis='normalised').to_csv(Path(out_dir) / 'spacing_vs_radius_binned_norm.csv')


def report(d):
    print(f'\n\nSPACING VS RADIUS   ({len(d):,} image-file rows, {d.image_name.nunique():,} images)')
    rho_abs, p_abs = spearmanr(d.mean_radius_um, d.packing_efficiency)
    rho_norm, p_norm = spearmanr(d.norm_radius, d.packing_efficiency)
    print(f'  packing efficiency vs absolute radius_um   : rho={rho_abs:+.3f}  p={p_abs:.2e}')
    print(f'  packing efficiency vs normalised radius    : rho={rho_norm:+.3f}  p={p_norm:.2e}')

    rho_step, _ = spearmanr(d.mean_radius_um, d.arc_step_um)
    rho_dia, _ = spearmanr(d.mean_radius_um, d.mean_diameter_um)
    print(f'  arc-step vs absolute radius_um             : rho={rho_step:+.3f}  (spacing grows with radius)')
    print(f'  cell diameter vs absolute radius_um        : rho={rho_dia:+.3f}  (cell size, same axis)')


def main():
    # python -m code.statistics.spacing_vs_radius
    class cfg:
        long_csv = str(LONG_CSV)
        out_dir = str(OUT_DIR)


    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load(Path(cfg.long_csv))
    report(d)
    plot(d, out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
