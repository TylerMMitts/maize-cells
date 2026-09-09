# Does a longer cell-file ring hold more cells, or bigger ones?
#
# Two complementary tests, run on the per-file table written by
# cells_per_cell_file.py.
#
# TEST 1 -- PACKING EFFICIENCY
#     If cells tile a ring edge to edge, the centre-to-centre spacing along the
#     arc equals one cell diameter. So
#
#         packing efficiency = mean cell diameter / arc step
#
#     where the arc step is the median angular gap between neighbouring cells
#     turned into a distance at that ring's radius. A value of 1 means the ring
#     is packed solid; below 1 means gaps; above 1 means the cells overlap in
#     projection, which in a real section means the file is not a single clean
#     ring.
#
#     The spacing is used rather than "arc length / number of cells" on purpose.
#     A quadrant image cuts the ring at both ends and the span is measured
#     centroid to centroid, so the visible arc is about 79 degrees rather than
#     90 -- dividing by a nominal quarter turn would bias efficiency down by
#     roughly 13%. Spacing between adjacent cells is immune to where the cut
#     fell and to a missing cell at one end.
#
#     What makes this a test rather than a description: efficiency should be
#     CONSTANT if rings are simply packed. Any drift with file number or with
#     root size is a real departure from geometric packing.
#
# TEST 2 -- WITHIN-ROOT SLOPES
#     Test 1 pools every root together, so a correlation between ring length and
#     cell size could come entirely from large roots having both, with no such
#     relationship inside any individual root (Simpson's paradox). This fits
#
#         mean cell diameter ~ arc length
#
#     separately WITHIN each root, across that root's own files, and aggregates
#     the slopes. Root size is held constant by construction, so a slope that is
#     consistently non-zero is a real within-root effect.
#
#     Spearman correlation is reported alongside the slope, since the
#     relationship is not assumed to be linear.
#
# Outputs land in results/ring_packing/.

import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

LONG_CSV = PROJECT_ROOT / 'results' / 'cells_per_file' / 'cells_per_file_long.csv'
OUT_DIR = PROJECT_ROOT / 'results' / 'ring_packing'

MIN_CELLS_IN_FILE = 6      # need several gaps for a stable median spacing
MIN_FILES_PER_ROOT = 5     # need several points to fit a within-root slope


def root_id(image_name):
    return re.sub(r'^(BL|BR|TL|TR)_', '', str(image_name))


def load(long_csv=LONG_CSV):
    d = pd.read_csv(long_csv)
    if 'arc_step_um' not in d.columns:
        raise SystemExit('arc_step_um missing -- rerun cells_per_cell_file.py --force')
    d = d[(d['n_cells'] >= MIN_CELLS_IN_FILE) &
          d['arc_step_um'].notna() & (d['arc_step_um'] > 0) &
          (d['mean_diameter_um'] > 0)].copy()
    d['packing_efficiency'] = d['mean_diameter_um'] / d['arc_step_um']
    d['root_id'] = d['image_name'].map(root_id)
    # a ring cannot be packed 5x over; those are file-detection failures
    d = d[(d['packing_efficiency'] > 0.05) & (d['packing_efficiency'] < 5)]
    return d


# Test 1

def packing_test(d, out_dir):
    eff = d['packing_efficiency']
    print(f'\n\nTEST 1 -- PACKING EFFICIENCY  (1.0 = cells tile the ring edge to edge)')
    print(f'  rings analysed : {len(d):,} across {d["root_id"].nunique():,} roots')
    print(f'  median         : {eff.median():.3f}')
    print(f'  mean +/- sd    : {eff.mean():.3f} +/- {eff.std():.3f}')
    print(f'  IQR            : {eff.quantile(.25):.3f} - {eff.quantile(.75):.3f}')

    # drift with file number
    by_file = d.groupby('cell_file')['packing_efficiency'].agg(['median', 'size'])
    by_file = by_file[by_file['size'] >= 30]
    r_file = stats.spearmanr(d['cell_file'], eff)
    print(f'\n  vs FILE NUMBER : Spearman rho={r_file.statistic:+.3f}  p={r_file.pvalue:.3g}')

    # drift with root size
    sub = d[d['n_files'] > 0]
    r_size = stats.spearmanr(sub['n_files'], sub['packing_efficiency'])
    print(f'  vs ROOT SIZE   : Spearman rho={r_size.statistic:+.3f}  p={r_size.pvalue:.3g}'
          f'   (root size proxied by file count)')

    verdict = ('CONSTANT -- consistent with simple geometric packing'
               if abs(r_file.statistic) < 0.1
               else 'DRIFTS with depth -- packing is not uniform through the cortex')
    print(f'  verdict        : {verdict}')

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.hist(eff, bins=80, color='steelblue', edgecolor='none')
    ax.axvline(1.0, color='crimson', ls='--', lw=2, label='edge-to-edge packing')
    ax.axvline(eff.median(), color='darkorange', ls='-', lw=2,
               label=f'median {eff.median():.2f}')
    ax.set_xlabel('packing efficiency  (cell diameter / arc spacing)')
    ax.set_ylabel('rings')
    ax.set_title('Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    q = d.groupby('cell_file')['packing_efficiency'].quantile([.25, .5, .75]).unstack()
    q = q[q.index.isin(by_file.index)]
    ax.fill_between(q.index, q[0.25], q[0.75], alpha=0.25, color='steelblue')
    ax.plot(q.index, q[0.5], 'o-', color='steelblue', lw=2)
    ax.axhline(1.0, color='crimson', ls='--', lw=1.5)
    ax.set_xlabel('cell file number (0 = innermost)')
    ax.set_ylabel('packing efficiency')
    ax.set_title(f'Against depth   (Spearman rho={r_file.statistic:+.3f})')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(d['arc_step_um'], d['mean_diameter_um'], s=3, alpha=0.08, color='steelblue')
    lim = np.nanpercentile(d['arc_step_um'], 99)
    ax.plot([0, lim], [0, lim], 'r--', lw=2, label='diameter = spacing')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('arc spacing between neighbouring cells (um)')
    ax.set_ylabel('mean cell diameter (um)')
    ax.set_title('Cell size against the room available')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = Path(out_dir) / 'packing_efficiency.png'
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved {p}')

    by_file.to_csv(Path(out_dir) / 'packing_efficiency_by_file.csv')
    return dict(median=float(eff.median()), rho_file=float(r_file.statistic),
                p_file=float(r_file.pvalue), rho_size=float(r_size.statistic))


# Test 2

def within_root_slopes(d, out_dir):
    rows = []
    for rid, g in d.groupby('root_id'):
        g = g.dropna(subset=['arc_length_um', 'mean_diameter_um'])
        if len(g) < MIN_FILES_PER_ROOT:
            continue
        x = g['arc_length_um'].to_numpy(float)
        y = g['mean_diameter_um'].to_numpy(float)
        if x.std() == 0:
            continue
        lr = stats.linregress(x, y)
        sp = stats.spearmanr(x, y)
        rows.append({'root_id': rid, 'n_files': len(g),
                     'slope_um_per_um': lr.slope, 'intercept_um': lr.intercept,
                     'r2': lr.rvalue ** 2, 'p': lr.pvalue,
                     'spearman_rho': sp.statistic,
                     'mean_arc_um': float(x.mean()),
                     'population': g['population'].iloc[0] if 'population' in g else None})
    slopes = pd.DataFrame(rows)
    if slopes.empty:
        print('\nTEST 2 -- no roots had enough files to fit')
        return None

    s = slopes['slope_um_per_um']
    rho = slopes['spearman_rho'].dropna()
    t = stats.ttest_1samp(s, 0.0)
    w = stats.wilcoxon(s) if len(s) > 10 else None
    pos = float((s > 0).mean())

    print(f'\n\nTEST 2 -- WITHIN-ROOT SLOPES  (cell diameter vs ring arc length)')
    print(f'  roots fitted   : {len(slopes):,}  (>= {MIN_FILES_PER_ROOT} files each)')
    print(f'  median slope   : {s.median():+.5f} um diameter per um of arc')
    print(f'                   i.e. {s.median()*1000:+.2f} um per 1000 um of extra arc')
    print(f'  mean slope     : {s.mean():+.5f}')
    print(f'  slopes > 0     : {100*pos:.1f}% of roots')
    print(f'  t-test vs 0    : t={t.statistic:+.2f}  p={t.pvalue:.3g}')
    if w is not None:
        print(f'  Wilcoxon vs 0  : p={w.pvalue:.3g}')
    print(f'  median Spearman: {rho.median():+.3f}   ({100*(rho>0).mean():.1f}% positive)')
    print(f'  median R2      : {slopes["r2"].median():.3f}')

    direction = 'LARGER' if s.median() > 0 else 'SMALLER'
    print(f'  verdict        : within a root, longer rings hold {direction} cells')

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    lo, hi = np.percentile(s, [1, 99])
    ax.hist(s.clip(lo, hi), bins=70, color='seagreen', edgecolor='none')
    ax.axvline(0, color='crimson', ls='--', lw=2, label='no relationship')
    ax.axvline(s.median(), color='darkorange', lw=2, label=f'median {s.median():+.4f}')
    ax.set_xlabel('within-root slope (um diameter per um arc)')
    ax.set_ylabel('roots')
    ax.set_title(f'Slope distribution\n{100*pos:.0f}% positive, p={t.pvalue:.2g}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(rho.clip(-1, 1), bins=60, color='slateblue', edgecolor='none')
    ax.axvline(0, color='crimson', ls='--', lw=2)
    ax.axvline(rho.median(), color='darkorange', lw=2, label=f'median {rho.median():+.2f}')
    ax.set_xlabel('within-root Spearman rho')
    ax.set_ylabel('roots')
    ax.set_title('Rank correlation (no linearity assumed)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    if 'population' in slopes.columns and slopes['population'].notna().any():
        pops = sorted(slopes['population'].dropna().unique())
        data = [slopes.loc[slopes['population'] == p, 'slope_um_per_um'].clip(lo, hi) for p in pops]
        ax.boxplot(data, labels=pops, showfliers=False)
        ax.axhline(0, color='crimson', ls='--', lw=1.5)
        ax.set_ylabel('within-root slope')
        ax.set_title('By population')
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    p = Path(out_dir) / 'within_root_slopes.png'
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved {p}')

    slopes.to_csv(Path(out_dir) / 'within_root_slopes.csv', index=False)
    return slopes


def main():
    # python -m code.statistics.ring_packing_analysis
    class cfg:
        long_csv = str(LONG_CSV)
        out_dir = str(OUT_DIR)


    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load(Path(cfg.long_csv))
    print(f'loaded {len(d):,} rings (>= {MIN_CELLS_IN_FILE} cells each)')

    packing_test(d, out_dir)
    within_root_slopes(d, out_dir)

    d.to_csv(out_dir / 'ring_packing_rows.csv', index=False)
    print(f'\nrow-level data: {out_dir / "ring_packing_rows.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
