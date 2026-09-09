# Where the spline feature space stops being physically realisable.
#
# The six shape features are not independent. Three of them are secant slopes
# inverted back into heights, so a combination of values can describe a curve
# that dips below zero normalised cell area -- which cannot exist, area being
# non-negative. root_model.html clamps such combinations when it reconstructs,
# silently, so a slider can sit at a value the model then refuses to honour.
#
# This maps that boundary and puts the real dataset on top of it.
#
# THE CONSTRAINTS
# Reconstruction (buildCurve in root_model.html) inverts the forward definitions:
#
#     start height   y0 = peak_height - rise_slope  x (peak_position + 0.001)
#     minima height  mh = peak_height + decay_slope x (minima_position - peak_position + 0.001)
#     edge height    yEnd = min(mh + outer_rise, peak_height)
#
# Requiring y0 >= 0, mh >= 0 and the peak to remain the curve's maximum gives
#
#     C1   rise_slope  <=  peak_height / (peak_position + 0.001)
#     C2   decay_slope >= -peak_height / (minima_position - peak_position + 0.001)
#     C3   rise_slope >= 0  and  decay_slope <= 0
#     C4   outer_rise  <=  peak_height - minima_height
#
# C1 and C2 are hyperbolic ceilings: the closer the peak sits to the stele, the
# less rise a given peak height can support, because the curve has less distance
# in which to climb.
#
# The analysis uses the four core features -- peak height, peak position, rise
# slope, decay slope. C2 also needs minima_position, so for the feasibility MAPS
# that is held at the dataset median, while real roots are always judged against
# their own value.
#
# A note on interpretation: a root landing outside the boundary does not mean the
# root is impossible. It means the spline fitted to it was, and the feature vector
# inherited that. The rise slope is a secant from the curve's start to its peak,
# so inverting it returns the start height exactly -- a negative result says the
# fitted spline undershot below zero, which is a smoothing artefact rather than
# anatomy.
#
# C4's plane doesn't need minima_position separately -- it only cares about the
# height the curve has actually reached by the minimum (minima_height, itself
# already floored at zero), so it plots directly in (minima_height, peak_height).
#
# Every figure is stamped with the literal constraint it visualises, in the
# same plain form as the docstring above, so the boundary being drawn is never
# left implicit.
#
# Outputs in results/feature_limits/: the 2x2 ceiling/floor grids for C1 and
# C2, a 3D surface + utilisation heatmap for each of C1/C2/C4, the morphospace
# cluster figure, and a feasibility CSV.

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FEATURE_TABLE = PROJECT_ROOT / 'results' / 'feature_table.csv'
OUT_DIR = PROJECT_ROOT / 'results' / 'feature_limits'

EPS = 0.001            # the same guard term the forward definitions use
CORE = ['peak_height', 'peak_position', 'rise_slope', 'decay_slope']

# readable versions of the constraints, stamped directly onto every figure
CONSTRAINT_TEXT = {
    'C1': 'C1:  rise_slope  <=  peak_height / peak_position',
    'C2': 'C2:  decay_slope  >=  -peak_height / (minima_position - peak_position)',
    'C4': 'C4:  outer_rise  <=  peak_height - minima_height',
}


def _stamp(fig, key, loc=(0.03, 0.02)):
    fig.text(*loc, CONSTRAINT_TEXT[key], fontsize=10.5, family='monospace',
              bbox=dict(boxstyle='round', fc='white', ec='crimson', alpha=0.9))


def load(prefix='radius', feature_table=FEATURE_TABLE):
    cols = {k: f'{prefix}_{k}' for k in
            ['peak_height', 'peak_position', 'rise_slope', 'decay_slope',
             'minima_position', 'outer_rise_magnitude']}
    ft = pd.read_csv(feature_table)
    missing = [c for c in cols.values() if c not in ft.columns]
    if missing:
        raise SystemExit(f'missing columns: {missing}')

    d = ft.dropna(subset=list(cols.values())).copy()
    for k, c in cols.items():
        d[k] = d[c]

    d['start_height'] = d.peak_height - d.rise_slope * (d.peak_position + EPS)
    gap = (d.minima_position - d.peak_position).clip(lower=0)
    d['minima_height_raw'] = np.where(d.minima_position > d.peak_position,
                                       d.peak_height + d.decay_slope * (gap + EPS),
                                       d.peak_height)
    d['minima_height'] = d.minima_height_raw.clip(lower=0)
    d['outer_overshoot'] = (d.minima_height + d.outer_rise_magnitude) - d.peak_height

    d['viol_start'] = d.start_height < 0
    d['viol_minima'] = d.minima_height_raw < 0
    d['viol_outer'] = d.outer_overshoot > 1e-9
    d['infeasible'] = d.viol_start | d.viol_minima | d.viol_outer

    # how far into the forbidden zone: 1.0 sits exactly on the boundary
    d['rise_utilisation'] = d.rise_slope * (d.peak_position + EPS) / d.peak_height.replace(0, np.nan)
    d['decay_utilisation'] = (-d.decay_slope) * (gap + EPS) / d.peak_height.replace(0, np.nan)
    d['outer_utilisation'] = d.outer_rise_magnitude / (d.peak_height - d.minima_height).replace(0, np.nan)
    return d, cols


def report(d, prefix):
    n = len(d)
    print(f'\n\nFEATURE SPACE FEASIBILITY -- {prefix}_*   ({n:,} roots)')
    for lab, col in [('C1 start height < 0 (rise too steep for the peak)', 'viol_start'),
                     ('C2 minima height < 0 (decay too steep for the gap)', 'viol_minima'),
                     ('C4 outer rise above the peak', 'viol_outer')]:
        k = int(d[col].sum())
        print(f'  {lab:52} {k:5,}  ({100*k/n:5.1f}%)')
    k = int(d.infeasible.sum())
    print(f'  {"ANY constraint clamped":52} {k:5,}  ({100*k/n:5.1f}%)')
    print(f'\n  rise utilisation  (>1 breaches C1): median {d.rise_utilisation.median():.2f}'
          f'   90th pct {d.rise_utilisation.quantile(.9):.2f}   max {d.rise_utilisation.max():.2f}')
    print(f'  decay utilisation (>1 breaches C2): median {d.decay_utilisation.median():.2f}'
          f'   90th pct {d.decay_utilisation.quantile(.9):.2f}   max {d.decay_utilisation.max():.2f}')
    print(f'  outer utilisation (>1 breaches C4): median {d.outer_utilisation.median():.2f}'
          f'   90th pct {d.outer_utilisation.quantile(.9):.2f}   max {d.outer_utilisation.max():.2f}')


def fig_rise_limits(d, out_dir, prefix, heights=(0.2, 0.4, 0.6, 0.8), band=0.08):
    # C1: rise_slope ceiling as peak_height is held fixed.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    pp = np.linspace(0.001, 1.0, 400)

    for ax, ph in zip(axes.flat, heights):
        ceiling = ph / (pp + EPS)
        top = min(np.nanpercentile(d.rise_slope, 99.5), 4)
        ax.fill_between(pp, ceiling, top, color='crimson', alpha=0.18,
                        label='impossible: curve starts below zero')
        ax.plot(pp, ceiling, color='crimson', lw=2, label=r'ceiling  $s_{rise}=h/x_{peak}$')

        sub = d[(d.peak_height - ph).abs() <= band]
        ok, bad = sub[~sub.viol_start], sub[sub.viol_start]
        ax.scatter(ok.peak_position, ok.rise_slope, s=9, alpha=0.45,
                   color='steelblue', label=f'roots, reconstructable (n={len(ok)})')
        ax.scatter(bad.peak_position, bad.rise_slope, s=16, alpha=0.75,
                   color='darkred', marker='x', label=f'roots, clamped (n={len(bad)})')

        ax.set_xlim(0, 1); ax.set_ylim(0, top)
        ax.set_xlabel('peak position'); ax.set_ylabel('rise slope')
        ax.set_title(f'peak height fixed at {ph:.1f}  (roots within +/-{band})')
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(f'C1 -- how much rise a given peak height can support ({prefix}_*)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _stamp(fig, 'C1')
    p = Path(out_dir) / f'{prefix}_C1_rise_slope_limits.png'
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  saved {p}')


def fig_decay_limits(d, out_dir, prefix, heights=(0.2, 0.4, 0.6, 0.8), band=0.08):
    # C2: decay_slope floor against the peak-to-minima gap.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    gap = np.linspace(0.001, 1.0, 400)

    for ax, ph in zip(axes.flat, heights):
        floor = -ph / (gap + EPS)
        bottom = max(np.nanpercentile(d.decay_slope, 0.5), -4)
        ax.fill_between(gap, bottom, floor, color='crimson', alpha=0.18,
                        label='impossible: minima below zero')
        ax.plot(gap, floor, color='crimson', lw=2, label=r'floor  $s_{decay}=-h/\Delta x$')

        sub = d[(d.peak_height - ph).abs() <= band].copy()
        sub['gap'] = (sub.minima_position - sub.peak_position).clip(lower=0)
        ok, bad = sub[~sub.viol_minima], sub[sub.viol_minima]
        ax.scatter(ok.gap, ok.decay_slope, s=9, alpha=0.45, color='seagreen',
                   label=f'roots, reconstructable (n={len(ok)})')
        ax.scatter(bad.gap, bad.decay_slope, s=16, alpha=0.75, color='darkred',
                   marker='x', label=f'roots, clamped (n={len(bad)})')

        ax.set_xlim(0, 1); ax.set_ylim(bottom, 0.05)
        ax.set_xlabel('peak-to-minima gap')
        ax.set_ylabel('decay slope')
        ax.set_title(f'peak height fixed at {ph:.1f}  (roots within +/-{band})')
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='lower right')

    fig.suptitle(f'C2 -- how much decay a given peak height can support ({prefix}_*)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _stamp(fig, 'C2')
    p = Path(out_dir) / f'{prefix}_C2_decay_slope_limits.png'
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  saved {p}')


def fig_3d_surface_c1(d, out_dir, prefix):
    # The C1 ceiling as a surface over (peak position, peak height).
    fig = plt.figure(figsize=(15, 6.5))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    pp = np.linspace(0.02, 1.0, 60)
    ph = np.linspace(0.05, 1.0, 60)
    PP, PH = np.meshgrid(pp, ph)
    CEIL = np.minimum(PH / (PP + EPS), 4)
    ax.plot_surface(PP, PH, CEIL, cmap='autumn', alpha=0.55, linewidth=0,
                    rstride=2, cstride=2)
    ok = d[~d.viol_start]; bad = d[d.viol_start]
    ax.scatter(ok.peak_position, ok.peak_height, ok.rise_slope.clip(upper=4),
               s=5, alpha=0.35, color='steelblue', label='reconstructable')
    ax.scatter(bad.peak_position, bad.peak_height, bad.rise_slope.clip(upper=4),
               s=14, alpha=0.85, color='darkred', marker='x', label='clamped')
    ax.set_xlabel('peak position'); ax.set_ylabel('peak height'); ax.set_zlabel('rise slope')
    ax.set_title('C1 ceiling surface\n(points above the sheet are impossible)')
    ax.legend(fontsize=8, loc='upper left')
    ax.view_init(elev=22, azim=-125)

    ax = fig.add_subplot(1, 2, 2)
    sc = ax.scatter(d.peak_position, d.peak_height, c=d.rise_utilisation.clip(0, 2),
                    s=9, cmap='RdYlBu_r', vmin=0, vmax=2)
    plt.colorbar(sc, ax=ax, label='rise utilisation  (>1 impossible)')
    ax.set_xlabel('peak position'); ax.set_ylabel('peak height')
    ax.set_title('How close each root sits to its own ceiling')
    ax.grid(alpha=0.3)

    fig.suptitle(f'C1 ceiling in 3D ({prefix}_*)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _stamp(fig, 'C1')
    p = Path(out_dir) / f'{prefix}_C1_surface_3d.png'
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  saved {p}')


def fig_3d_surface_c2(d, out_dir, prefix):
    # The C2 floor as a surface over (peak-to-minima gap, peak height).
    d = d.copy()
    d['gap'] = (d.minima_position - d.peak_position).clip(lower=0)

    fig = plt.figure(figsize=(15, 6.5))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    gap = np.linspace(0.02, 1.0, 60)
    ph = np.linspace(0.05, 1.0, 60)
    GAP, PH = np.meshgrid(gap, ph)
    FLOOR = np.maximum(-PH / (GAP + EPS), -4)
    ax.plot_surface(GAP, PH, FLOOR, cmap='winter', alpha=0.55, linewidth=0,
                    rstride=2, cstride=2)
    ok = d[~d.viol_minima]; bad = d[d.viol_minima]
    ax.scatter(ok.gap, ok.peak_height, ok.decay_slope.clip(lower=-4),
               s=5, alpha=0.35, color='seagreen', label='reconstructable')
    ax.scatter(bad.gap, bad.peak_height, bad.decay_slope.clip(lower=-4),
               s=14, alpha=0.85, color='darkred', marker='x', label='clamped')
    ax.set_xlabel('peak-to-minima gap'); ax.set_ylabel('peak height'); ax.set_zlabel('decay slope')
    ax.set_title('C2 floor surface\n(points below the sheet are impossible)')
    ax.legend(fontsize=8, loc='upper left')
    ax.view_init(elev=22, azim=-125)

    ax = fig.add_subplot(1, 2, 2)
    sc = ax.scatter(d.gap, d.peak_height, c=d.decay_utilisation.clip(0, 2),
                    s=9, cmap='RdYlBu_r', vmin=0, vmax=2)
    plt.colorbar(sc, ax=ax, label='decay utilisation  (>1 impossible)')
    ax.set_xlabel('peak-to-minima gap'); ax.set_ylabel('peak height')
    ax.set_title('How close each root sits to its own floor')
    ax.grid(alpha=0.3)

    fig.suptitle(f'C2 floor in 3D ({prefix}_*)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _stamp(fig, 'C2')
    p = Path(out_dir) / f'{prefix}_C2_surface_3d.png'
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  saved {p}')


def fig_3d_surface_c4(d, out_dir, prefix):
    # The C4 ceiling as a plane over (minima height, peak height).
    fig = plt.figure(figsize=(15, 6.5))
    top = min(np.nanpercentile(d.outer_rise_magnitude, 99.5), 1.0)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    mh = np.linspace(0, 1.0, 40)
    ph = np.linspace(0, 1.0, 40)
    MH, PH = np.meshgrid(mh, ph)
    CEIL = np.clip(PH - MH, 0, top)
    ax.plot_surface(MH, PH, CEIL, cmap='summer', alpha=0.55, linewidth=0,
                    rstride=2, cstride=2)
    ok = d[~d.viol_outer]; bad = d[d.viol_outer]
    ax.scatter(ok.minima_height, ok.peak_height, ok.outer_rise_magnitude.clip(upper=top),
               s=5, alpha=0.35, color='darkorange', label='reconstructable')
    ax.scatter(bad.minima_height, bad.peak_height, bad.outer_rise_magnitude.clip(upper=top),
               s=14, alpha=0.85, color='darkred', marker='x', label='clamped')
    ax.set_xlabel('minima height'); ax.set_ylabel('peak height'); ax.set_zlabel('outer rise magnitude')
    ax.set_title('C4 ceiling surface\n(points above the sheet are impossible)')
    ax.legend(fontsize=8, loc='upper left')
    ax.view_init(elev=22, azim=-125)

    ax = fig.add_subplot(1, 2, 2)
    sc = ax.scatter(d.minima_height, d.peak_height, c=d.outer_utilisation.clip(0, 2),
                    s=9, cmap='RdYlBu_r', vmin=0, vmax=2)
    plt.colorbar(sc, ax=ax, label='outer-rise utilisation  (>1 impossible)')
    ax.set_xlabel('minima height'); ax.set_ylabel('peak height')
    ax.set_title('How close each root sits to its own ceiling')
    ax.grid(alpha=0.3)

    fig.suptitle(f'C4 ceiling in 3D ({prefix}_*)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _stamp(fig, 'C4')
    p = Path(out_dir) / f'{prefix}_C4_surface_3d.png'
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  saved {p}')


def fig_morphospace(d, out_dir, prefix, k=4, seed=0):
    # Where the dataset actually lives in the four-feature space.
    X = d[CORE].to_numpy(float)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xs)
    d = d.copy(); d['cluster'] = km.labels_
    pca = PCA(n_components=2).fit(Xs)
    P = pca.transform(Xs)
    d['pc1'], d['pc2'] = P[:, 0], P[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colours = plt.cm.tab10(np.linspace(0, 1, k))

    ax = axes[0]
    for i in range(k):
        s = d[d.cluster == i]
        ax.scatter(s.pc1, s.pc2, s=10, alpha=0.55, color=colours[i],
                   label=f'cluster {i} (n={len(s)}, {100*s.infeasible.mean():.0f}% clamped)')
    ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.0f}% var)')
    ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.0f}% var)')
    ax.set_title('Morphospace of the four core features')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    pp = np.linspace(0.001, 1, 300)
    for ph_line, style in [(0.3, ':'), (0.5, '--'), (0.7, '-')]:
        ax.plot(pp, np.minimum(ph_line / (pp + EPS), 4), style, color='crimson', lw=1.4,
                label=f'C1 ceiling at h={ph_line}')
    for i in range(k):
        s = d[d.cluster == i]
        ax.scatter(s.peak_position, s.rise_slope, s=10, alpha=0.55, color=colours[i])
    ax.set_xlim(0, 1); ax.set_ylim(0, min(np.nanpercentile(d.rise_slope, 99.5), 4))
    ax.set_xlabel('peak position'); ax.set_ylabel('rise slope')
    ax.set_title('Clusters against the C1 ceiling')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    prof = d.groupby('cluster')[CORE + ['infeasible']].mean()
    prof_z = (prof[CORE] - d[CORE].mean()) / d[CORE].std()
    im = ax.imshow(prof_z.to_numpy(), cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
    ax.set_xticks(range(len(CORE)))
    ax.set_xticklabels([c.replace('_', '\n') for c in CORE], fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f'cluster {i}\n{100*prof.infeasible[i]:.0f}% clamped' for i in range(k)],
                       fontsize=8)
    for i in range(k):
        for j, c in enumerate(CORE):
            ax.text(j, i, f'{prof[c][i]:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, label='SD from dataset mean')
    ax.set_title('Cluster centres (labels are raw values)')

    plt.tight_layout()
    p = Path(out_dir) / f'{prefix}_morphospace_clusters.png'
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  saved {p}')

    print(f'\n  morphospace clusters (k={k}):')
    print(prof.round(3).to_string())
    return d


def main():
    # python -m code.statistics.feature_space_limits
    class cfg:
        prefix = 'radius'
        clusters = 4
        out_dir = str(OUT_DIR)


    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    d, _ = load(cfg.prefix)
    report(d, cfg.prefix)

    fig_rise_limits(d, out_dir, cfg.prefix)
    fig_decay_limits(d, out_dir, cfg.prefix)
    fig_3d_surface_c1(d, out_dir, cfg.prefix)
    fig_3d_surface_c2(d, out_dir, cfg.prefix)
    fig_3d_surface_c4(d, out_dir, cfg.prefix)
    d = fig_morphospace(d, out_dir, cfg.prefix, k=cfg.clusters)

    keep = ['root_id'] if 'root_id' in d.columns else []
    keep += CORE + ['minima_position', 'outer_rise_magnitude', 'start_height',
                    'minima_height_raw', 'outer_overshoot', 'rise_utilisation',
                    'decay_utilisation', 'outer_utilisation', 'viol_start', 'viol_minima',
                    'viol_outer', 'infeasible', 'cluster']
    keep = [c for c in keep if c in d.columns]
    p = out_dir / f'{cfg.prefix}_feature_feasibility.csv'
    d[keep].to_csv(p, index=False)
    print(f'\n  saved {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
