# How the cortical cell pattern emerges across root development.
#
# Standalone -- this reads a Roboflow COCO export directly and does not touch the
# main pipeline, its master summary, or its feature table.
#
# THE DATA
# data/age_experiment/train/ holds transverse sections taken at five positions
# along the root axis, which correspond to five developmental stages:
#
#     root cap -> meristem -> early elongation -> elongation -> late elongation
#
# Each stage is imaged with several different cell-wall stains (calcofluor
# white, arabinoxylan, xyloglucan, ...). The stain changes what fluoresces, not
# the anatomy, so for this question the stains act as replicates of the same
# stage and are pooled -- but they are kept in the per-image outputs so a stain
# effect would be visible if one existed.
#
# Unlike the main pipeline these images are not split into quadrants, and only a
# single cell file is annotated in each -- roughly ten cells running from the
# stele outward to the epidermis.
#
# HOW A PROFILE IS BUILT
#     1. cell area comes from the annotated polygon (shoelace), not the bbox
#     2. the cell whose centroid is nearest the image centre is taken as the
#        innermost cell -- the start of the cortex
#     3. every cell's radius is its centroid distance from that innermost cell,
#        so the innermost cell sits at radius 0 and the file runs outward
#     4. radius and area are min-max normalised within the image, matching the
#        convention the main pipeline uses in visualize_splines.py
#
# A NOTE ON UNITS
# The panels carry scale bars but they do not read reliably at this resolution,
# so no micron calibration is applied. Normalised profiles -- the primary
# result -- are unaffected, since normalisation divides magnification out. Raw
# pixel areas are also reported, and those DO assume every panel shares a
# magnification; treat them as indicative rather than measured.
#
# Outputs (results/age_experiment/):
#     pattern_emergence.png               per-stage normalised profiles (headline)
#     absolute_cell_size.png              the same stages in raw pixel area
#     stage_small_multiples.png           each stage's individual files, normalised
#     stage_small_multiples_absolute.png  the same, in raw pixel area
#     stage_metrics.png                   peak position / hump / size vs stage
#     cells_long.csv                      one row per annotated cell
#     stage_summary.csv                   per-stage fitted metrics
#     per_file_fits.csv                   per-file fitted metrics

import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / 'data' / 'age_experiment' / 'train'
OUT_DIR = PROJECT_ROOT / 'results' / 'age_experiment'

# Developmental order, tip first. The keys are as they appear in the filenames.
STAGES = ['Root-cap', 'Meristem', 'Early-elongation', 'Elongation', 'Late-elongation']
STAGE_LABEL = {
    'Root-cap': 'Root cap',
    'Meristem': 'Meristem',
    'Early-elongation': 'Early elongation',
    'Elongation': 'Elongation',
    'Late-elongation': 'Late elongation',
}
STAGE_RE = re.compile('|'.join(STAGES))

MIN_CELLS = 5      # a cubic needs 4 points; below this a profile says nothing
N_GRID = 200


def cubic(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def polygon_area(pts):
    # Shoelace area of one polygon ring.
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_centroid(pts):
    # Area-weighted centroid; falls back to the vertex mean if degenerate.
    x, y = pts[:, 0], pts[:, 1]
    cross = x * np.roll(y, 1) - np.roll(x, 1) * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return pts.mean(axis=0)
    cx = ((x + np.roll(x, 1)) * cross).sum() / (6 * a)
    cy = ((y + np.roll(y, 1)) * cross).sum() / (6 * a)
    return np.array([cx, cy])


def parse_stage(name):
    m = STAGE_RE.search(name)
    return m.group(0) if m else None


def parse_stain(name):
    # Everything between the grid reference and the stage is the stain.
    m = re.match(r'.*?_R(\d+)C(\d+)_(.+?)_(?:' + '|'.join(STAGES) + ')', name)
    return (m.group(3).replace('_', ' '), int(m.group(1))) if m else ('unknown', -1)


def load_cells(data_dir=DATA_DIR):
    coco = json.load(open(Path(data_dir) / '_annotations.coco.json'))
    images = {im['id']: im for im in coco['images']}

    # the export carries a container category alongside the real one
    cell_ids = {c['id'] for c in coco['categories'] if c['name'].lower() == 'cell'}

    per_image = {}
    for a in coco['annotations']:
        if cell_ids and a['category_id'] not in cell_ids:
            continue
        seg = a.get('segmentation') or []
        if not seg or len(seg[0]) < 6:
            continue
        pts = np.asarray(seg[0], dtype=float).reshape(-1, 2)
        per_image.setdefault(a['image_id'], []).append(
            dict(area_px=polygon_area(pts), centroid=polygon_centroid(pts)))

    rows = []
    for img_id, cells in per_image.items():
        im = images[img_id]
        name = im.get('extra', {}).get('name', im['file_name'])
        stage = parse_stage(name)
        if stage is None or len(cells) < MIN_CELLS:
            continue
        stain, row_no = parse_stain(name)

        cen = np.array([c['centroid'] for c in cells])
        # The most centred annotated cell is the innermost one: the file runs
        # from the stele outward, so the end nearest the image centre starts it.
        img_centre = np.array([im['width'] / 2.0, im['height'] / 2.0])
        inner = int(np.argmin(np.linalg.norm(cen - img_centre, axis=1)))
        radius = np.linalg.norm(cen - cen[inner], axis=1)

        area = np.array([c['area_px'] for c in cells])
        order = np.argsort(radius)
        for rank, i in enumerate(order):
            rows.append(dict(
                image=name, stage=stage, stain=stain, stain_row=row_no,
                n_cells=len(cells), rank=rank,
                radius_px=radius[i], area_px=area[i],
                cx=cen[i][0], cy=cen[i][1],
                is_innermost=(i == inner),
            ))

    df = pd.DataFrame(rows)

    # min-max within each image, the same convention visualize_splines.py uses
    g = df.groupby('image')
    df['radius_norm'] = g.radius_px.transform(lambda s: (s - s.min()) / (s.max() - s.min())
                                              if s.max() > s.min() else np.nan)
    df['area_norm'] = g.area_px.transform(lambda s: (s - s.min()) / (s.max() - s.min())
                                          if s.max() > s.min() else np.nan)
    # area relative to the innermost cell keeps a magnitude the normalisation drops
    inner_area = df[df.is_innermost].set_index('image').area_px
    df['area_rel_inner'] = df.area_px / df.image.map(inner_area)

    df['stage'] = pd.Categorical(df.stage, categories=STAGES, ordered=True)
    return df.dropna(subset=['radius_norm', 'area_norm']).sort_values(['stage', 'image', 'rank'])


def fit_profile(x, y):
    # Cubic fit plus the landmarks that say whether a hump exists.
    if len(x) < 5 or np.ptp(x) == 0:
        return None
    try:
        popt, _ = curve_fit(cubic, x, y, p0=[0.1, -0.1, 1.0, 0.5], maxfev=20000)
    except Exception:
        return None

    gx = np.linspace(0, 1, N_GRID)
    gy = cubic(gx, *popt)
    k = int(np.argmax(gy))
    resid = y - cubic(np.asarray(x), *popt)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return dict(
        popt=popt, grid_x=gx, grid_y=gy,
        peak_position=float(gx[k]), peak_value=float(gy[k]),
        start_value=float(gy[0]), end_value=float(gy[-1]),
        # how far the profile climbs before turning over: 0 means no hump
        hump=float(gy[k] - gy[0]),
        # an interior peak is the signature of the pattern
        interior_peak=bool(0.05 < gx[k] < 0.95),
        r2=float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        n=len(x),
    )


def per_image_fits(df):
    out = []
    for (image, stage, stain), g in df.groupby(['image', 'stage', 'stain'], observed=True):
        f = fit_profile(g.radius_norm.to_numpy(), g.area_norm.to_numpy())
        if f is None:
            continue
        out.append(dict(image=image, stage=stage, stain=stain, n_cells=len(g),
                        peak_position=f['peak_position'], hump=f['hump'],
                        interior_peak=f['interior_peak'], r2=f['r2'],
                        mean_area_px=g.area_px.mean(),
                        outer_over_inner=g.sort_values('rank').area_rel_inner.iloc[-1]))
    d = pd.DataFrame(out)
    d['stage'] = pd.Categorical(d.stage, categories=STAGES, ordered=True)
    return d.sort_values('stage')


def stage_fits(df, ycol='area_norm'):
    fits = {}
    for stage, g in df.groupby('stage', observed=True):
        f = fit_profile(g.radius_norm.to_numpy(), g[ycol].to_numpy())
        if f:
            f['n_images'] = g.image.nunique()
            fits[stage] = f
    return fits


# ---------------------------------------------------------------- figures

def fig_emergence(df, fits, out_dir):
    # Normalised profiles only -- shape, with magnification divided out.
    fig, ax = plt.subplots(figsize=(8.4, 6))
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(STAGES)))

    for c, stage in zip(colours, STAGES):
        if stage not in fits:
            continue
        f = fits[stage]
        g = df[df.stage == stage]
        # Binned means rather than raw points: min-max normalising each file
        # pins exactly one of its cells to 0 and one to 1, so a raw scatter
        # draws dense stripes along both edges that mean nothing.
        b = g.groupby(pd.cut(g.radius_norm, np.linspace(0, 1, 7), include_lowest=True),
                      observed=True).agg(x=('radius_norm', 'mean'),
                                         y=('area_norm', 'mean'),
                                         e=('area_norm', 'sem')).dropna(subset=['x', 'y'])
        ax.errorbar(b.x, b.y, yerr=b.e, fmt='o', ms=4, alpha=0.55, color=c,
                    capsize=2, lw=1)
        ax.plot(f['grid_x'], f['grid_y'], lw=2.6, color=c,
                label=f"{STAGE_LABEL[stage]}  (n={f['n_images']} files)")
        ax.plot(f['peak_position'], f['peak_value'], 'o', ms=7, color=c,
                mec='white', mew=1.4, zorder=5)

    ax.set_xlabel('Normalised position along the cell file\n(0 = innermost cell, 1 = epidermis)')
    ax.set_ylabel('Normalised cell area')
    ax.set_title('Emergence of the cortical cell pattern\nalong root development',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower center')

    plt.tight_layout()
    p = Path(out_dir) / 'pattern_emergence.png'
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  saved {p}')


def fig_absolute_size(df, out_dir):
    # The magnitude the normalisation deliberately removes.
    fig, ax = plt.subplots(figsize=(8.4, 6))
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(STAGES)))

    for c, stage in zip(colours, STAGES):
        g = df[df.stage == stage]
        if g.empty:
            continue
        b = g.groupby(pd.cut(g.radius_norm, np.linspace(0, 1, 7), include_lowest=True),
                      observed=True).agg(x=('radius_norm', 'mean'),
                                         y=('area_px', 'median'),
                                         lo=('area_px', lambda s: s.quantile(0.25)),
                                         hi=('area_px', lambda s: s.quantile(0.75))).dropna()
        n = g.image.nunique()
        ax.plot(b.x, b.y, 'o-', ms=4, lw=2, color=c,
                label=f'{STAGE_LABEL[stage]}  (n={n} files)')
        ax.fill_between(b.x, b.lo, b.hi, color=c, alpha=0.13)

    ax.set_xlabel('Normalised position along the cell file\n(0 = innermost cell, 1 = epidermis)')
    ax.set_ylabel('Cell area (pixels$^2$)')
    ax.set_title('Absolute cell size along the file\n(median and IQR; assumes a shared magnification)',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    p = Path(out_dir) / 'absolute_cell_size.png'
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  saved {p}')


def fig_small_multiples(df, fits, out_dir, ycol='area_norm',
                        ylabel='normalised cell area', fname='stage_small_multiples.png',
                        title='Individual annotated cell files, by stage (grey) with the stage fit'):
    # Every annotated file drawn on its own stage's axes.
    #
    # A shared y axis is what makes the absolute version readable: the point
    # there is that the meristem sits flat and low while the elongation stages
    # climb, and independent axes would hide exactly that.
    fig, axes = plt.subplots(1, len(STAGES), figsize=(19, 3.9), sharex=True, sharey=True)
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(STAGES)))

    for ax, c, stage in zip(axes, colours, STAGES):
        g = df[df.stage == stage]
        for _, gi in g.groupby('image'):
            gi = gi.sort_values('radius_norm')
            ax.plot(gi.radius_norm, gi[ycol], '-', lw=1, alpha=0.45, color='grey')
            ax.plot(gi.radius_norm, gi[ycol], '.', ms=4, alpha=0.5, color='grey')
        if stage in fits:
            f = fits[stage]
            ax.plot(f['grid_x'], f['grid_y'], lw=2.8, color=c)
            ax.axvline(f['peak_position'], color=c, ls='--', lw=1.2, alpha=0.8)
            ax.set_title(f"{STAGE_LABEL[stage]}\npeak at {f['peak_position']:.2f}",
                         fontsize=10)
        else:
            ax.set_title(STAGE_LABEL[stage], fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel('position along file')
    axes[0].set_ylabel(ylabel)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    p = Path(out_dir) / fname
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  saved {p}')


def fig_metrics(per_img, fits, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    xs = np.arange(len(STAGES))
    labels = [STAGE_LABEL[s] for s in STAGES]

    def series(col):
        m = per_img.groupby('stage', observed=False)[col].agg(['mean', 'sem', 'size'])
        return m.reindex(STAGES)

    ax = axes[0]
    m = series('peak_position')
    ax.errorbar(xs, m['mean'], yerr=m['sem'], fmt='o-', lw=2, ms=7, capsize=4,
                color='#0f6b57')
    pooled = [fits[s]['peak_position'] if s in fits else np.nan for s in STAGES]
    ax.plot(xs, pooled, 's--', ms=5, lw=1.2, color='grey', label='pooled fit')
    ax.set_ylim(0, 1)
    ax.set_ylabel('peak position along the file')
    ax.set_title('Where the largest cells sit')
    ax.legend(fontsize=8)

    ax = axes[1]
    m = series('hump')
    ax.errorbar(xs, m['mean'], yerr=m['sem'], fmt='o-', lw=2, ms=7, capsize=4,
                color='#c2510a')
    ax.axhline(0, color='grey', lw=1, ls=':')
    ax.set_ylabel('rise before the turnover')
    ax.set_title('How pronounced the hump is')

    ax = axes[2]
    m = series('mean_area_px')
    ax.errorbar(xs, m['mean'], yerr=m['sem'], fmt='o-', lw=2, ms=7, capsize=4,
                color='#334155')
    ax.set_ylabel('mean cell area (pixels$^2$)')
    ax.set_title('Cell size\n(assumes a shared magnification)')

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('Pattern landmarks across development (mean +/- SE over stains)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    p = Path(out_dir) / 'stage_metrics.png'
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  saved {p}')


def report(df, per_img, fits):
    print(f'\n\nCORTICAL CELL PATTERN ACROSS DEVELOPMENT')
    print(f'{df.image.nunique()} annotated files, {len(df)} cells, '
          f'{df.stain.nunique()} stains\n')
    print(f'{"stage":18} {"files":>5} {"cells":>6} {"peak":>6} {"hump":>7} '
          f'{"R2":>6} {"interior":>9}')
    for s in STAGES:
        g = df[df.stage == s]
        if s not in fits:
            print(f'{STAGE_LABEL[s]:18} {g.image.nunique():5} {len(g):6}'
                  f'{"":>6}{"":>7}{"":>6}   (no fit)')
            continue
        f = fits[s]
        pi = per_img[per_img.stage == s]
        frac = pi.interior_peak.mean() if len(pi) else np.nan
        print(f'{STAGE_LABEL[s]:18} {g.image.nunique():5} {len(g):6} '
              f'{f["peak_position"]:6.2f} {f["hump"]:7.3f} {f["r2"]:6.2f} '
              f'{frac:8.0%}')
    print('\n  peak     = where the fitted profile is highest (0 = innermost)')
    print('  hump     = how far it climbs above the innermost cell before turning over')
    print('  interior = share of individual files whose own peak is not at an edge')


def main():
    # python -m code.age_experiment.pattern_development
    class cfg:
        data_dir = str(DATA_DIR)
        out_dir = str(OUT_DIR)


    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_cells(Path(cfg.data_dir))
    if df.empty:
        raise SystemExit('no usable annotations found')

    fits = stage_fits(df)
    per_img = per_image_fits(df)
    report(df, per_img, fits)

    fig_emergence(df, fits, out_dir)
    fig_absolute_size(df, out_dir)
    fig_small_multiples(df, fits, out_dir)

    # the same per-file view in absolute units, against fits in those units
    abs_fits = stage_fits(df, ycol='area_px')
    fig_small_multiples(
        df, abs_fits, out_dir, ycol='area_px',
        ylabel='cell area (pixels$^2$)',
        fname='stage_small_multiples_absolute.png',
        title='Individual annotated cell files, by stage (grey), in absolute cell area')

    fig_metrics(per_img, fits, out_dir)

    df.to_csv(out_dir / 'cells_long.csv', index=False)
    rows = []
    for s in STAGES:
        if s not in fits:
            continue
        f = fits[s]
        pi = per_img[per_img.stage == s]
        rows.append(dict(stage=s, n_files=f['n_images'], n_cells=f['n'],
                         peak_position=f['peak_position'], hump=f['hump'],
                         start_value=f['start_value'], end_value=f['end_value'],
                         r2=f['r2'],
                         peak_position_mean=pi.peak_position.mean(),
                         peak_position_sem=pi.peak_position.sem(),
                         hump_mean=pi.hump.mean(), hump_sem=pi.hump.sem(),
                         interior_peak_frac=pi.interior_peak.mean(),
                         mean_area_px=pi.mean_area_px.mean()))
    pd.DataFrame(rows).to_csv(out_dir / 'stage_summary.csv', index=False)
    per_img.to_csv(out_dir / 'per_file_fits.csv', index=False)
    print(f'\n  saved {out_dir / "cells_long.csv"}, stage_summary.csv, per_file_fits.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
