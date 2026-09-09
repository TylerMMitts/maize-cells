# Package the root model for the public website, with a curated image set.
#
# results/root_model/root_model.html ships every root in the dataset (1,648 of
# them) and points at data/chosen_results/detections/ for the photographs. That
# is right for local use and wrong for a public site: the images are several
# hundred megabytes and none of them would be pushed, so every individual-root
# preset would render a broken image.
#
# This builds a web copy that carries its own images. It picks a small,
# defensible set of examples per species, downsizes those photographs, and
# rewrites the preset list so the dropdown offers exactly the roots whose images
# actually shipped -- no broken thumbnails, and a menu short enough to demo from.
#
# The aggregate presets (dataset median, per-treatment, per-root-type,
# per-population) are always kept: they carry no photograph, so they cost
# nothing and they are the sensible default view.
#
# WHAT MAKES A ROOT A GOOD EXAMPLE
#   faithful     the reconstruction must reproduce the root it claims to show.
#                The individual-root presets are built from the file_* features
#                (generate_root_model.py), so feasibility is judged in that
#                basis, not the radius_* one -- they disagree substantially.
#                Rather than a hard feasible/infeasible cut, a root qualifies if
#                its worst constraint violation is within TOLERANCE of its own
#                peak height; most real violations are a fraction of a percent
#                and are invisible on screen, while a hard cut would throw away
#                most of the smaller species for no visible gain.
#   complete     all four quadrants measured, not a partial root
#   representative  shape close to that species' own median
#
# Within a species, picks are restricted to the middle of the observed
# cell-file-count range (MID_FRAC) before ranking -- the smallest- and
# largest-file roots looked worse in the tool, so both tails are excluded
# rather than deliberately included.
#
# Species with too few usable roots are reported and skipped rather than being
# padded out. Tomato is the standing case: it is hand-segmented, only 2 of its
# 19 roots have file_* features at all (they carry 2-4 cell files, too few to
# fit a profile over file index), and none has a measured stele diameter -- the
# generator would substitute a default one, so any tomato preset would be drawn
# against a stele that was never measured.
#
# Settings live in the cfg block in main(); there are no command-line flags.
#     python -m code.root_model.build_web_bundle

import io
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

SOURCE_HTML = PROJECT_ROOT / 'results' / 'root_model' / 'root_model.html'
FEATURE_TABLE = PROJECT_ROOT / 'results' / 'feature_table.csv'
DETECTIONS = PROJECT_ROOT / 'data' / 'chosen_results' / 'detections'

# Written inside the project by default. Point cfg.out at the public/ folder of
# a website checkout when publishing; it used to default to a personal Desktop
# path, which only existed on one machine.
DEFAULT_OUT = PROJECT_ROOT / 'results' / 'root_model' / 'web_bundle'

# Where the shipped photographs live, relative to the generated page.
WEB_IMAGE_DIR = 'roots/'

# Downsize for the web: these are scanned at 2-3 MP and a phone at a poster
# session does not need that. Longest edge in pixels.
MAX_EDGE = 1000
JPEG_QUALITY = 82

EPS = 0.001  # the guard term the forward feature definitions use
SHAPE = ['peak_height', 'peak_position', 'rise_slope', 'decay_slope']

# Individual-root presets are built from the file_* features, so that is the
# basis reconstruction fidelity has to be judged in.
BASIS = 'file'

# Worst allowed constraint violation, as a fraction of the root's own peak
# height. 0.05 keeps the reconstruction visually indistinguishable from the
# root it represents while retaining enough of the smaller species to show.
TOLERANCE = 0.05

# Fraction of each species' observed n_files range to keep, centered on the
# middle -- e.g. 0.5 keeps the middle half and drops the smallest and largest
# quarter of file counts before ranking.
MID_FRAC = 0.5

# Presented in this order; the label is what the dropdown shows.
SPECIES_ORDER = [
    ('Zea mays', 'Maize'),
    ('Pearl Millet', 'Pearl millet'),
    ('Sorghum', 'Sorghum'),
    ('tomato', 'Tomato'),
]

# Held out of the bundle regardless of what the data would allow, with the
# reason shown in the build report. Tomato has no measured stele diameter on
# any of its 19 roots, so the generator substitutes a default one; a preset
# built on that would be drawn against a stele nobody measured.
EXCLUDED_SPECIES = {'tomato': 'no measured stele diameter'}


def load_html(path):
    text = io.open(path, encoding='utf-8').read()
    m = re.search(r'const DATA\s*=\s*(\{.*?\});\s*\n', text, re.S)
    if not m:
        raise SystemExit(f'could not find the DATA block in {path}')
    return text, json.loads(m.group(1)), m.span(1)


def infidelity(row, prefix=BASIS):
    # Worst constraint violation, as a fraction of the root's own peak height.
    #
    # 0 means the reconstruction reproduces the root exactly; NaN means the
    # features needed to build a preset are missing. See feature_space_limits.py
    # for where the three constraints come from.
    ph = row.get(f'{prefix}_peak_height')
    pp = row.get(f'{prefix}_peak_position')
    rs = row.get(f'{prefix}_rise_slope')
    ds = row.get(f'{prefix}_decay_slope')
    mp = row.get(f'{prefix}_minima_position')
    orm = row.get(f'{prefix}_outer_rise_magnitude')
    if any(pd.isna(v) for v in (ph, pp, rs, ds, mp, orm)) or not ph:
        return float('nan')

    start = ph - rs * (pp + EPS)                        # C1
    gap = max(mp - pp, 0.0)
    minima = ph + ds * (gap + EPS) if mp > pp else ph    # C2
    over = (max(minima, 0.0) + orm) - ph                 # C4
    return max(max(-start, 0.0), max(-minima, 0.0), max(over, 0.0)) / ph


def score(df):
    # Rank within a species: faithful, complete, and typical.
    d = df.copy()
    z = lambda s: (s - s.median()) / (s.std() if s.std() else 1.0)

    dist = np.zeros(len(d))
    for f in SHAPE:
        col = f'{BASIS}_{f}'
        if col in d:
            dist += z(d[col]).fillna(0).abs().to_numpy()
    d['typicality'] = -dist / max(len(SHAPE), 1)

    d['complete'] = (d.get('n_quadrants', 0) == 4).astype(float)
    # fidelity dominates: a root that renders as itself beats a tidier one
    d['fidelity'] = -(d['infidelity'] / TOLERANCE)

    d['score'] = 3.0 * d['fidelity'] + 2.0 * d['complete'] + 1.5 * d['typicality']
    return d.sort_values('score', ascending=False)


def spread_by_files(df, k, mid_frac=MID_FRAC):
    # Best up-to-k roots by score, restricted to the middle of the file-count range.
    #
    # The very smallest and largest file counts per species looked worse in the
    # tool, so both tails are excluded before ranking -- regardless of how many
    # roots that leaves. A species with few roots to begin with (e.g. sorghum)
    # may end up offering fewer than k, which is preferable to including a root
    # from an excluded tail just to hit the count.
    d = score(df)
    files = d['n_files'].astype(float)
    lo, hi = files.min(), files.max()
    span = hi - lo
    if span > 0:
        pad = span * (1 - mid_frac) / 2
        mid = d[(files >= lo + pad) & (files <= hi - pad)]
        if len(mid) >= 3:  # only apply the restriction if it leaves enough to choose from
            d = mid

    return d.sort_values('n_files') if len(d) <= k else d.head(k).sort_values('n_files')


def choose(per_species, available=None, feature_table=FEATURE_TABLE, detections=DETECTIONS):
    # Pick the best examples per species.
    #
    # `available` is the set of root ids that actually exist as individual-root
    # presets in the source page. It matters: a root needs a stele diameter to
    # set the reconstruction's absolute scale, and hand-segmented species do not
    # have one, so they never became presets. Selecting such a root would ship a
    # photograph the dropdown can never reach.
    ft = pd.read_csv(feature_table)
    have = set(os.listdir(detections)) if Path(detections).exists() else set()
    ft['image'] = ft.root_id.astype(str) + '_full.jpg'
    ft['infidelity'] = ft.apply(infidelity, axis=1)

    picked, report = {}, []
    for key, label in SPECIES_ORDER:
        sub = ft[ft.species == key]
        total = len(sub)

        if key in EXCLUDED_SPECIES:
            report.append((label, 0, total, None, EXCLUDED_SPECIES[key]))
            continue

        sub = sub[sub.image.isin(have)]
        if available is not None:
            sub = sub[sub.root_id.astype(str).isin(available)]
        if not len(sub):
            report.append((label, 0, total, None,
                           f'{BASIS}_* shape features unavailable'))
            continue

        sub = sub[sub.infidelity <= TOLERANCE]
        if not len(sub):
            report.append((label, 0, total, None,
                           f'no root reconstructs within {TOLERANCE:.0%}'))
            continue

        top = spread_by_files(sub, per_species)
        picked[label] = list(top.root_id.astype(str))
        files = top['n_files'].astype(float)
        report.append((label, len(top), total,
                       (files.min(), files.max()), ''))
    return picked, report


def copy_images(picked, out_dir, detections=DETECTIONS, dry_run=False):
    from PIL import Image

    img_dir = Path(out_dir) / WEB_IMAGE_DIR.strip('/')
    if not dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for label, roots in picked.items():
        for rid in roots:
            src = Path(detections) / f'{rid}_full.jpg'
            dst = img_dir / f'{rid}_full.jpg'
            if dry_run:
                total += src.stat().st_size if src.exists() else 0
                continue
            im = Image.open(src)
            im.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.save(dst, 'JPEG', quality=JPEG_QUALITY, optimize=True, progressive=True)
            total += dst.stat().st_size
    return img_dir, total


def rebuild_presets(data, picked):
    # Aggregates first, then one group per species holding only shipped roots.
    keep_index = {}
    for label, roots in picked.items():
        for rid in roots:
            keep_index[rid] = label

    out, dropped = [], 0
    for p in data['presets']:
        if p.get('group') != 'Individual roots':
            out.append(p)

    by_species = {label: [] for label in picked}
    for p in data['presets']:
        if p.get('group') != 'Individual roots':
            continue
        label = keep_index.get(p.get('label'))
        if label is None:
            dropped += 1
            continue
        q = dict(p)
        q['group'] = label
        by_species[label].append(q)

    for label in picked:
        # keep the curated order the ranking produced
        order = {rid: i for i, rid in enumerate(picked[label])}
        by_species[label].sort(key=lambda q: order.get(q['label'], 1e9))
        out.extend(by_species[label])

    return out, dropped


def main():
    # python -m code.root_model.build_web_bundle
    class cfg:
        source = str(SOURCE_HTML)
        # website folder that receives reconstruction.html and roots/
        out = str(DEFAULT_OUT)
        per_species = 10
        dry_run = False


    out_dir = Path(cfg.out)
    text, data, (lo, hi) = load_html(Path(cfg.source))

    available = {p['label'] for p in data['presets']
                 if p.get('group') == 'Individual roots'}

    picked, report = choose(cfg.per_species, available=available)
    print(f'{"species":14} {"picked":>7} {"in dataset":>11}  cell files')
    for label, n, pool, frange, reason in report:
        if reason:
            note = f'  -- skipped: {reason}'
        else:
            note = f'  {frange[0]:.0f} - {frange[1]:.0f}'
            if n < cfg.per_species:
                note += '   (all that qualify)'
        print(f'{label:14} {n:7} {pool:11}{note}')

    presets, dropped = rebuild_presets(data, picked)
    total_roots = sum(len(v) for v in picked.values())

    # every shipped image must be reachable from the dropdown, and every
    # individual preset must have a shipped image -- neither orphan is allowed
    kept_ids = {p['label'] for p in presets if p.get('group') in picked}
    all_picked = {r for v in picked.values() for r in v}
    if kept_ids != all_picked:
        raise SystemExit(f'preset/image mismatch: {sorted(all_picked - kept_ids)[:5]}')

    print(f'\npresets: {len(data["presets"])} -> {len(presets)}  '
          f'({dropped} individual roots dropped, {total_roots} kept)')

    img_dir, nbytes = copy_images(picked, out_dir, dry_run=cfg.dry_run)
    print(f'images : {total_roots} -> {img_dir}  ({nbytes/1e6:.1f} MB'
          f'{" before resize" if cfg.dry_run else " after resize"})')

    if cfg.dry_run:
        print('\ndry run - nothing written')
        return 0

    data['presets'] = presets
    data['imageDir'] = WEB_IMAGE_DIR
    new_text = text[:lo] + json.dumps(data, separators=(',', ':')) + text[hi:]

    dst = out_dir / 'reconstruction.html'
    io.open(dst, 'w', encoding='utf-8').write(new_text)
    before = len(text) / 1e6
    after = len(new_text) / 1e6
    print(f'html   : {dst}  ({before:.2f} MB -> {after:.2f} MB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
