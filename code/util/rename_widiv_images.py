# Rename the widiv slide images into the naming convention the pipeline parses.
#
# The pipeline recovers per-image metadata from the filename via
# util.file_utils.parse_image_name, so a widiv file called `2003_XS.jpg` yields
# nothing usable. This renames each image to the same shape the maize data uses,
# carrying the fields widiv actually has.
#
#     2003_XS.jpg     ->  2003_WS_1 (1).jpg
#     {PLOT}_XS.jpg   ->  {PLOT}_{TRT}_{root} ({rep}).jpg
#
# After root detection and quadrant splitting the pipeline sees, for example,
# `BR_2003_WS_1 (1)_root1`, from which parse_image_name recovers:
#
#     plot_number         2003     (from the leading number)
#     treatment           WS       (from the metadata)
#     root_number         1        (one root per image)
#     technical_replicate 1        (the parenthesised number)
#     plant_number        None     widiv has no plant id
#     root_type           None     widiv roots are not typed W2/W3/W4
#
# plant_number and root_type are deliberately left absent rather than filled with
# a placeholder: inventing a `P1` would make fabricated values indistinguishable
# from real plant numbers everywhere downstream.
#
# TRT comes from Widiv_metadata.xlsx, matched on PLOT. Sixteen images have plot
# numbers that do not appear in the metadata at all; per instruction these are
# treated as WW, and are flagged in the mapping CSV so the assumption stays
# visible rather than being silently baked in.
#
# The xlsx is read with the standard library (it is a zip of XML) so this does
# not depend on openpyxl, which is not installed here.
#
# Nothing is renamed unless --apply is passed. A mapping CSV is always written,
# so the rename can be reversed.

import csv
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from code.config import DATA_FOLDER, RESULTS_FOLDER

IMAGES = DATA_FOLDER / 'widiv'

# The spreadsheet is not in the repository; drop it here before running. It
# used to be read from a personal Downloads folder, which meant the script
# only worked on one machine.
METADATA = DATA_FOLDER / 'widiv' / 'Widiv_metadata.xlsx'

MAP_OUT = RESULTS_FOLDER / 'widiv' / 'rename_map.csv'

# Applied when a plot has no metadata row (per instruction).
DEFAULT_TREATMENT = 'WW'

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def read_xlsx(path):
    # Minimal xlsx reader -- returns a list of dicts keyed by header row.
    z = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        shared = [''.join(t.text or '' for t in si.iter(NS + 't'))
                  for si in ET.fromstring(z.read('xl/sharedStrings.xml')).iter(NS + 'si')]
    rows = []
    for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).iter(NS + 'row'):
        cells = {}
        for c in row.iter(NS + 'c'):
            col = re.match(r'[A-Z]+', c.get('r')).group()
            v = c.find(NS + 'v')
            if v is not None:
                cells[col] = shared[int(v.text)] if c.get('t') == 's' else v.text
            else:
                cells[col] = ''
        rows.append(cells)
    if not rows:
        return []
    header = rows[0]
    return [{header[c]: r.get(c, '') for c in header if header.get(c)} for r in rows[1:]]


def build_plan(images_dir=IMAGES, metadata=METADATA):
    meta = {r['PLOT']: r for r in read_xlsx(metadata) if r.get('PLOT')}

    files = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))

    # Group by plot so repeat images of one plot become replicate 1, 2, ...
    by_plot = defaultdict(list)
    unparsed = []
    for p in files:
        m = re.match(r'^(\d+)_', p.stem)
        if m:
            by_plot[m.group(1)].append(p)
        else:
            unparsed.append(p)

    plan, missing = [], []
    for plot in sorted(by_plot, key=int):
        for rep, src in enumerate(sorted(by_plot[plot]), start=1):
            row = meta.get(plot)
            if row:
                trt = (row.get('TRT') or DEFAULT_TREATMENT).strip()
                assumed = False
            else:
                trt = DEFAULT_TREATMENT
                assumed = True
                missing.append(src.name)
            new_name = f'{plot}_{trt}_1 ({rep}){src.suffix.lower()}'
            plan.append({
                'old_name': src.name,
                'new_name': new_name,
                'plot': plot,
                'treatment': trt,
                'treatment_assumed': assumed,
                'taxa': (row or {}).get('TAXA', ''),
                'block': (row or {}).get('BLOCK', ''),
                'rep_field': (row or {}).get('REP', ''),
                'replicate_index': rep,
            })
    return plan, missing, unparsed


def main():
    # python -m code.util.rename_widiv_images
    class cfg:
        images = str(IMAGES)
        metadata = str(METADATA)
        map_out = str(MAP_OUT)
        # actually rename (default: dry run)
        apply = False


    images = Path(cfg.images)
    plan, missing, unparsed = build_plan(images, Path(cfg.metadata))

    print(f'images found      : {len(plan) + len(unparsed)}')
    print(f'planned renames   : {len(plan)}')
    print(f'no metadata match : {len(missing)}  -> treatment defaulted to {DEFAULT_TREATMENT}')
    if unparsed:
        print(f'UNPARSED (skipped): {len(unparsed)}')
        for p in unparsed[:10]:
            print(f'    {p.name}')

    # collisions would silently destroy data, so check before touching anything
    news = [r['new_name'] for r in plan]
    dupes = {n for n in news if news.count(n) > 1}
    if dupes:
        raise SystemExit(f'ABORT: duplicate target names: {sorted(dupes)[:10]}')
    existing = {p.name for p in images.iterdir()}
    olds = {r['old_name'] for r in plan}
    clash = (set(news) & existing) - olds
    if clash:
        raise SystemExit(f'ABORT: target names already exist: {sorted(clash)[:10]}')

    print('\nsample:')
    for r in plan[:6]:
        flag = '  [treatment assumed]' if r['treatment_assumed'] else ''
        print(f'  {r["old_name"]:24} -> {r["new_name"]}{flag}')

    out = Path(cfg.map_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(plan[0].keys()))
        w.writeheader()
        w.writerows(plan)
    print(f'\nmapping written   : {out}')

    if not cfg.apply:
        print('\nDRY RUN -- nothing renamed. Re-run with --apply.')
        return 0

    renamed = 0
    for r in plan:
        src = images / r['old_name']
        dst = images / r['new_name']
        if src.exists() and src != dst:
            src.rename(dst)
            renamed += 1
    print(f'\nrenamed {renamed} files in {images}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
