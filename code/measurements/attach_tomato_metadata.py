# Fills in master_summary metadata for the hand-annotated tomato set.
#
# The values come from the dataset's own CrossSections.csv, because tomato was
# annotated by hand and never went through the stages that normally record
# species, population and treatment.
#
# The maize filenames encode plot / plant / root / treatment / root-type, and
# parse_image_name() digs them back out. The tomato filenames follow a different
# convention entirely --
#     {plant_id}_{segment}_{coloration}_{cut}_{zoom}_{repetition}
# -- so parse_image_name recovers only the quadrant and (inconsistently) a plant
# number. This script matches each tomato image back to its CrossSections.csv row
# and fills the columns properly.
#
# Matching is deliberately tolerant, because the image filenames disagree with
# CrossSections.csv in several small ways (all verified by hand):
#   * coloration written 'T' instead of 'To', or omitted entirely
#   * the trailing repetition omitted, or different from the CSV's
#   * one segment written 'D1' where the CSV has 'D'
# So we match on (plant_id, segment, cut, zoom) first -- which is unique in this
# CSV -- then retry with trailing digits stripped from the segment. An exact
# CS_id match is always preferred when one exists.
#
# treatment and root_type are intentionally left empty: this experiment has
# neither, and inventing values would make tomato look like a maize treatment arm
# in the grouped analyses.

import json
import os
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Columns copied verbatim from CrossSections.csv so nothing is lost to the
# lossy int-only columns above them.
RAW_COLUMNS = ['plant_id', 'segment', 'coloration', 'cut', 'zoom', 'repetition', 'age_days']


def root_key_from_image(image_name):
    # 'BL_P11_D1_To_C01_Z20_01 exort1_root1' -> 'P11_D1_To_C01_Z20_01'.
    s = os.path.splitext(image_name)[0]
    s = re.sub(r'^(BL|BR|TL|TR)_', '', s)
    # ' export1_root1', ' exort1_root1' (typo in one source file), ' export_root1'
    s = re.sub(r'\s*ex[a-z]*\d*_root\d+$', '', s)
    return s


def parse_components(s):
    parts = s.split('_')
    plant = parts[0] if parts else None
    segment = parts[1] if len(parts) > 1 else None
    cut = next((p for p in parts if re.fullmatch(r'C\d+', p)), None)
    zoom = next((p for p in parts if re.fullmatch(r'Z\d+', p)), None)
    return plant, segment, cut, zoom


def build_lookup(cross_sections_csv):
    csv = pd.read_csv(cross_sections_csv, dtype=str)
    csv['age'] = pd.to_numeric(csv['age'], errors='coerce')
    by_id = {r.CS_id: r for r in csv.itertuples(index=False)}
    by_key4, by_key3 = {}, {}
    for r in csv.itertuples(index=False):
        by_key4.setdefault((r.plant_id, r.segment, r.cut, r.zoom), []).append(r)
        seg3 = re.sub(r'\d+$', '', r.segment or '')
        by_key3.setdefault((r.plant_id, seg3, r.cut, r.zoom), []).append(r)
    return by_id, by_key4, by_key3


def match_row(root_key, by_id, by_key4, by_key3):
    # Return (csv_row, how) or (None, reason).
    if root_key in by_id:
        return by_id[root_key], 'exact'
    plant, segment, cut, zoom = parse_components(root_key)
    hits = by_key4.get((plant, segment, cut, zoom))
    if hits and len(hits) == 1:
        return hits[0], 'plant/segment/cut/zoom'
    if hits:
        return hits[0], f'plant/segment/cut/zoom AMBIGUOUS({len(hits)})'
    seg3 = re.sub(r'\d+$', '', segment or '')
    hits = by_key3.get((plant, seg3, cut, zoom))
    if hits and len(hits) == 1:
        return hits[0], 'segment-digit-stripped'
    if hits:
        return hits[0], f'segment-digit-stripped AMBIGUOUS({len(hits)})'
    return None, 'NO MATCH'


def attach(master_summary_path, cross_sections_csv, species='tomato', dry_run=False):
    ms = pd.read_csv(master_summary_path)
    if 'species' not in ms.columns:
        raise SystemExit('master_summary.csv has no species column')

    target = ms['species'] == species
    if not target.any():
        print(f"No rows with species == '{species}'; nothing to do.")
        return ms

    by_id, by_key4, by_key3 = build_lookup(cross_sections_csv)

    for col in RAW_COLUMNS:
        if col not in ms.columns:
            ms[col] = pd.NA

    # Stable integer ids: plant_number per distinct plant_id, root_number from
    # the cut number (the cut is what identifies a given cross-section).
    keys = {i: root_key_from_image(ms.at[i, 'image_name']) for i in ms.index[target]}
    matches = {i: match_row(k, by_id, by_key4, by_key3) for i, k in keys.items()}

    plant_ids = sorted({row.plant_id for row, how in matches.values() if row is not None})
    plant_number_of = {p: n + 1 for n, p in enumerate(plant_ids)}

    report, unmatched = [], []
    for i, (row, how) in matches.items():
        if row is None:
            unmatched.append((keys[i], how))
            continue
        ms.at[i, 'plant_number'] = plant_number_of[row.plant_id]
        cut_num = re.sub(r'\D', '', row.cut or '')
        ms.at[i, 'root_number'] = int(cut_num) if cut_num else pd.NA
        rep = re.sub(r'\D', '', row.repetition or '')
        ms.at[i, 'technical_replicate'] = int(rep) if rep else pd.NA
        # treatment / root_type deliberately untouched -- see module docstring.
        ms.at[i, 'plant_id'] = row.plant_id
        ms.at[i, 'segment'] = row.segment
        ms.at[i, 'coloration'] = row.coloration
        ms.at[i, 'cut'] = row.cut
        ms.at[i, 'zoom'] = row.zoom
        ms.at[i, 'repetition'] = row.repetition
        ms.at[i, 'age_days'] = row.age
        report.append((keys[i], row.CS_id, how, row.age))

    seen = {}
    for key, cs_id, how, age in report:
        seen.setdefault((key, cs_id, how, age), 0)
        seen[(key, cs_id, how, age)] += 1
    print(f'{"IMAGE ROOT":34} {"CS_id":30} {"HOW":26} {"AGE":>6}  n')
    for (key, cs_id, how, age), n in sorted(seen.items()):
        print(f'{key:34} {cs_id:30} {how:26} {age:>6}  {n}')

    print(f'\nmatched {len(report)} rows across {len(seen)} distinct roots; '
          f'plant_number map: {plant_number_of}')
    if unmatched:
        print(f'UNMATCHED ({len(unmatched)}):')
        for k, why in unmatched:
            print(f'  {k}: {why}')

    if dry_run:
        print('\n(dry run -- master_summary.csv not written)')
        return ms

    ms.to_csv(master_summary_path, index=False)
    print(f'\nWrote {master_summary_path}')
    return ms


def main():
    # python -m code.measurements.attach_tomato_metadata
    class cfg:
        master_summary = str(PROJECT_ROOT / 'results' / 'master_summary.csv')
        cross_sections = None
        species = 'tomato'
        dry_run = False

    attach(cfg.master_summary, cfg.cross_sections, cfg.species, cfg.dry_run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
