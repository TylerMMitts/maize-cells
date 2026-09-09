# Five printable roots that differ only in how cell size is distributed.
#
# The question these are built to answer is whether the cortical cell pattern
# does anything structural. To ask that cleanly, everything except the pattern
# has to be held still:
#
#     fixed     stele diameter and cell file count (8)
#     fixed     the absolute size range a file may take (min/max file area)
#     varying   the pattern -- how cell size is distributed across the eight files
#     derived   root radius, which follows from the cells rather than being set
#
# SIZING THE STELE
# The stele is fixed across all five, but its value is not simply the dataset
# median. That median (792 um radius) belongs to roots carrying about thirteen
# cell files and a 655 um cortex; dropping to eight files shrinks the cortex
# without shrinking the stele, and the stele ends up swallowing the root -- a
# stele/root radius ratio near 0.79 against the 0.546 that 1,688 real roots
# actually show (0.546-0.554 across all three species measured).
#
# So the stele is instead sized to land the reference pattern on that measured
# ratio. In scaleMode 'true' the summed file thickness depends only on the cell
# areas, never on the stele, so this resolves in two passes: build once to learn
# the thickness, solve
#
#     r_stele = ratio / (1 - ratio) * sum(thickness)
#
# then rebuild every pattern against that one stele. The other four patterns
# then sit slightly either side of the target ratio, which is correct -- they
# are meant to differ in size.
#
# That last point is the reason these are worth printing. Each file's radial
# thickness is the diameter of a cell of that area, so a root packed with large
# cells is genuinely wider than one packed with small cells. Pinning the root
# radius instead would rescale every pattern back onto the same disc and throw
# away the difference, so scaleMode is 'true' (radius = stele + sum of file
# thicknesses) rather than the fitted mode.
#
# THE FIVE PATTERNS
#     median      the dataset-median shape -- small at the stele, peak about a
#                 third out, decaying toward the epidermis
#     all_large   every file at the maximum area
#     all_small   every file at the minimum area
#     linear_out  smallest at the stele, growing linearly to the epidermis
#     linear_in   largest at the stele, shrinking linearly to the epidermis
#
# Each is expressed in the model's own six shape features rather than by
# patching the curve afterwards, so the geometry comes out of exactly the code
# the web page runs:
#
#     constant c    peak_height=c, rise_slope=0, decay_slope=0, outer_rise=0
#     ramp up       peak at x=1, rise_slope inverted to put the start at 0
#     ramp down     peak at x=0, decay_slope inverted to put the end at 0
#
# WHY NODE
# buildCurve() and computeFiles() are lifted out of root_model/template.html and
# executed as-is. Re-implementing them in Python would leave two definitions of
# the same geometry free to drift apart; running the shipped source means the
# printed solid is the one the page draws.
#
# A SHARED PRINT SCALE
# All five are written at one micron-per-mm scale, chosen so the widest of them
# lands on --largest-mm. Scaling each to a fixed diameter instead would undo the
# whole point: the size difference between the patterns is a result, not an
# inconvenience.
#
# A SHARED HEIGHT
# Height is deliberately NOT left at the page's default of three mean cell
# diameters. That default is pattern-dependent -- a root of large cells would
# come out half again as tall as one of small cells -- and disc height drives
# bending stiffness on its own, so it would sit in the results as a confound
# that has nothing to do with the pattern. Every disc is extruded to the same
# height instead, taken from the median pattern unless --height-um says
# otherwise.
#
# Settings live in the cfg block in main(); there are no command-line flags.
#     python -m code.root_model.build_pattern_stls

import json
import re
import struct
import subprocess
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from code.root_model import root_to_fea_stl

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

TEMPLATE = SCRIPT_DIR / 'template.html'
MODEL_HTML = PROJECT_ROOT / 'results' / 'root_model' / 'root_model.html'
OUT_DIR = PROJECT_ROOT / 'results' / 'pattern_stls'

N_FILES = 8            # held fixed across all five
EPS = 0.001            # the guard term buildCurve() uses inside its inversions

# Median stele radius / root radius over the 1,688 roots that carry both
# measurements. Consistent across species (maize 0.546, sorghum 0.554, pearl
# millet 0.549), so it is a property of root anatomy rather than of one crop.
STELE_RATIO = 0.546

# Which pattern the stele is sized against; the rest inherit that same stele.
REFERENCE_PATTERN = 'median'

# Peak at the far edge with the start driven to zero, and its mirror. The
# +EPS matches the guard inside buildCurve so the inversion lands on 0 exactly.
RAMP_SLOPE = 1.0 / (1.0 + EPS)


def pattern_settings(median):
    # The five feature vectors, all sharing the median's absolute size range.
    base = dict(
        n_files=N_FILES,
        stele_diameter_um=median['stele_diameter_um'],
        root_radius_um=median['root_radius_um'],   # unused: scaleMode is 'true'
        min_file_area_um2=median['min_file_area_um2'],
        max_file_area_um2=median['max_file_area_um2'],
        scaleMode='true',
    )

    def flat(level):
        # every anchor collapses onto peak_height, so the curve is constant
        return dict(peak_height=level, peak_position=0.5, rise_slope=0.0,
                    decay_slope=0.0, minima_position=0.75,
                    outer_rise_magnitude=0.0)

    return {
        'median': dict(base, **{k: median[k] for k in (
            'peak_height', 'peak_position', 'rise_slope', 'decay_slope',
            'minima_position', 'outer_rise_magnitude')}),

        'all_large': dict(base, **flat(1.0)),
        'all_small': dict(base, **flat(0.0)),

        # peak sits at the epidermis; rise_slope inverts back to a start of 0
        'linear_out': dict(base, peak_height=1.0, peak_position=1.0,
                           rise_slope=RAMP_SLOPE, decay_slope=0.0,
                           minima_position=1.0, outer_rise_magnitude=0.0),

        # peak sits at the stele; decay_slope inverts back to an edge of 0
        'linear_in': dict(base, peak_height=1.0, peak_position=0.0,
                          rise_slope=0.0, decay_slope=-RAMP_SLOPE,
                          minima_position=1.0, outer_rise_magnitude=0.0),
    }


def extract_js(template=TEMPLATE):
    # Pull makeInterp / buildCurve / computeFiles out of the page verbatim.
    src = Path(template).read_text(encoding='utf-8')
    out = []
    for name in ('makeInterp', 'buildCurve', 'computeFiles'):
        m = re.search(r'^function %s\s*\(' % name, src, re.M)
        if not m:
            raise SystemExit(f'could not find function {name}() in {template}')
        i = src.index('{', m.end() - 1)
        depth, j = 0, i
        while j < len(src):                     # brace-match to the function end
            if src[j] == '{':
                depth += 1
            elif src[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(src[m.start():j + 1])
    return '\n\n'.join(out)


NODE_DRIVER = r'''
%(functions)s

const settings = JSON.parse(process.argv[2]);
const curve = buildCurve(settings);
const model = computeFiles(settings, curve);

// same ring layout exportGeometryJson() writes
const cells = [];
for (const f of model.files) {
  const rMid = (f.rInner + f.rOuter) / 2;
  const step = (2 * Math.PI) / f.cells;
  for (let j = 0; j < f.cells; j++) {
    const a = (j + 0.5) * step;
    cells.push({ x: +(rMid * Math.cos(a)).toFixed(4),
                 y: +(rMid * Math.sin(a)).toFixed(4),
                 r: +f.cellR.toFixed(4), file: f.i });
  }
}
let sum = 0, n = 0;
for (const f of model.files) { sum += f.cells * 2 * f.cellR; n += f.cells; }
const meanCellDiameterUm = n > 0 ? sum / n : 0;

process.stdout.write(JSON.stringify({
  preset: settings.__label,
  scaleMode: settings.scaleMode,
  settings,
  units: 'um',
  rSteleUm: model.rStele,
  rEdgeUm: model.rEdge,
  heightUm: 3 * meanCellDiameterUm,
  meanCellDiameterUm,
  nFiles: model.n,
  totalCells: model.totalCells,
  cortexAreaUm2: model.totalCortexArea,
  lumenAreaUm2: model.totalCellArea,
  clampedBy: curve.clampedBy,
  y0Floored: curve.y0Floored,
  minimaFloored: curve.minimaFloored,
  files: model.files.map(f => ({ i: f.i, rInner: f.rInner, rOuter: f.rOuter,
                                 thickness: f.thickness, cellR: f.cellR,
                                 cells: f.cells, areaUm2: f.area })),
  cells,
}));
'''


def run_node(js, settings, label, workdir):
    driver = Path(workdir) / '_driver.js'
    driver.write_text(NODE_DRIVER % {'functions': js}, encoding='utf-8')
    payload = dict(settings, __label=label)
    res = subprocess.run(['node', str(driver), json.dumps(payload)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f'node failed for {label}:\n{res.stderr}')
    return json.loads(res.stdout)


def stl_volume_cm3(path):
    # Signed volume of a binary STL, by the divergence theorem.
    #
    # Read back from the written file rather than carried over from the builder,
    # so the number in the spreadsheet describes the mesh that actually shipped.
    with open(path, 'rb') as f:
        f.read(80)
        n = struct.unpack('<I', f.read(4))[0]
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    v = raw[:, 12:48].copy().view('<f4').reshape(n, 3, 3)
    a, b, c = v[:, 0], v[:, 1], v[:, 2]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0 / 1000.0)


# A minimal xlsx: one sheet, inline strings, no shared-string table. openpyxl
# is not installed in any interpreter on this machine, and a spreadsheet that
# opens on a double-click is worth more than a dependency.
_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
    '</Types>')
_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
    '</Relationships>')
_WORKBOOK = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    '<sheets><sheet name="patterns" sheetId="1" r:id="rId1"/></sheets></workbook>')
_WB_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
    '</Relationships>')


def _col(i):
    # 0 -> A, 25 -> Z, 26 -> AA.
    s = ''
    while True:
        s = chr(ord('A') + i % 26) + s
        i = i // 26 - 1
        if i < 0:
            return s


def write_xlsx(path, header, rows, widths=None):
    cells = []
    for ri, row in enumerate([header] + rows, start=1):
        parts = []
        for ci, val in enumerate(row):
            ref = f'{_col(ci)}{ri}'
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                parts.append(f'<c r="{ref}"><v>{val}</v></c>')
            else:
                parts.append(f'<c r="{ref}" t="inlineStr"><is><t>'
                             f'{escape(str(val))}</t></is></c>')
        cells.append(f'<row r="{ri}">' + ''.join(parts) + '</row>')

    cols = ''
    if widths:
        cols = '<cols>' + ''.join(
            f'<col min="{i+1}" max="{i+1}" width="{w}" customWidth="1"/>'
            for i, w in enumerate(widths)) + '</cols>'

    sheet = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             + cols + '<sheetData>' + ''.join(cells) + '</sheetData></worksheet>')

    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', _CONTENT_TYPES)
        z.writestr('_rels/.rels', _RELS)
        z.writestr('xl/workbook.xml', _WORKBOOK)
        z.writestr('xl/_rels/workbook.xml.rels', _WB_RELS)
        z.writestr('xl/worksheets/sheet1.xml', sheet)


def write_summary(out_dir, geoms, mm_per_um, height_um, stele_diameter):
    header = ['pattern', 'diameter (mm)', 'height (mm)', 'stele diameter (mm)',
              'volume (cm3)', 'file areas (um2)', 'cells', 'stele/root']
    rows = []
    for label, g in geoms.items():
        stl = Path(out_dir) / f'root_{label}.stl'
        rows.append([
            label,
            round(2 * g['rEdgeUm'] * mm_per_um, 2),
            round(height_um * mm_per_um, 2),
            round(stele_diameter * mm_per_um, 2),
            round(stl_volume_cm3(stl), 1) if stl.exists() else '',
            ' '.join(f'{f["areaUm2"]:.0f}' for f in g['files']),
            g['totalCells'],
            round(g['rSteleUm'] / g['rEdgeUm'], 3),
        ])

    xlsx = Path(out_dir) / 'pattern_summary.xlsx'
    write_xlsx(xlsx, header, rows, widths=[13, 14, 12, 19, 13, 46, 8, 11])

    csv = Path(out_dir) / 'pattern_summary.csv'
    with open(csv, 'w', encoding='utf-8', newline='') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(f'"{v}"' if isinstance(v, str) and ' ' in str(v)
                             else str(v) for v in r) + '\n')
    return xlsx, csv


def median_settings(model_html=MODEL_HTML):
    src = Path(model_html).read_text(encoding='utf-8')
    data = json.loads(re.search(r'const DATA\s*=\s*(\{.*?\});\s*\n', src, re.S).group(1))
    for p in data['presets']:
        if p.get('group') == 'Summary':
            return p['values']
    raise SystemExit('no Summary/dataset-median preset found')


def main():
    # python -m code.root_model.build_pattern_stls
    class cfg:
        out_dir = str(OUT_DIR)
        # printed diameter of the WIDEST pattern; the rest keep their true
        # relative size at that same scale
        largest_mm = 50.0
        # target solid fraction, passed through to root_to_fea_stl
        wall_fraction = 0.15
        # common disc height for all five (default: whatever the median
        # pattern works out to, applied to every pattern)
        height_um = None
        stele_ratio = STELE_RATIO
        # keep the intermediate geometry JSON files
        keep_json = False


    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    median = median_settings()
    js = extract_js()
    patterns = pattern_settings(median)

    # Pass 1: the reference pattern's summed file thickness. Under scaleMode
    # 'true' this depends only on the cell areas, so the stele it is about to
    # determine cannot feed back into it.
    probe = run_node(js, patterns[REFERENCE_PATTERN], REFERENCE_PATTERN, out_dir)
    cortex = probe['rEdgeUm'] - probe['rSteleUm']
    ratio = cfg.stele_ratio
    r_stele = ratio / (1 - ratio) * cortex
    stele_diameter = 2 * r_stele

    print(f'stele sizing   : {REFERENCE_PATTERN} cortex is {cortex:.1f} um; '
          f'targeting stele/root = {ratio:.3f}')
    print(f'                 -> stele radius {r_stele:.1f} um '
          f'(was {median["stele_diameter_um"] / 2:.1f} um, the raw dataset median)\n')

    for s in patterns.values():
        s['stele_diameter_um'] = stele_diameter

    # Pass 2: every geometry against the settled stele. The shared print scale
    # cannot be chosen until the widest of the five is known.
    geoms = {}
    for label, s in patterns.items():
        geoms[label] = run_node(js, s, label, out_dir)

    widest = max(g['rEdgeUm'] for g in geoms.values())
    mm_per_um = cfg.largest_mm / (2 * widest)
    height_um = cfg.height_um if cfg.height_um is not None \
        else geoms['median']['heightUm']

    print(f'stele diameter : {stele_diameter:.1f} um   (fixed)')
    print(f'cell files     : {N_FILES}                (fixed)')
    print(f'file area range: {median["min_file_area_um2"]:.0f} - '
          f'{median["max_file_area_um2"]:.0f} um2  (fixed)')
    print(f'disc height    : {height_um:.1f} um -> {height_um * mm_per_um:.2f} mm  '
          f'(fixed, so it cannot confound the comparison)')
    print(f'print scale    : 1 um -> {mm_per_um:.6f} mm  '
          f'(widest pattern = {cfg.largest_mm:.1f} mm)\n')

    print(f'{"pattern":12} {"cells":>6} {"root r um":>10} {"cortex um":>10} '
          f'{"stele/root":>11} {"print mm":>9}  file areas (um2)')
    for label, g in geoms.items():
        areas = ' '.join(f'{f["areaUm2"]:.0f}' for f in g['files'])
        print(f'{label:12} {g["totalCells"]:6} {g["rEdgeUm"]:10.1f} '
              f'{g["rEdgeUm"] - g["rSteleUm"]:10.1f} '
              f'{g["rSteleUm"] / g["rEdgeUm"]:11.3f} '
              f'{2 * g["rEdgeUm"] * mm_per_um:9.1f}  {areas}')
        if g['clampedBy'] > 1e-9 or g['y0Floored'] or g['minimaFloored']:
            print(f'{"":12} WARNING: curve was clamped for {label}')

    print()
    for label, g in geoms.items():
        g['mmPerUm'] = mm_per_um
        jpath = out_dir / f'root_geometry_{label}.json'
        jpath.write_text(json.dumps(g), encoding='utf-8')

        stl = out_dir / f'root_{label}.stl'
        print(f'--- {label} ---')
        # Called directly rather than as a subprocess: with argument parsing
        # gone there are no flags to pass, and a plain call surfaces a failure
        # as a traceback instead of hiding it behind a captured exit code.
        root_to_fea_stl.main(
            geometry_json=jpath,
            out=stl,
            pattern='real',
            wall_fraction=cfg.wall_fraction,
            height_um=height_um,
            units='mm',
        )

        if not cfg.keep_json:
            jpath.unlink()

    xlsx, csv = write_summary(out_dir, geoms, mm_per_um, height_um, stele_diameter)
    print(f'\nwrote {xlsx.name} and {csv.name}')
    print(f'STLs in {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
