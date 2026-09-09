# Turns a geometry export into a watertight wall-solid STL.
#
# The printed material is the cell wall network, not the cells: the solid is
# the cortex disc minus the cell lumens minus the stele vessels. Wall
# thickness is solved to hit a target solid fraction so different patterns
# can be compared at equal material. Writes one .stl next to its input.

import json
import math
from pathlib import Path

import numpy as np
import shapely
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CIRCLE_SEGMENTS = 256      # resolution of the cortex outer boundary
MITRE = 2                  # shapely join_style=2 (mitre) keeps convex corners sharp

# Stele vessel ring, kept identical to root_model.html's buildStele() so the
# printed and analysed solids describe the same anatomy. Fractions of the stele
# radius; the same stele for every root.
STELE_VESSEL_COUNT = 12
STELE_VESSEL_RING = 0.78   # vessel centres
STELE_VESSEL_R = 0.13      # vessel radius


def stele_vessel_lumens(r_stele, count=STELE_VESSEL_COUNT):
    # Xylem vessels as voids.
    #
    # Only the vessels are voided, not the surrounding parenchyma: a vessel is an
    # open conduit and genuinely hollow, whereas the parenchyma are living cells
    # whose walls and contents both carry load at this scale. Voiding several
    # hundred tiny parenchyma would also drive the mesh count up sharply for very
    # little change in stiffness.
    #
    # The vessel circle is the lumen itself -- the wall is the material left
    # between neighbouring vessels -- so it is not shrunk the way the cortex
    # lumens are.
    if r_stele <= 0 or count <= 0:
        return []
    ring = STELE_VESSEL_RING * r_stele
    rad = STELE_VESSEL_R * r_stele
    out = []
    for i in range(count):
        a = (i + 0.5) * (2 * math.pi / count)
        out.append(Point(ring * math.cos(a), ring * math.sin(a)).buffer(rad, quad_segs=16))
    return out


# geometry

def load_geometry(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    cells = data['cells']
    xy = np.array([[c['x'], c['y']] for c in cells], dtype=float)
    r = np.array([c['r'] for c in cells], dtype=float)
    return data, xy, r


def apply_pattern(data, xy, r, pattern, seed=0):
    # Return (xy, r) for the requested control, area-matched to the real one.
    if pattern == 'real':
        return xy, r

    rng = np.random.default_rng(seed)
    target_area = float(np.sum(np.pi * r ** 2))

    if pattern == 'uniform':
        # Cell size here is set by the SPACING of the centres (the lumens come
        # from a Voronoi of the centres, not from the recorded radii), so an
        # equal-size control has to re-lay the centres on constant spacing --
        # keeping the original positions and only equalising radii would
        # reproduce the real geometry exactly and test nothing.
        r_stele = float(data['rSteleUm'])
        r_edge = float(data['rEdgeUm'])
        cortex_area = math.pi * (r_edge ** 2 - r_stele ** 2)
        n_target = len(r)

        def lay_out(spacing):
            n_rings = max(1, int(round((r_edge - r_stele) / spacing)))
            dr = (r_edge - r_stele) / n_rings
            pts = []
            for j in range(n_rings):
                rm = r_stele + (j + 0.5) * dr
                count = max(3, int(round(2 * math.pi * rm / spacing)))
                for q in range(count):
                    a = (q + 0.5) * (2 * math.pi / count)
                    pts.append((rm * math.cos(a), rm * math.sin(a)))
            return np.array(pts)

        # bisect the spacing so the control carries the same number of cells
        lo, hi = math.sqrt(cortex_area / n_target) * 0.4, math.sqrt(cortex_area / n_target) * 2.5
        pts = lay_out((lo + hi) / 2)
        for _ in range(40):
            mid = (lo + hi) / 2
            pts = lay_out(mid)
            if len(pts) == n_target:
                break
            if len(pts) > n_target:
                lo = mid
            else:
                hi = mid
        r_uniform = math.sqrt(target_area / (np.pi * max(len(pts), 1)))
        return pts, np.full(len(pts), r_uniform)

    if pattern == 'random':
        # Same radii, positions re-drawn uniformly by AREA across the cortex
        # annulus, rejecting overlaps so lumens stay disjoint.
        r_stele = float(data['rSteleUm'])
        r_edge = float(data['rEdgeUm'])
        order = np.argsort(-r)               # place largest first: easier packing
        placed = np.zeros((len(r), 2))
        radii = r[order]
        for i, ri in enumerate(radii):
            lo, hi = r_stele + ri, r_edge - ri
            if hi <= lo:
                lo, hi = r_stele, max(r_stele + 1e-6, r_edge)
            for _ in range(400):
                # sqrt keeps the draw uniform per unit AREA, not per unit radius
                rad = math.sqrt(rng.uniform(lo ** 2, hi ** 2))
                ang = rng.uniform(0, 2 * math.pi)
                p = np.array([rad * math.cos(ang), rad * math.sin(ang)])
                if i == 0:
                    placed[i] = p
                    break
                d = np.hypot(*(placed[:i] - p).T)
                if np.all(d > radii[:i] + ri):
                    placed[i] = p
                    break
            else:
                placed[i] = p                 # give up on this one; overlap is rare
        out_xy = np.zeros_like(placed)
        out_r = np.zeros_like(radii)
        out_xy[order] = placed
        out_r[order] = radii
        return out_xy, out_r

    raise SystemExit(f'unknown pattern: {pattern}')


def voronoi_cells(xy, boundary):
    # Voronoi region per input point, clipped to `boundary`.
    #
    # Guard points placed far outside make every real region finite, so no
    # region has to be reconstructed from infinite ridges.
    minx, miny, maxx, maxy = boundary.bounds
    span = max(maxx - minx, maxy - miny)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    guards = np.array([[cx + 3 * span * math.cos(a), cy + 3 * span * math.sin(a)]
                       for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)])
    vor = Voronoi(np.vstack([xy, guards]))

    prepared = prep(boundary)
    cells = []
    for i in range(len(xy)):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            cells.append(None)
            continue
        poly = Polygon(vor.vertices[region])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not prepared.intersects(poly):
            cells.append(None)
            continue
        clipped = poly.intersection(boundary)
        if clipped.is_empty:
            cells.append(None)
            continue
        if clipped.geom_type == 'MultiPolygon':
            clipped = max(clipped.geoms, key=lambda p: p.area)
        cells.append(clipped if clipped.geom_type == 'Polygon' else None)
    return cells


def lumens_for_wall(cells, wall_um):
    # Shrink each Voronoi cell inward by half the wall thickness.
    #
    # Half from each side of a shared edge sums to a full wall between
    # neighbours, which is what makes the walls come out at `wall_um`.
    valid = [c for c in cells if c is not None]
    if not valid:
        return []
    shrunk = shapely.buffer(np.array(valid, dtype=object), -wall_um / 2.0, join_style=MITRE)
    out = []
    for g in shrunk:
        if g.is_empty:
            continue
        if g.geom_type == 'MultiPolygon':
            g = max(g.geoms, key=lambda p: p.area)
        if g.area > 0:
            out.append(g)
    return out


def solve_wall_for_fraction(cells, cortex_area, target_solid_fraction, tol=1e-3, iters=40):
    # Bisect wall thickness until solid/cortex hits the target relative density.
    valid = [c for c in cells if c is not None]
    widths = np.array([2 * math.sqrt(c.area / math.pi) for c in valid])
    lo, hi = 0.0, float(widths.max())          # hi closes every cell completely
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        lum = lumens_for_wall(cells, mid)
        lumen_area = float(sum(p.area for p in lum))
        frac = 1.0 - lumen_area / cortex_area
        best = (mid, frac)
        if abs(frac - target_solid_fraction) < tol:
            break
        if frac < target_solid_fraction:
            lo = mid
        else:
            hi = mid
    return best


# meshing

def ring_coords(ring):
    # A ring's coordinates, open (no repeated closing point).
    c = np.asarray(ring.coords)
    if np.allclose(c[0], c[-1]):
        c = c[:-1]
    return c


def constrained_triangles(region, grids=(0, 5e-7, 2e-6, 5e-6, 2e-5, 1e-4)):
    # Constrained Delaunay, with a precision-snap fallback.
    #
    # GEOS raises "Unable to find a convex corner" when a ring carries an edge
    # orders of magnitude shorter than the geometry containing it. The mitre
    # joins used to shrink the Voronoi cells can leave edges around 1e-2 um
    # inside a 600 um disc, which is enough to trip it -- and it only shows up
    # for some geometries, so the failure looks arbitrary until you measure the
    # edge lengths.
    #
    # Snapping coordinates onto a fine grid removes those slivers. The grids are
    # fractions of the geometry's own extent and stay far below any wall
    # thickness, so the solid does not move measurably; the first entry is 0,
    # meaning the untouched geometry is always tried first and nothing changes
    # for the cases that already worked.
    #
    # simplify() does NOT fix this -- it drops collinear vertices but leaves the
    # offending short edges in place, and GEOS still throws.
    x0, y0, x1, y1 = region.bounds
    extent = max(x1 - x0, y1 - y0)
    last = None
    for g in grids:
        candidate = region if g == 0 else shapely.set_precision(region, g * extent)
        if candidate.is_empty or candidate.geom_type != 'Polygon':
            continue
        # Orient AFTER snapping, never before: set_precision is free to reverse
        # a ring, and a hole that comes back CCW gives its wall the outward
        # winding of the exterior, which unpairs every edge along it.
        candidate = shapely.geometry.polygon.orient(candidate, 1.0)
        try:
            tris = shapely.constrained_delaunay_triangles(candidate)
            if g:
                print(f'note            : snapped to a {g * extent:.4g} um grid so the '
                      f'triangulator could run (area change '
                      f'{abs(candidate.area - region.area) / region.area * 100:.4f}%)')
            return candidate, tris
        except shapely.errors.GEOSException as exc:
            last = exc
    raise SystemExit(f'constrained Delaunay failed at every precision: {last}')


def build_mesh(region, height):
    # Extrude a polygon-with-holes to `height` as a watertight triangle soup.
    #
    # The caps come from a CONSTRAINED Delaunay triangulation, which is the part
    # that matters: an ordinary Delaunay over the boundary vertices is free to
    # cut across the boundary, so the cap edges stop matching the ring edges the
    # side walls are built from and the extrusion ends up full of cracks. GEOS's
    # constrained version keeps every boundary edge, so caps and walls share
    # exactly the same edges and the result seals.
    #
    # orient() normalises the exterior ring to CCW and holes to CW, which makes
    # one wall-winding rule correct for both: outward on the outside, into the
    # void on the inside.
    region = shapely.geometry.polygon.orient(region, 1.0)
    # the snapped region is returned too: the caps and the side walls must be
    # built from the same coordinates or the extrusion will not seal
    region, tri_geoms = constrained_triangles(region)

    verts, index = [], {}

    def vid(p):
        key = (round(p[0], 6), round(p[1], 6))
        got = index.get(key)
        if got is None:
            got = index[key] = len(verts)
            verts.append((p[0], p[1]))
        return got

    faces = []
    for g in tri_geoms.geoms:
        c = np.asarray(g.exterior.coords)[:3]
        # keep every cap triangle CCW so the +z/-z cap windings below hold
        if (c[1][0] - c[0][0]) * (c[2][1] - c[0][1]) - (c[1][1] - c[0][1]) * (c[2][0] - c[0][0]) < 0:
            c = c[::-1]
        faces.append([vid(c[0]), vid(c[1]), vid(c[2])])

    rings = [ring_coords(region.exterior)] + [ring_coords(r) for r in region.interiors]
    ring_ids = [[vid(p) for p in ring] for ring in rings]

    pts2d = np.asarray(verts, dtype=float)
    n = len(pts2d)
    bottom = np.hstack([pts2d, np.zeros((n, 1))])
    top = np.hstack([pts2d, np.full((n, 1), height)])

    tris = []
    for a, b, c in faces:
        tris.append([bottom[a], bottom[c], bottom[b]])   # -z
        tris.append([top[a], top[b], top[c]])            # +z
    for ids in ring_ids:
        m = len(ids)
        for i in range(m):
            a, b = ids[i], ids[(i + 1) % m]
            tris.append([bottom[a], bottom[b], top[b]])
            tris.append([bottom[a], top[b], top[a]])
    return np.array(tris, dtype=np.float64)


def write_binary_stl(path, tris, scale=1.0):
    t = tris * scale
    n = len(t)
    normals = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    normals /= lengths[:, None]

    rec = np.zeros(n, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    rec['n'] = normals
    rec['v'] = t
    with open(path, 'wb') as fh:
        fh.write(b'root FEA wall solid'.ljust(80, b'\0'))
        fh.write(np.uint32(n).tobytes())
        fh.write(rec.tobytes())
    return 84 + n * 50


def mesh_report(tris):
    # Watertight check: every directed edge must have exactly one opposite twin.
    q = np.round(tris, 6)
    edges = {}
    for tri in q:
        for i in range(3):
            a = tuple(tri[i]); b = tuple(tri[(i + 1) % 3])
            edges[(a, b)] = edges.get((a, b), 0) + 1
    unpaired = sum(1 for (a, b) in edges if (b, a) not in edges)
    doubled = sum(1 for v in edges.values() if v > 1)
    return dict(triangles=len(tris), unpaired_edges=unpaired,
                duplicated_edges=doubled, watertight=(unpaired == 0 and doubled == 0))


def _default_geometry():
    # Whichever geometry has most recently been exported from the model page.
    # Returning None rather than raising keeps import cheap; main() reports the
    # missing input properly when it is actually run.
    exports = PROJECT_ROOT / 'results' / 'root_model' / 'exports'
    if not exports.exists():
        return None
    found = sorted(exports.glob('root_geometry_*.json'))
    return found[-1] if found else None


def main(**overrides):
    # python -m code.root_model.root_to_fea_stl
    #
    # Accepts keyword overrides so another script can drive it as a function
    # call. build_pattern_stls used to invoke this file as a subprocess with
    # command-line flags; with argument parsing gone that contract no longer
    # exists, and a direct call is cheaper and fails loudly instead of through
    # a captured exit code.
    class cfg:
        # from root_model.html 'Export FEA JSON'. Defaults to whatever geometry
        # has been exported into results/root_model/exports/.
        geometry_json = _default_geometry()
        # output .stl (default: alongside the json)
        out = None
        pattern = 'real'
        # measured wall thickness in um (overrides --wall-fraction)
        wall_um = None
        # target solid fraction of the cortex (default 0.15)
        wall_fraction = 0.15
        # override extrusion height (default: 3x mean cell diameter)
        height_um = None
        # override printed diameter in mm (default: as exported)
        target_mm = None
        # RNG seed for --pattern random
        seed = 0
        # keep the stele solid instead of voiding its 12 vessels
        solid_stele = False
        # STL units: 'mm' scales for printing, 'um' keeps true size
        units = 'mm'

    for _k, _v in overrides.items():
        setattr(cfg, _k, _v)

    if cfg.geometry_json is None:
        raise SystemExit(
            'No geometry JSON given and none found in results/root_model/exports/. '
            "Export one from the model page, or set cfg.geometry_json.")


    data, xy, r = load_geometry(cfg.geometry_json)
    xy, r = apply_pattern(data, xy, r, cfg.pattern, cfg.seed)

    r_stele = float(data['rSteleUm'])
    r_edge = float(data['rEdgeUm'])
    height = cfg.height_um if cfg.height_um is not None else float(data['heightUm'])

    print(f"preset          : {data.get('preset')}")
    print(f"pattern         : {cfg.pattern}")
    print(f"cells           : {len(xy)}")
    print(f"root radius     : {r_edge:.1f} um   stele radius: {r_stele:.1f} um")
    print(f"height          : {height:.1f} um  (3x mean cell diameter unless overridden)")

    # Two different domains, and the distinction matters:
    #   disc   -- everything that ends up as solid material (stele included;
    #             it is vascular tissue, not lumen, so it stays filled)
    #   cortex -- the annulus the lumens are allowed to occupy
    # Clipping the Voronoi to the disc instead of the annulus lets the
    # innermost cells swallow the whole stele, which both hollows out tissue
    # that should be solid and wildly inflates those cells' size.
    disc = Point(0, 0).buffer(r_edge, quad_segs=CIRCLE_SEGMENTS // 4)
    stele = Point(0, 0).buffer(r_stele, quad_segs=CIRCLE_SEGMENTS // 4) if r_stele > 0 else None
    cortex = disc.difference(stele) if stele is not None else disc
    cortex_area = math.pi * (r_edge ** 2 - r_stele ** 2)

    cells = voronoi_cells(xy, cortex)
    n_ok = sum(1 for c in cells if c is not None)
    print(f"voronoi cells   : {n_ok} usable of {len(cells)}")

    if cfg.wall_um is not None:
        wall = cfg.wall_um
        lumens = lumens_for_wall(cells, wall)
        frac = 1.0 - float(sum(p.area for p in lumens)) / cortex_area
    else:
        wall, frac = solve_wall_for_fraction(cells, cortex_area, cfg.wall_fraction)
        lumens = lumens_for_wall(cells, wall)
    print(f"wall thickness  : {wall:.2f} um  -> solid fraction {frac:.3f}")

    if not lumens:
        raise SystemExit('wall thickness closed every cell -- lower it')

    # Spread of lumen sizes: this is what 'uniform' is meant to flatten, so it
    # doubles as a check that the control really is a different geometry.
    areas = np.array([p.area for p in lumens])
    print(f"lumen size      : mean {areas.mean():.0f} um2, CV {areas.std() / areas.mean():.3f}")

    # The stele is identical across real/uniform/random, so it cannot confound
    # a comparison between them -- only the cortex pattern differs.
    vessels = [] if cfg.solid_stele else stele_vessel_lumens(r_stele)
    if vessels:
        print(f"stele vessels   : {len(vessels)} voided "
              f"(r={STELE_VESSEL_R * r_stele:.1f} um at {STELE_VESSEL_RING * r_stele:.1f} um)")
    else:
        print("stele vessels   : none (solid stele)")

    region = disc.difference(unary_union(lumens + vessels))
    if region.geom_type == 'MultiPolygon':
        # Shouldn't happen (lumens never disconnect the disc), but if a wall
        # setting ever did sever it, print the largest piece rather than a
        # silently truncated model.
        parts = sorted(region.geoms, key=lambda p: p.area, reverse=True)
        print(f'WARNING: solid split into {len(parts)} pieces; keeping the largest')
        region = parts[0]
    print(f"lumens meshed   : {len(region.interiors)}")

    tris = build_mesh(region, height)
    report = mesh_report(tris)
    print(f"mesh            : {report['triangles']} triangles, "
          f"watertight={report['watertight']} "
          f"(unpaired={report['unpaired_edges']}, duplicated={report['duplicated_edges']})")

    if cfg.units == 'mm':
        mm_per_um = data['mmPerUm']
        if cfg.target_mm is not None:
            mm_per_um = cfg.target_mm / (2 * r_edge)
        scale = mm_per_um
        print(f"scale           : 1 um -> {scale:.5f} mm "
              f"(disc {2 * r_edge * scale:.1f} mm, height {height * scale:.2f} mm, "
              f"wall {wall * scale:.3f} mm)")
    else:
        scale = 1.0
        print('scale           : none (STL in um)')

    out = Path(cfg.out) if cfg.out else \
        Path(cfg.geometry_json).with_name(
            Path(cfg.geometry_json).stem.replace('root_geometry_', 'root_fea_')
            + f'_{cfg.pattern}_wall{wall:.0f}um.stl')
    size = write_binary_stl(out, tris, scale)
    print(f"wrote           : {out}  ({size / 1048576:.1f} MB)")

    if cfg.units == 'mm':
        print(f"solid volume    : {region.area * height * scale ** 3 / 1000:.2f} cm3")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
