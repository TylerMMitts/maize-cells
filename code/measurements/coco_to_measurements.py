# Converts hand-annotated COCO polygons into the pipeline's measurement CSVs.
#
# Writes the identical per-quadrant format cell_measurements produces -- same
# columns, same plant-centre convention -- so hand-annotated species are
# analysed by exactly the same code rather than down a parallel path.
# produces, so a hand-annotated dataset can enter the pipeline at step 3
# (master summary) and run through the rest unchanged.
#
# This exists because the trained cell-segmentation model does not generalize to
# every species in the dataset (tomato in particular). Rather than accept bad
# automatic segmentations, cells are annotated by hand in Roboflow, exported as
# COCO, and converted here.
#
# Two deliberate differences from the model-driven path:
#
#   * No exclusion (inner-part) model is run. Its two jobs -- dropping cells that
#     shouldn't be measured, and supplying the image-level stele_area_um2 /
#     root_radius_um -- are handled differently here: the hand annotations
#     already exclude unwanted cells, and the stele/outer boundaries simply are
#     not annotated, so those two fields are written as 0. Downstream code
#     guards on `> 0` (build_feature_table.py, the statistics scripts), so rows
#     produced here are automatically excluded from stele-area and root-radius
#     analyses instead of polluting them with fabricated geometry.
#
#   * Cell area comes from rasterizing the annotation polygon, not from a
#     predicted mask -- matching calculate_area_from_mask's pixel-count
#     semantics so areas are directly comparable to model-derived rows.
#
# Everything else (plant-centre convention, radius/angle, um conversion, local
# density, neighbour graph, the _centers.jpg overlay) reuses the exact same
# helpers the model path uses.

import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from code.measurements.cell_measurements import add_local_densities_to_measurements
from code.measurements.mask_neighbors import get_neighbors_from_masks
from code.util.visualize_cell_centers import get_plant_center_from_filename

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Roboflow emits a placeholder category 0 for the dataset itself; real
# instances carry the 'cell' category.
CELL_CATEGORY_NAME = 'cell'


def polygons_to_mask(segmentation, height, width):
    # Rasterize COCO polygon(s) for one instance into a full-frame uint8 mask.
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        if len(poly) < 6:  # need >= 3 points
            continue
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask


def mask_centroid(mask):
    # Centroid of a binary mask, or None if empty.
    m = cv2.moments(mask, binaryImage=True)
    if m['m00'] == 0:
        return None
    return (m['m10'] / m['m00'], m['m01'] / m['m00'])


def convert(coco_json, images_folder, output_dir, pixel_to_um,
            expansion_pixels=2, max_neighbors=8, overwrite=False):
    coco = json.loads(Path(coco_json).read_text(encoding='utf-8'))
    pixel_to_um_squared = pixel_to_um * pixel_to_um

    cell_cat_ids = {c['id'] for c in coco.get('categories', [])
                    if c.get('name') == CELL_CATEGORY_NAME}
    if not cell_cat_ids:
        raise SystemExit(f"No '{CELL_CATEGORY_NAME}' category in {coco_json}; "
                         f"found {[c.get('name') for c in coco.get('categories', [])]}")

    anns_by_image = defaultdict(list)
    for ann in coco['annotations']:
        if ann['category_id'] in cell_cat_ids:
            anns_by_image[ann['image_id']].append(ann)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_folder = Path(images_folder)

    written, skipped = [], []

    for img_info in coco['images']:
        # Roboflow mangles file_name with a content hash; extra.name preserves
        # the original, which is what the rest of the pipeline keys on.
        original = img_info.get('extra', {}).get('name') or img_info['file_name']
        base_name = os.path.splitext(original)[0]

        out_csv = output_dir / f'{base_name}_measurements.csv'
        out_img = output_dir / f'{base_name}_centers.jpg'
        if out_csv.exists() and not overwrite:
            skipped.append((base_name, 'already exists'))
            continue

        anns = anns_by_image.get(img_info['id'], [])
        if not anns:
            skipped.append((base_name, 'no cell annotations'))
            continue

        height, width = img_info['height'], img_info['width']

        # The source image is only needed for the overlay; fall back to a blank
        # frame so a missing image doesn't cost us the measurements.
        img_path = images_folder / original
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((height, width, 3), dtype=np.uint8)
        elif img.shape[:2] != (height, width):
            # Trust the actual raster over the annotation header.
            height, width = img.shape[:2]

        center_x, center_y, quadrant = get_plant_center_from_filename(base_name, width, height)
        if center_x is None:
            skipped.append((base_name, 'no BL/BR/TL/TR quadrant in name'))
            continue

        cell_masks, cell_centers = [], []
        for ann in anns:
            mask = polygons_to_mask(ann['segmentation'], height, width)
            centroid = mask_centroid(mask)
            if centroid is None:
                continue
            cell_masks.append(mask)
            cell_centers.append(centroid)

        if not cell_masks:
            skipped.append((base_name, 'all annotations rasterized empty'))
            continue

        measurements = []
        for i, (center, mask) in enumerate(zip(cell_centers, cell_masks)):
            area_pixels = float(np.count_nonzero(mask))
            dx = center[0] - float(center_x)
            dy = center[1] - float(center_y)
            radius_pixels = float(np.hypot(dx, dy))

            measurements.append({
                'cell_id': i + 1,
                'x_pixels': round(center[0], 2),
                'y_pixels': round(center[1], 2),
                'x_um': round(center[0] * pixel_to_um, 2),
                'y_um': round(center[1] * pixel_to_um, 2),
                'radius_pixels': round(radius_pixels, 2),
                'radius_um': round(radius_pixels * pixel_to_um, 2),
                'angle_degrees': round(float(np.degrees(np.arctan2(dy, dx))), 2),
                'area_pixels': round(area_pixels, 2),
                'area_um2': round(area_pixels * pixel_to_um_squared, 2),
                'quadrant': quadrant,
                # Not annotated -- see module docstring.
                'stele_area_um2': 0.0,
                'root_radius_um': 0.0,
            })

        measurements = add_local_densities_to_measurements(measurements, cell_centers, radius_scale=100)
        try:
            measurements = get_neighbors_from_masks(
                measurements, cell_masks, cell_centers,
                expansion_pixels=expansion_pixels, max_neighbors=max_neighbors, verbose=False)
        except Exception as e:
            print(f'  WARNING: neighbour detection failed for {base_name}: {e}')

        pd.DataFrame(measurements).to_csv(out_csv, index=False)

        overlay = img.copy()
        for center in cell_centers:
            cv2.circle(overlay, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)
        cv2.circle(overlay, (int(center_x), int(center_y)), 8, (255, 0, 0), -1)
        cv2.putText(overlay, f'Cells: {len(cell_centers)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(overlay, f'Quadrant: {quadrant}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imwrite(str(out_img), overlay)

        written.append((base_name, len(measurements)))
        print(f'  {base_name}: {len(measurements)} cells')

    print(f'\nWrote {len(written)} measurement files to {output_dir}')
    if written:
        counts = [n for _, n in written]
        print(f'  cells per image: min={min(counts)} median={int(np.median(counts))} '
              f'max={max(counts)} total={sum(counts)}')
    if skipped:
        print(f'Skipped {len(skipped)}:')
        for name, why in skipped[:20]:
            print(f'  {name}: {why}')
    return written, skipped


def main():
    # python -m code.measurements.coco_to_measurements
    class cfg:
        # COCO _annotations.coco.json
        coco_json = None
        # Folder holding the ORIGINAL (un-hashed) quadrant images
        images_folder = None
        output_dir = str(PROJECT_ROOT / 'results' / 'measurements_all')
        # Default: config.DEFAULT_PIXEL_TO_UM
        pixel_to_um = None
        overwrite = False


    pixel_to_um = cfg.pixel_to_um
    if pixel_to_um is None:
        from code import config
        pixel_to_um = config.DEFAULT_PIXEL_TO_UM

    print(f'COCO:      {cfg.coco_json}')
    print(f'Images:    {cfg.images_folder}')
    print(f'Output:    {cfg.output_dir}')
    print(f'Scale:     pixel_to_um={pixel_to_um} (stele/root radius left as 0)\n')

    convert(cfg.coco_json, cfg.images_folder, cfg.output_dir,
            pixel_to_um, overwrite=cfg.overwrite)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
