# Entry point for the COCO annotation converter.
#
# Thin wrapper; the conversion itself lives in the convert_coco package.

from ultralytics.data.converter import convert_coco

# Converts coco annotations to YOLO format for segmentation tasks
convert_coco(
    labels_dir="convert_coco/annotations/",
    save_dir="convert_coco/converted/",
    use_segments=True,
    cls91to80=False,
)
