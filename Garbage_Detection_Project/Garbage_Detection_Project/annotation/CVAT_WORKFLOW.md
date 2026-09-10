# Manual CVAT workflow

Create three CVAT tasks from `dataset_yolo_annotation_ready/images/train`, `val`, and `test`. Create rectangle labels in this exact ID order: battery (0), biological (1), cardboard (2), clothes (3), glass (4), metal (5), paper (6), plastic (7), shoes (8), trash (9).

Never create a `clean` label. A clean image has no boxes only after review confirms that none of the ten target objects are present. Label every visible target object in a garbage image, including multiple objects/classes. Do not label people, phones, furniture, vehicles, or other non-target COCO objects.

Export in **Ultralytics YOLO** format. Put exported `.txt` labels into the matching `labels/train`, `labels/val`, or `labels/test` directory without changing image basenames. Do not use the original `dataset/labels` directory.

Before training, run the annotation audit with `--require-boxes --verify-source-class`. Training is blocked until it reports zero errors.
