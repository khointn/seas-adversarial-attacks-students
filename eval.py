from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from utils import COCO_CLASSES, letterbox_image
from yolov3_model import YOLOv3Detector


def _coco_name_to_id(coco: COCO) -> dict[str, int]:
    """Map COCO class name -> official category_id (1..90 with gaps)."""
    return {cat["name"]: cat["id"] for cat in coco.loadCats(coco.getCatIds())}


def _detections_to_coco_results(
    detections: np.ndarray,
    image_id: int,
    letterbox_meta: tuple[int, int, int, int, float],
    orig_size: tuple[int, int],
    class_index_to_coco_id: dict[int, int],
) -> list[dict]:
    """
    Convert detector rows to COCO result dicts.

    Detector format per row:
      [class_idx, score, logits..., xmin, ymin, xmax, ymax]
    Boxes are absolute xyxy in letterboxed model space (e.g. 416x416).
    COCO expects xywh in original image pixels.
    """
    if detections.size == 0:
        return []

    left, top, _, _, scale = letterbox_meta
    orig_w, orig_h = orig_size
    results = []

    for det in detections:
        class_idx = int(det[0])
        score = float(det[1])
        xmin, ymin, xmax, ymax = map(float, det[-4:])

        # undo letterbox padding + scale
        xmin = (xmin - left) / scale
        ymin = (ymin - top) / scale
        xmax = (xmax - left) / scale
        ymax = (ymax - top) / scale

        xmin = float(np.clip(xmin, 0, orig_w))
        ymin = float(np.clip(ymin, 0, orig_h))
        xmax = float(np.clip(xmax, 0, orig_w))
        ymax = float(np.clip(ymax, 0, orig_h))

        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue

        results.append(
            {
                "image_id": int(image_id),
                "category_id": int(class_index_to_coco_id[class_idx]),
                "bbox": [xmin, ymin, w, h],
                "score": score,
            }
        )
    return results


def evaluate_coco_map(
    detector: YOLOv3Detector,
    coco_root: str | Path = "data/coco",
    split: str = "val2017",
    max_images: Optional[int] = None,
    confidence_threshold: float = 0.001,
    iou_threshold: float = 0.45,
) -> dict[str, float]:
    """
    Run the detector on original (benign) COCO images and compute official COCO mAP.

    Returns a dict with the usual COCOeval summary stats, e.g.:
      mAP, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l
    """
    coco_root = Path(coco_root)
    ann_path = coco_root / "annotations" / f"instances_{split}.json"
    image_dir = coco_root / split

    coco_gt = COCO(str(ann_path))
    name_to_coco_id = _coco_name_to_id(coco_gt)
    # detector class index 0..79 follows COCO_CLASSES order/names
    class_index_to_coco_id = {
        i: name_to_coco_id[name] for i, name in enumerate(COCO_CLASSES)
    }

    image_ids = sorted(coco_gt.getImgIds())
    if max_images is not None:
        image_ids = image_ids[:max_images]

    results: list[dict] = []
    for image_id in tqdm(image_ids, desc=f"COCO {split} benign mAP"):
        info = coco_gt.loadImgs(image_id)[0]
        image_path = image_dir / info["file_name"]
        pil_image = Image.open(image_path).convert("RGB")

        x_query, meta = letterbox_image(pil_image, size=detector.model_image_size)
        detections = detector.detect(
            x_query,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        results.extend(
            _detections_to_coco_results(
                detections,
                image_id=image_id,
                letterbox_meta=meta,
                orig_size=(info["width"], info["height"]),
                class_index_to_coco_id=class_index_to_coco_id,
            )
        )

    if not results:
        raise RuntimeError("No detections produced; cannot run COCOeval.")

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # coco_eval.stats:
    keys = ["mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l"]
    return {k: float(v) for k, v in zip(keys, coco_eval.stats[:6])}