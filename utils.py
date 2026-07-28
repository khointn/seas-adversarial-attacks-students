"""Shared constants and utilities for preprocessing, plotting, and TOG targets."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import softmax

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

MislabelingMode = Literal["most_likely", "least_likely"]


def letterbox_image(image: Image.Image, size: tuple[int, int] = (416, 416)) -> tuple[np.ndarray, tuple[int, int, int, int, float]]:
    """Resize an image with aspect-ratio-preserving padding and return NHWC float data."""
    source = image.copy()
    source_width, source_height = source.size
    target_width, target_height = size
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = int(source_width * scale)
    resized_height = int(source_height * scale)

    source = source.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    padded = Image.new("RGB", size, (0, 0, 0))
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    padded.paste(source, (left, top))

    image_array = np.asarray(padded, dtype=np.float32)[None, ...] / 255.0
    metadata = (left, top, left + resized_width, top + resized_height, scale)
    return image_array, metadata


def visualize_detections(detection_sets: dict) -> None:
    """Plot one or more images with detector outputs in the repository detection format."""
    if not detection_sets:
        raise ValueError("detection_sets must contain at least one item.")

    num_colors = max(21, max(len(item[3]) for item in detection_sets.values()))
    colors = plt.cm.hsv(np.linspace(0, 1, num_colors)).tolist()
    plt.figure(figsize=(3 * len(detection_sets), 3))

    for plot_index, (title, values) in enumerate(detection_sets.items(), start=1):
        image, detections, model_image_size, class_names = values
        image = image[0] if image.ndim == 4 else image
        axis = plt.subplot(1, len(detection_sets), plot_index)
        axis.set_title(title)
        axis.imshow(image)

        for detection in detections:
            class_id = int(detection[0])
            xmin = max(int(detection[-4] * image.shape[1] / model_image_size[1]), 0)
            ymin = max(int(detection[-3] * image.shape[0] / model_image_size[0]), 0)
            xmax = min(int(detection[-2] * image.shape[1] / model_image_size[1]), image.shape[1])
            ymax = min(int(detection[-1] * image.shape[0] / model_image_size[0]), image.shape[0])
            color = colors[class_id % num_colors]
            label = f"{class_names[class_id]}: {detection[1]:.2f}"
            axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2)
            )
            axis.text(xmin, ymin, label, size="small", color="black", bbox={"facecolor": color, "alpha": 1.0})
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def generate_attack_targets(
    detections: np.ndarray,
    mode: MislabelingMode,
    confidence_threshold: float,
    source_class_id: int | None = None,
) -> np.ndarray:
    """Replace detected class IDs with TOG most- or least-likely target classes."""
    if mode not in ("most_likely", "least_likely"):
        raise ValueError("mode must be 'most_likely' or 'least_likely'.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    detections = np.asarray(detections, dtype=np.float32)
    if detections.ndim != 2 or detections.shape[0] == 0:
        raise ValueError("Mislabeling requires at least one valid detection.")

    targets = detections.copy()
    class_logits = targets[:, 2:-4].copy()
    if class_logits.shape[1] == 0:
        raise ValueError("Detections do not contain class logits.")

    if mode == "least_likely":
        target_class_ids = np.argmin(class_logits, axis=1)
    else:
        confident_classes = softmax(class_logits, axis=1) > confidence_threshold
        class_logits[confident_classes] = -np.inf
        target_class_ids = np.argmax(class_logits, axis=1)

    if source_class_id is not None:
        source_mask = targets[:, 0].astype(np.int64) == source_class_id
        if not source_mask.any():
            raise ValueError(f"No detections found for source class {source_class_id}.")
        target_class_ids = np.where(source_mask, target_class_ids, targets[:, 0].astype(np.int64))

    targets[:, 0] = target_class_ids.astype(np.float32)
    targets[:, 1] = 1.0
    return targets
