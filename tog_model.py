"""YOLOv3-specific losses and image gradients required by TOG attacks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from yolov3_model import ANCHOR_MASKS, YOLOv3Detector, to_nchw


def box_iou_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between center-format xywh boxes."""
    boxes1 = boxes1.unsqueeze(-2)
    boxes2 = boxes2.unsqueeze(0)
    boxes1_min = boxes1[..., :2] - boxes1[..., 2:] / 2
    boxes1_max = boxes1[..., :2] + boxes1[..., 2:] / 2
    boxes2_min = boxes2[..., :2] - boxes2[..., 2:] / 2
    boxes2_max = boxes2[..., :2] + boxes2[..., 2:] / 2
    intersection_min = torch.maximum(boxes1_min, boxes2_min)
    intersection_max = torch.minimum(boxes1_max, boxes2_max)
    intersection_size = (intersection_max - intersection_min).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]
    return intersection / (area1 + area2 - intersection + 1e-6)


def encode_yolo_targets(
    boxes: np.ndarray,
    input_shape: tuple[int, int],
    anchors: np.ndarray,
    num_classes: int,
) -> list[np.ndarray]:
    """Encode absolute xyxy boxes and class IDs into three YOLO target tensors."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 3 or boxes.shape[-1] != 5:
        raise ValueError("boxes must have shape (batch, num_boxes, 5).")
    if boxes.size and ((boxes[..., 4] < 0).any() or (boxes[..., 4] >= num_classes).any()):
        raise ValueError("Every class ID must be within the detector class range.")

    input_shape_array = np.asarray(input_shape, dtype=np.int32)
    normalized_boxes = boxes.copy()
    box_centers = (normalized_boxes[..., :2] + normalized_boxes[..., 2:4]) / 2
    box_sizes = normalized_boxes[..., 2:4] - normalized_boxes[..., :2]
    normalized_boxes[..., :2] = box_centers / input_shape_array[::-1]
    normalized_boxes[..., 2:4] = box_sizes / input_shape_array[::-1]

    grid_shapes = [input_shape_array // stride for stride in (32, 16, 8)]
    targets = [
        np.zeros(
            (boxes.shape[0], grid_height, grid_width, len(mask), 5 + num_classes),
            dtype=np.float32,
        )
        for (grid_height, grid_width), mask in zip(grid_shapes, ANCHOR_MASKS)
    ]

    anchor_boxes = anchors[None, ...]
    anchor_min = -anchor_boxes / 2
    anchor_max = anchor_boxes / 2
    valid_mask = (box_sizes[..., 0] > 0) & (box_sizes[..., 1] > 0)

    for batch_index in range(boxes.shape[0]):
        valid_indices = np.flatnonzero(valid_mask[batch_index])
        if valid_indices.size == 0:
            continue
        valid_sizes = box_sizes[batch_index, valid_indices, None, :]
        box_min = -valid_sizes / 2
        box_max = valid_sizes / 2
        intersection_size = np.maximum(np.minimum(box_max, anchor_max) - np.maximum(box_min, anchor_min), 0)
        intersection = intersection_size[..., 0] * intersection_size[..., 1]
        box_area = valid_sizes[..., 0] * valid_sizes[..., 1]
        anchor_area = anchor_boxes[..., 0] * anchor_boxes[..., 1]
        best_anchors = np.argmax(intersection / (box_area + anchor_area - intersection), axis=-1)

        for valid_position, anchor_index in enumerate(best_anchors):
            box_index = valid_indices[valid_position]
            for layer_index, anchor_mask in enumerate(ANCHOR_MASKS):
                if anchor_index not in anchor_mask:
                    continue
                grid_height, grid_width = grid_shapes[layer_index]
                grid_x = int(np.clip(np.floor(normalized_boxes[batch_index, box_index, 0] * grid_width), 0, grid_width - 1))
                grid_y = int(np.clip(np.floor(normalized_boxes[batch_index, box_index, 1] * grid_height), 0, grid_height - 1))
                mask_index = anchor_mask.index(int(anchor_index))
                class_id = int(normalized_boxes[batch_index, box_index, 4])
                targets[layer_index][batch_index, grid_y, grid_x, mask_index, :4] = normalized_boxes[
                    batch_index, box_index, :4
                ]
                targets[layer_index][batch_index, grid_y, grid_x, mask_index, 4] = 1.0
                targets[layer_index][batch_index, grid_y, grid_x, mask_index, 5 + class_id] = 1.0
    return targets


class YOLOv3TOGModel:
    """Wrap a detector with YOLOv3 loss and gradient operations for TOG."""

    def __init__(self, detector: YOLOv3Detector):
        """Store the detector used for inference and differentiable gradients."""
        self.detector = detector
        self.device = detector.device
        self.model_image_size = detector.model_image_size
        self.confidence_threshold = detector.confidence_threshold
        self.num_classes = detector.num_classes
        self.anchors = detector.anchors

    def detect(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Delegate inference to the separate object detector."""
        return self.detector.detect(image, **kwargs)

    def _decode_for_loss(
        self,
        prediction: torch.Tensor,
        anchors: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode grid and normalized boxes needed by the YOLO training loss."""
        _, num_anchors, grid_height, grid_width, _ = prediction.shape
        anchor_tensor = torch.as_tensor(anchors, dtype=prediction.dtype, device=prediction.device).view(
            1, num_anchors, 1, 1, 2
        )
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_height, device=prediction.device),
            torch.arange(grid_width, device=prediction.device),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, grid_height, grid_width, 2).to(prediction.dtype)
        grid_size = torch.tensor([grid_width, grid_height], device=prediction.device, dtype=prediction.dtype)
        input_size = torch.tensor(
            [self.model_image_size[1], self.model_image_size[0]],
            device=prediction.device,
            dtype=prediction.dtype,
        )
        box_xy = (torch.sigmoid(prediction[..., :2]) + grid) / grid_size
        box_wh = torch.exp(prediction[..., 2:4].clamp(max=10)) * anchor_tensor / input_size
        return grid, box_xy, box_wh

    def _objectness_loss(self, predictions: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
        """Compute summed objectness binary cross-entropy across all YOLO scales."""
        loss = predictions[0].new_zeros(())
        for prediction, target in zip(predictions, targets):
            target = target.permute(0, 3, 1, 2, 4).contiguous()
            loss += F.binary_cross_entropy_with_logits(prediction[..., 4:5], target[..., 4:5], reduction="sum")
        return loss

    def _full_yolo_loss(self, predictions: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
        """Compute the YOLOv3 box, objectness, and class loss used by TOG."""
        batch_size = float(predictions[0].shape[0])
        total_loss = predictions[0].new_zeros(())

        for prediction, target, anchor_mask in zip(predictions, targets, ANCHOR_MASKS):
            target = target.permute(0, 3, 1, 2, 4).contiguous()
            object_mask = target[..., 4:5]
            true_classes = target[..., 5:]
            grid, predicted_xy, predicted_wh = self._decode_for_loss(prediction, self.anchors[anchor_mask])
            predicted_boxes = torch.cat([predicted_xy, predicted_wh], dim=-1)
            grid_height, grid_width = prediction.shape[2:4]

            raw_true_xy = target[..., :2] * torch.tensor(
                [grid_width, grid_height], device=prediction.device, dtype=prediction.dtype
            ) - grid
            anchor_tensor = torch.as_tensor(
                self.anchors[anchor_mask], dtype=prediction.dtype, device=prediction.device
            ).view(1, len(anchor_mask), 1, 1, 2)
            raw_true_wh = torch.log(
                target[..., 2:4]
                * torch.tensor(
                    [self.model_image_size[1], self.model_image_size[0]],
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                / anchor_tensor
                + 1e-16
            )
            raw_true_wh = torch.where(object_mask.bool(), raw_true_wh, torch.zeros_like(raw_true_wh))
            box_loss_scale = 2.0 - target[..., 2:3] * target[..., 3:4]

            ignore_mask = torch.ones_like(object_mask)
            for batch_index in range(prediction.shape[0]):
                true_boxes = target[batch_index, ..., :4][object_mask[batch_index, ..., 0] > 0.5]
                if true_boxes.numel() == 0:
                    continue
                best_iou = box_iou_xywh(predicted_boxes[batch_index], true_boxes).max(dim=-1).values
                ignore_mask[batch_index, ..., 0] = (best_iou < 0.45).to(prediction.dtype)

            xy_loss = object_mask * box_loss_scale * F.binary_cross_entropy_with_logits(
                prediction[..., :2], raw_true_xy, reduction="none"
            )
            wh_loss = object_mask * box_loss_scale * 0.5 * (raw_true_wh - prediction[..., 2:4]) ** 2
            confidence_loss = object_mask * F.binary_cross_entropy_with_logits(
                prediction[..., 4:5], object_mask, reduction="none"
            ) + (1 - object_mask) * F.binary_cross_entropy_with_logits(
                prediction[..., 4:5], object_mask, reduction="none"
            ) * ignore_mask
            class_loss = object_mask * F.binary_cross_entropy_with_logits(
                prediction[..., 5:], true_classes, reduction="none"
            )
            total_loss += # TODO: Implement the total loss here using the loss components
        return total_loss

    def _targets_from_detections(self, detections: np.ndarray | None) -> list[torch.Tensor]:
        """Convert detector rows into YOLO target tensors on the detector device."""
        if detections is None or np.asarray(detections).size == 0:
            boxes = np.empty((1, 0, 5), dtype=np.float32)
        else:
            detections = np.asarray(detections, dtype=np.float32)
            if detections.ndim != 2 or detections.shape[1] < 6:
                raise ValueError("detections must be a two-dimensional detector output array.")
            boxes = detections[:, [-4, -3, -2, -1, 0]][None, ...]
        encoded = encode_yolo_targets(boxes, self.model_image_size, self.anchors, self.num_classes)
        return [torch.from_numpy(target).to(self.device) for target in encoded]

    def _image_gradient(
        self,
        image: np.ndarray,
        loss_function: Callable[[list[torch.Tensor]], torch.Tensor],
    ) -> np.ndarray:
        """Differentiate a supplied loss with respect to an NHWC input image."""
        input_tensor = to_nchw(image, self.device).detach().requires_grad_(True)
        self.detector.network.zero_grad(set_to_none=True)
        loss = loss_function(self.detector.network(input_tensor))
        loss.backward()
        if input_tensor.grad is None:
            raise RuntimeError("The attack loss did not produce an input gradient.")
        return input_tensor.grad.detach().permute(0, 2, 3, 1).cpu().numpy()

    def compute_object_vanishing_gradient(self, image: np.ndarray) -> np.ndarray:
        """Compute the gradient that minimizes objectness for every prediction cell."""
        targets = self._targets_from_detections(None)
        return self._image_gradient(image, lambda predictions: self._objectness_loss(predictions, targets))

    def compute_object_fabrication_gradient(self, image: np.ndarray) -> np.ndarray:
        """Compute the gradient that maximizes objectness for every prediction cell."""
        targets = self._targets_from_detections(None)
        for target in targets:
            target[..., 4] = 1.0
        return self._image_gradient(image, lambda predictions: self._objectness_loss(predictions, targets))

    def compute_object_untargeted_gradient(self, image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """Compute the negative YOLO loss gradient for benign detections."""
        targets = self._targets_from_detections(detections)
        return self._image_gradient(image, lambda predictions: -self._full_yolo_loss(predictions, targets))

    def compute_object_mislabeling_gradient(self, image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """Compute the YOLO loss gradient toward specified target detections."""
        targets = self._targets_from_detections(detections)
        return self._image_gradient(image, lambda predictions: self._full_yolo_loss(predictions, targets))
