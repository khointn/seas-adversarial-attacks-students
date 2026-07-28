"""PyTorch YOLOv3 network, Darknet weight loading, and detector inference."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms

from utils import COCO_CLASSES

DEFAULT_ANCHORS = np.asarray(
    [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]],
    dtype=np.float32,
)
ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


def to_nchw(image: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert NHWC or NCHW image data to a validated NCHW float tensor."""
    tensor = torch.from_numpy(image.astype(np.float32, copy=False)) if isinstance(image, np.ndarray) else image.float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError("image must have three or four dimensions.")
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    elif tensor.shape[1] != 3:
        raise ValueError("image must contain exactly three channels.")
    if not torch.isfinite(tensor).all():
        raise ValueError("image contains non-finite values.")
    return tensor.to(device)


class ConvBNLeaky(nn.Module):
    """Apply a Darknet convolution, batch normalization, and leaky ReLU."""

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, stride: int = 1):
        """Initialize one Darknet convolution block."""
        super().__init__()
        padding = (kernel_size - 1) // 2 if stride == 1 else 0
        self.asymmetric_padding = nn.ZeroPad2d((1, 0, 1, 0)) if stride == 2 else None
        self.convolution = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(output_channels, momentum=0.03, eps=1e-4)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the convolution block."""
        if self.asymmetric_padding is not None:
            inputs = self.asymmetric_padding(inputs)
        return self.activation(self.batch_norm(self.convolution(inputs)))


class ResidualBlock(nn.Module):
    """Apply one Darknet-53 residual block."""

    def __init__(self, channels: int):
        """Initialize a bottleneck residual block."""
        super().__init__()
        self.reduce = ConvBNLeaky(channels, channels // 2, kernel_size=1)
        self.expand = ConvBNLeaky(channels // 2, channels, kernel_size=3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add the residual transformation to its input."""
        return inputs + self.expand(self.reduce(inputs))


class ResidualStage(nn.Module):
    """Downsample once and apply a sequence of residual blocks."""

    def __init__(self, input_channels: int, output_channels: int, num_blocks: int):
        """Initialize one Darknet-53 stage."""
        super().__init__()
        self.downsample = ConvBNLeaky(input_channels, output_channels, kernel_size=3, stride=2)
        self.blocks = nn.Sequential(*[ResidualBlock(output_channels) for _ in range(num_blocks)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the downsampling and residual blocks."""
        return self.blocks(self.downsample(inputs))


class Darknet53(nn.Module):
    """Produce Darknet-53 feature maps at strides 8, 16, and 32."""

    def __init__(self):
        """Initialize the Darknet-53 backbone."""
        super().__init__()
        self.stem = ConvBNLeaky(3, 32)
        self.stage1 = ResidualStage(32, 64, 1)
        self.stage2 = ResidualStage(64, 128, 2)
        self.stage3 = ResidualStage(128, 256, 8)
        self.stage4 = ResidualStage(256, 512, 8)
        self.stage5 = ResidualStage(512, 1024, 4)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return small-, medium-, and large-scale backbone features."""
        inputs = self.stem(inputs)
        inputs = self.stage1(inputs)
        inputs = self.stage2(inputs)
        small_features = self.stage3(inputs)
        medium_features = self.stage4(small_features)
        large_features = self.stage5(medium_features)
        return small_features, medium_features, large_features


class YOLOHead(nn.Module):
    """Build one YOLOv3 multi-convolution prediction head."""

    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int):
        """Initialize a YOLOv3 detection head."""
        super().__init__()
        self.features = nn.Sequential(
            ConvBNLeaky(input_channels, hidden_channels, kernel_size=1),
            ConvBNLeaky(hidden_channels, hidden_channels * 2, kernel_size=3),
            ConvBNLeaky(hidden_channels * 2, hidden_channels, kernel_size=1),
            ConvBNLeaky(hidden_channels, hidden_channels * 2, kernel_size=3),
            ConvBNLeaky(hidden_channels * 2, hidden_channels, kernel_size=1),
        )
        self.prediction = nn.Sequential(
            ConvBNLeaky(hidden_channels, hidden_channels * 2, kernel_size=3),
            nn.Conv2d(hidden_channels * 2, output_channels, 1, bias=True),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return intermediate features and raw predictions."""
        features = self.features(inputs)
        return features, self.prediction(features)


class YOLOv3(nn.Module):
    """Return raw YOLOv3 predictions for three object scales."""

    def __init__(self, num_classes: int = 80, anchors_per_scale: int = 3):
        """Initialize the YOLOv3 backbone and prediction heads."""
        super().__init__()
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        output_channels = anchors_per_scale * (5 + num_classes)

        self.backbone = Darknet53()
        self.large_head = YOLOHead(1024, 512, output_channels)
        self.large_upsample = nn.Sequential(
            ConvBNLeaky(512, 256, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.medium_head = YOLOHead(768, 256, output_channels)
        self.medium_upsample = nn.Sequential(
            ConvBNLeaky(256, 128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.small_head = YOLOHead(384, 128, output_channels)

    def _reshape_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """Reshape a convolution output to batch-anchor-grid-channel layout."""
        batch_size, _, grid_height, grid_width = prediction.shape
        prediction = prediction.view(
            batch_size,
            self.anchors_per_scale,
            5 + self.num_classes,
            grid_height,
            grid_width,
        )
        return prediction.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Return large-, medium-, and small-object raw predictions."""
        small_features, medium_features, large_features = self.backbone(inputs)
        large_head_features, large_prediction = self.large_head(large_features)
        medium_input = torch.cat([self.large_upsample(large_head_features), medium_features], dim=1)
        medium_head_features, medium_prediction = self.medium_head(medium_input)
        small_input = torch.cat([self.medium_upsample(medium_head_features), small_features], dim=1)
        _, small_prediction = self.small_head(small_input)
        return [
            self._reshape_prediction(large_prediction),
            self._reshape_prediction(medium_prediction),
            self._reshape_prediction(small_prediction),
        ]


def _conv_blocks(module: nn.Module) -> Iterator[ConvBNLeaky]:
    """Yield Darknet convolution blocks in module registration order."""
    for child in module.modules():
        if isinstance(child, ConvBNLeaky):
            yield child


def _ordered_weight_layers(model: YOLOv3) -> Iterator[tuple[nn.Conv2d, nn.BatchNorm2d | None]]:
    """Yield convolution layers in the order used by Darknet weight files."""
    for block in _conv_blocks(model.backbone):
        yield block.convolution, block.batch_norm

    for head, upsample in (
        (model.large_head, model.large_upsample),
        (model.medium_head, model.medium_upsample),
        (model.small_head, None),
    ):
        for block in _conv_blocks(head.features):
            yield block.convolution, block.batch_norm
        prediction_block = head.prediction[0]
        yield prediction_block.convolution, prediction_block.batch_norm
        yield head.prediction[1], None
        if upsample is not None:
            for block in _conv_blocks(upsample):
                yield block.convolution, block.batch_norm


def load_darknet_weights(model: YOLOv3, weights_path: str | Path) -> None:
    """Load an official Darknet YOLOv3 binary weight file exactly."""
    with Path(weights_path).open("rb") as file:
        header = np.fromfile(file, dtype=np.int32, count=5)
        weights = np.fromfile(file, dtype=np.float32)
    if header.size != 5:
        raise RuntimeError("Invalid Darknet weight header.")

    pointer = 0

    def take(parameter: torch.Tensor) -> torch.Tensor:
        """Read and reshape the next parameter-sized slice from the weight array."""
        nonlocal pointer
        size = parameter.numel()
        if pointer + size > weights.size:
            raise RuntimeError("Darknet weight file ended unexpectedly.")
        values = torch.from_numpy(weights[pointer:pointer + size]).to(parameter.device, parameter.dtype)
        pointer += size
        return values.view_as(parameter)

    with torch.no_grad():
        for convolution, batch_norm in _ordered_weight_layers(model):
            if batch_norm is not None:
                batch_norm.bias.copy_(take(batch_norm.bias))
                batch_norm.weight.copy_(take(batch_norm.weight))
                batch_norm.running_mean.copy_(take(batch_norm.running_mean))
                batch_norm.running_var.copy_(take(batch_norm.running_var))
            else:
                if convolution.bias is None:
                    raise RuntimeError("Expected a bias tensor for a prediction convolution.")
                convolution.bias.copy_(take(convolution.bias))
            convolution.weight.copy_(take(convolution.weight))

    if pointer != weights.size:
        raise RuntimeError(f"Darknet weight mismatch: consumed {pointer}/{weights.size} values.")


class YOLOv3Detector:
    """Load YOLOv3 weights and expose detector-only inference operations."""

    def __init__(
        self,
        weights: str | Path = "weights/yolov3.weights",
        model_image_size: tuple[int, int] = (416, 416),
        confidence_threshold: float = 0.20,
        num_classes: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        anchors: Optional[np.ndarray] = None,
    ):
        """Initialize the network, detector settings, and Darknet weights."""
        self.model_image_size = tuple(model_image_size)
        self.confidence_threshold = confidence_threshold
        self.class_names = list(class_names) if class_names is not None else list(COCO_CLASSES)
        self.num_classes = num_classes if num_classes is not None else len(self.class_names)
        if len(self.class_names) != self.num_classes:
            raise ValueError("class_names length must equal num_classes.")
        self.anchors = np.asarray(anchors, dtype=np.float32) if anchors is not None else DEFAULT_ANCHORS.copy()
        if self.anchors.shape != (9, 2):
            raise ValueError("anchors must have shape (9, 2).")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network = YOLOv3(num_classes=self.num_classes).to(self.device)
        load_darknet_weights(self.network, weights)
        self.network.eval()

    def _decode_layer(
        self,
        prediction: torch.Tensor,
        anchors: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode one raw YOLO layer into boxes, class scores, and class logits."""
        batch_size, num_anchors, grid_height, grid_width, _ = prediction.shape
        if batch_size != 1:
            raise ValueError("detect currently supports a batch size of one.")

        anchor_tensor = torch.as_tensor(anchors, dtype=prediction.dtype, device=prediction.device).view(
            1, num_anchors, 1, 1, 2
        )
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_height, device=prediction.device),
            torch.arange(grid_width, device=prediction.device),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, grid_height, grid_width, 2).to(prediction.dtype)
        raw = prediction[0]
        grid_size = torch.tensor([grid_width, grid_height], device=prediction.device, dtype=prediction.dtype)
        input_size = torch.tensor(
            [self.model_image_size[1], self.model_image_size[0]],
            device=prediction.device,
            dtype=prediction.dtype,
        )
        box_xy = (torch.sigmoid(raw[..., :2]) + grid[0]) / grid_size
        box_wh = torch.exp(raw[..., 2:4].clamp(max=10)) * anchor_tensor[0] / input_size
        confidence = torch.sigmoid(raw[..., 4:5])
        class_logits = raw[..., 5:]
        class_scores = confidence * torch.sigmoid(class_logits)

        box_min = box_xy - box_wh / 2
        box_max = box_xy + box_wh / 2
        boxes = torch.cat([box_min, box_max], dim=-1) * torch.tensor(
            [self.model_image_size[1], self.model_image_size[0], self.model_image_size[1], self.model_image_size[0]],
            device=prediction.device,
            dtype=prediction.dtype,
        )
        return (
            boxes.reshape(-1, 4),
            class_scores.reshape(-1, self.num_classes),
            class_logits.reshape(-1, self.num_classes),
        )

    @torch.no_grad()
    def detect(
        self,
        image: np.ndarray,
        iou_threshold: float = 0.45,
        confidence_threshold: float | None = None,
        max_detections_per_class: int = 400,
    ) -> np.ndarray:
        """Detect objects and return class, score, logits, and xyxy box columns."""
        threshold = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")
        if max_detections_per_class <= 0:
            raise ValueError("max_detections_per_class must be positive.")

        predictions = self.network(to_nchw(image, self.device))
        decoded = [
            self._decode_layer(prediction, self.anchors[mask])
            for prediction, mask in zip(predictions, ANCHOR_MASKS)
        ]
        boxes = torch.cat([item[0] for item in decoded], dim=0)
        scores = torch.cat([item[1] for item in decoded], dim=0)
        logits = torch.cat([item[2] for item in decoded], dim=0)

        rows = []
        for class_id in range(self.num_classes):
            class_scores = scores[:, class_id]
            candidate_mask = class_scores >= threshold
            if not candidate_mask.any():
                continue
            candidate_boxes = boxes[candidate_mask]
            candidate_scores = class_scores[candidate_mask]
            candidate_logits = logits[candidate_mask]
            kept_indices = nms(candidate_boxes, candidate_scores, iou_threshold)[:max_detections_per_class]
            for index in kept_indices:
                rows.append(
                    torch.cat(
                        [
                            torch.tensor([float(class_id), float(candidate_scores[index])], device=self.device),
                            candidate_logits[index],
                            candidate_boxes[index],
                        ]
                    ).cpu().numpy()
                )

        if not rows:
            return np.empty((0, 2 + self.num_classes + 4), dtype=np.float32)
        detections = np.stack(rows).astype(np.float32)
        return detections[np.argsort(detections[:, 1])[::-1]]
