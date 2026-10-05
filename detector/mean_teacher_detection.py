"""
Architecture-agnostic Mean Teacher baseline for object detection.

The trainer itself does not assume YOLO, Faster R-CNN, DETR, or any other
specific detector. To use a new model, implement the DetectionAdapter interface:

    predict(model, images) -> List[Dict[str, Tensor]]
    compute_loss(model, images, targets) -> Dict[str, Tensor]

Required prediction fields:
    boxes:  FloatTensor[N, 4] in xyxy coordinates of the INPUT image
    scores: FloatTensor[N]
    labels: LongTensor[N]

Required target fields passed to the student:
    boxes:  FloatTensor[M, 4] in xyxy coordinates
    labels: LongTensor[M]

The default paired augmentation applies the same horizontal flip to teacher
and student images, then uses weak photometric augmentation for the teacher
and stronger photometric augmentation for the student. Because both branches
share the same geometry, teacher pseudo boxes can be used directly for the
student.

Typical source-free target-domain adaptation:
    1. Initialize `student` from the same off-the-shelf detector used by the
       proposed method.
    2. Initialize `teacher` as an exact copy of `student`.
    3. Generate high-confidence pseudo labels with the teacher.
    4. Train the student on strongly augmented target images.
    5. Update the teacher using an exponential moving average of the student.

This file is intended as a clean Mean Teacher comparison baseline. It does not
use temporal tracking, cross-view association, or ground-truth target labels.
"""

from __future__ import annotations

import copy
import math
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms
import torchvision.transforms.functional as TF


Prediction = Dict[str, Tensor]
Target = Dict[str, Tensor]
ImageList = List[Tensor]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MeanTeacherConfig:
    # Pseudo-label filtering
    score_threshold: float = 0.70
    nms_iou_threshold: float = 0.60
    min_box_size: float = 2.0
    max_detections_per_image: int = 300
    allowed_class_ids: Optional[Tuple[int, ...]] = None

    # EMA teacher
    ema_decay: float = 0.999
    ema_warmup: bool = True

    # Paired augmentation
    horizontal_flip_probability: float = 0.5
    weak_brightness: float = 0.05
    weak_contrast: float = 0.05
    weak_saturation: float = 0.05
    strong_brightness: float = 0.30
    strong_contrast: float = 0.30
    strong_saturation: float = 0.30
    strong_grayscale_probability: float = 0.10
    strong_blur_probability: float = 0.20
    strong_noise_std: float = 0.02

    # Optimization
    gradient_clip_norm: Optional[float] = 10.0
    skip_images_without_pseudo_labels: bool = True
    amp: bool = False

    # Logging/checkpointing
    log_interval: int = 20
    checkpoint_interval_epochs: int = 1


# ---------------------------------------------------------------------------
# Model adapter
# ---------------------------------------------------------------------------

class DetectionAdapter(ABC):
    """
    Minimal interface separating the Mean Teacher algorithm from the detector.

    Implement these two methods for your detector. This is the only
    architecture-specific part of the baseline.
    """

    @abstractmethod
    @torch.no_grad()
    def predict(self, model: nn.Module, images: ImageList) -> List[Prediction]:
        """
        Run inference.

        Each returned dictionary must contain:
            boxes:  [N, 4], xyxy in the coordinates of the supplied image
            scores: [N]
            labels: [N]
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        images: ImageList,
        targets: List[Target],
    ) -> Dict[str, Tensor]:
        """
        Compute the detector's training loss on pseudo targets.

        Return a dictionary of scalar losses. The trainer sums all values.
        """
        raise NotImplementedError


class TorchvisionDetectionAdapter(DetectionAdapter):
    """
    Ready-to-use adapter for torchvision detection models such as Faster R-CNN,
    RetinaNet, FCOS, and SSD, assuming the standard torchvision API.
    """

    @torch.no_grad()
    def predict(self, model: nn.Module, images: ImageList) -> List[Prediction]:
        was_training = model.training
        model.eval()
        outputs = model(images)
        if was_training:
            model.train()
        return outputs

    def compute_loss(
        self,
        model: nn.Module,
        images: ImageList,
        targets: List[Target],
    ) -> Dict[str, Tensor]:
        model.train()
        losses = model(images, targets)
        if not isinstance(losses, Mapping):
            raise TypeError(
                "Torchvision training mode must return a dictionary of losses."
            )
        return dict(losses)


# Example skeleton for a custom YOLO/DETR implementation:
#
# class MyDetectorAdapter(DetectionAdapter):
#     @torch.no_grad()
#     def predict(self, model, images):
#         raw_outputs = model.inference(images)
#         predictions = []
#         for output in raw_outputs:
#             predictions.append({
#                 "boxes": output.xyxy,
#                 "scores": output.confidence,
#                 "labels": output.class_id.long(),
#             })
#         return predictions
#
#     def compute_loss(self, model, images, targets):
#         batch = convert_images_and_targets_to_model_format(images, targets)
#         loss_output = model.training_step(batch)
#         return {
#             "box_loss": loss_output.box_loss,
#             "class_loss": loss_output.class_loss,
#             "objectness_loss": loss_output.objectness_loss,
#         }


# ---------------------------------------------------------------------------
# Paired teacher/student augmentation
# ---------------------------------------------------------------------------

def _sample_factor(amount: float) -> float:
    if amount <= 0:
        return 1.0
    return random.uniform(max(0.0, 1.0 - amount), 1.0 + amount)


def _photometric_jitter(
    image: Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
) -> Tensor:
    """
    Applies tensor-compatible photometric jitter.

    Expected image format:
        float Tensor[C, H, W], normally in [0, 1].
    """
    operations = [
        lambda x: TF.adjust_brightness(x, _sample_factor(brightness)),
        lambda x: TF.adjust_contrast(x, _sample_factor(contrast)),
        lambda x: TF.adjust_saturation(x, _sample_factor(saturation)),
    ]
    random.shuffle(operations)
    for operation in operations:
        image = operation(image)
    return image


class PairedDetectionAugmenter:
    """
    Produces teacher and student views with identical geometry.

    The current implementation intentionally limits geometric augmentation to a
    shared horizontal flip. This keeps pseudo-box alignment exact and makes the
    baseline independent of any detector-specific transform pipeline.
    """

    def __init__(self, config: MeanTeacherConfig):
        self.config = config

    def __call__(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        if image.ndim != 3:
            raise ValueError(
                f"Each image must have shape [C, H, W], received {tuple(image.shape)}."
            )
        if not image.is_floating_point():
            image = image.float() / 255.0

        # Shared geometry: both branches receive exactly the same flip.
        if random.random() < self.config.horizontal_flip_probability:
            image = TF.hflip(image)

        teacher_image = _photometric_jitter(
            image.clone(),
            brightness=self.config.weak_brightness,
            contrast=self.config.weak_contrast,
            saturation=self.config.weak_saturation,
        )

        student_image = _photometric_jitter(
            image.clone(),
            brightness=self.config.strong_brightness,
            contrast=self.config.strong_contrast,
            saturation=self.config.strong_saturation,
        )

        if random.random() < self.config.strong_grayscale_probability:
            student_image = TF.rgb_to_grayscale(
                student_image, num_output_channels=student_image.shape[0]
            )

        if random.random() < self.config.strong_blur_probability:
            # Kernel 5 is valid for ordinary OR frames and keeps the code simple.
            student_image = TF.gaussian_blur(student_image, kernel_size=[5, 5])

        if self.config.strong_noise_std > 0:
            noise = torch.randn_like(student_image) * self.config.strong_noise_std
            student_image = student_image + noise

        teacher_image = teacher_image.clamp(0.0, 1.0)
        student_image = student_image.clamp(0.0, 1.0)
        return teacher_image, student_image


# ---------------------------------------------------------------------------
# Pseudo-label filtering
# ---------------------------------------------------------------------------

def _empty_target(device: torch.device) -> Target:
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
        "labels": torch.empty((0,), dtype=torch.long, device=device),
        "pseudo_scores": torch.empty((0,), dtype=torch.float32, device=device),
    }


def filter_pseudo_prediction(
    prediction: Prediction,
    config: MeanTeacherConfig,
) -> Target:
    required = {"boxes", "scores", "labels"}
    missing = required.difference(prediction)
    if missing:
        raise KeyError(f"Prediction is missing required fields: {sorted(missing)}")

    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"].long()

    if boxes.numel() == 0:
        return _empty_target(boxes.device)

    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must have shape [N, 4], got {tuple(boxes.shape)}")

    keep = scores >= config.score_threshold

    if config.allowed_class_ids is not None:
        allowed = torch.as_tensor(
            config.allowed_class_ids,
            dtype=labels.dtype,
            device=labels.device,
        )
        keep = keep & (labels[:, None] == allowed[None, :]).any(dim=1)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = keep & (widths >= config.min_box_size) & (heights >= config.min_box_size)

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if boxes.numel() == 0:
        return _empty_target(prediction["boxes"].device)

    keep_indices = batched_nms(
        boxes,
        scores,
        labels,
        iou_threshold=config.nms_iou_threshold,
    )
    keep_indices = keep_indices[: config.max_detections_per_image]

    return {
        "boxes": boxes[keep_indices].detach(),
        "labels": labels[keep_indices].detach(),
        "pseudo_scores": scores[keep_indices].detach(),
    }


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema_teacher(
    teacher: nn.Module,
    student: nn.Module,
    decay: float,
) -> None:
    """
    EMA update for parameters and buffers.

    Floating-point buffers are averaged. Integer buffers, such as
    num_batches_tracked, are copied directly.
    """
    teacher_parameters = dict(teacher.named_parameters())
    student_parameters = dict(student.named_parameters())

    if teacher_parameters.keys() != student_parameters.keys():
        raise RuntimeError("Teacher and student parameter structures do not match.")

    for name, teacher_parameter in teacher_parameters.items():
        student_parameter = student_parameters[name]
        teacher_parameter.mul_(decay).add_(
            student_parameter.detach(), alpha=1.0 - decay
        )

    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())

    if teacher_buffers.keys() != student_buffers.keys():
        raise RuntimeError("Teacher and student buffer structures do not match.")

    for name, teacher_buffer in teacher_buffers.items():
        student_buffer = student_buffers[name].detach()
        if teacher_buffer.is_floating_point():
            teacher_buffer.mul_(decay).add_(student_buffer, alpha=1.0 - decay)
        else:
            teacher_buffer.copy_(student_buffer)


def ema_decay_for_step(
    configured_decay: float,
    global_step: int,
    warmup: bool,
) -> float:
    if not warmup:
        return configured_decay

    # Common Mean Teacher warm-up: use a faster-moving teacher initially.
    step_decay = 1.0 - 1.0 / float(global_step + 1)
    return min(configured_decay, step_decay)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MeanTeacherDetectionTrainer:
    def __init__(
        self,
        student: nn.Module,
        adapter: DetectionAdapter,
        optimizer: Optimizer,
        device: Union[str, torch.device],
        config: Optional[MeanTeacherConfig] = None,
        scheduler: Optional[Any] = None,
        augmenter: Optional[Any] = None,
    ):
        self.device = torch.device(device)
        self.student = student.to(self.device)
        self.teacher = copy.deepcopy(student).to(self.device)
        self.adapter = adapter
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or MeanTeacherConfig()
        self.augmenter = (
            augmenter
            if augmenter is not None
            else PairedDetectionAugmenter(self.config)
        )

        self.teacher.eval()
        self.teacher.requires_grad_(False)

        # Gradient checkpointing is useful only for the trainable student.
        # The EMA teacher always runs under no_grad(), so checkpointing provides
        # no memory benefit and old PyTorch versions emit:
        # "None of the inputs have requires_grad=True".
        for module in self.teacher.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = False

        self.global_step = 0
        self.current_epoch = 0

        amp_enabled = self.config.amp and self.device.type == "cuda"
        # torch.cuda.amp.GradScaler is retained for compatibility with both
        # older and newer PyTorch releases.
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    @staticmethod
    def _split_nested_tensor(value: Any) -> ImageList:
        """Convert a DETR NestedTensor-like object into unpadded image tensors."""
        tensors = getattr(value, "tensors", None)
        mask = getattr(value, "mask", None)

        if not isinstance(tensors, Tensor):
            raise TypeError(
                "NestedTensor-like input must expose a Tensor through `.tensors`."
            )

        if tensors.ndim == 3:
            return [tensors]

        if tensors.ndim != 4:
            raise ValueError(
                "NestedTensor `.tensors` must have shape [C,H,W] or [B,C,H,W], "
                f"received {tuple(tensors.shape)}."
            )

        if mask is None:
            return list(tensors)

        if not isinstance(mask, Tensor) or mask.ndim != 3:
            raise ValueError(
                "NestedTensor `.mask` must have shape [B,H,W] when provided."
            )

        images: ImageList = []
        for tensor, image_mask in zip(tensors, mask):
            valid = ~image_mask.bool()

            valid_rows = torch.nonzero(valid.any(dim=1), as_tuple=False)
            valid_cols = torch.nonzero(valid.any(dim=0), as_tuple=False)

            if valid_rows.numel() == 0 or valid_cols.numel() == 0:
                raise ValueError("NestedTensor contains an image with no valid pixels.")

            height = int(valid_rows[-1].item()) + 1
            width = int(valid_cols[-1].item()) + 1
            images.append(tensor[:, :height, :width])

        return images

    @classmethod
    def _coerce_image_entry(cls, image: Any) -> ImageList:
        """Convert one dataset image entry into one or more plain tensors."""
        if isinstance(image, Tensor):
            if image.ndim == 3:
                return [image]
            if image.ndim == 4:
                return list(image)
            raise ValueError(
                "Image tensor must have shape [C,H,W] or [B,C,H,W], "
                f"received {tuple(image.shape)}."
            )

        if hasattr(image, "tensors"):
            return cls._split_nested_tensor(image)

        # Some custom datasets wrap an image as (tensor, metadata) or [tensor].
        if isinstance(image, (tuple, list)) and len(image) > 0:
            first = image[0]
            if isinstance(first, Tensor) or hasattr(first, "tensors"):
                return cls._coerce_image_entry(first)

        raise TypeError(
            "Could not convert dataset image entry to a tensor. "
            f"Received type {type(image).__module__}.{type(image).__name__}. "
            "Expected Tensor, DETR NestedTensor, or a tuple/list whose first "
            "element is one of these."
        )

    def _move_images(self, images: Any) -> ImageList:
        # Support a complete DETR NestedTensor passed directly.
        if hasattr(images, "tensors"):
            extracted = self._split_nested_tensor(images)
        elif isinstance(images, Tensor):
            extracted = self._coerce_image_entry(images)
        elif isinstance(images, Sequence):
            extracted = []
            for image in images:
                extracted.extend(self._coerce_image_entry(image))
        else:
            raise TypeError(
                "Could not interpret the raw image batch. "
                f"Received type {type(images).__module__}.{type(images).__name__}."
            )

        if len(extracted) == 0:
            raise ValueError("The image batch is empty.")

        return [
            image.to(self.device, non_blocking=True)
            for image in extracted
        ]

    def _make_teacher_student_views(
        self,
        images: ImageList,
    ) -> Tuple[ImageList, ImageList]:
        teacher_images: ImageList = []
        student_images: ImageList = []

        for image in images:
            teacher_image, student_image = self.augmenter(image)
            teacher_images.append(teacher_image)
            student_images.append(student_image)

        return teacher_images, student_images

    @torch.no_grad()
    def generate_pseudo_targets(
        self,
        teacher_images: ImageList,
    ) -> List[Target]:
        self.teacher.eval()
        predictions = self.adapter.predict(self.teacher, teacher_images)

        if len(predictions) != len(teacher_images):
            raise RuntimeError(
                "Adapter returned a different number of predictions and images."
            )

        return [
            filter_pseudo_prediction(prediction, self.config)
            for prediction in predictions
        ]

    def _select_trainable_samples(
        self,
        student_images: ImageList,
        pseudo_targets: List[Target],
    ) -> Tuple[ImageList, List[Target]]:
        if not self.config.skip_images_without_pseudo_labels:
            return student_images, pseudo_targets

        selected_images: ImageList = []
        selected_targets: List[Target] = []

        for image, target in zip(student_images, pseudo_targets):
            if target["boxes"].shape[0] > 0:
                selected_images.append(image)
                selected_targets.append(target)

        return selected_images, selected_targets

    def train_step(self, raw_images: Sequence[Tensor]) -> Dict[str, float]:
        images = self._move_images(raw_images)
        teacher_images, student_images = self._make_teacher_student_views(images)
        pseudo_targets = self.generate_pseudo_targets(teacher_images)

        student_images, pseudo_targets = self._select_trainable_samples(
            student_images, pseudo_targets
        )

        pseudo_box_count = sum(target["boxes"].shape[0] for target in pseudo_targets)

        if len(student_images) == 0:
            return {
                "loss": 0.0,
                "pseudo_boxes": 0.0,
                "used_images": 0.0,
                "skipped_update": 1.0,
            }

        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)

        amp_enabled = self.scaler.is_enabled()
        # AMP is enabled only on CUDA; the disabled context is safe on CPU.
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss_dict = self.adapter.compute_loss(
                self.student,
                student_images,
                pseudo_targets,
            )

            if not loss_dict:
                raise RuntimeError("Adapter returned an empty loss dictionary.")

            total_loss = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(total_loss):
            printable = {
                name: float(value.detach().cpu())
                for name, value in loss_dict.items()
            }
            raise FloatingPointError(
                f"Non-finite loss at step {self.global_step}: {printable}"
            )

        self.scaler.scale(total_loss).backward()

        if self.config.gradient_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.config.gradient_clip_norm,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.global_step += 1
        decay = ema_decay_for_step(
            self.config.ema_decay,
            self.global_step,
            self.config.ema_warmup,
        )
        update_ema_teacher(self.teacher, self.student, decay)

        metrics = {
            name: float(value.detach().cpu())
            for name, value in loss_dict.items()
        }
        metrics.update(
            {
                "loss": float(total_loss.detach().cpu()),
                "pseudo_boxes": float(pseudo_box_count),
                "used_images": float(len(student_images)),
                "ema_decay": float(decay),
                "skipped_update": 0.0,
            }
        )
        return metrics

    def fit(
        self,
        dataloader: DataLoader,
        epochs: int,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
        if checkpoint_path is not None:
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            running_loss = 0.0
            update_count = 0

            for batch_index, batch in enumerate(dataloader):
                raw_images = unpack_unlabeled_batch(batch)
                metrics = self.train_step(raw_images)

                if metrics["skipped_update"] == 0.0:
                    running_loss += metrics["loss"]
                    update_count += 1

                if batch_index % self.config.log_interval == 0:
                    mean_loss = running_loss / max(update_count, 1)
                    print(
                        f"[Epoch {epoch + 1}/{epochs}] "
                        f"[Batch {batch_index + 1}/{len(dataloader)}] "
                        f"loss={metrics['loss']:.4f} "
                        f"mean_loss={mean_loss:.4f} "
                        f"pseudo_boxes={int(metrics['pseudo_boxes'])} "
                        f"used_images={int(metrics['used_images'])}"
                    )

            if self.scheduler is not None:
                self.scheduler.step()

            if (
                checkpoint_path is not None
                and (epoch + 1) % self.config.checkpoint_interval_epochs == 0
            ):
                self.save_checkpoint(
                    checkpoint_path / f"mean_teacher_epoch_{epoch + 1:03d}.pth"
                )

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "student": self.student.state_dict(),
                "teacher": self.teacher.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None
                ),
                "scaler": self.scaler.state_dict(),
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "config": asdict(self.config),
            },
            path,
        )

    def load_checkpoint(
        self,
        path: Union[str, Path],
        strict: bool = True,
    ) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        self.student.load_state_dict(checkpoint["student"], strict=strict)
        self.teacher.load_state_dict(checkpoint["teacher"], strict=strict)
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.global_step = int(checkpoint.get("global_step", 0))
        self.current_epoch = int(checkpoint.get("current_epoch", 0))


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def detection_collate_fn(batch: Sequence[Any]) -> Tuple[List[Any], ...]:
    """
    Generic collate function for variable-size detection images.

    Examples:
        Dataset returns image:
            batch -> ([image_1, image_2, ...],)

        Dataset returns (image, image_id):
            batch -> ([images...], [image_ids...])
    """
    return tuple(map(list, zip(*batch)))


def unpack_unlabeled_batch(batch: Any) -> Sequence[Tensor]:
    """
    Extract images from common dataloader batch structures.

    Supported:
        List[Tensor]
        Tuple[List[Tensor], ...]
        Dict with key "images"
        Tensor[B, C, H, W]
    """
    if isinstance(batch, Mapping):
        if "images" not in batch:
            raise KeyError('Dictionary batch must contain an "images" key.')
        images = batch["images"]
    elif isinstance(batch, tuple):
        if len(batch) == 0:
            raise ValueError("Received an empty tuple batch.")
        images = batch[0]
    else:
        images = batch

    if isinstance(images, Tensor):
        if images.ndim != 4:
            raise ValueError(
                f"Batched image tensor must be [B, C, H, W], got {images.shape}."
            )
        return list(images)

    if not isinstance(images, Sequence):
        raise TypeError("Could not extract a sequence of images from the batch.")

    return images


# ---------------------------------------------------------------------------
# Minimal usage example
# ---------------------------------------------------------------------------

def example_torchvision_usage(
    pretrained_detector: nn.Module,
    target_dataset: torch.utils.data.Dataset,
    device: str = "cuda",
) -> MeanTeacherDetectionTrainer:
    """
    Example only. Replace the detector, optimizer, and dataset with yours.

    The dataset may return:
        image
    or:
        (image, any_metadata)

    Images should be float tensors [C, H, W], ideally in [0, 1].
    """
    dataloader = DataLoader(
        target_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=detection_collate_fn,
    )

    student = pretrained_detector
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=1e-5,
        weight_decay=1e-4,
    )

    config = MeanTeacherConfig(
        score_threshold=0.70,
        nms_iou_threshold=0.60,
        ema_decay=0.999,
        amp=True,
    )

    trainer = MeanTeacherDetectionTrainer(
        student=student,
        adapter=TorchvisionDetectionAdapter(),
        optimizer=optimizer,
        device=device,
        config=config,
    )

    trainer.fit(
        dataloader=dataloader,
        epochs=5,
        checkpoint_dir="./mean_teacher_checkpoints",
    )
    return trainer


if __name__ == "__main__":
    print(
        "Import this module, implement DetectionAdapter for your detector, "
        "and initialize MeanTeacherDetectionTrainer with your pretrained model."
    )
