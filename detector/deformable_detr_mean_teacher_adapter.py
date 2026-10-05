"""Deformable DETR adapter for the generic Mean Teacher detector trainer.

This module bridges the standard Deformable DETR API used by the supplied
``main.py`` and ``mean_teacher_detection.py``.

Expected Deformable DETR API
----------------------------
``build_model(args)`` returns ``model, criterion, postprocessors``.

* ``model(images)`` returns the raw DETR output dictionary.
* ``criterion(outputs, targets)`` returns the unweighted DETR loss dictionary.
* ``criterion.weight_dict`` contains the coefficients used by the original
  ``train_one_epoch`` implementation.
* ``postprocessors["bbox"](outputs, target_sizes)`` converts model outputs to
  absolute ``xyxy`` detections.

The generic Mean Teacher code stores pseudo boxes as absolute ``xyxy`` boxes.
Before calling the DETR criterion, this adapter converts them to normalized
``cxcywh`` boxes, which is the target representation expected by DETR's
Hungarian matcher and box losses.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torchvision.transforms.functional as TF

from misc import nested_tensor_from_tensor_list

from mean_teacher_detection import (
    DetectionAdapter,
    ImageList,
    MeanTeacherConfig,
    PairedDetectionAugmenter,
    Prediction,
    Target,
)


class DeformableDETRAdapter(DetectionAdapter):
    """Adapter for the Deformable DETR model/criterion/postprocessor API."""

    def __init__(
        self,
        criterion: nn.Module,
        postprocessors: Mapping[str, object],
        bbox_postprocessor_key: str = "bbox",
    ) -> None:
        if bbox_postprocessor_key not in postprocessors:
            raise KeyError(
                f"postprocessors does not contain {bbox_postprocessor_key!r}. "
                f"Available keys: {list(postprocessors.keys())}"
            )

        self.criterion = criterion
        self.bbox_postprocessor = postprocessors[bbox_postprocessor_key]

    @staticmethod
    def _image_sizes(images: Sequence[Tensor]) -> Tensor:
        """Return ``[[height, width], ...]`` on the images' device."""
        if len(images) == 0:
            raise ValueError("Cannot infer sizes from an empty image list.")

        return torch.as_tensor(
            [[image.shape[-2], image.shape[-1]] for image in images],
            dtype=torch.float32,
            device=images[0].device,
        )

    @staticmethod
    def _to_nested_tensor(images: ImageList):
        """Pack variable-size image tensors for this Deformable DETR model.

        The local model implementation expects an object exposing
        ``.tensors`` and ``.mask`` and does not perform this conversion inside
        ``forward``. Augmentation is still applied to individual tensors; they
        are packed immediately before the model call.
        """
        if len(images) == 0:
            raise ValueError("Cannot build a NestedTensor from an empty image list.")

        for index, image in enumerate(images):
            if not isinstance(image, Tensor) or image.ndim != 3:
                raise TypeError(
                    f"Image {index} must be a [C,H,W] Tensor before packing; "
                    f"received {type(image).__name__} with shape "
                    f"{getattr(image, 'shape', None)}."
                )

        return nested_tensor_from_tensor_list(images)

    @torch.no_grad()
    def predict(self, model: nn.Module, images: ImageList) -> List[Prediction]:
        """Run DETR and convert predictions to absolute ``xyxy`` boxes."""
        was_training = model.training
        model.eval()

        samples = self._to_nested_tensor(images)
        outputs = model(samples)
        image_sizes = self._image_sizes(images)
        processed = self.bbox_postprocessor(outputs, image_sizes)

        if was_training:
            model.train()

        predictions: List[Prediction] = []
        for result in processed:
            required = {"boxes", "scores", "labels"}
            missing = required.difference(result.keys())
            if missing:
                raise KeyError(
                    "The DETR bbox postprocessor output is missing fields: "
                    f"{sorted(missing)}"
                )

            predictions.append(
                {
                    "boxes": result["boxes"],
                    "scores": result["scores"],
                    "labels": result["labels"].long(),
                }
            )

        return predictions

    @staticmethod
    def _xyxy_to_normalized_cxcywh(boxes: Tensor, image: Tensor) -> Tensor:
        """Convert absolute ``xyxy`` boxes to normalized DETR ``cxcywh``."""
        height, width = image.shape[-2:]

        boxes = boxes.to(dtype=torch.float32)
        x0, y0, x1, y1 = boxes.unbind(-1)

        # Clamp minor postprocessing overshoots before normalization.
        x0 = x0.clamp(min=0.0, max=float(width))
        x1 = x1.clamp(min=0.0, max=float(width))
        y0 = y0.clamp(min=0.0, max=float(height))
        y1 = y1.clamp(min=0.0, max=float(height))

        cx = (x0 + x1) * 0.5 / float(width)
        cy = (y0 + y1) * 0.5 / float(height)
        box_width = (x1 - x0) / float(width)
        box_height = (y1 - y0) / float(height)

        return torch.stack((cx, cy, box_width, box_height), dim=-1)

    def _convert_targets(
        self,
        images: ImageList,
        pseudo_targets: List[Target],
    ) -> List[Dict[str, Tensor]]:
        if len(images) != len(pseudo_targets):
            raise ValueError(
                "The number of student images and pseudo targets does not match: "
                f"{len(images)} vs. {len(pseudo_targets)}."
            )

        detr_targets: List[Dict[str, Tensor]] = []
        for image_index, (image, target) in enumerate(zip(images, pseudo_targets)):
            boxes_xyxy = target["boxes"]
            labels = target["labels"].long()

            if boxes_xyxy.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Image {image_index}: boxes and labels have different lengths."
                )

            boxes_cxcywh = self._xyxy_to_normalized_cxcywh(
                boxes_xyxy, image
            )

            # The generic filter already removes tiny/invalid boxes. Keep a
            # defensive check here because invalid targets can destabilize the
            # Hungarian matcher and GIoU loss.
            valid = (
                torch.isfinite(boxes_cxcywh).all(dim=1)
                & (boxes_cxcywh[:, 2] > 0)
                & (boxes_cxcywh[:, 3] > 0)
            )
            boxes_cxcywh = boxes_cxcywh[valid]
            labels = labels[valid]
            boxes_xyxy = boxes_xyxy[valid].to(dtype=torch.float32)

            height, width = image.shape[-2:]
            size = torch.as_tensor(
                [height, width], dtype=torch.int64, device=image.device
            )

            # Keep the pseudo target compatible with the COCO target dictionary
            # consumed by this repository's custom criterion. Pseudo detections
            # are ordinary instances, so every iscrowd value is zero.
            area = (
                (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0.0)
                * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0.0)
            )
            iscrowd = torch.zeros(
                (boxes_cxcywh.shape[0],),
                dtype=torch.int64,
                device=image.device,
            )

            detr_target: Dict[str, Tensor] = {
                "boxes": boxes_cxcywh,
                "labels": labels,
                "area": area,
                "iscrowd": iscrowd,
                "size": size,
                "orig_size": size.clone(),
                "image_id": torch.as_tensor(
                    [image_index], dtype=torch.int64, device=image.device
                ),
            }

            if "pseudo_scores" in target:
                detr_target["pseudo_scores"] = target["pseudo_scores"][valid]

            detr_targets.append(detr_target)

        return detr_targets

    def compute_loss(
        self,
        model: nn.Module,
        images: ImageList,
        targets: List[Target],
    ) -> Dict[str, Tensor]:
        """Compute the same weighted DETR losses as ``train_one_epoch``."""
        model.train()
        detr_targets = self._convert_targets(images, targets)
        samples = self._to_nested_tensor(images)
        outputs = model(samples)
        raw_loss_dict = self.criterion(outputs, detr_targets)

        if not isinstance(raw_loss_dict, Mapping):
            raise TypeError("The DETR criterion must return a loss dictionary.")

        weight_dict = getattr(self.criterion, "weight_dict", None)
        if weight_dict is None:
            # Fallback for a custom criterion that already returns weighted
            # losses. Standard Deformable DETR always provides weight_dict.
            return {
                name: value
                for name, value in raw_loss_dict.items()
                if torch.is_tensor(value) and value.ndim == 0
            }

        weighted_losses: Dict[str, Tensor] = {}
        for name, value in raw_loss_dict.items():
            if name in weight_dict:
                weighted_losses[name] = value * weight_dict[name]

        if not weighted_losses:
            raise RuntimeError(
                "No criterion losses matched criterion.weight_dict. "
                f"Loss keys: {list(raw_loss_dict.keys())}; "
                f"weight keys: {list(weight_dict.keys())}."
            )

        return weighted_losses


class ImageNetNormalizedPairedAugmenter:
    """Paired augmentation for tensors normalized by DETR's COCO pipeline.

    Standard Deformable DETR datasets normalize images using ImageNet mean and
    standard deviation. The generic augmenter expects raw values in ``[0, 1]``.
    This wrapper therefore denormalizes first, applies the paired weak/strong
    augmentation, and normalizes the two views again.
    """

    def __init__(
        self,
        config: MeanTeacherConfig,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.base_augmenter = PairedDetectionAugmenter(config)
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)

    def _denormalize(self, image: Tensor) -> Tensor:
        mean = image.new_tensor(self.mean)[:, None, None]
        std = image.new_tensor(self.std)[:, None, None]
        return (image * std + mean).clamp(0.0, 1.0)

    def _normalize(self, image: Tensor) -> Tensor:
        return TF.normalize(image, mean=self.mean, std=self.std)

    def __call__(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        raw_image = self._denormalize(image)
        teacher_raw, student_raw = self.base_augmenter(raw_image)
        return self._normalize(teacher_raw), self._normalize(student_raw)


def mean_teacher_collate_fn(batch):
    """Keep individual images instead of applying DETR's NestedTensor collate.

    Standard DETR datasets return ``(image, target)``. Some local variants may
    return additional metadata. Only the first element of each sample is used
    for unlabeled Mean Teacher training.
    """
    images = []

    for sample_index, sample in enumerate(batch):
        if isinstance(sample, Mapping):
            if "image" in sample:
                image = sample["image"]
            elif "images" in sample:
                image = sample["images"]
            else:
                raise KeyError(
                    f"Sample {sample_index} is a mapping without `image` or `images`."
                )
        elif isinstance(sample, (tuple, list)):
            if len(sample) == 0:
                raise ValueError(f"Sample {sample_index} is empty.")
            image = sample[0]
        else:
            image = sample

        images.append(image)

    # Returning a one-element tuple keeps compatibility with
    # unpack_unlabeled_batch(), while discarding ground-truth targets.
    return (images,)
