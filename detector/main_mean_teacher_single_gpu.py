# ------------------------------------------------------------------------
# Deformable DETR Mean Teacher baseline
# Based on the supplied Deformable DETR main.py.
# ------------------------------------------------------------------------

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import config
import misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from detr import build_model

from mean_teacher_detection import MeanTeacherConfig, MeanTeacherDetectionTrainer
from deformable_detr_mean_teacher_adapter import (
    DeformableDETRAdapter,
    ImageNetNormalizedPairedAugmenter,
    mean_teacher_collate_fn,
)


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Deformable DETR Mean Teacher baseline", add_help=False
    )

    # Optimizer settings retained from the original main.py.
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument(
        "--lr_backbone_names", default=["backbone.0"], type=str, nargs="+"
    )
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument("--clip_max_norm", default=0.1, type=float)
    parser.add_argument("--sgd", action="store_true")

    # Deformable DETR variants.
    parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--two_stage", default=False, action="store_true")
    parser.add_argument("--frozen_weights", type=str, default=None)

    # Backbone.
    parser.add_argument("--backbone", default="resnet50", type=str)
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
    )
    parser.add_argument("--position_embedding_scale", default=2 * np.pi, type=float)
    parser.add_argument("--num_feature_levels", default=4, type=int)

    # Transformer.
    parser.add_argument("--enc_layers", default=6, type=int)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=512, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--num_queries", default=200, type=int)
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)
    parser.add_argument("--use_checkpoint", action="store_true")

    # Segmentation/loss/matcher arguments required by build_model(args).
    parser.add_argument("--masks", action="store_true")
    parser.add_argument("--no_aux_loss", dest="aux_loss", action="store_false")
    parser.set_defaults(aux_loss=True)
    parser.add_argument("--set_cost_class", default=2, type=float)
    parser.add_argument("--set_cost_bbox", default=5, type=float)
    parser.add_argument("--set_cost_giou", default=2, type=float)
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    # Dataset/runtime.
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", default="./data/coco", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--cache_mode", default=False, action="store_true")

    # Project-specific build_model arguments retained from the supplied file.
    parser.add_argument("--dense_query", default=0, type=int)
    parser.add_argument("--rectified_attention", default=0, type=int)
    parser.add_argument("--aps", default=0, type=int)

    # Initialization and Mean Teacher checkpointing.
    parser.add_argument(
        "--pretrained",
        default=(
            "/home2020/home/icube/keqichen/code/Iter-Deformable-DETR/"
            "weights/iter-d-detr-swinl.pth"
        ),
        type=str,
        help="Source/off-the-shelf detector checkpoint used to initialize both models.",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="Resume a checkpoint produced by mean_teacher_detection.py.",
    )

    # Mean Teacher hyperparameters.
    parser.add_argument("--mt_score_threshold", default=0.70, type=float)
    parser.add_argument("--mt_nms_iou_threshold", default=0.60, type=float)
    parser.add_argument("--mt_min_box_size", default=2.0, type=float)
    parser.add_argument("--mt_max_detections", default=300, type=int)
    parser.add_argument(
        "--mt_person_class_id",
        default=None,
        type=int,
        help=(
            "Keep only this class in teacher pseudo labels. Leave unset if the "
            "detector contains only the person class or to retain all classes."
        ),
    )
    parser.add_argument("--mt_ema_decay", default=0.999, type=float)
    parser.add_argument("--mt_no_ema_warmup", action="store_true")

    parser.add_argument("--mt_flip_probability", default=0.5, type=float)
    parser.add_argument("--mt_weak_brightness", default=0.05, type=float)
    parser.add_argument("--mt_weak_contrast", default=0.05, type=float)
    parser.add_argument("--mt_weak_saturation", default=0.05, type=float)
    parser.add_argument("--mt_strong_brightness", default=0.30, type=float)
    parser.add_argument("--mt_strong_contrast", default=0.30, type=float)
    parser.add_argument("--mt_strong_saturation", default=0.30, type=float)
    parser.add_argument("--mt_grayscale_probability", default=0.10, type=float)
    parser.add_argument("--mt_blur_probability", default=0.20, type=float)
    parser.add_argument("--mt_noise_std", default=0.02, type=float)

    parser.add_argument(
        "--mt_amp",
        action="store_true",
        help=(
            "Enable CUDA AMP. Do not use this with the repository's default "
            "MSDeformAttn extension, which has no FP16 CUDA implementation."
        ),
    )
    parser.add_argument(
        "--mt_no_amp",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--mt_keep_empty_images", action="store_true")
    parser.add_argument("--mt_log_interval", default=20, type=int)

    return parser


def _load_model_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    unexpected_keys = [
        key
        for key in unexpected_keys
        if not (key.endswith("total_params") or key.endswith("total_ops"))
    ]
    if missing_keys:
        print("Missing keys when loading pretrained detector:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys when loading pretrained detector:", unexpected_keys)


def _match_name_keywords(name, keywords):
    return any(keyword in name for keyword in keywords)


def _build_optimizer(model, args):
    parameter_groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not _match_name_keywords(name, args.lr_backbone_names)
                and not _match_name_keywords(name, args.lr_linear_proj_names)
                and parameter.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if _match_name_keywords(name, args.lr_backbone_names)
                and parameter.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if _match_name_keywords(name, args.lr_linear_proj_names)
                and parameter.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]

    if args.sgd:
        return torch.optim.SGD(
            parameter_groups,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )

    return torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def main(args):
    # This Mean Teacher implementation is intentionally single-process.
    # Do NOT call utils.init_distributed_mode(args) here: an sbatch/srun job
    # defines SLURM_PROCID, which would make the original DETR utility enter
    # torch.distributed.init_process_group(init_method="env://") and wait for
    # ranks that are not launched.
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.gpu = 0

    print("Running Mean Teacher in single-process mode on", args.device)
    if args.mt_amp and not args.mt_no_amp:
        print(
            "WARNING: AMP was explicitly enabled. The default MSDeformAttn "
            "CUDA extension normally requires FP32 and may fail on Half tensors."
        )
    else:
        print("AMP disabled: training MSDeformAttn in FP32.")
    print(args)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model, criterion, postprocessors = build_model(args)
    _load_model_checkpoint(model, args.pretrained)
    model.to(device)
    criterion.to(device)

    number_of_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print("number of params:", number_of_parameters)

    # The target annotations returned by this dataset are intentionally ignored.
    # A COCO JSON containing only image records is sufficient if build_dataset
    # permits images with zero annotations.
    dataset_train = build_dataset(image_set="train", args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=mean_teacher_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = _build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    allowed_class_ids = (
        None
        if args.mt_person_class_id is None
        else (args.mt_person_class_id,)
    )
    mt_config = MeanTeacherConfig(
        score_threshold=args.mt_score_threshold,
        nms_iou_threshold=args.mt_nms_iou_threshold,
        min_box_size=args.mt_min_box_size,
        max_detections_per_image=args.mt_max_detections,
        allowed_class_ids=allowed_class_ids,
        ema_decay=args.mt_ema_decay,
        ema_warmup=not args.mt_no_ema_warmup,
        horizontal_flip_probability=args.mt_flip_probability,
        weak_brightness=args.mt_weak_brightness,
        weak_contrast=args.mt_weak_contrast,
        weak_saturation=args.mt_weak_saturation,
        strong_brightness=args.mt_strong_brightness,
        strong_contrast=args.mt_strong_contrast,
        strong_saturation=args.mt_strong_saturation,
        strong_grayscale_probability=args.mt_grayscale_probability,
        strong_blur_probability=args.mt_blur_probability,
        strong_noise_std=args.mt_noise_std,
        gradient_clip_norm=args.clip_max_norm,
        skip_images_without_pseudo_labels=not args.mt_keep_empty_images,
        amp=args.mt_amp and not args.mt_no_amp,
        log_interval=args.mt_log_interval,
        checkpoint_interval_epochs=1,
    )

    adapter = DeformableDETRAdapter(
        criterion=criterion,
        postprocessors=postprocessors,
    )
    augmenter = ImageNetNormalizedPairedAugmenter(mt_config)

    trainer = MeanTeacherDetectionTrainer(
        student=model,
        adapter=adapter,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=mt_config,
        augmenter=augmenter,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume, strict=True)
        print(
            f"Resumed Mean Teacher checkpoint at epoch {trainer.current_epoch}, "
            f"step {trainer.global_step}."
        )

    checkpoint_dir = args.output_dir if args.output_dir else None
    print("Start Mean Teacher adaptation")
    trainer.fit(
        dataloader=data_loader_train,
        epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
    )

    if checkpoint_dir:
        final_path = Path(checkpoint_dir) / "mean_teacher_final.pth"
        trainer.save_checkpoint(final_path)
        print("Saved final Mean Teacher checkpoint to", final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Deformable DETR Mean Teacher training script",
        parents=[get_args_parser()],
    )
    args = parser.parse_args()
    args.output_dir = os.path.join(config.snapshot_dir, args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
