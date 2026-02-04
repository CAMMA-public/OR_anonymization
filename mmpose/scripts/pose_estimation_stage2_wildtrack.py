# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

import time

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

import pickle
import json

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def split_instances(instances):
    """Convert instances into a list where each element is a dict that contains
    information about one instance."""
    results = []

    # return an empty list if there is no instance detected by the model
    if instances is None:
        return results

    for i in range(len(instances.keypoints)):
        result = dict(
            keypoints=instances.keypoints[i].tolist(),
            keypoint_scores=instances.keypoint_scores[i].tolist(),
        )
        if 'bboxes' in instances:
            result['bbox'] = instances.bboxes[i].tolist(),
            if 'bbox_scores' in instances:
                result['bbox_score'] = instances.bbox_scores[i]
        results.append(result)

    return results


def process_one_image(args,
                      img,
                      bboxes,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    width = 1920
    height = 1080
    img_dir = '/media/chen/hdd1/chen/datasets/wildtrack/sequence1/output/frames'
    anno_dir = '/media/chen/hdd1/chen/datasets/wildtrack/sequence1/output'
    anno_names = ['gt_eval.json', 'gt_test.json', 'gt_train.json']
    cam_num = 7
    for anno_name in anno_names:
        anno_path = os.path.join(anno_dir, anno_name)
        with open(anno_path, 'r') as fp:
            frames = json.load(fp)
        frames_id = list(map(int, frames.keys()))
        final_output = {}
        for f_id in frames_id:
            bboxes = frames[str(f_id)]
            final_output[str(f_id)] = []
            bboxes_dict = {}
            for bbox in bboxes:
                x, y, w, h, tid, cid = bbox
                if cid not in bboxes_dict:
                    bboxes_dict[cid] = []
                bboxes_dict[cid].append([x, y, w, h, tid, cid])
            for c in range(cam_num):
                if c not in bboxes_dict:
                    continue
                input_bboxes = []
                for data in bboxes_dict[c]:
                    x1 = data[0]
                    y1 = data[1]
                    x2 = data[0] + data[2]
                    y2 = data[1] + data[3]
                    input_bboxes.append([x1, y1, x2, y2])
                img_name = str(f_id) + '_' + str(c) + '.jpg'
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imread(img_path)
                start = time.time()
                pred_instances = process_one_image(args, img, input_bboxes, pose_estimator, None)
                end = time.time()
                print(end - start, flush=True)

                for i, data in enumerate(bboxes_dict[c]):
                    new_data = data.copy()
                    keypoints = pred_instances.keypoints[i].tolist()
                    for keypoint in keypoints:
                        new_data += keypoint
                    keypoint_scores = pred_instances.keypoint_scores[i].tolist()
                    new_data += keypoint_scores
                    final_output[str(f_id)].append(new_data)
                    # print(new_data)
                    # exit(1)
        
        output_path = os.path.join(anno_dir, 'pose_' + anno_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        

                # if output_file:
                #     img_vis = visualizer.get_image()
                #     mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)


if __name__ == '__main__':
    main()
