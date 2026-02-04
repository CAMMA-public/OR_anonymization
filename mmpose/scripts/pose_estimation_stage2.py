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
    vid_dir = '/media/chen/hdd1/chen/datasets/BERN_OR_VIDEOS/cut_CH3'
    anno_dir = '/media/chen/hdd1/chen/datasets/BERN_OR_VIDEOS/ByteTrack'
    pose_dir = '/media/chen/hdd1/chen/datasets/BERN_OR_VIDEOS/ByteTrack/pose'
    interval = 8
    vid_list = os.listdir(vid_dir)
    for vid_name in vid_list:
        output_path = os.path.join(pose_dir, vid_name.rsplit('.', 1)[0] + '.pkl')
        # if os.path.exists(output_path):
        #     continue
        # ['1000123960242_1_CH1', '1000123960242_2_CH1', '1000124877273_0_CH1', '1000125087258_0_CH1', '1000126775985_0_CH1']: interval = 4
        # The rest: interval = 16

        vid_path = os.path.join(vid_dir, vid_name)
        anno_path = os.path.join(anno_dir, vid_name.rsplit('.', 1)[0] + '.txt')
        if not os.path.exists(anno_path):
            continue

        # read ByteTrack results
        with open(anno_path, 'r') as f:
            data = f.readlines()
        anno_dict = {}
        for line in data:
            parts = line.strip().split(',')
            f_id = int(parts[0])
            t_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = float(parts[6])
            # x1 = max(int(x), 0)
            # y1 = max(int(y), 0)
            # x2 = min(int(x + w), width)
            # y2 = min(int(y + h), height)

            if f_id not in anno_dict:
                anno_dict[f_id] = []
            anno_dict[f_id].append([f_id, t_id, x, y, x + w, y + h, score])

        # read video frame and estimate poses
        cap = cv2.VideoCapture(vid_path)
        count = 0
        final_output = {}
        while 1:
            ret, frame = cap.read()
            if ret:
                if count % interval != 0:
                    count += 1
                    continue
                if count % 100 == 0:
                    print(vid_name + ' Frame: ' + str(count), flush=True)
                if count in anno_dict:
                    lines = anno_dict[count]
                    bboxes = []
                    for line in lines:
                        x1, y1, x2, y2 = line[2], line[3], line[4], line[5]
                        bboxes.append([x1, y1, x2, y2])
                    start = time.time()
                    pred_instances = process_one_image(args, frame, bboxes, pose_estimator, None)
                    end = time.time()
                    print(interval / (end - start), flush=True)

                    if count not in final_output:
                        final_output[count] = []
                    for i, line in enumerate(lines):
                        new_line = line.copy()
                        keypoints = pred_instances.keypoints[i].tolist()
                        for keypoint in keypoints:
                            new_line += keypoint
                        keypoint_scores = pred_instances.keypoint_scores[i].tolist()
                        new_line += keypoint_scores
                        final_output[count].append(new_line)
                        # print(new_line)
                        # exit(1)
                count += 1
            else:
                break
        
        with open(output_path, 'wb') as handle:
            pickle.dump(final_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

                # if output_file:
                #     img_vis = visualizer.get_image()
                #     mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)


if __name__ == '__main__':
    main()
