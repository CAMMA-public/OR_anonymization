# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
from collections import defaultdict

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

import time

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

import pickle
import h5py

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


def process_one_image(img,
                      bboxes,
                      pose_estimator):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict keypoints
    
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    # start = time.time()
    data_samples = merge_data_samples(pose_results)
    # end = time.time()
    # print(end - start, flush=True)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        "--path", default="../ByteTrack/001_OF_JPG_tracking_prompted.json", type=str, help="path to detections"
    )
    parser.add_argument(
        "--anno_path", default='/home/kchen/code/eyes_annotation/face_annotations_v2/001_OF_JPG.json', type=str, help="path to images"
    )
    parser.add_argument(
        "--root_dir", default='/media/camma-monitor/Storage_backup/001_OF_JPG', type=str, help="path to images"
    )
    parser.add_argument(
        "--sub_dirs", default=['117222250956', '309622300656', '309622301491'], type=list, help="path to images"
    )
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--box_thresh',
        type=float,
        default=0.5,
        help='Bounding box score threshold')
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
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    dataset_id = 6
    img_dicts_pkl = f'/home2020/home/icube/keqichen/datasets/4D-OR/export_holistic_take{dataset_id}_processed/img_dicts.pkl'
    with open(img_dicts_pkl, mode='rb') as f:
        img_dicts = pickle.load(f)
    cam_num = len(img_dicts)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=False))))
    
    with open(args.path, 'r') as fp:
        frames = json.load(fp)
    # frames_id = list(map(int, frames.keys()))
    
    # prepare ground-truth data
    with open(args.anno_path, 'r') as fp:
        gt = json.load(fp)
    img_keys = list(gt.keys())
    
    preds_dicts = {}
    for _, img_dict in enumerate(img_dicts):
        for f_id, img_name in img_dict.items():
            if img_name not in img_keys:
                continue
            if str(f_id) in frames:
                nodes = frames[str(f_id)]
                for node in nodes:
                    x, y, w, h, t_id, c_id, score = node
                    if score > args.box_thresh:
                        if c_id not in preds_dicts:
                            preds_dicts[c_id] = defaultdict(list)
                        preds_dicts[c_id][f_id].append([x, y, x + w, y + h, score])
    cams_id = list(preds_dicts.keys())
    
    # outputs = {}
    file = h5py.File(os.path.basename(args.path).rsplit('.', 1)[0] + '.hdf5', 'w')
    for c_id in cams_id:
        output = {}
        # img_dir = os.path.join(args.root_dir, args.sub_dirs[c_id])
        # print(img_dir, flush=True)
        img_dir = args.root_dir
        preds_dict = preds_dicts[c_id]
        for f_id, lines in preds_dict.items():
            print(f_id, flush=True)
            if len(lines):
                img_name = img_dicts[c_id][f_id]
                # img_name = 'color-' + str(f_id).zfill(8) + '.jpg'
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imread(img_path)
                
                boxes = []
                boxes_score = []
                for line in lines:
                    x1, y1, x2, y2, score = line
                    boxes.append([x1, y1, x2, y2])
                    boxes_score.append(score)
                
                pred_instances = process_one_image(img, boxes, pose_estimator)
                
                fgroup = file.create_group(img_name)
                
                save_boxes = np.array(boxes, dtype=int)
                save_boxes_score = np.array(boxes_score, dtype=float)
                save_kpts = np.zeros((len(boxes), 133, 2), dtype=int)
                save_kpts_score = np.zeros((len(boxes), 133), dtype=float)
                for i, box in enumerate(boxes):
                    save_kpts[i] = pred_instances.keypoints[i]
                    save_kpts_score[i] = pred_instances.keypoint_scores[i]
                
                fgroup.create_dataset('boxes', data=save_boxes)
                fgroup.create_dataset('boxes_score', data=save_boxes_score)
                fgroup.create_dataset('kpts', data=save_kpts)
                fgroup.create_dataset('kpts_score', data=save_kpts_score)
                
                # if f_id not in output:
                #     output[f_id] = []
                # for i, box in enumerate(boxes):
                #     data = box.copy()
                #     keypoints = pred_instances.keypoints[i].flatten().tolist() # shape: (133, 2)
                #     data += keypoints
                #     keypoint_scores = pred_instances.keypoint_scores[i].tolist() # shape: (133)
                #     data += keypoint_scores
                #     output[f_id].append(data)
        # outputs[args.sub_dirs[c_id]] = output
        
    file.close()
    # with open(os.path.basename(args.root_dir) + '_baseline.json', 'w') as fp:
    #     json.dump(outputs, fp)


if __name__ == '__main__':
    main()
