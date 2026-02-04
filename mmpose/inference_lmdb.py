# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm

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
import lmdb
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
        "--path", default="/media/camma-monitor/Storage_postprocessing2/pose_detection/detections/008_OT_JPG_temp.lmdb", type=str, help="path to detections"
    )
    parser.add_argument(
        "--save_dir", default="/media/camma-monitor/Storage_postprocessing2/pose_detection", type=str, help="path to output"
    )
    # /media/camma-monitor/Storage_postprocessing2/005_CG_JPG/005_CG
    # /media/camma-monitor/Storage_postprocessing2/006_RS_JPG/006_RS
    # /media/camma-monitor/Storage_postprocessing3/007_CL_JPG
    # /media/camma-monitor/Storage_postprocessing3/008_OT_JPG
    # /media/camma-monitor/Storage_postprocessing3/009_SV_JPG
    parser.add_argument(
        "--root_dir", default='/media/camma-monitor/Storage_postprocessing3/008_OT_JPG', type=str, help="path to images"
    )
    parser.add_argument(
        "--anno_path", default='/media/camma-monitor/Storage_postprocessing2/pose_detection_utils/code/anonymization/evaluation/annotations/008_OT_JPG.json', type=str, help="path to images"
    )
    parser.add_argument(
        "--save_suffix", default="_partial.lmdb", type=str, help="path to output"
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

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=False))))
    
    sub_dirs = []
    for dir_name in os.listdir(args.root_dir):
        if dir_name == '547758':
            continue
        dir_path = os.path.join(args.root_dir, dir_name)
        if os.path.isdir(dir_path):
            sub_dirs.append(dir_name)
    sub_dirs.sort()
    
    with open(args.anno_path, 'r') as fp:
        gt = json.load(fp)
    img_keys = list(gt.keys())
    
    base_name = args.path.rsplit(os.sep, 1)[1].rsplit('.', 1)[0]
    pose_dir = os.path.join(args.save_dir, 'pose')
    os.makedirs(pose_dir, exist_ok=True)
    pose_path = os.path.join(pose_dir, base_name + '_pose' + args.save_suffix)
    face_dir = os.path.join(args.save_dir, 'face')
    os.makedirs(face_dir, exist_ok=True)
    face_path = os.path.join(face_dir, base_name + '_face' + args.save_suffix)
    
    # maximum 10GB
    map_size = int(10 * 1024 * 1024 * 1024)
    flush_interval = 1000
    
    env = lmdb.open(args.path, readonly=True, lock=False, subdir=False)
    pose_env = lmdb.open(pose_path, map_size=map_size, subdir=False, meminit=False, max_dbs=0)
    face_env = lmdb.open(face_path, map_size=map_size, subdir=False, meminit=False, max_dbs=0)
    with env.begin() as txn:
        # Use a cursor for efficient iteration if desired.
        cursor = txn.cursor()
        
        count = 0
        pose_txn = pose_env.begin(write=True)
        face_txn = face_env.begin(write=True)
        for img_key in tqdm(img_keys):
            sub_dir, img_name = img_key.split('/')
            img_dir = os.path.join(args.root_dir, sub_dir)
            key_kpts = f"{img_key}/kpts".encode('utf-8')
            key_kpts_scores = f"{img_key}/kpts_scores".encode('utf-8')
            key_kpts_vis = f"{img_key}/kpts_vis".encode('utf-8')
            key_boxes_scores = f"{img_key}/boxes_scores".encode('utf-8')
            # Check if this record already exists (for resume capability)
            if pose_txn.get(key_kpts_vis) is not None and pose_txn.get(key_kpts) is not None and pose_txn.get(key_kpts_scores) is not None and pose_txn.get(key_boxes_scores) is not None and face_txn.get(key_kpts_vis) is not None and face_txn.get(key_kpts) is not None and face_txn.get(key_kpts_scores) is not None and pose_txn.get(key_boxes_scores) is not None:
                continue  # Skip already written records
            
            key_boxes = f"{img_key}/boxes".encode('utf-8')
            key_scores = f"{img_key}/scores".encode('utf-8')
            boxes_data = txn.get(key_boxes)
            scores_data = txn.get(key_scores)
            if boxes_data is not None and scores_data is not None:
                # Deserialize using pickle.
                boxes = pickle.loads(boxes_data)
                scores = pickle.loads(scores_data)
                if len(boxes):
                    img_path = os.path.join(img_dir, img_name)
                    img = cv2.imread(img_path)
                    pred_instances = process_one_image(img, boxes, pose_estimator)
                    
                    pose_kpts = np.zeros((len(boxes), 17, 2), dtype=float)
                    pose_score = np.zeros((len(boxes), 17), dtype=float)
                    pose_vis = np.zeros((len(boxes), 17), dtype=float)
                    face_kpts = np.zeros((len(boxes), 3, 2), dtype=float)
                    face_score = np.zeros((len(boxes), 3), dtype=float)
                    face_vis = np.zeros((len(boxes), 3), dtype=float)
                    for i, box in enumerate(boxes):
                        pose_kpts[i] = pred_instances.keypoints[i][:17]
                        pose_score[i] = pred_instances.keypoint_scores[i][:17]
                        pose_vis[i] = pred_instances.keypoints_visible[i][:17]
                        face_kpts[i][:2] = pred_instances.keypoints[i][1:3]
                        face_score[i][:2] = pred_instances.keypoint_scores[i][1:3]
                        face_vis[i][:2] = pred_instances.keypoints_visible[i][1:3]
                        face_kpts[i][2] = pred_instances.keypoints[i][31]
                        face_score[i][2] = pred_instances.keypoint_scores[i][31]
                        face_vis[i][2] = pred_instances.keypoints_visible[i][31]
                    
                    # Serialize with pickle.
                    pose_txn.put(key_boxes, pickle.dumps(boxes))
                    pose_txn.put(key_kpts, pickle.dumps(pose_kpts))
                    pose_txn.put(key_boxes_scores, pickle.dumps(scores))
                    pose_txn.put(key_kpts_scores, pickle.dumps(pose_score))
                    pose_txn.put(key_kpts_vis, pickle.dumps(pose_vis))
                    face_txn.put(key_boxes, pickle.dumps(boxes))
                    face_txn.put(key_kpts, pickle.dumps(face_kpts))
                    face_txn.put(key_boxes_scores, pickle.dumps(scores))
                    face_txn.put(key_kpts_scores, pickle.dumps(face_score))
                    face_txn.put(key_kpts_vis, pickle.dumps(face_vis))
                    
                    key_matching_scores = f"{img_key}/matching_scores".encode('utf-8')
                    matching_scores_data = txn.get(key_matching_scores)
                    if matching_scores_data is not None:
                        matching_scores = pickle.loads(matching_scores_data)
                        pose_txn.put(key_matching_scores, pickle.dumps(matching_scores))
                        face_txn.put(key_matching_scores, pickle.dumps(matching_scores))
                    
                    key_tracking_ids = f"{img_key}/tracking_ids".encode('utf-8')
                    tracking_ids_data = txn.get(key_tracking_ids)
                    if tracking_ids_data is not None:
                        tracking_ids = pickle.loads(tracking_ids_data)
                        pose_txn.put(key_tracking_ids, pickle.dumps(tracking_ids))
                        face_txn.put(key_tracking_ids, pickle.dumps(tracking_ids))
                
                count += 1
                if count % flush_interval == 0:
                    pose_txn.commit()
                    pose_txn = pose_env.begin(write=True)
                    face_txn.commit()
                    face_txn = face_env.begin(write=True)
        pose_txn.commit()
        face_txn.commit()
    env.close()
    pose_env.close()
    face_env.close()


if __name__ == '__main__':
    main()
