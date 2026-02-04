import argparse
import os
import os.path as osp
import time
import cv2
import torch
from torchvision.ops import nms
import json
from collections import defaultdict
import numpy as np
import lmdb
import pickle
from tqdm import tqdm

from loguru import logger

from my_tracker import BYTETracker
from timer import Timer


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    # 005_CG_JPG.lmdb
    # 006_RS_JPG.lmdb
    # 007_CL_JPG.lmdb
    # 008_OT_JPG.lmdb
    # 009_SV_JPG.lmdb
    parser.add_argument(
        "--path", default="/media/camma-monitor/Storage_postprocessing2/pose_detection/detections/008_OT_JPG_iter2.lmdb", type=str, help="path to detections"
    )
    parser.add_argument(
        "--experiment_name", default="monitor", help="experiment name"
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
        "--image_shape", default=[480, 853], type=list, help="image shape"
    )
    parser.add_argument(
        "--output_dir", default="./YOLOX_outputs", help="path to images or video"
    )
    parser.add_argument(
        "--save_image",
        action="store_true",
        help="whether to save the inference result of image",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="whether to save the inference result of video",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--fps", default=15, type=int, help="frame rate (fps)")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def multi_imageflow_demo(preds_dict, frame_range, vis_folder, current_time, args, save_name, img_dir):
    min_frame_id, max_frame_id = frame_range
    if vis_folder is not None:
        save_path = osp.join(vis_folder, save_name + ".mp4")
        height, width = args.image_shape
        
        if args.save_image:
            image_save_dir = osp.join(vis_folder, save_name)
            os.makedirs(image_save_dir, exist_ok=True)
        
        if args.save_video:
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (int(width), int(height))
            )

    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    
    final_dict = {}
    
    for frame_id in range(min_frame_id, max_frame_id + 1):
        # img_name = 'color-' + str(frame_id).zfill(8) + '.jpg'
        # img_path = os.path.join(img_dir, img_name)
        # if not os.path.exists(img_path):
        #     continue
        
        if frame_id in preds_dict:
            outputs = preds_dict[frame_id]
        else:
            outputs = []
        if len(outputs):
            outputs = np.array(outputs, dtype=float)
            # online_targets, _ = tracker.update(outputs, img_path)
            online_targets = tracker.update(outputs)
            
            online_results = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                tlbr = t.tlbr
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_results.append(tlbr.tolist() + [float(t.score)])
            if len(online_results):
                if frame_id not in final_dict:
                    final_dict[frame_id] = online_results
                else:
                    final_dict[frame_id] += online_results
            
            # img = cv2.imread(img_path)
            # for line in online_results:
            #     cv2.rectangle(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), color=(0, 255, 0), thickness=2)
            # save_path = os.path.join('debug', img_name)
            # cv2.imwrite(save_path, img)
                    
            timer.toc()
        else:
            timer.toc()
    
    # new_boxes_dict = {}
    # # new_boxes: [x1, y1, x2, y2]
    
    for frame_id in range(max_frame_id, min_frame_id - 1, -1):
        # img_name = 'color-' + str(frame_id).zfill(8) + '.jpg'
        # img_path = os.path.join(img_dir, img_name)
        # if not os.path.exists(img_path):
        #     continue
        
        if frame_id in preds_dict:
            outputs = preds_dict[frame_id]
        else:
            outputs = []
        if len(outputs):
            outputs = np.array(outputs, dtype=float)
            # online_targets, new_boxes = tracker.update(outputs, img_path)
            online_targets = tracker.update(outputs)
            
            online_results = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                tlbr = t.tlbr
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_results.append(tlbr.tolist() + [float(t.score)])
            if len(online_results):
                if frame_id not in final_dict:
                    final_dict[frame_id] = online_results
                else:
                    final_dict[frame_id] += online_results
            timer.toc()
        else:
            timer.toc()
    
    for f_id, all_boxes in final_dict.items():
        if len(all_boxes):
            inputs_all = torch.tensor(all_boxes, dtype=torch.float32)
            inputs_all_boxes = inputs_all[:, :4]
            inputs_all_scores = inputs_all[:, 4]
            
            indices = nms(inputs_all_boxes, inputs_all_scores, 0.6)
            outputs = inputs_all[indices].numpy()
            final_dict[f_id] = outputs
    
    
    # for f_id, new_boxes in new_boxes_dict.items():
    #     if len(new_boxes):
    #         inputs_new = torch.tensor(new_boxes, dtype=torch.float32)
    #         inputs_new_boxes = inputs_new[:, :4]
    #         inputs_new_scores = inputs_new[:, 4]
    #         if f_id in final_dict and len(final_dict[f_id]):
    #             inputs_existing = torch.tensor(final_dict[f_id], dtype=torch.float32)
    #             inputs_existing_boxes = inputs_existing[:, :4]
    #             inputs_existing_scores = torch.ones(len(inputs_existing_boxes), dtype=torch.float32)
    #             inputs_new = torch.cat([inputs_new, inputs_existing], dim=0)
    #             inputs_new_boxes = torch.cat([inputs_new_boxes, inputs_existing_boxes], dim=0)
    #             inputs_new_scores = torch.cat([inputs_new_scores, inputs_existing_scores], dim = 0)
    #         indices = nms(inputs_new_boxes, inputs_new_scores, 0.6)
    #         outputs = inputs_new[indices]
    #         outputs = outputs.tolist()
    #         final_dict[f_id] = outputs
    
    return final_dict

def main(args):
    vis_folder = None
    if args.save_video or args.save_image:
        output_dir = osp.join(args.output_dir, args.experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))
    
    img_dicts_pkl = os.path.join(args.root_dir, 'img_dicts.pkl')
    with open(img_dicts_pkl, mode='rb') as f:
        img_dicts = pickle.load(f)
    cam_num = len(img_dicts)
    
    base_name = args.path.rsplit(os.sep, 1)[1].rsplit('.', 1)[0]
    tracking_dir = os.path.join(args.save_dir, 'tracking')
    os.makedirs(tracking_dir, exist_ok=True)
    tracking_path = os.path.join(tracking_dir, base_name + '_tracking.lmdb')
    
    env = lmdb.open(args.path, readonly=True, lock=False, subdir=False)
    preds_dicts = {}
    for c_id in range(cam_num):
        if c_id not in preds_dicts:
            preds_dicts[c_id] = defaultdict(list)
    min_frame_id = 0
    max_frame_id = 0
    with env.begin() as txn:
        # Use a cursor for efficient iteration if desired.
        cursor = txn.cursor()
        
        count = 0
        for c_id, img_dict in enumerate(img_dicts):
            for f_id, img_name in img_dict.items():
                if f_id > max_frame_id:
                    max_frame_id = f_id
                img_key = str(f_id) + '_' + str(c_id)
                key_boxes = f"{img_key}/boxes".encode('utf-8')
                key_scores = f"{img_key}/scores".encode('utf-8')
                boxes_data = txn.get(key_boxes)
                scores_data = txn.get(key_scores)
                if boxes_data is not None and scores_data is not None:
                    # Deserialize using pickle.
                    boxes = pickle.loads(boxes_data)
                    scores = pickle.loads(scores_data)
                    if len(boxes):
                        boxes_scores = np.concatenate([np.copy(boxes), np.copy(scores)[:, None]], axis=1)
                        preds_dicts[c_id][f_id] = boxes_scores
    env.close()

    current_time = time.localtime()
    
    final_dicts = {}
    for c_id in range(cam_num):
        # if c_id == 1 or c_id == 2:
        #     continue
        img_dir = os.path.join(args.root_dir, 'colorimage')
        final_dict = multi_imageflow_demo(preds_dicts[c_id], [min_frame_id, max_frame_id], vis_folder, current_time, args, str(c_id), img_dir)
        final_dicts[c_id] = final_dict
    # exit(1)
    
    # maximum 10GB
    map_size = int(10 * 1024 * 1024 * 1024)
    flush_interval = 1000
    
    tracking_env = lmdb.open(tracking_path, map_size=map_size, subdir=False, meminit=False, max_dbs=0)
    tracking_txn = tracking_env.begin(write=True)
    count = 0
    for c_id in range(cam_num):
        final_dict = final_dicts[c_id]
        for f_id, preds in final_dict.items():
            img_key = str(f_id) + '_' + str(c_id)
            key_boxes = f"{img_key}/boxes".encode('utf-8')
            key_scores = f"{img_key}/scores".encode('utf-8')
            
            boxes_scores = np.array(preds)
            boxes = boxes_scores[:, :4]
            scores = boxes_scores[:, 4]
            
            tracking_txn.put(key_boxes, pickle.dumps(boxes))
            tracking_txn.put(key_scores, pickle.dumps(scores))
            
            count += 1
            if count % flush_interval == 0:
                tracking_txn.commit()
                tracking_txn = tracking_env.begin(write=True)
    tracking_txn.commit()
    tracking_env.close()
    
    print('All the files saved!')


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
