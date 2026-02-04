import os
from config import config, update_config
# os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_GPUS
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import OR_Loader
from model import *
import logging
import time
from utils import *
from torchvision.ops import nms
import numpy as np
import argparse
import pprint
import json
import pickle
import lmdb
from torchreid.utils import FeatureExtractor
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms as T
import cv2


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    parser.add_argument(
        "--path", default="/media/camma-monitor/Storage_postprocessing2/pose_detection/detections/008_OT_JPG_iter2.lmdb", type=str, help="path to detections"
    )
    parser.add_argument(
        "--tracking_path", default="/media/camma-monitor/Storage_postprocessing2/pose_detection/tracking/008_OT_JPG_iter2_tracking.lmdb", type=str, help="path to detections"
    )
    parser.add_argument(
        "--save_dir", default="/media/camma-monitor/Storage_postprocessing2/pose_detection", type=str, help="path to output"
    )
    parser.add_argument(
        "--save_suffix", default="_mva_tracking.lmdb", type=str, help="path to output"
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
    
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args

def model_forward(args, model, reid_model, device, img_key, boxes, scores, height, width, c_id, dino):
    input_boxes = torch.from_numpy(np.copy(boxes)).float()

    if reid_model is not None:
        anchor_img_path = os.path.join(args.root_dir, img_key)
        img = cv2.imread(anchor_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = T.ToTensor()(img)  # (C, H, W), float
        cropped_imgs = torch.zeros(len(input_boxes), 3, 224, 224, dtype=torch.float32)
        for i, box in enumerate(input_boxes):
            y1 = max(int(box[1]), 0)
            y2 = int(box[3])
            x1 = max(int(box[0]), 0)
            x2 = int(box[2])
            cropped_img = img[:, y1:y2, x1:x2]
            cropped_img = T.Resize((224, 224))(cropped_img)
            cropped_img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(cropped_img)
            cropped_imgs[i] = cropped_img
        cropped_imgs = cropped_imgs.to(device)
        with torch.no_grad():
            if dino:
                anchor_reid = reid_model(cropped_imgs).last_hidden_state.mean(dim=1)
            else:
                anchor_reid = reid_model(cropped_imgs)

    # normalization
    input_boxes[:, 0] /= width
    input_boxes[:, 1] /= height
    input_boxes[:, 2] /= width
    input_boxes[:, 3] /= height
    input_boxes = input_boxes.to(device)
    cam_batch = torch.ones(len(input_boxes), dtype=torch.int64, device=device) * c_id
    with torch.no_grad():
        feas, preds = model.module.encode_decode(cam_batch, (height, width), input_boxes)
    
    if reid_model is not None:
        return [boxes, scores, feas, anchor_reid]
    else:
        return [boxes, scores, feas]


def fix_boxes(boxes, scores):
    if len(boxes):
        filter1 = boxes[:, 1] < boxes[:, 3]
        boxes = boxes[filter1]
        scores = scores[filter1]
    if len(boxes):
        filter2 = boxes[:, 0] < boxes[:, 2]
        boxes = boxes[filter2]
        scores = scores[filter2]
    return boxes, scores


def inference(args):
    gpus = [int(i) for i in config.GPUS.split(",")]
    device = torch.device('cuda')
    train_mode = config.TRAIN.TRAIN_MODE
    ssl_prediction = config.SSL_PREDICTION
    depth = config.DEPTH
    multi_view_model = True
    fully_supervised = config.TRAIN.FULLY_SUPERVISED
    thresh = config.TEST.THRESH
    batch_size = config.TRAIN.BATCH_SIZE
    fix_view_id = config.FIX_VIEW_ID
    test_adaptive = False
    reid = config.TRAIN.REID
    dino = config.TRAIN.DINO
    vis = config.TEST.VIS
    alpha = config.DATASET.ALPHA
    view_num = len(config.DATASET.VIEW_IDS)

    reid_model = None
    if reid:
        if dino:
            processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            reid_model = AutoModel.from_pretrained('facebook/dinov2-base')
            reid_model.eval()
            reid_model.to(device)
        else:
            reid_model = FeatureExtractor(
                model_name='osnet_ain_x1_0',
                model_path='weights/osnet_ain_ms_d_c.pth.tar',
                device='cuda'
            )

    if fully_supervised:
        if ssl_prediction:
            model = Multi_View_Predictor_prompt(view_num, mode=config.DATASET.PROMPT_MODE)
            # model = Multi_View_Predictor(view_num)
        else:
            model = FeaturesDeitBase_P3DE(view_num)
    elif train_mode == 'triplet':
        if ssl_prediction:
            if multi_view_model:
                # model = Multi_View_Predictor_bbox(view_num)
                model = Multi_View_Predictor_prompt(view_num, mode=config.DATASET.PROMPT_MODE)
                # model = FeaturesDeitBase_P3DE_Predictor(view_num)
        else:
            if depth:
                model = FeaturesDeitBase_P3DE_depth(view_num)
            else:
                model = FeaturesDeitBase_P3DE(view_num)
                # model = Multi_View_Predictor_P3DE(view_num)
    
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    checkpoint_path = config.TEST.CKP_PATH
    ckp = torch.load(checkpoint_path)
    model.module.load_state_dict(ckp['model'])
    model.eval()
    
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
    img_names = set()
    for img_key in img_keys:
        sub_dir, img_name = img_key.split('/')
        if img_name not in img_names:
            img_names.add(img_name)
    
    base_name = args.path.rsplit(os.sep, 1)[1].rsplit('.', 1)[0]
    mva_dir = os.path.join(args.save_dir, 'mva')
    os.makedirs(mva_dir, exist_ok=True)
    mva_path = os.path.join(mva_dir, base_name + args.save_suffix)
    
    demo_img_dir = os.path.join(args.root_dir, sub_dirs[0])
    demo_img_names = os.listdir(demo_img_dir)
    for demo_img_name in demo_img_names:
        if demo_img_name[-3:] != 'jpg' or demo_img_name[0] != 'c':
            continue
        demo_img_path = os.path.join(demo_img_dir, demo_img_name)
        demo_img = cv2.imread(demo_img_path)
        height, width, _ = demo_img.shape
        break
    
    # full_query = True >>> using both tracking and detection boxes as assicoation queries
    full_query = False 
    
    # maximum 10GB
    map_size = int(10 * 1024 * 1024 * 1024)
    flush_interval = 1000
    
    env = lmdb.open(args.path, readonly=True, lock=False, subdir=False)
    tracking_env = lmdb.open(args.tracking_path, readonly=True, lock=False, subdir=False)
    mva_env = lmdb.open(mva_path, map_size=map_size, subdir=False, meminit=False, max_dbs=0)
    with env.begin() as txn:
        # Use a cursor for efficient iteration if desired.
        cursor = txn.cursor()
        with tracking_env.begin() as tracking_txn:
            # Use a cursor for efficient iteration if desired.
            tracking_cursor = tracking_txn.cursor()
        
            count = 0
            mva_txn = mva_env.begin(write=True)
            for img_name in tqdm(img_names):
                run_flag = False
                for c_id, sub_dir in enumerate(sub_dirs):
                    img_key = os.path.join(sub_dir, img_name)
                    key_boxes = f"{img_key}/boxes".encode('utf-8')
                    key_scores = f"{img_key}/scores".encode('utf-8')
                    # Check if this record already exists (for resume capability)
                    if mva_txn.get(key_boxes) is None or mva_txn.get(key_scores) is None:
                        run_flag = True
                        break
                if not run_flag:
                    continue
                all_features = {}
                all_tracking_features = {}
                for c_id, sub_dir in enumerate(sub_dirs):
                    img_dir = os.path.join(args.root_dir, sub_dir)
                    img_key = os.path.join(sub_dir, img_name)
                    key_boxes = f"{img_key}/boxes".encode('utf-8')
                    key_scores = f"{img_key}/scores".encode('utf-8')
                    boxes_data = txn.get(key_boxes)
                    scores_data = txn.get(key_scores)
                    run_det_box = False
                    if boxes_data is not None and scores_data is not None:
                        # Deserialize using pickle.
                        boxes = pickle.loads(boxes_data)
                        scores = pickle.loads(scores_data)
                        boxes, scores = fix_boxes(boxes, scores)
                        if len(boxes):
                            run_det_box = True
                    
                    tracking_boxes_data = tracking_txn.get(key_boxes)
                    tracking_scores_data = tracking_txn.get(key_scores)
                    run_tracking_box = False
                    if tracking_boxes_data is not None and tracking_scores_data is not None:
                        # Deserialize using pickle.
                        tracking_boxes = pickle.loads(tracking_boxes_data)
                        tracking_scores = pickle.loads(tracking_scores_data)
                        if len(tracking_boxes):
                            run_tracking_box = True
                    
                    if run_det_box and run_tracking_box:
                        all_boxes = np.concatenate([boxes, tracking_boxes], axis=0)
                        all_scores = np.concatenate([scores, tracking_scores], axis=0)
                        all_output = model_forward(args, model, reid_model, device, img_key, all_boxes, all_scores, height, width, c_id, dino)
                        if full_query:
                            all_features[c_id] = all_output
                        else:
                            all_features[c_id] = [item[:len(boxes)] for item in all_output]
                        all_tracking_features[c_id] = [item[len(boxes):] for item in all_output]
                    elif run_det_box:
                        all_output = model_forward(args, model, reid_model, device, img_key, boxes, scores, height, width, c_id, dino)
                        all_features[c_id] = all_output
                    elif run_tracking_box:
                        all_tracking_output = model_forward(args, model, reid_model, device, img_key, tracking_boxes, tracking_scores, height, width, c_id, dino)
                        all_tracking_features[c_id] = all_tracking_output
                        
                
                selected_indexes = {}
                for c_id, sub_dir in enumerate(sub_dirs):
                    if c_id not in all_features or c_id not in all_tracking_features:
                        continue
                    boxes = all_features[c_id][0]
                    scores = all_features[c_id][1]
                    feas = all_features[c_id][2]
                    
                    anchor_feas = all_tracking_features[c_id][2]
                    anchor_feas_reid = None
                    if reid_model is not None:
                        feas_reid = all_features[c_id][3]
                        anchor_feas_reid = all_tracking_features[c_id][3]
                    if len(anchor_feas):
                        # base_indexes = np.arange(len(boxes) - len(anchor_feas), len(boxes))
                        # if c_id not in selected_indexes:
                        #     selected_indexes[c_id] = base_indexes
                        # else:
                        #     new_indexes = np.concatenate([selected_indexes[c_id], base_indexes], axis=None)
                        #     selected_indexes[c_id] = new_indexes
                        
                        for sample_c_id, sample_sub_dir in enumerate(sub_dirs):
                            if sample_c_id == c_id or sample_c_id not in all_features:
                                continue
                            sample_boxes = all_features[sample_c_id][0]
                            sample_scores = all_features[sample_c_id][1]
                            sample_feas = all_features[sample_c_id][2]
                            sample_feas_reid = None
                            if reid_model is not None:
                                sample_feas_reid = all_features[sample_c_id][3]
                            matches_scores, matches = get_matching_dis(anchor_feas, sample_feas, reid_fea1=anchor_feas_reid, reid_fea2=sample_feas_reid, mode='l2', thresh=0.0, alpha=alpha, return_score=True)
                            matches_x, matches_y = matches
                            matches_scores = matches_scores.cpu().numpy()
                            if sample_c_id not in selected_indexes:
                                selected_indexes[sample_c_id] = (matches_y, matches_scores)
                            else:
                                existing_matches_y, existing_matches_scores = selected_indexes[sample_c_id]
                                new_indexes = np.concatenate([existing_matches_y, matches_y], axis=None)
                                new_scores = np.concatenate([existing_matches_scores, matches_scores], axis=None)
                                selected_indexes[sample_c_id] = (new_indexes, new_scores)
                            
                            # true_or_false = np.ones(len(matches_x))
                            # anchor_img_path = os.path.join(args.root_dir, sub_dir, img_name)
                            # anchor_save_path = os.path.join('debug', img_name.rsplit('.', 1)[0] + str(c_id) + str(sample_c_id) + '_anchor')
                            # visualize(anchor_img_path, anchor_save_path, true_or_false, matches_x, all_tracking_features[c_id][0], scores=matches_scores, normalize=False)
                            # sample_img_path = os.path.join(args.root_dir, sample_sub_dir, img_name)
                            # sample_save_path = os.path.join('debug', img_name.rsplit('.', 1)[0] + str(c_id) + str(sample_c_id) + '_sample')
                            # visualize(sample_img_path, sample_save_path, true_or_false, matches_y, sample_boxes, scores=matches_scores, normalize=False)
                
                for c_id, sub_dir in enumerate(sub_dirs):
                    # if c_id not in selected_indexes or c_id not in all_features:
                    #     continue
                    # final_indexes = np.unique(selected_indexes[c_id])
                    # boxes = all_features[c_id][0]
                    # scores = all_features[c_id][1]
                    # final_boxes = boxes[final_indexes]
                    # final_scores = scores[final_indexes]
                    
                    tracking_boxes = None
                    tracking_scores = None
                    if c_id in all_tracking_features:
                        tracking_boxes = all_tracking_features[c_id][0]
                        tracking_scores = all_tracking_features[c_id][1]
                    
                    matched_boxes = None
                    matched_scores = None
                    mva_matches_scores = None
                    if c_id in selected_indexes and c_id in all_features:
                        all_indexes, all_scores = selected_indexes[c_id]
                        # final_indexes = np.unique(all_indexes)
                        data_dict = {}
                        for i in range(len(all_indexes)):
                            if all_indexes[i] not in data_dict:
                                data_dict[all_indexes[i]] = all_scores[i]
                            else:
                                if all_scores[i] > data_dict[all_indexes[i]]:
                                    data_dict[all_indexes[i]] = all_scores[i]
                        final_indexes = []
                        final_scores = []
                        for key, value in data_dict.items():
                            # if value >= 0.3:
                            final_indexes.append(key)
                            final_scores.append(value)
                        final_scores = np.array(final_scores)
                        mva_matches_scores = final_scores
                        
                        boxes = all_features[c_id][0]
                        scores = all_features[c_id][1]
                        matched_boxes = boxes[final_indexes]
                        matched_scores = scores[final_indexes]
                    
                    if tracking_boxes is not None and matched_boxes is not None:
                        final_boxes = np.concatenate([matched_boxes, tracking_boxes], axis=0)
                        final_scores = np.concatenate([matched_scores, tracking_scores], axis=0)
                        left_mva_tracking_scores = np.ones(len(tracking_scores), dtype=float) * -1
                        final_matching_scores = np.concatenate([mva_matches_scores, left_mva_tracking_scores], axis=0)
                    elif tracking_boxes is not None:
                        final_boxes = tracking_boxes
                        final_scores = tracking_scores
                        final_matching_scores = np.ones(len(tracking_scores), dtype=float) * -1
                    elif matched_boxes is not None:
                        final_boxes = matched_boxes
                        final_scores = matched_scores
                        final_matching_scores = mva_matches_scores
                    else:
                        continue
                    torch_boxes = torch.from_numpy(final_boxes).float()
                    torch_score = torch.from_numpy(final_scores).float()
                    indices = nms(torch_boxes, torch_score, 0.9)
                    indices = indices.numpy()
                    
                    new_boxes = final_boxes[indices]
                    new_boxes_score = final_scores[indices]
                    new_matching_scores = final_matching_scores[indices]
                    
                    img_key = os.path.join(sub_dir, img_name)
                    
                    # img_path = os.path.join(args.root_dir, img_key)
                    # img = cv2.imread(img_path)
                    # for line in final_boxes:
                    #     cv2.rectangle(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), color=(0, 255, 0), thickness=2)
                    # save_dir = os.path.join('debug', sub_dir)
                    # os.makedirs(save_dir, exist_ok=True)
                    # save_path = os.path.join(save_dir, img_name)
                    # cv2.imwrite(save_path, img)
                    
                    
                    key_boxes = f"{img_key}/boxes".encode('utf-8')
                    key_scores = f"{img_key}/scores".encode('utf-8')
                    key_matching_scores = f"{img_key}/matching_scores".encode('utf-8')
                    mva_txn.put(key_boxes, pickle.dumps(new_boxes))
                    mva_txn.put(key_scores, pickle.dumps(new_boxes_score))
                    mva_txn.put(key_matching_scores, pickle.dumps(new_matching_scores))
                
                count += 1
                if count % flush_interval == 0:
                    mva_txn.commit()
                    mva_txn = mva_env.begin(write=True)
            mva_txn.commit()
    env.close()
    tracking_env.close()
    mva_env.close()


if __name__ == '__main__':
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, "inference")
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    inference(args)