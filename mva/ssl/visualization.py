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
import lmdb
import pickle
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms as T

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    parser.add_argument(
        "--path", default="/media/camma-monitor/Storage_postprocessing2/pose_detection/detections/005_CG_JPG.lmdb", type=str, help="path to detections"
    )
    parser.add_argument(
        "--anno_path", default='/media/camma-monitor/Storage_postprocessing2/pose_detection_utils/code/eyes_annotation/face_annotations_v2/005_CG.json', type=str, help="path to images"
    )
    parser.add_argument(
        "--root_dir", default='/media/camma-monitor/Storage_postprocessing2/005_CG_JPG/005_CG', type=str, help="path to images"
    )
    parser.add_argument(
        "--image_shape", default=[480, 853], type=list, help="image shape"
    )
    parser.add_argument(
        '--box_thresh',
        type=float,
        default=0.5,
        help='Bounding box score threshold')

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


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
    vis = config.TEST.VIS
    reid = config.TRAIN.REID
    view_num = len(config.DATASET.VIEW_IDS)
    
    reid_model = None
    if reid:
        # reid_model = FeatureExtractor(
        #     model_name='osnet_ain_x1_0',
        #     model_path='weights/osnet_ain_ms_d_c.pth.tar',
        #     device='cuda'
        # )
        # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        reid_model = AutoModel.from_pretrained('facebook/dinov2-base')
        reid_model.eval()
        reid_model.to(device)
        

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
    ckp = torch.load(checkpoint_path, weights_only=True)
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
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join('vis_pred', sub_dir), exist_ok=True)
        
    # prepare ground-truth data
    with open(args.anno_path, 'r') as fp:
        gt = json.load(fp)
    img_keys = list(gt.keys())
    
    final_dict = {}
    
    height, width = args.image_shape
    env = lmdb.open(args.path, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        # Use a cursor for efficient iteration if desired.
        cursor = txn.cursor()
        
        count = 0
        for img_key in img_keys:
            key_boxes = f"{img_key}/boxes".encode('utf-8')
            key_scores = f"{img_key}/scores".encode('utf-8')
            boxes_data = txn.get(key_boxes)
            scores_data = txn.get(key_scores)
            if boxes_data is not None and scores_data is not None:
                boxes = pickle.loads(boxes_data)
                scores = pickle.loads(scores_data)
                mask = scores > 0.6
                left_boxes = boxes[mask]
                
                if not len(left_boxes):
                    continue
            
                sub_dir, img_name = img_key.rsplit('/', 1)
                prefix = img_name.rsplit('.', 1)[0]
                f_id = int(prefix.rsplit('-', 1)[1])
                anchor_img_path = os.path.join(args.root_dir, img_key)
                input_boxes = torch.from_numpy(left_boxes[:, :4]).clone()
                
                if reid_model is not None:
                    img = cv2.imread(anchor_img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = T.ToTensor()(img)  # (C, H, W), float
                    
                    cropped_imgs = torch.zeros(len(input_boxes), 3, 224, 224, dtype=torch.float32)
                    for i, box in enumerate(input_boxes):
                        cropped_img = img[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        cropped_img = T.Resize((224, 224))(cropped_img)
                        cropped_img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(cropped_img)
                        cropped_imgs[i] = cropped_img
                    cropped_imgs = cropped_imgs.to(device)
                    
                    with torch.no_grad():
                        anchor_reid = reid_model(cropped_imgs).last_hidden_state.mean(dim=1)
                
                # normalization
                input_boxes[:, 0] /= width
                input_boxes[:, 1] /= height
                input_boxes[:, 2] /= width
                input_boxes[:, 3] /= height
                cam_index = sub_dirs.index(sub_dir)
                cam_batch = torch.ones(len(input_boxes), dtype=torch.int64) * cam_index
                input_boxes = input_boxes.to(device)
                cam_batch = cam_batch.to(device)
                with torch.no_grad():
                    _, preds = model.module.encode_decode(cam_batch, (height, width), input_boxes, anchor_reid)
                # preds = preds.cpu().detach().numpy()
                
                preds[:, :, :, 0] *= width
                preds[:, :, :, 1] *= height
                preds[:, :, :, 2] *= width
                preds[:, :, :, 3] *= height
                
                for i in range(len(cam_batch)):
                    cam_id = cam_batch[i].item()
                    color = get_color(0)
                    anchor_img = cv2.imread(anchor_img_path)
                    cv2.rectangle(anchor_img, [int(left_boxes[i, 0]), int(left_boxes[i, 1])], [int(left_boxes[i, 2]), int(left_boxes[i, 3])], color=color, thickness=3)
                    anchor_save_path = os.path.join('vis_pred', sub_dir, prefix + '_anchor_' + str(i) + '.jpg')
                    cv2.imwrite(anchor_save_path, anchor_img)
                    for c in range(view_num):
                        if c == cam_id:
                            continue
                        anchor_img = cv2.imread(anchor_img_path)
                        
                        new_sub_dir = sub_dirs[c]
                        pred_img_path = os.path.join(args.root_dir, new_sub_dir, img_name)
                        pred_img = cv2.imread(pred_img_path)
                        
                        prompts = preds[i, c]
                        for prompt in prompts:
                            cv2.rectangle(pred_img, [int(prompt[0]), int(prompt[1])], [int(prompt[2]), int(prompt[3])], color=color, thickness=3)
                        pred_save_path = os.path.join('vis_pred', sub_dir, prefix + '_pred_' + str(i) + '_' + str(c) + '.jpg')
                        cv2.imwrite(pred_save_path, pred_img)
    env.close()
        


if __name__ == '__main__':
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, "inference")
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    inference(args)