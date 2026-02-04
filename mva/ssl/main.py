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
from scipy.optimize import linear_sum_assignment
from torchvision.ops import roi_align
from torchvision.utils import save_image
import numpy as np
from sklearn.metrics import average_precision_score
import argparse
import pprint
from torchreid.utils import FeatureExtractor
from transformers import AutoImageProcessor, AutoModel


logger = logging.getLogger(__name__)


def fully_supervised_triplet_loss(out1, out2, labels1, labels2, anchor_prompts, pos_prompts, anchor_view_id, sample_view_id, ssl_prediction=False):
    if ssl_prediction:
        fea1, anchor_preds = out1
        fea2, pos_preds = out2
    else:
        fea1 = out1
        fea2 = out2
    anchor_idxes = []
    pos_idxes = []
    neg_idxes = []
    for i, label1 in enumerate(labels1):
        for j, label2 in enumerate(labels2):
            if label1 == label2:
                for k, neg_label2 in enumerate(labels2):
                    if j == k:
                        continue
                    anchor_idxes.append(i)
                    pos_idxes.append(j)
                    neg_idxes.append(k)
    if len(anchor_idxes) == 0:
        return -1
    anchor_fea = fea1[anchor_idxes]
    pos_fea = fea2[pos_idxes]
    neg_fea = fea2[neg_idxes]
    pos_dis = torch.sqrt((anchor_fea - pos_fea).pow(2).sum(-1))
    neg_dis = torch.sqrt((anchor_fea - neg_fea).pow(2).sum(-1))
    M = 1.0
    loss = torch.nn.functional.relu(pos_dis - neg_dis + M).mean()
    if ssl_prediction:
        idxes1 = []
        idxes2 = []
        for i, label1 in enumerate(labels1):
            for j, label2 in enumerate(labels2):
                if label1 == label2:
                    idxes1.append(i)
                    idxes2.append(j)
        
        anchor_loss1 = (anchor_preds[:, anchor_view_id[0]] - anchor_prompts).abs().sum(-1).mean()
        anchor_loss2 = (anchor_preds[idxes1, sample_view_id[0]] - pos_prompts[idxes2]).abs().sum(-1).mean()
        pos_loss1 = (pos_preds[idxes2, anchor_view_id[0]] - anchor_prompts[idxes1]).abs().sum(-1).mean()
        pos_loss2 = (pos_preds[:, sample_view_id[0]] - pos_prompts).abs().sum(-1).mean()

        loss += (anchor_loss1 + anchor_loss2 + pos_loss1 + pos_loss2)
    return loss

max_num_person = 0

def train_p3de(config, epoch, model, dataloader_train, optimizer, device, train_mode='triplet', ssl_prediction=True, multi_view_model=False, fully_supervised=False, thresh=0.0, test_adaptive=False, reid_model=None, dino=False):
    alpha = config.DATASET.ALPHA
    tracking = config.TRAIN.TRACKING

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    epoch_loss = 0
    for step_i, data in enumerate(dataloader_train):
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        if fully_supervised:
            anchor_info = data[0]
            pos_info = data[1]
            anchor_labels = anchor_info['labels']
            anchor_bboxes = anchor_info['bboxes']
            pos_labels = pos_info['labels']
            pos_bboxes = pos_info['bboxes']
            if len(anchor_bboxes) == 0 or len(pos_bboxes) == 0:
                continue
            anchor_prompts = anchor_bboxes.squeeze(0)
            pos_prompts = pos_bboxes.squeeze(0)
            anchor_prompts = anchor_prompts.to(device)
            pos_prompts = pos_prompts.to(device)

            if 'cropped_imgs' in anchor_info:
                anchor_imgs = anchor_info['cropped_imgs']
                pos_imgs = pos_info['cropped_imgs']
                anchor_imgs = anchor_imgs.to(device)
                pos_imgs = pos_imgs.to(device)
                anchor_imgs_input = anchor_imgs.squeeze(0)
                pos_imgs_input = pos_imgs.squeeze(0)

            labels1 = torch.cat(anchor_labels, dim=0)
            labels1 = labels1.to(device)
            labels2 = torch.cat(pos_labels, dim=0)
            labels2 = labels2.to(device)

            anchor_view_id = anchor_info['cam_id']
            sample_view_id = pos_info['cam_id']

            if ssl_prediction:
                anchor_h = anchor_info['height'][0].item()
                anchor_w = anchor_info['width'][0].item()
                pos_h = pos_info['height'][0].item()
                pos_w = pos_info['width'][0].item()

                anchor_out = model.module.encode_decode(anchor_view_id, (anchor_h, anchor_w), anchor_prompts)
                pos_out = model.module.encode_decode(sample_view_id, (pos_h, pos_w), pos_prompts)
            else:
                anchor_out = model(anchor_imgs_input, anchor_prompts, anchor_view_id)
                pos_out = model(pos_imgs_input, pos_prompts, sample_view_id)

            loss = fully_supervised_triplet_loss(anchor_out, pos_out, labels1, labels2, anchor_prompts, pos_prompts, anchor_view_id, sample_view_id, ssl_prediction)
            if loss == -1:
                continue

        elif train_mode == 'triplet':
            anchor_info, pos_info, neg_info = data
            anchor_bboxes = anchor_info['bboxes']
            pos_bboxes = pos_info['bboxes']
            neg_bboxes = neg_info['bboxes']
            if len(anchor_bboxes) == 0 or len(pos_bboxes) == 0 or len(neg_bboxes) == 0:
                continue
            anchor_prompts = anchor_bboxes.squeeze(0)
            pos_prompts = pos_bboxes.squeeze(0)
            neg_prompts = neg_bboxes.squeeze(0)
            anchor_prompts = anchor_prompts.to(device)
            pos_prompts = pos_prompts.to(device)
            neg_prompts = neg_prompts.to(device)
            
            anchor_scores = anchor_info['scores']
            anchor_scores = anchor_scores.squeeze(0)
            pos_scores = pos_info['scores']
            pos_scores = pos_scores.squeeze(0)
            neg_scores = neg_info['scores']
            neg_scores = neg_scores.squeeze(0)
            
            if tracking:
                anchor_tracking_bboxes = anchor_info['tracking_bboxes']
                pos_tracking_bboxes = pos_info['tracking_bboxes']
                neg_tracking_bboxes = neg_info['tracking_bboxes']
                if len(anchor_tracking_bboxes) == 0 or len(pos_tracking_bboxes) == 0 or len(neg_tracking_bboxes) == 0:
                    continue
                anchor_tracking_prompts = anchor_tracking_bboxes.squeeze(0)
                pos_tracking_prompts = pos_tracking_bboxes.squeeze(0)
                neg_tracking_prompts = neg_tracking_bboxes.squeeze(0)
                anchor_tracking_prompts = anchor_tracking_prompts.to(device)
                pos_tracking_prompts = pos_tracking_prompts.to(device)
                neg_tracking_prompts = neg_tracking_prompts.to(device)
            else:
                anchor_mask = anchor_scores > 0.6
                pos_mask = pos_scores > 0.6
                neg_mask = neg_scores > 0.6
                if not len(anchor_prompts[anchor_mask]) or not len(pos_prompts[pos_mask]) or not len(neg_scores[neg_mask]):
                    continue
                    
            use_center_loss = False
            if 'center_bboxes' in anchor_info:
                anchor_center_bboxes = anchor_info['center_bboxes']
                pos_center_bboxes = pos_info['center_bboxes']
                neg_center_bboxes = neg_info['center_bboxes']
                if len(anchor_center_bboxes) and len(pos_center_bboxes) and len(neg_center_bboxes):
                    anchor_center_prompts = anchor_center_bboxes.squeeze(0)
                    pos_center_prompts = pos_center_bboxes.squeeze(0)
                    neg_center_prompts = neg_center_bboxes.squeeze(0)
                    anchor_center_prompts = anchor_center_prompts.to(device)
                    pos_center_prompts = pos_center_prompts.to(device)
                    neg_center_prompts = neg_center_prompts.to(device)
                    use_center_loss = True

            # global max_num_person
            # if len(anchor_prompts) > max_num_person:
            #     max_num_person = len(anchor_prompts)
            #     print(max_num_person, flush=True)
            # if len(pos_prompts) > max_num_person:
            #     max_num_person = len(pos_prompts)
            #     print(max_num_person, flush=True)

            if 'cropped_imgs' in anchor_info:
                anchor_imgs = anchor_info['cropped_imgs']
                pos_imgs = pos_info['cropped_imgs']
                neg_imgs = neg_info['cropped_imgs']
                anchor_imgs = anchor_imgs.to(device)
                pos_imgs = pos_imgs.to(device)
                neg_imgs = neg_imgs.to(device)
                anchor_imgs_input = anchor_imgs.squeeze(0)
                pos_imgs_input = pos_imgs.squeeze(0)
                neg_imgs_input = neg_imgs.squeeze(0)
                
                if tracking:
                    anchor_tracking_imgs = anchor_info['cropped_tracking_imgs']
                    pos_tracking_imgs = pos_info['cropped_tracking_imgs']
                    neg_tracking_imgs = neg_info['cropped_tracking_imgs']
                    anchor_tracking_imgs = anchor_tracking_imgs.to(device)
                    pos_tracking_imgs = pos_tracking_imgs.to(device)
                    neg_tracking_imgs = neg_tracking_imgs.to(device)
                    anchor_tracking_imgs_input = anchor_tracking_imgs.squeeze(0)
                    pos_tracking_imgs_input = pos_tracking_imgs.squeeze(0)
                    neg_tracking_imgs_input = neg_tracking_imgs.squeeze(0)
            
            if not ssl_prediction:
                anchor_view_id = anchor_info['cam_id']
                sample_view_id = pos_info['cam_id']

                anchor_ins_fea = model(anchor_imgs_input, anchor_prompts, anchor_view_id)
                pos_ins_fea = model(pos_imgs_input, pos_prompts, sample_view_id)
                neg_ins_fea = model(neg_imgs_input, neg_prompts, sample_view_id)

                pos_dis, _ = get_matching_dis(anchor_ins_fea, pos_ins_fea, thresh=thresh)
                neg_dis, _ = get_matching_dis(anchor_ins_fea, neg_ins_fea, thresh=thresh)

                M = 1.0
                loss = torch.nn.functional.relu(pos_dis - neg_dis + M).mean()
            else:
                if multi_view_model:
                    anchor_view_id = anchor_info['cam_id']
                    sample_view_id = pos_info['cam_id']
                    
                    if reid_model is not None:
                        with torch.no_grad():
                            if tracking:
                                all_anchor_imgs_input = torch.cat([anchor_imgs_input, anchor_tracking_imgs_input], dim=0)
                                all_pos_imgs_input = torch.cat([pos_imgs_input, pos_tracking_imgs_input], dim=0)
                                all_neg_imgs_input = torch.cat([neg_imgs_input, neg_tracking_imgs_input], dim=0)
                                if dino:
                                    all_anchor_reid = reid_model(all_anchor_imgs_input).last_hidden_state.mean(dim=1)
                                    all_pos_reid = reid_model(all_pos_imgs_input).last_hidden_state.mean(dim=1)
                                    all_neg_reid = reid_model(all_neg_imgs_input).last_hidden_state.mean(dim=1)
                                else:
                                    all_anchor_reid = reid_model(all_anchor_imgs_input)
                                    all_pos_reid = reid_model(all_pos_imgs_input)
                                    all_neg_reid = reid_model(all_neg_imgs_input)
                                anchor_reid = all_anchor_reid[:len(anchor_imgs_input)]
                                pos_reid = all_pos_reid[:len(pos_imgs_input)]
                                neg_reid = all_neg_reid[:len(neg_imgs_input)]
                                anchor_tracking_reid = all_anchor_reid[len(anchor_imgs_input):]
                                pos_tracking_reid = all_pos_reid[len(pos_imgs_input):]
                                neg_tracking_reid = all_neg_reid[len(neg_imgs_input):]
                            else:
                                if dino:
                                    anchor_reid = reid_model(anchor_imgs_input).last_hidden_state.mean(dim=1)
                                    pos_reid = reid_model(pos_imgs_input).last_hidden_state.mean(dim=1)
                                    neg_reid = reid_model(neg_imgs_input).last_hidden_state.mean(dim=1)
                                else:
                                    anchor_reid = reid_model(anchor_imgs_input)
                                    pos_reid = reid_model(pos_imgs_input)
                                    neg_reid = reid_model(neg_imgs_input)
                                    
                    if test_adaptive:
                        anchor_img = anchor_info['img'].to(device)
                        pos_img = pos_info['img'].to(device)
                        neg_img = neg_info['img'].to(device)
                        input_imgs = torch.cat([anchor_img, pos_img, neg_img], dim=0)

                        z_cam_tokens, projections = model.module.get_image_view(input_imgs)

                        anchor_fea, anchor_preds = model(anchor_imgs_input, anchor_prompts, z_cam_tokens[0], projections)
                        pos_fea, pos_preds = model(pos_imgs_input, pos_prompts, z_cam_tokens[1], projections)
                        neg_fea, neg_preds = model(neg_imgs_input, neg_prompts, z_cam_tokens[2], projections)
                    else:
                        anchor_h = anchor_info['height'][0].item()
                        anchor_w = anchor_info['width'][0].item()
                        pos_h = pos_info['height'][0].item()
                        pos_w = pos_info['width'][0].item()
                        neg_h = neg_info['height'][0].item()
                        neg_w = neg_info['width'][0].item()
                        if tracking:
                            all_anchor_prompts = torch.cat([anchor_prompts, anchor_tracking_prompts], dim=0)
                            all_pos_prompts = torch.cat([pos_prompts, pos_tracking_prompts], dim=0)
                            all_neg_prompts = torch.cat([neg_prompts, neg_tracking_prompts], dim=0)
                            
                            all_anchor_cam_batch = torch.ones(len(all_anchor_prompts), dtype=torch.int64, device=device) * anchor_view_id[0]
                            all_pos_cam_batch = torch.ones(len(all_pos_prompts), dtype=torch.int64, device=device) * sample_view_id[0]
                            all_neg_cam_batch = torch.ones(len(all_neg_prompts), dtype=torch.int64, device=device) * sample_view_id[0]
                            
                            all_anchor_fea, all_anchor_preds = model.module.encode_decode(all_anchor_cam_batch, (anchor_h, anchor_w), all_anchor_prompts)
                            all_pos_fea, all_pos_preds = model.module.encode_decode(all_pos_cam_batch, (pos_h, pos_w), all_pos_prompts)
                            all_neg_fea, all_neg_preds = model.module.encode_decode(all_neg_cam_batch, (neg_h, neg_w), all_neg_prompts)
                            
                            anchor_fea = all_anchor_fea[:len(anchor_prompts)]
                            pos_fea = all_pos_fea[:len(pos_prompts)]
                            neg_fea = all_neg_fea[:len(neg_prompts)]
                            anchor_tracking_fea = all_anchor_fea[len(anchor_prompts):]
                            pos_tracking_fea = all_pos_fea[len(pos_prompts):]
                            neg_tracking_fea = all_neg_fea[len(neg_prompts):]
                        else:
                            anchor_cam_batch = torch.ones(len(anchor_prompts), dtype=torch.int64, device=device) * anchor_view_id[0]
                            pos_cam_batch = torch.ones(len(pos_prompts), dtype=torch.int64, device=device) * sample_view_id[0]
                            neg_cam_batch = torch.ones(len(neg_prompts), dtype=torch.int64, device=device) * sample_view_id[0]

                            anchor_fea, anchor_preds = model.module.encode_decode(anchor_cam_batch, (anchor_h, anchor_w), anchor_prompts)
                            pos_fea, pos_preds = model.module.encode_decode(pos_cam_batch, (pos_h, pos_w), pos_prompts)
                            neg_fea, neg_preds = model.module.encode_decode(neg_cam_batch, (neg_h, neg_w), neg_prompts)
                            # anchor_fea, anchor_preds = model.module.encode_decode(anchor_cam_batch, (anchor_h, anchor_w), anchor_prompts, anchor_reid)
                            # pos_fea, pos_preds = model.module.encode_decode(pos_cam_batch, (pos_h, pos_w), pos_prompts, pos_reid)
                            # neg_fea, neg_preds = model.module.encode_decode(neg_cam_batch, (neg_h, neg_w), neg_prompts, neg_reid)
                    
                    if reid_model is not None:
                        # pos_dis, pos_matches = get_matching_dis(anchor_fea, pos_fea, reid_fea1=anchor_reid, reid_fea2=pos_reid, thresh=thresh, alpha=alpha)
                        # neg_dis, neg_matches = get_matching_dis(anchor_fea, neg_fea, reid_fea1=anchor_reid, reid_fea2=neg_reid, thresh=thresh, alpha=alpha)
                        if tracking:
                            pos_dis1, _ = get_matching_dis(anchor_tracking_fea, pos_fea, reid_fea1=anchor_tracking_reid, reid_fea2=pos_reid, thresh=thresh, alpha=alpha)
                            neg_dis1, _ = get_matching_dis(anchor_tracking_fea, neg_fea, reid_fea1=anchor_tracking_reid, reid_fea2=neg_reid, thresh=thresh, alpha=alpha)
                            pos_dis2, _ = get_matching_dis(anchor_fea, pos_tracking_fea, reid_fea1=anchor_reid, reid_fea2=pos_tracking_reid, thresh=thresh, alpha=alpha)
                            neg_dis2, _ = get_matching_dis(anchor_fea, neg_tracking_fea, reid_fea1=anchor_reid, reid_fea2=neg_tracking_reid, thresh=thresh, alpha=alpha)
                        else:
                            pos_dis1, _ = get_matching_dis(anchor_fea[anchor_mask], pos_fea, reid_fea1=anchor_reid[anchor_mask], reid_fea2=pos_reid, thresh=thresh, alpha=alpha)
                            neg_dis1, _ = get_matching_dis(anchor_fea[anchor_mask], neg_fea, reid_fea1=anchor_reid[anchor_mask], reid_fea2=neg_reid, thresh=thresh, alpha=alpha)
                            pos_dis2, _ = get_matching_dis(anchor_fea, pos_fea[pos_mask], reid_fea1=anchor_reid, reid_fea2=pos_reid[pos_mask], thresh=thresh, alpha=alpha)
                            neg_dis2, _ = get_matching_dis(anchor_fea, neg_fea[neg_mask], reid_fea1=anchor_reid, reid_fea2=neg_reid[neg_mask], thresh=thresh, alpha=alpha)
                    else:
                        # pos_dis, pos_matches = get_matching_dis(anchor_fea, pos_fea, thresh=thresh)
                        # neg_dis, neg_matches = get_matching_dis(anchor_fea, neg_fea, thresh=thresh)
                        if tracking:
                            pos_dis1, _ = get_matching_dis(anchor_tracking_fea, pos_fea, thresh=thresh)
                            neg_dis1, _ = get_matching_dis(anchor_tracking_fea, neg_fea, thresh=thresh)
                            pos_dis2, _ = get_matching_dis(anchor_fea, pos_tracking_fea, thresh=thresh)
                            neg_dis2, _ = get_matching_dis(anchor_fea, neg_tracking_fea, thresh=thresh)
                        else:
                            pos_dis1, _ = get_matching_dis(anchor_fea[anchor_mask], pos_fea, thresh=thresh)
                            neg_dis1, _ = get_matching_dis(anchor_fea[anchor_mask], neg_fea, thresh=thresh)
                            pos_dis2, _ = get_matching_dis(anchor_fea, pos_fea[pos_mask], thresh=thresh)
                            neg_dis2, _ = get_matching_dis(anchor_fea, neg_fea[neg_mask], thresh=thresh)
                        # reprojection_dis1, reprojection_dis2, pos_dis1, pos_dis2 = get_matching_dis_reprojection_loss(anchor_prompts, anchor_fea, anchor_preds[:, sample_view_id[0]], pos_prompts, pos_fea, pos_preds[:, anchor_view_id[0]])
                        # _, _, neg_dis1, neg_dis2 = get_matching_dis_reprojection_loss(anchor_prompts, anchor_fea, anchor_preds[:, sample_view_id[0]], neg_prompts, neg_fea, neg_preds[:, anchor_view_id[0]])
                    # pos_dis, pos_matches = get_matching_dis(anchor_fea, pos_fea, thresh=thresh)
                    # neg_dis, neg_matches = get_matching_dis(anchor_fea, neg_fea, thresh=thresh)
                    # pos_dis = get_clustering_dis(anchor_fea, pos_fea)
                    # neg_dis = get_clustering_dis(anchor_fea, neg_fea)

                    M = 1.0
                    # triplet_loss = torch.nn.functional.relu(pos_dis - neg_dis + M).mean()
                    triplet_loss = torch.nn.functional.relu(pos_dis1 - neg_dis1 + M).mean() + torch.nn.functional.relu(pos_dis2 - neg_dis2 + M).mean()
                    # loss = triplet_loss

                    # anchor_xywh = xyxy2xywh(anchor_prompts)
                    # pos_xywh = xyxy2xywh(pos_prompts)

                    # anchor_gt_heatmap = get_heatmaps(anchor_xywh[:, :2])
                    # pos_gt_heatmap = get_heatmaps(pos_xywh[:, :2])
                    # # save_image(anchor_gt_heatmap, 'vis/target_heatmaps1.jpg')
                    # # print(anchor_info['image_path'])
                    # anchor_pred_heatmap1 = get_heatmaps(anchor_preds[:, anchor_view_id[0]])
                    # anchor_pred_heatmap2 = get_heatmaps(anchor_preds[:, sample_view_id[0]])
                    # pos_pred_heatmap1 = get_heatmaps(pos_preds[:, anchor_view_id[0]])
                    # pos_pred_heatmap2 = get_heatmaps(pos_preds[:, sample_view_id[0]])
                    # anchor_heatmap_loss = F.mse_loss(anchor_gt_heatmap, anchor_pred_heatmap1) + F.mse_loss(pos_gt_heatmap, anchor_pred_heatmap2)
                    # pos_heatmap_loss = F.mse_loss(anchor_gt_heatmap, pos_pred_heatmap1) + F.mse_loss(pos_gt_heatmap, pos_pred_heatmap2)

                    # anchor_loss = cross_view_matching_loss(anchor_preds[:, anchor_view_id[0]], anchor_xywh) + cross_view_matching_loss(anchor_preds[pos_matches[0], sample_view_id[0]], pos_xywh)
                    # pos_loss = cross_view_matching_loss(pos_preds[pos_matches[1], anchor_view_id[0]], anchor_xywh) + cross_view_matching_loss(pos_preds[:, sample_view_id[0]], pos_xywh)
                    # anchor_loss1, _ = get_matching_dis(anchor_preds[:, anchor_view_id[0]], anchor_xywh[:, :2], mode='l1', thresh=thresh) 
                    # anchor_loss2, _ = get_matching_dis(anchor_preds[:, sample_view_id[0]], pos_xywh[:, :2], mode='l1', thresh=thresh)
                    # pos_loss1, _ = get_matching_dis(pos_preds[:, anchor_view_id[0]], anchor_xywh[:, :2], mode='l1', thresh=thresh)
                    # pos_loss2, _ = get_matching_dis(pos_preds[:, sample_view_id[0]], pos_xywh[:, :2], mode='l1', thresh=thresh)
                    if test_adaptive:
                        anchor_loss1 = get_dis(anchor_preds[:, 0], anchor_prompts, mode='l1') 
                        # anchor_loss2, anchor_bbox_matches2 = get_matching_dis(anchor_preds[:, 1], pos_prompts, mode='l1', thresh=thresh)
                        # pos_loss1, pos_bbox_matches1 = get_matching_dis(pos_preds[:, 0], anchor_prompts, mode='l1', thresh=thresh)
                        pos_loss2 = get_dis(pos_preds[:, 1], pos_prompts, mode='l1')
                    else:
                        if tracking:
                            anchor_loss1 = get_dis(all_anchor_preds[:len(anchor_prompts), anchor_view_id[0]], anchor_prompts, mode='l1')
                            pos_loss2 = get_dis(all_pos_preds[:len(pos_prompts), sample_view_id[0]], pos_prompts, mode='l1')
                        else:
                            anchor_loss1 = get_dis(anchor_preds[:, anchor_view_id[0]], anchor_prompts, mode='l1')
                            # anchor_loss2, anchor_bbox_matches2 = get_matching_dis(anchor_preds[:, sample_view_id[0]], pos_prompts, mode='giou', thresh=thresh, dummy=True)
                            # pos_loss1, pos_bbox_matches1 = get_matching_dis(pos_preds[:, anchor_view_id[0]], anchor_prompts, mode='giou', thresh=thresh, dummy=True)
                            # anchor_loss2 = get_dis(anchor_preds[pos_matches[0], sample_view_id[0]], pos_prompts[pos_matches[1]], mode='l1')
                            # pos_loss1 = get_dis(pos_preds[pos_matches[1], anchor_view_id[0]], anchor_prompts[pos_matches[0]], mode='l1')
                            
                            # anchor_cycle_fea, anchor_cycle_preds = model(cam_id=sample_view_id, original_size=(pos_h, pos_w), boxes=anchor_preds[:, sample_view_id[0]])
                            # pos_cycle_fea, pos_cycle_preds = model(cam_id=anchor_view_id, original_size=(anchor_h, anchor_w), boxes=pos_preds[:, anchor_view_id[0]])
                            # anchor_cycle_loss = get_dis(anchor_cycle_preds[:, anchor_view_id[0]], anchor_prompts, mode='l1', percentage=0.4)
                            # pos_cycle_loss = get_dis(pos_cycle_preds[:, sample_view_id[0]], pos_prompts, mode='l1', percentage=0.4)
                            # if reid_model is not None:
                            #     pos_dis2, _ = get_matching_dis(anchor_cycle_fea, pos_cycle_fea, reid_fea1=anchor_reid, reid_fea2=pos_reid, thresh=thresh, alpha=alpha)
                            # else:
                            #     pos_dis2, _ = get_matching_dis(anchor_cycle_fea, pos_cycle_fea, thresh=thresh)
                            # triplet_loss2 = torch.nn.functional.relu(pos_dis2 - neg_dis + M).mean()

                            pos_loss2 = get_dis(pos_preds[:, sample_view_id[0]], pos_prompts, mode='l1')
                            # anchor_loss2, _ = get_matching_dis(anchor_preds[:, sample_view_id[0]], pos_prompts, mode='giou', thresh=0.0)
                            # pos_loss1, _ = get_matching_dis(pos_preds[:, anchor_view_id[0]], anchor_prompts, mode='giou', thresh=0.0)
                    # loss = (anchor_loss1 + anchor_loss2 + pos_loss1 + pos_loss2) + triplet_loss# + (anchor_cycle_loss + pos_cycle_loss)
                    loss = (anchor_loss1 + pos_loss2) + triplet_loss# + triplet_loss2# + anchor_cycle_loss + pos_cycle_loss
                    # loss = triplet_loss
                    # loss = (anchor_loss1 + pos_loss2)
                    if use_center_loss:
                        pts1, pts2 = get_center_prompts(anchor_center_prompts, pos_center_prompts, pos_matches[0], pos_matches[1])
                        pts3, pts4 = get_center_prompts(anchor_center_prompts, neg_center_prompts, neg_matches[0], neg_matches[1])
                        center_triplet_loss = 0
                        if len(pts1) and len(pts2) and len(pts3) and len(pts4):
                            pts1_cam_batch = torch.ones(len(pts1), dtype=torch.int64, device=device) * anchor_view_id[0]
                            pts2_cam_batch = torch.ones(len(pts2), dtype=torch.int64, device=device) * sample_view_id[0]
                            pts3_cam_batch = torch.ones(len(pts3), dtype=torch.int64, device=device) * anchor_view_id[0]
                            pts4_cam_batch = torch.ones(len(pts4), dtype=torch.int64, device=device) * sample_view_id[0]
                        
                            pts1_fea, pts1_preds = model.module.encode_decode(pts1_cam_batch, (anchor_h, anchor_w), pts1)
                            pts2_fea, pts2_preds = model.module.encode_decode(pts2_cam_batch, (pos_h, pos_w), pts2)
                            center_pos_dis = torch.sqrt((pts1_fea - pts2_fea).pow(2).sum(-1)).mean(-1)

                            pts3_fea, pts3_preds = model.module.encode_decode(pts3_cam_batch, (anchor_h, anchor_w), pts3)
                            pts4_fea, pts4_preds = model.module.encode_decode(pts4_cam_batch, (neg_h, neg_w), pts4)
                            center_neg_dis = torch.sqrt((pts3_fea - pts4_fea).pow(2).sum(-1)).mean(-1)

                            center_triplet_loss = torch.nn.functional.relu(center_pos_dis - center_neg_dis + M).mean()

                            # pts1_loss1 = get_dis(pts1_preds[:, anchor_view_id[0]], pts1, mode='l1')
                            # pts2_loss1 = get_dis(pts2_preds[:, sample_view_id[0]], pts2, mode='l1')
                            # pts3_loss1 = get_dis(pts3_preds[:, anchor_view_id[0]], pts3, mode='l1')
                            # pts4_loss1 = get_dis(pts4_preds[:, sample_view_id[0]], pts4, mode='l1')

                            loss = loss + center_triplet_loss# + pts1_loss1 + pts2_loss1 + pts3_loss1 + pts4_loss1
                    # if use_center_loss:
                    #     input_anchor_center_prompts, anchor_center_labels = get_center_prompts_labels(anchor_center_prompts, anchor_preds)
                    #     input_pos_center_prompts, pos_center_labels = get_center_prompts_labels(pos_center_prompts, pos_preds)
                    #     input_neg_center_prompts, neg_center_labels = get_center_prompts_labels(neg_center_prompts, neg_preds)
                        
                    #     if len(input_anchor_center_prompts):
                    #         anchor_center_cam_batch = torch.ones(len(input_anchor_center_prompts), dtype=torch.int64, device=device) * anchor_view_id[0]
                    #         _, anchor_center_preds = model.module.encode_decode(anchor_center_cam_batch, (anchor_h, anchor_w), input_anchor_center_prompts)
                    #         anchor_center_loss = get_dis(anchor_center_preds, anchor_center_labels, mode='giou').mean()
                    #         loss += anchor_center_loss
                    #     if len(input_pos_center_prompts):
                    #         pos_center_cam_batch = torch.ones(len(input_pos_center_prompts), dtype=torch.int64, device=device) * sample_view_id[0]
                    #         _, pos_center_preds = model.module.encode_decode(pos_center_cam_batch, (pos_h, pos_w), input_pos_center_prompts)
                    #         pos_center_loss = get_dis(pos_center_preds, pos_center_labels, mode='giou').mean()
                    #         loss += pos_center_loss
                    #     if len(input_neg_center_prompts):
                    #         neg_center_cam_batch = torch.ones(len(input_neg_center_prompts), dtype=torch.int64, device=device) * sample_view_id[0]
                    #         _, neg_center_preds = model.module.encode_decode(neg_center_cam_batch, (neg_h, neg_w), input_neg_center_prompts)
                    #         neg_center_loss = get_dis(neg_center_preds, neg_center_labels, mode='giou').mean()
                    #         loss += neg_center_loss
                else:
                    _, pred_bboxes = model(anchor_imgs_input, anchor_prompts)

                    pos_dis = cross_view_matching_loss(pred_bboxes, pos_prompts)
                    neg_dis = cross_view_matching_loss(pred_bboxes, neg_prompts)
                    M = 0.1
                    triplet_loss = torch.nn.functional.relu(pos_dis - neg_dis + M)
                    loss = pos_dis * 0.1 + triplet_loss
                    # loss = triplet_loss


        elif multi_view_model:
            anchor_info, pos_info = data
            # anchor_full_img = anchor_info['image']
            # pos_full_img = pos_info['image']
            anchor_bboxes = anchor_info['bboxes']
            pos_bboxes = pos_info['bboxes']
            anchor_view_id = anchor_info['cam_id']
            sample_view_id = pos_info['cam_id']

            if len(anchor_bboxes) == 0 or len(pos_bboxes) == 0:
                continue
            
            if 'cropped_imgs' in anchor_info:
                anchor_imgs = anchor_info['cropped_imgs']
                pos_imgs = pos_info['cropped_imgs']
                anchor_imgs = anchor_imgs.to(device)
                pos_imgs = pos_imgs.to(device)
                anchor_imgs_input = anchor_imgs.squeeze(0)
                pos_imgs_input = pos_imgs.squeeze(0)
            
            # anchor_full_img = anchor_full_img.to(device)
            # pos_full_img = pos_full_img.to(device)

            anchor_prompts = anchor_bboxes.squeeze(0)
            pos_prompts = pos_bboxes.squeeze(0)
            anchor_prompts = anchor_prompts.to(device)
            pos_prompts = pos_prompts.to(device)
            # anchor_full_img = anchor_full_img.repeat(len(anchor_imgs_input), 1, 1, 1)
            # pos_full_img = pos_full_img.repeat(len(pos_imgs_input), 1, 1, 1)
            if test_adaptive:
                # anchor_img = anchor_info['img'].to(device)
                # pos_img = pos_info['img'].to(device)
                # input_imgs = torch.cat([anchor_img, pos_img], dim=0)
                # z_cam_tokens, projections = model.module.get_image_view(input_imgs)

                anchor_img = anchor_info['depth_fea'].to(device)
                pos_img = pos_info['depth_fea'].to(device)
                input_imgs = torch.cat([anchor_img, pos_img], dim=0) # shape: [2, 2442, 1024]
                z_cam_tokens, projections = model.module.get_image_view(input_imgs)

                anchor_fea, anchor_preds = model(anchor_imgs_input, anchor_prompts, z_cam_tokens[0], projections)
                pos_fea, pos_preds = model(pos_imgs_input, pos_prompts, z_cam_tokens[1], projections)
            else:
                anchor_fea, anchor_preds = model(anchor_imgs_input, anchor_prompts, anchor_view_id)
                pos_fea, pos_preds = model(pos_imgs_input, pos_prompts, sample_view_id)

            # anchor_xywh = xyxy2xywh(anchor_prompts)
            # pos_xywh = xyxy2xywh(pos_prompts)
            
            # anchor_gt_heatmap = get_heatmaps(anchor_xywh[:, :2])
            # pos_gt_heatmap = get_heatmaps(pos_xywh[:, :2])
            # # save_image(anchor_gt_heatmap, 'vis/target_heatmaps1.jpg')
            # # print(anchor_info['image_path'])
            # anchor_pred_heatmap1 = get_heatmaps(anchor_preds[:, 0])
            # anchor_pred_heatmap2 = get_heatmaps(anchor_preds[:, 1])
            # pos_pred_heatmap1 = get_heatmaps(pos_preds[:, 0])
            # pos_pred_heatmap2 = get_heatmaps(pos_preds[:, 1])
            # anchor_heatmap_loss = F.mse_loss(anchor_gt_heatmap, anchor_pred_heatmap1) + F.mse_loss(pos_gt_heatmap, anchor_pred_heatmap2)
            # pos_heatmap_loss = F.mse_loss(anchor_gt_heatmap, pos_pred_heatmap1) + F.mse_loss(pos_gt_heatmap, pos_pred_heatmap2)

            # # anchor_loss = get_matching_dis(min_max_norm(anchor_preds[:, 0]), min_max_norm(anchor_xywh[:, :2]), mode='l1') + get_matching_dis(min_max_norm(anchor_preds[:, 1]), min_max_norm(pos_xywh[:, :2]), mode='l1')
            # # pos_loss = get_matching_dis(min_max_norm(pos_preds[:, 0]), min_max_norm(anchor_xywh[:, :2]), mode='l1') + get_matching_dis(min_max_norm(pos_preds[:, 1]), min_max_norm(pos_xywh[:, :2]), mode='l1')
            # anchor_loss = get_matching_dis(anchor_preds[:, 0], anchor_xywh[:, :2], mode='l1', thresh=thresh) + get_matching_dis(anchor_preds[:, 1], pos_xywh[:, :2], mode='l1', thresh=thresh)
            # pos_loss = get_matching_dis(pos_preds[:, 0], anchor_xywh[:, :2], mode='l1', thresh=thresh) + get_matching_dis(pos_preds[:, 1], pos_xywh[:, :2], mode='l1', thresh=thresh)
            # # anchor_loss = get_clustering_dis(anchor_preds[:, 0], anchor_xywh[:, :2], mode='l1') + get_clustering_dis(anchor_preds[:, 1], pos_xywh[:, :2], mode='l1')
            # # pos_loss = get_clustering_dis(pos_preds[:, 0], anchor_xywh[:, :2], mode='l1') + get_clustering_dis(pos_preds[:, 1], pos_xywh[:, :2], mode='l1')
            # loss = (anchor_loss + pos_loss) + (anchor_heatmap_loss + pos_heatmap_loss)

            if test_adaptive:
                anchor_loss1, _ = get_matching_dis(anchor_preds[:, 0], anchor_prompts, mode='l1', thresh=thresh) 
                anchor_loss2, _ = get_matching_dis(anchor_preds[:, 1], pos_prompts, mode='l1', thresh=thresh)
                pos_loss1, _ = get_matching_dis(pos_preds[:, 0], anchor_prompts, mode='l1', thresh=thresh)
                pos_loss2, _ = get_matching_dis(pos_preds[:, 1], pos_prompts, mode='l1', thresh=thresh)
                loss = (anchor_loss1 + anchor_loss2 + pos_loss1 + pos_loss2)
            else:
                anchor_loss1, _ = get_matching_dis(anchor_preds[:, anchor_view_id[0]], anchor_prompts, mode='l1', thresh=thresh) 
                anchor_loss2, _ = get_matching_dis(anchor_preds[:, sample_view_id[0]], pos_prompts, mode='l1', thresh=thresh)
                pos_loss1, _ = get_matching_dis(pos_preds[:, anchor_view_id[0]], anchor_prompts, mode='l1', thresh=thresh)
                pos_loss2, _ = get_matching_dis(pos_preds[:, sample_view_id[0]], pos_prompts, mode='l1', thresh=thresh)
            # anchor_loss = get_clustering_dis(anchor_preds[:, 0], anchor_xywh[:, :2], mode='l1') + get_clustering_dis(anchor_preds[:, 1], pos_xywh[:, :2], mode='l1')
            # pos_loss = get_clustering_dis(pos_preds[:, 0], anchor_xywh[:, :2], mode='l1') + get_clustering_dis(pos_preds[:, 1], pos_xywh[:, :2], mode='l1')
                loss = (anchor_loss1 + anchor_loss2 + pos_loss1 + pos_loss2)# + (anchor_heatmap_loss + pos_heatmap_loss)
        else:
            pass
        
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if step_i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = (
                "Epoch: [{0}][{1}/{2}]\t"
                "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                "Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
                "Loss: {loss.val:.6f} ({loss.avg:.6f})\t"
                "Memory {memory:.1f}".format(
                    epoch,
                    step_i,
                    len(dataloader_train),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    memory=gpu_memory_usage,
                )
            )
            logger.info(msg)


    return epoch_loss / len(dataloader_train)


def evaluate_p3de(config, model, dataloader_test, device, ssl_prediction=True, fully_supervised=False, thresh=0.0, fix_view_id=-1, test_adaptive=False, reid_model=None, vis=False):
    alpha = config.DATASET.ALPHA
    # tracking = config.TRAIN.TRACKING
    
    correct_sum = 0
    precision_total_sum = 0
    recall_total_sum = 0
    scores = []
    labels = []
    IPAA_correct_sum = 0
    IPAA_total_sum = 0
    IPAA_80_sum = 0
    IPAA_90_sum = 0
    IPAA_100_sum = 0
    IPAA_img_sum = 0
    
    for step_i, data in enumerate(dataloader_test):
        view_num = len(data)
        feas_all_views = []
        preds_all_views = []
        prompts_all_views = []
        scores_all_views = []
        if reid_model is not None:
            reid_feas = []
        for i in range(view_num):
            info = data[i]
            bboxes = info['bboxes']
            if len(bboxes) == 0:
                feas_all_views.append([])
                preds_all_views.append([])
                prompts_all_views.append([])
                scores_all_views.append([])
                if reid_model is not None:
                    reid_feas.append([])
                continue
            prompts = bboxes.squeeze(0)
            prompts = prompts.to(device)
            # if tracking:
            #     tracking_bboxes = info['tracking_bboxes']
            #     if len(tracking_bboxes) == 0:
            #         feas_all_views.append([])
            #         preds_all_views.append([])
            #         prompts_all_views.append([])
            #         scores_all_views.append([])
            #         if reid_model is not None:
            #             reid_feas.append([])
            #         continue
            #     tracking_prompts = tracking_bboxes.squeeze(0)
            #     tracking_prompts = tracking_prompts.to(device)
            if 'cropped_imgs' in info:
                imgs = info['cropped_imgs']
                imgs = imgs.to(device)
                imgs_input = imgs.squeeze(0)
                if reid_model is not None:
                    imgs_input = imgs_input.to(device)
                    reid_feature = reid_model(imgs_input)# .last_hidden_state.mean(dim=1)
                    reid_feas.append(reid_feature)
            view_id = info['cam_id']
            if fully_supervised and not ssl_prediction:
                feas = model(imgs_input, prompts, view_id)
            elif not ssl_prediction:
                feas = model(imgs_input, prompts, view_id)
            else:
                if test_adaptive:
                    # ori_img = info['img'].to(device)
                    # z_cam_tokens, projections = model.module.get_image_view(ori_img)
                    ori_img = info['depth_fea'].to(device)
                    z_cam_tokens, projections = model.module.get_image_view(ori_img)
                    feas, preds = model(imgs_input, prompts, z_cam_tokens[0], projections)
                else:
                    # feas, preds = model(imgs_input, prompts, view_id)
                    cam_batch = torch.ones(len(prompts), dtype=torch.int64, device=device) * view_id[0]
                    h = info['height'][0].item()
                    w = info['width'][0].item()
                    feas, preds = model.module.encode_decode(cam_batch, (h, w), prompts)
                    # feas, preds = model.module.encode_decode(cam_batch, (h, w), prompts, reid_feature)
            one_view_scores = info['scores']
            one_view_scores = one_view_scores.squeeze(0)
            scores_all_views.append(one_view_scores)
            feas_all_views.append(feas)
            preds_all_views.append(preds)
            prompts_all_views.append(prompts)
        if fix_view_id == -1:
            cam_pairs = []
            for i in range(view_num - 1):
                for j in range(i + 1, view_num):
                    cam_pairs.append((i, j))
        else:
            cam_pairs = []
            for j in range(view_num):
                if j == fix_view_id:
                    continue
                cam_pairs.append((fix_view_id, j))

        # view1 = 0
        # view2 = 1
        # view3 = 2
        # labels1 = torch.cat(data[view1]['labels'], dim=0)
        # labels1 = labels1.to(device)
        # labels2 = torch.cat(data[view2]['labels'], dim=0)
        # labels2 = labels2.to(device)
        # labels3 = torch.cat(data[view3]['labels'], dim=0)
        # labels3 = labels3.to(device)
        # bboxes1 = data[view1]['bboxes']
        # bboxes2 = data[view2]['bboxes']
        # bboxes3 = data[view3]['bboxes']
        # bboxes = [bboxes1.squeeze(0), bboxes2.squeeze(0), bboxes3.squeeze(0)]
        # img_path1 = data[view1]['image_path'][0]
        # img_path2 = data[view2]['image_path'][0]
        # img_path3 = data[view3]['image_path'][0]
        # img_paths = [img_path1, img_path2, img_path3]

        # result12 = cross_view_matching_evaluation(feas[view1], feas[view2], labels1, labels2, reid_fea1=reid_feas[view1], reid_fea2=reid_feas[view2], mode='l2', thresh=thresh, alpha=alpha)
        # result23 = cross_view_matching_evaluation(feas[view1], feas[view3], labels1, labels3, reid_fea1=reid_feas[view1], reid_fea2=reid_feas[view3], mode='l2', thresh=thresh, alpha=alpha)

        # matches_x12 = result12['matches_x']
        # matches_y12 = result12['matches_y']
        
        # matches_x23 = result23['matches_x']
        # matches_y23 = result23['matches_y']

        # for k in range(len(matches_x12)):
        #     matches_x12[k] = (0, matches_x12[k])
        # for k in range(len(matches_y12)):
        #     matches_y12[k] = (1, matches_y12[k])
        # for k in range(len(matches_x23)):
        #     matches_x23[k] = (0, matches_x23[k])
        # for k in range(len(matches_y23)):
        #     matches_y23[k] = (2, matches_y23[k])
        # pairs = [(matches_x12, matches_y12), (matches_x23, matches_y23)]
        # multi_view_associations = merge_cross_view_associations_with_constraints(pairs)
        # visualize_multi_view_associations(img_paths, multi_view_associations, bboxes)
        # continue

        for cam_pair in cam_pairs:
            view1, view2 = cam_pair
            anchor_fea = feas_all_views[view1]
            pos_fea = feas_all_views[view2]
            if len(anchor_fea) == 0 or len(pos_fea) == 0:
                continue
            anchor_pred = preds_all_views[view1][:, view2]
            pos_pred = preds_all_views[view2][:, view1]
            anchor_prompts = prompts_all_views[view1]
            pos_prompts = prompts_all_views[view2]
            anchor_scores = scores_all_views[view1]
            pos_scores = scores_all_views[view2]

            anchor_info = data[view1]
            pos_info = data[view2]
            anchor_labels = anchor_info['labels']
            pos_labels = pos_info['labels']

            labels1 = torch.cat(anchor_labels, dim=0)
            labels1 = labels1.to(device)
            labels2 = torch.cat(pos_labels, dim=0)
            labels2 = labels2.to(device)

            mask = anchor_scores > 0.6
            if not len(anchor_fea[mask]):
                continue

            if reid_model is not None:
                anchor_reid_fea = reid_feas[view1]
                pos_reid_fea = reid_feas[view2]
                result = cross_view_matching_evaluation(anchor_fea[mask], pos_fea, labels1[mask], labels2, reid_fea1=anchor_reid_fea[mask], reid_fea2=pos_reid_fea, mode='l2', thresh=thresh, alpha=alpha)
                # result = cross_view_matching_evaluation(anchor_fea[mask], pos_fea, labels1[mask], labels2, mode='l2', thresh=thresh)
            else:
                result = cross_view_matching_evaluation(anchor_fea[mask], pos_fea, labels1[mask], labels2, mode='l2', thresh=thresh)
                # result = cross_view_matching_reprojection_evaluation(anchor_fea, anchor_pred, pos_fea, pos_prompts, labels1, labels2, mode='l2', thresh=thresh)
                # result = cross_view_matching_evaluation(anchor_preds[:, 1], xyxy2xywh(pos_prompts)[:, :2], labels1, labels2, mode='l1')
            correct_sum += result['correct']
            precision_total_sum += result['precision_total']
            recall_total_sum += result['recall_total']
            scores.append(result['scores'])
            labels.append(result['labels'])
            IPAA_correct_sum += result['IPAA_correct']
            IPAA_total_sum += result['IPAA_total']
            ratio = result['IPAA_correct'] / result['IPAA_total']
            IPAA_100_sum += (ratio == 1.)
            IPAA_90_sum += (ratio >= 0.9)
            IPAA_80_sum += (ratio >= 0.8)
            IPAA_img_sum += 1

            if vis:
                anchor_img_path = anchor_info['image_path'][0]
                pos_img_path = pos_info['image_path'][0]
                true_or_false = result['true_or_false']
                matches_x = result['matches_x']
                matches_y = result['matches_y']
                matches_scores = result['matches_scores']
                anchor_bboxes = anchor_info['bboxes']
                pos_bboxes = pos_info['bboxes']
                anchor_bboxes = anchor_bboxes.squeeze(0)
                pos_bboxes = pos_bboxes.squeeze(0)

                anchor_name = anchor_img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
                save_path = 'vis/' + anchor_name + '_' + str(view1) + '_' + str(view1) + str(view2)
                visualize(anchor_img_path, save_path, true_or_false, matches_x, anchor_bboxes, scores=matches_scores)
                # visualize(anchor_img_path, save_path, true_or_false, matches_x, anchor_bboxes)
                # visualize_highlights(anchor_img_path, save_path, true_or_false, matches_x, anchor_bboxes)

                pos_name = pos_img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
                save_path = 'vis/' + pos_name + '_' + str(view2) + '_' + str(view1) + str(view2)
                visualize(pos_img_path, save_path, true_or_false, matches_y, pos_bboxes, scores=matches_scores)
                # visualize(pos_img_path, save_path, true_or_false, matches_y, pos_bboxes)
                # visualize_highlights(pos_img_path, save_path, true_or_false, matches_y, pos_bboxes)
    
    if precision_total_sum:
        precision = correct_sum / precision_total_sum
    else:
        precision = 0
    recall = correct_sum / recall_total_sum

    labels = np.concatenate(labels)
    scores = np.concatenate(scores)
    ap = average_precision_score(labels, scores)
    # print('Average Precision: {}'.format(ap))
    fpr_95 = FPR_95(labels, scores)
    # print('FPR-95: {}'.format(fpr_95))
    match_acc = IPAA_correct_sum / IPAA_total_sum
    IPAA_100 = IPAA_100_sum / IPAA_img_sum
    IPAA_90 = IPAA_90_sum / IPAA_img_sum
    IPAA_80 = IPAA_80_sum / IPAA_img_sum
    # print('Match Accuracy: {}'.format(match_acc))
    # print('IPAA 100: {}'.format(IPAA_100))
    # print('IPAA 90: {}'.format(IPAA_90))
    # print('IPAA 80: {}'.format(IPAA_80))
    # print('Instance matching precision: {}'.format(precision))
    # print('Instance matching recall: {}'.format(recall))

    msg = (
        "Average Precision: {0:.4f}\t"
        "FPR-95: {1:.4f}\t"
        "Precision: {2:.4f}\t"
        "Recall: {3:.4f}\t"
        "Accuracy: {4:.4f}\t"
        "IPAA 100: {5:.4f}\t"
        "IPAA 90: {6:.4f}\t"
        "IPAA 80: {7:.4f}".format(
            ap,
            fpr_95,
            precision,
            recall,
            match_acc,
            IPAA_100,
            IPAA_90,
            IPAA_80
        )
    )
    logger.info(msg)

def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    parser.add_argument("--test", help="test mode", action='store_true', default=False)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args

def train():
    gpus = [int(i) for i in config.GPUS.split(",")]
    device = torch.device('cuda')
    train_mode = config.TRAIN.TRAIN_MODE
    ssl_prediction = config.SSL_PREDICTION
    # dataset choices: wildtrack / 4dor / mvmhat / mvor / panoptic / volleyball / shelf
    dataset_name = config.DATASET.TRAIN_DATASET
    depth = config.DEPTH
    multi_view_model = True
    fully_supervised = config.TRAIN.FULLY_SUPERVISED
    thresh = config.TRAIN.THRESH
    batch_size = config.TRAIN.BATCH_SIZE
    fix_view_id = config.FIX_VIEW_ID
    test_adaptive = False
    reid = config.TRAIN.REID
    dino = config.TRAIN.DINO
    alpha = config.DATASET.ALPHA
    
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

    dataset_train = OR_Loader(config=config, mode='train', box_thresh=0.1)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=6, pin_memory=True, shuffle=True)

    view_num = dataset_train.get_view_num()

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

    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH
    last_epoch = -1
    if config.TRAIN.RESUME:
        checkpoint_path = config.TRAIN.CKP_PATH
        if len(checkpoint_path):
            ckp = torch.load(checkpoint_path)
            model.module.load_state_dict(ckp['model'])
        else:
            checkpoint_path = os.path.join(final_output_dir, 'checkpoint.pth.tar')
            ckp = torch.load(checkpoint_path)
            model.module.load_state_dict(ckp['model'])
            optimizer.load_state_dict(ckp['optimizer'])
            last_epoch = ckp['epoch']
            start_epoch = last_epoch

    max_loss = 1e8

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch=last_epoch
    )
    

    for epoch_i in range(start_epoch, end_epoch):
        logger.info("learning rate for this epoch {}".format(lr_scheduler.get_last_lr()))
        model.train()
        epoch_loss = train_p3de(config, epoch_i, model, dataloader_train, optimizer, device, train_mode, ssl_prediction, multi_view_model, fully_supervised, thresh, test_adaptive, reid_model, dino)
        model.eval()
        lr_scheduler.step()
        print('Epoch: {}'.format(epoch_i))
        print('Epoch loss: {}'.format(epoch_loss))
        is_best = False
        if epoch_loss < max_loss:
            is_best = True
            max_loss = epoch_loss

        save_checkpoint(
            {
                "epoch": epoch_i,
                'loss': epoch_loss,
                'optimizer': optimizer.state_dict(),
                'model': model.module.state_dict()
            },
            is_best,
            final_output_dir,
        )
        logger.info("=> saving checkpoint to {} (Best: {})".format(final_output_dir, is_best))

def test():
    gpus = [int(i) for i in config.GPUS.split(",")]
    device = torch.device('cuda')
    train_mode = config.TRAIN.TRAIN_MODE
    ssl_prediction = config.SSL_PREDICTION
    # dataset choices: wildtrack / 4dor / mvmhat / mvor / panoptic / volleyball / shelf
    dataset_name = config.DATASET.TEST_DATASET
    depth = config.DEPTH
    multi_view_model = True
    fully_supervised = config.TRAIN.FULLY_SUPERVISED
    thresh = config.TEST.THRESH
    batch_size = config.TRAIN.BATCH_SIZE
    fix_view_id = config.FIX_VIEW_ID
    test_adaptive = False
    reid = config.TRAIN.REID
    vis = config.TEST.VIS

    dataset_test = OR_Loader(config=config, mode='test', inference=True, box_thresh=0.1)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=6, pin_memory=True, shuffle=False)

    view_num = dataset_test.get_view_num()

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

    reid_model = None
    if reid:
        reid_model = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path='weights/osnet_ain_ms_d_c.pth.tar',
            device='cuda'
        )
        # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        # reid_model = AutoModel.from_pretrained('facebook/dinov2-base')
        # reid_model.eval()
        # reid_model.to(device)
        

    model.eval()
    with torch.no_grad():
        evaluate_p3de(config, model, dataloader_test, device, ssl_prediction, fully_supervised, thresh, fix_view_id, test_adaptive, reid_model, vis)


if __name__ == '__main__':
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, "train")
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    if args.test:
        test()
    else:
        train()