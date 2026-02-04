import os
import logging
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
import cv2
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.transform import Rotation as R
import cv2
import random
from collections import defaultdict
from typing import Tuple, List


hm_xx = torch.tensor([i for i in range(224)]).to(
    dtype=torch.float32, device='cuda'
)
hm_yy = torch.tensor([i for i in range(224)]).to(
    dtype=torch.float32, device='cuda'
)
hm_yy, hm_xx = torch.meshgrid(hm_yy, hm_xx, indexing="ij")
hm_xx, hm_yy = hm_xx.view(1, *hm_xx.shape), hm_yy.view(1, *hm_yy.shape)

def get_heatmaps(pts, fuse=True):
    # poses.shape: [num_view, bs, num_person, 15, 2]
    # pts.shape: [num_pt, 2]
    sigma_hm = 3
    _x, _y = pts[:, 0] * 224, pts[:, 1] * 224
    _x = _x.view(-1, 1, 1)
    _y = _y.view(-1, 1, 1)
    xx, yy = hm_xx.clone(), hm_yy.clone()
    # heatmaps.shape: [num_pt, height, width]
    heatmaps = torch.exp(
        -(((xx - _x) / sigma_hm) ** 2) / 2
        - (((yy - _y) / sigma_hm) ** 2) / 2
    )
    if fuse:
        heatmaps = torch.clamp(torch.sum(heatmaps, 0), min=0.0, max=1.0)
    else:
        heatmaps = torch.clamp(heatmaps, min=0.0, max=1.0)

    return heatmaps

def visualize(img_path, save_path, true_or_false, matches, bboxes, pts=[], scores=None, normalize=True):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    for k in range(len(matches)):
        idx = matches[k]
        one_bbox = bboxes[idx]
        if true_or_false[k]:
            label_color = (0, 255, 0)
        else:
            label_color = (0, 0, 255)
        color = get_color(k)
        if normalize:
            x1 = int(one_bbox[0] * w)
            y1 = int(one_bbox[1] * h)
            x2 = int(one_bbox[2] * w)
            y2 = int(one_bbox[3] * h)
        else:
            x1 = int(one_bbox[0])
            y1 = int(one_bbox[1])
            x2 = int(one_bbox[2])
            y2 = int(one_bbox[3])
        cv2.rectangle(img, [x1, y1], [x2, y2], color=color, thickness=3)
        cv2.putText(img, str(k), [x1, y2], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)
        if scores is not None:
            cv2.putText(img, str(round(scores[k], 2)), [x1, y1 + 20], cv2.FONT_HERSHEY_PLAIN, 2, label_color, thickness=2)
        if len(pts):
            x = int(pts[k][0] * w)
            y = int(pts[k][1] * h)
            cv2.circle(img, [x, y], 4, color, -1)
            cv2.putText(img, str(k), [x, y], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)
    cv2.imwrite(save_path + '.jpg', img)

def visualize_highlights(img_path, save_path, true_or_false, matches, bboxes, pts=[], scores=None):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    black_rect = np.ones(img.shape, dtype=np.uint8) * 0
    alpha = 0.4
    for k in range(len(matches)):
        idx = matches[k]
        one_bbox = bboxes[idx]
        if true_or_false[k]:
            label_color = (0, 255, 0)
        else:
            label_color = (0, 0, 255)
        color = get_color(k)
        x1 = int(one_bbox[0] * w)
        y1 = int(one_bbox[1] * h)
        x2 = int(one_bbox[2] * w)
        y2 = int(one_bbox[3] * h)

        sub_img = img[y1:y2, x1:x2]
        res = cv2.addWeighted(img, alpha, black_rect, 1 - alpha, 1.0)
        res[y1:y2, x1:x2] = sub_img

        cv2.putText(res, str(k), [x1, y2], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)
        if scores is not None:
            cv2.putText(res, str(round(scores[k], 2)), [x1, y1 + 20], cv2.FONT_HERSHEY_PLAIN, 2, label_color, thickness=2)
        if len(pts):
            x = int(pts[k][0] * w)
            y = int(pts[k][1] * h)
            cv2.circle(img, [x, y], 4, color, -1)
            cv2.putText(img, str(k), [x, y], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)

        cv2.imwrite(save_path + '_' + str(k) + '.jpg', res)

def visualize_cluster(img_path, save_path, true_or_false, labels, bboxes, pts=[]):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    for k in range(len(labels)):
        label = labels[k]
        one_bbox = bboxes[k]
        if true_or_false[k]:
            label_color = (0, 255, 0)
        else:
            label_color = (0, 0, 255)
        color = get_color(label)
        x1 = int(one_bbox[0] * w)
        y1 = int(one_bbox[1] * h)
        x2 = int(one_bbox[2] * w)
        y2 = int(one_bbox[3] * h)
        cv2.rectangle(img, [x1, y1], [x2, y2], color=color, thickness=3)
        cv2.putText(img, str(label), [x1, y2], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)
        if len(pts):
            x = int(pts[k][0] * w)
            y = int(pts[k][1] * h)
            cv2.circle(img, [x, y], 4, color, -1)
            cv2.putText(img, str(k), [x, y], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)
    cv2.imwrite(save_path, img)

def min_max_norm(data):
    min_v, _ = data.min(dim=0, keepdim=True)
    max_v, _ = data.max(dim=0, keepdim=True)
    new_data = (data - min_v) / (max_v - min_v + 1e-6)
    return new_data

def find_pair(ground_x, ground_y):
    pairs = []
    center_points = []
    for i in range(len(ground_x)):
        for j in range(i + 1, len(ground_x)):
            x = (ground_x[i] + ground_x[j]) * 0.5
            y = (ground_y[i] + ground_y[j]) * 0.5
            threshold = torch.pow(ground_x[i] - x, 2) + torch.pow(ground_y[i] - y, 2)
            flag = True
            for k in range(len(ground_x)):
                if k == i or k == j:
                    continue
                dis = torch.pow(ground_x[k] - x, 2) + torch.pow(ground_y[k] - y, 2)
                if dis < threshold:
                    flag = False
                    break
            if flag:
                pairs.append([i, j])
                center_points.append([x, y])
    return pairs, center_points

def get_dis_matrix(pred_pts, gt_pts, mode='l2'):
    if mode == 'cosine':
        # Normalize embeddings for cosine similarity calculation
        norm_a = F.normalize(pred_pts, p=2, dim=1)
        norm_b = F.normalize(gt_pts, p=2, dim=1)

        # Calculate cosine similarity matrix (N, M)
        similarity_matrix = torch.mm(norm_a, norm_b.t()) # Transpose B for matrix multiplication

        # Clamp similarity to avoid potential floating point issues slightly outside [-1, 1]
        similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)

        # Calculate the cost matrix for real matches
        cost_matrix = 1.0 - similarity_matrix # Shape (N, M)
    elif mode == 'l2':
        cost_matrix = torch.cdist(pred_pts, gt_pts, p=2.0)
    elif mode == 'l1':
        cost_matrix = torch.cdist(pred_pts, gt_pts, p=1.0)
    else:
        pts1 = pred_pts.repeat_interleave(len(gt_pts), 0)
        pts2 = gt_pts.repeat(len(pred_pts), 1)
        if mode == 'iou':
            cost_matrix = 1 - bbox_iou(pts1.T, pts2.T, x1y1x2y2=True).reshape(len(pred_pts), len(gt_pts))
        elif mode == 'giou':
            cost_matrix = 1 - bbox_iou(pts1.T, pts2.T, x1y1x2y2=True, GIoU=True).reshape(len(pred_pts), len(gt_pts))
        elif mode == 'diou':
            cost_matrix = 1 - bbox_iou(pts1.T, pts2.T, x1y1x2y2=True, DIoU=False).reshape(len(pred_pts), len(gt_pts))
    return cost_matrix

def dis_matrix_normalization(dis_matrix, min_max=False):
    # gamma = 1.0
    # similarity = torch.exp(-gamma * dis_matrix)
    # cost_matrix = 1.0 - similarity # Bounded [0, 1)
    # return cost_matrix
    min_v = 0
    if min_max:
        min_v = dis_matrix.min()
    max_v = dis_matrix.max()
    norm_dis_matrix = (dis_matrix - min_v) / (max_v - min_v + 1e-6)
    return norm_dis_matrix

def cross_view_matching_evaluation(pred_pts, gt_pts, labels1, labels2, reid_fea1=None, reid_fea2=None, mode='l2', thresh=0.6, matching_mode='hungarian', geo_dis_matrix=None, alpha=0.1):
    labels1_set = set(labels1.tolist())
    labels2_set = set(labels2.tolist())
    # id = -1 means no id label available
    if -1 in labels1_set:
        labels1_set.remove(-1)
    if -1 in labels2_set:
        labels2_set.remove(-1)
    recall_total = len(labels1_set.intersection(labels2_set))
    IPAA_total = len(labels1_set.union(labels2_set))
    correct = 0
    true_or_false = []
    IPAA_correct = 0
    result = {}
    result['recall_total'] = recall_total
    if len(pred_pts):
        dis_matrix = get_dis_matrix(pred_pts, gt_pts, mode=mode)
        print(dis_matrix, flush=True)
        norm_dis_matrix = dis_matrix_normalization(dis_matrix)
        print(norm_dis_matrix, flush=True)
        
        if reid_fea1 is not None:
            reid_dis_matrix = get_dis_matrix(reid_fea1, reid_fea2, mode=mode)
            norm_reid_dis_matrix = dis_matrix_normalization(reid_dis_matrix)
            norm_dis_matrix = norm_dis_matrix * (1 - alpha) + norm_reid_dis_matrix * alpha
        
        norm_dis_matrix = norm_dis_matrix.cpu().detach().numpy()
        if geo_dis_matrix is not None:
            norm_geo_dis_matrix = dis_matrix_normalization(geo_dis_matrix, True)
            norm_dis_matrix = norm_dis_matrix * 0.1 + norm_geo_dis_matrix * 0.9

        if matching_mode == 'hungarian':
            matches_x, matches_y = linear_sum_assignment(norm_dis_matrix)
            matches_x = list(matches_x)
            matches_y = list(matches_y)

            score_matrix = 1 - norm_dis_matrix
        
            for k in range(len(matches_x) - 1, -1, -1):
                if len(matches_x) == 1:
                    break
                if score_matrix[matches_x[k], matches_y[k]] < thresh:
                    del matches_x[k]
                    del matches_y[k]
        else:
            cost, x, y = lap.lapjv(norm_dis_matrix, extend_cost=True, cost_limit=(1-thresh))
            matches_x = []
            matches_y = []
            for i in range(len(x)):
                if x[i] < 0:
                    continue
                matches_x.append(i)
                matches_y.append(x[i])
            
            score_matrix = 1 - norm_dis_matrix

        # print(matches_x, flush=True)
        # print(matches_y, flush=True)
        # exit(1)
        # norm_dis_matrix, matches = augmented_hungarian_matching(pred_pts, gt_pts, cost_threshold=0.5)
        # matches_x, matches_y = matches
        # score_matrix = 1 - norm_dis_matrix.cpu().numpy()
        
        precision_total = len(matches_x)
        for k in range(precision_total):
            if labels1[matches_x[k]] == -1 and labels2[matches_y[k]] == -1:
                true_or_false.append(True)
                continue
            if labels1[matches_x[k]] == labels2[matches_y[k]]:
                true_or_false.append(True)
                correct += 1
            else:
                true_or_false.append(False)
        label_matrix = np.zeros((len(labels1), len(labels2)), dtype=bool)
        label_mask = np.ones((len(labels1), len(labels2)), dtype=bool)
        for i in range(len(labels1)):
            for j in range(len(labels2)):
                if labels1[i] == labels2[j]:
                    label_matrix[i, j] = True
                if labels1[i] == -1 or labels2[j] == -1:
                    label_mask[i, j] = False
        N = IPAA_total - precision_total # predicted unmatched instances
        P = precision_total
        TP = correct
        FP = precision_total - correct
        TN = 0
        for i in range(len(labels1)):
            if labels1[i] == -1:
                continue
            if i in matches_x:
                continue
            idx1 = labels1[i]
            if idx1 not in labels2:
                TN += 1
        for i in range(len(labels2)):
            if labels2[i] == -1:
                continue
            if i in matches_y:
                continue
            idx2 = labels2[i]
            if idx2 not in labels1:
                TN += 1
        IPAA_correct = (TN + correct)
        
        result['scores'] = score_matrix.flatten()[label_mask.flatten()]
        result['labels'] = label_matrix.flatten()[label_mask.flatten()]
        result['IPAA_total'] = IPAA_total
        result['IPAA_correct'] = IPAA_correct
        result['true_or_false'] = true_or_false
        result['matches_x'] = matches_x
        result['matches_y'] = matches_y
        result['matches_scores'] = score_matrix[matches_x, matches_y]
    else:
        precision_total = 0
    result['correct'] = correct
    result['precision_total'] = precision_total

    return result


def cross_view_matching_reprojection_evaluation(anchor_fea, anchor_pred, pos_fea, pos_prompts, labels1, labels2, reid_fea1=None, reid_fea2=None, mode='l2', thresh=0.6, matching_mode='hungarian', geo_dis_matrix=None, alpha=0.1):
    labels1_set = set(labels1.tolist())
    labels2_set = set(labels2.tolist())
    # id = -1 means no id label available
    if -1 in labels1_set:
        labels1_set.remove(-1)
    if -1 in labels2_set:
        labels2_set.remove(-1)
    recall_total = len(labels1_set.intersection(labels2_set))
    IPAA_total = len(labels1_set.union(labels2_set))
    correct = 0
    true_or_false = []
    IPAA_correct = 0
    result = {}
    result['recall_total'] = recall_total
    if len(anchor_fea):
        matches_x, matches_y, dis_matrix = get_matches_by_reprojection(anchor_pred, pos_prompts, return_dis_matrix=True)
        score_matrix = 1 - dis_matrix.cpu().numpy() / 2
        
        precision_total = len(matches_x)
        for k in range(precision_total):
            if labels1[matches_x[k]] == -1 and labels2[matches_y[k]] == -1:
                true_or_false.append(True)
                continue
            if labels1[matches_x[k]] == labels2[matches_y[k]]:
                true_or_false.append(True)
                correct += 1
            else:
                true_or_false.append(False)
        label_matrix = np.zeros((len(labels1), len(labels2)), dtype=bool)
        label_mask = np.ones((len(labels1), len(labels2)), dtype=bool)
        for i in range(len(labels1)):
            for j in range(len(labels2)):
                if labels1[i] == labels2[j]:
                    label_matrix[i, j] = True
                if labels1[i] == -1 or labels2[j] == -1:
                    label_mask[i, j] = False
        N = IPAA_total - precision_total # predicted unmatched instances
        P = precision_total
        TP = correct
        FP = precision_total - correct
        TN = 0
        for i in range(len(labels1)):
            if labels1[i] == -1:
                continue
            if i in matches_x:
                continue
            idx1 = labels1[i]
            if idx1 not in labels2:
                TN += 1
        for i in range(len(labels2)):
            if labels2[i] == -1:
                continue
            if i in matches_y:
                continue
            idx2 = labels2[i]
            if idx2 not in labels1:
                TN += 1
        IPAA_correct = (TN + correct)
        
        result['scores'] = score_matrix.flatten()[label_mask.flatten()]
        result['labels'] = label_matrix.flatten()[label_mask.flatten()]
        result['IPAA_total'] = IPAA_total
        result['IPAA_correct'] = IPAA_correct
        result['true_or_false'] = true_or_false
        result['matches_x'] = matches_x
        result['matches_y'] = matches_y
        result['matches_scores'] = score_matrix[matches_x, matches_y]
    else:
        precision_total = 0
    result['correct'] = correct
    result['precision_total'] = precision_total

    return result



def multi_view_matching_evaluation(feas, reid_feas=None, mode='l2', thresh=0.6, matching_mode='hungarian'):
    view_num = len(feas)
    cam_pairs = []
    for i in range(view_num - 1):
        for j in range(i + 1, view_num):
            cam_pairs.append((i, j))
    # Step 1: Compute pairwise similarity between views
    pairwise_similarity = {}
    for cam_pair in cam_pairs:
        view1, view2 = cam_pair
        anchor_fea = feas[view1]
        pos_fea = feas[view2]
        if len(anchor_fea) == 0 or len(pos_fea) == 0:
            continue
        dis_matrix = get_dis_matrix(anchor_fea, pos_fea, mode=mode)
        norm_dis_matrix = dis_matrix_normalization(dis_matrix)
        score_matrix = 1 - norm_dis_matrix.cpu().detach().numpy()
        pairwise_similarity[(view1, view2)] = score_matrix
        pairwise_similarity[(view2, view1)] = score_matrix.T
    
    # Step 2: Create a similarity graph where nodes are people in each view
    # Each node is connected to nodes in other views with similarity above threshold
    edges = []
    node_map = []
    total_nodes = 0
    
    for view, emb in enumerate(feas):
        node_map.extend([(view, i) for i in range(len(emb))])
        total_nodes += len(emb)
    
    # Map from nodes to graph indices
    node_idx_map = {node: idx for idx, node in enumerate(node_map)}

    for (view1, view2), sim_matrix in pairwise_similarity.items():
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                if sim_matrix[i, j] > thresh:
                    node1 = node_idx_map[(view1, i)]
                    node2 = node_idx_map[(view2, j)]
                    edges.append((node1, node2, sim_matrix[i, j]))  # Add edge with similarity score
    
    # Step 3: Cluster nodes based on the similarity graph for multi-view consistency
    edge_array = np.array([[e[0], e[1]] for e in edges])
    edge_weights = np.array([e[2] for e in edges])
    
    clustering = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="average", distance_threshold=1 - thresh)
    similarity_matrix = np.zeros((total_nodes, total_nodes))
    
    for (node1, node2, weight) in edges:
        similarity_matrix[node1, node2] = 1 - weight  # Use 1 - similarity as distance
    
    labels = clustering.fit_predict(similarity_matrix)

    # Step 4: Extract matches from clustering labels
    final_associations = [{} for _ in range(n_views)]
    for label in np.unique(labels):
        nodes_in_cluster = np.where(labels == label)[0]
        person_group = {}
        for node in nodes_in_cluster:
            view, person_idx = node_map[node]
            person_group[view] = person_idx
        
        # Add matches for each person in this cluster
        for view, person_idx in person_group.items():
            final_associations[view][person_idx] = label
    
    return final_associations


def matching_inference(pred_pts, gt_pts, mode='l1', thresh=0.6):
    result = {}
    if len(pred_pts):
        pts1 = pred_pts.repeat_interleave(len(gt_pts), 0)
        pts2 = gt_pts.repeat(len(pred_pts), 1)
        if mode == 'l1':
            dis_matrix = (pts1 - pts2).abs().sum(-1).reshape(len(pred_pts), len(gt_pts))
        else:
            dis_matrix = torch.sqrt((pts1 - pts2).pow(2).sum(-1)).reshape(len(pred_pts), len(gt_pts))
        norm_dis_matrix = dis_matrix.cpu().detach().numpy()
        matches_x, matches_y = linear_sum_assignment(norm_dis_matrix)
        # min_v = norm_dis_matrix.min(axis=1, keepdims=True)
        # max_v = norm_dis_matrix.max(axis=1, keepdims=True)
        # min_v = norm_dis_matrix.min()
        min_v = 0
        max_v = norm_dis_matrix.max()
        norm_dis_matrix = 1 - (norm_dis_matrix - min_v) / (max_v - min_v + 1e-6)
        matches_x = list(matches_x)
        matches_y = list(matches_y)
        for k in range(len(matches_x) - 1, -1, -1):
            if len(matches_x) == 1:
                break
            if norm_dis_matrix[matches_x[k], matches_y[k]] < thresh:
                del matches_x[k]
                del matches_y[k]
        result['matches_x'] = matches_x
        result['matches_y'] = matches_y
        result['true_or_false'] = [True for _ in range(len(matches_x))]
    else:
        result['matches_x'] = []
        result['matches_y'] = []
        result['true_or_false'] = []

    return result

def scalar_clip(x, min, max):
    """
    input: scalar
    """
    if x < min:
        return min
    if x > max:
        return max
    return x

def crop_feat(img_copy, bbox, zoomout_ratio=1.0):
    """
    input: img and reuqirement on zoomout ratio
    where img_size = (max_x, max_y)
    return: a single img crop
    """
    x1, y1, x2, y2 = bbox

    img_feat = None

    if zoomout_ratio == 1.0:
        img_feat = img_copy[int(y1):int(y2+1), int(x1):int(x2+1), :]
    elif zoomout_ratio > 1:
        h = y2 - y1
        w = x2 - x1
        img_feat = img_copy[int(max(0,y1-h*(zoomout_ratio-1)/2)):int(min(max_y,y2+1+h*(zoomout_ratio-1)/2)),
            int(max(0,x1-w*(zoomout_ratio-1)/2)):int(min(max_x,x2+1+w*(zoomout_ratio-1)/2)), :]
    return img_feat

def get_graph(bboxes, device):
    center_x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    center_y = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    u = []
    v = []
    avg_heights = []
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            # if abs(center_x[i] - center_x[j]) >= 0.24 or abs(center_y[i] - center_y[j]) >= 0.49:
            #     continue
            if abs(bboxes[i, 4] - bboxes[j, 4]) >= 5.:
                continue
            u.append(i)
            v.append(j)
            avg_height = (bboxes[i, 3] - bboxes[i, 1] + bboxes[j, 3] - bboxes[j, 1]) * 0.5
            avg_heights.append(avg_height)
    graph = dgl.graph((u + v, v + u), num_nodes=len(bboxes), idtype=torch.int32, device=device)  # undirected graph
    # dis_x = (center_x[u] - center_x[v]) * 0.25 # range: [-0.25, 0.25] * 1.0
    # dis_y = (center_y[u] - center_y[v]) * 0.25 # range: [-0.5, 0.5] * 0.5
    avg_heights = torch.stack(avg_heights)
    dis_x = (center_x[u] - center_x[v]) / avg_heights * 0.05
    dis_y = (center_y[u] - center_y[v]) / avg_heights * 0.05
    pos_embedding = 0.25 - torch.vstack((dis_x, dis_y)).T # range: [0, 0.5]
    neg_embedding = 0.75 - torch.vstack((dis_x, dis_y)).T # range: [0.5, 1]
    edge_feature = torch.cat((pos_embedding, neg_embedding), dim=0)
    return graph, edge_feature

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    if isinstance(box1, np.ndarray):
        box1 = np.asarray(box1, dtype=np.float32)
        box2 = np.asarray(box2, dtype=np.float32)
    
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    if isinstance(box1, torch.Tensor):
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    elif isinstance(box1, np.ndarray):
        inter = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
    else:
        raise ValueError('bbox must be array or tensor')

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        if isinstance(box1, torch.Tensor):
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        else:
            cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                if isinstance(box1, torch.Tensor):
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                else:
                    v = (4 / math.pi ** 2) * np.power(np.arctan(w2 / h2) - np.arctan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                alpha[np.isnan(alpha)] = 0.
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

def xyxy2xywh(bboxes):
    x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    y = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    return torch.stack([x, y, w, h], dim=1)


def get_dis(fea1, fea2, reid_fea1=None, reid_fea2=None, mode='l2', percentage=1):
    if len(fea1) and len(fea2):
        multi_prediction = False
        if len(fea1.shape) != len(fea2.shape):
            multi_prediction = True
        if multi_prediction:
            fea2 = fea2.unsqueeze(1)
        if mode == 'l2':
            distances = torch.sqrt((fea1 - fea2).pow(2).sum(-1))
        elif mode == 'l1':
            distances = (fea1 - fea2).abs().sum(-1)
        elif mode == 'giou':
            boxes1 = fea1.reshape(-1, 4)
            boxes2 = fea2.reshape(-1, 4)
            distances = 1 - bbox_iou(boxes1.T, boxes2.T, x1y1x2y2=True, GIoU=True)
        if multi_prediction:
            min_values, _ = torch.min(distances, dim=-1)
            return min_values.mean(-1)
        if percentage == 1:
            return distances.mean(-1)
        selected_num = int(len(fea1) * percentage)
        if selected_num == 0:
            selected_num = 1
        values, indices = torch.topk(distances, selected_num, largest=False)
        return values.mean(-1)
    else:
        return 0


def matching_nn_filter(dis_matrix, matches_x, matches_y):
    match_num = len(matches_x)
    if len(dis_matrix) == match_num:
        indices = np.argmin(dis_matrix, axis=1)
        mask = indices == matches_y
    else:
        indices = np.argmin(dis_matrix, axis=0)
        mask = indices == matches_x
    new_matches_x = matches_x[mask]
    new_matches_y = matches_y[mask]
    if len(new_matches_x):
        return new_matches_x, new_matches_y
    else:
        return matches_x, matches_y


def inter_cls_distance(fea, mode='l2'):
    avg_fea = fea.mean(dim=0)
    if mode == 'l2':
        dis = torch.sqrt((avg_fea - fea).pow(2).sum(-1)).mean()
    elif mode == 'l1':
        dis = (avg_fea - fea).abs().sum(-1).mean()
    return dis


def get_matching_dis(fea1, fea2, reid_fea1=None, reid_fea2=None, mode='l2', thresh=0.0, dummy=False, adaptive=False, matching_mode='hungarian', alpha=0.1, return_score=False):
    # mask = torch.where((fea1[:, 0] > 1.) | (fea1[:, 1] > 1.) | (fea1[:, 2] < 0) | (fea1[:, 3] < 0), False, True)
    # if mask.sum() != 0:
    #     fea1 = fea1[mask]
    #     fea1 = torch.clamp(fea1, min=0.0, max=1.0)
    
    if len(fea1) and len(fea2):
        if dummy:
            dis_matrix, matches = augmented_hungarian_matching(fea1, fea2, cost_threshold=1.0, mode=mode)
            matches_x, matches_y = matches
            if mode == 'giou':
                norm_dis_matrix = dis_matrix / 2
            else:
                norm_dis_matrix = dis_matrix_normalization(dis_matrix)
        else:
            matches_x = []
            matches_y = []
        if not len(matches_x):
            dis_matrix = get_dis_matrix(fea1, fea2, mode=mode)
            if mode == 'giou':
                norm_dis_matrix = dis_matrix / 2
            else:
                norm_dis_matrix = dis_matrix_normalization(dis_matrix)
            
            if reid_fea1 is not None:
                reid_dis_matrix = get_dis_matrix(reid_fea1, reid_fea2, mode=mode)
                norm_reid_dis_matrix = dis_matrix_normalization(reid_dis_matrix)
                norm_dis_matrix = norm_dis_matrix * (1 - alpha) + norm_reid_dis_matrix * alpha
            
            if matching_mode == 'hungarian':
                matches_x, matches_y = linear_sum_assignment(norm_dis_matrix.cpu().detach().numpy())
                # matches_x, matches_y = matching_nn_filter(norm_dis_matrix.cpu().detach().numpy(), matches_x, matches_y)
            else:
                cost, x, y = lap.lapjv(norm_dis_matrix.cpu().detach().numpy(), extend_cost=True, cost_limit=(1-thresh))
                matches_x = []
                matches_y = []
                for i in range(len(x)):
                    if x[i] < 0:
                        continue
                    matches_x.append(i)
                    matches_y.append(x[i])
                matches_x = np.asarray(matches_x, dtype=int)
                matches_y = np.asarray(matches_y, dtype=int)

        score_matrix = 1 - norm_dis_matrix.detach().clone()
        mask = (score_matrix[matches_x, matches_y] >= thresh).cpu().detach().numpy()

        best_dis = dis_matrix[matches_x, matches_y].sum() / len(matches_x)
        
        if return_score:
            return score_matrix[matches_x[mask], matches_y[mask]], (matches_x[mask], matches_y[mask])
        return best_dis, (matches_x[mask], matches_y[mask])

        if adaptive:
            scores = score_matrix[matches_x, matches_y]
            best_dis = (dis_matrix[matches_x, matches_y] * scores).sum() / len(matches_x)
            return best_dis, (matches_x, matches_y)
        
        if mode == 'iou':
            mask = (dis_matrix[matches_x, matches_y] != 1).cpu().detach().numpy()
        else:
            mask = (score_matrix[matches_x, matches_y] >= thresh).cpu().detach().numpy()
        
        if len(matches_x[mask]) == 0:
            # best_dis = dis_matrix[matches_x, matches_y].sum() / len(matches_x)
            # return best_dis, (matches_x, matches_y)
            return 0, ([], [])
        else:
            best_dis = dis_matrix[matches_x[mask], matches_y[mask]].sum() / len(matches_x[mask])
            return best_dis, (matches_x[mask], matches_y[mask])
    else:
        return 0, ([], [])


def get_matches_by_reprojection(anchor_preds, sample_prompts, return_dis_matrix=False):
    pred_dis_matrix = get_dis_matrix(anchor_preds, sample_prompts, mode='giou') # range: [0, 2]
    pred_matches_x, pred_matches_y = linear_sum_assignment(pred_dis_matrix.cpu().detach().numpy())
    matched_pred_dis = pred_dis_matrix[pred_matches_x, pred_matches_y]
    mask = matched_pred_dis < 1
    mask = mask.cpu().numpy()
    final_matched_pred_dis = matched_pred_dis[mask]
    if len(final_matched_pred_dis) == 0:
        if return_dis_matrix:
            matches_x = []
            matches_y = []
        else:
            _, select_match_id = torch.min(matched_pred_dis, dim=0)
            select_ids = [select_match_id.item()]
            matches_x = pred_matches_x[select_ids]
            matches_y = pred_matches_y[select_ids]
    else:
        matches_x = pred_matches_x[mask]
        matches_y = pred_matches_y[mask]
    if return_dis_matrix:
        return matches_x, matches_y, pred_dis_matrix
    else:
        reprojection_dis = get_dis(anchor_preds[matches_x], sample_prompts[matches_y], mode='l1') 
        return matches_x, matches_y, reprojection_dis

def get_matching_dis_reprojection_loss(anchor_prompts, anchor_fea, anchor_preds, sample_prompts, sample_fea, sample_preds, reid_fea1=None, reid_fea2=None, mode='l2', thresh=0.0, adaptive=False, matching_mode='hungarian', alpha=0.1):
    if len(anchor_fea) and len(sample_fea):
        matches_x1, matches_y1, reprojection_dis1 = get_matches_by_reprojection(anchor_preds, sample_prompts)
        matches_x2, matches_y2, reprojection_dis2 = get_matches_by_reprojection(anchor_prompts, sample_preds)

        view_dis1 = get_dis(anchor_fea[matches_x1], sample_fea[matches_y1], mode='l2')
        view_dis2 = get_dis(anchor_fea[matches_x2], sample_fea[matches_y2], mode='l2')

        return reprojection_dis1, reprojection_dis2, view_dis1, view_dis2

        dis_matrix = get_dis_matrix(fea1, fea2, mode=mode)
        norm_dis_matrix = dis_matrix_normalization(dis_matrix)
        
        if reid_fea1 is not None:
            reid_dis_matrix = get_dis_matrix(reid_fea1, reid_fea2, mode=mode)
            norm_reid_dis_matrix = dis_matrix_normalization(reid_dis_matrix)
            norm_dis_matrix = norm_dis_matrix * (1 - alpha) + norm_reid_dis_matrix * alpha
        
        if matching_mode == 'hungarian':
            matches_x, matches_y = linear_sum_assignment(norm_dis_matrix.cpu().detach().numpy())
        else:
            cost, x, y = lap.lapjv(norm_dis_matrix.cpu().detach().numpy(), extend_cost=True, cost_limit=(1-thresh))
            matches_x = []
            matches_y = []
            for i in range(len(x)):
                if x[i] < 0:
                    continue
                matches_x.append(i)
                matches_y.append(x[i])
            matches_x = np.asarray(matches_x, dtype=int)
            matches_y = np.asarray(matches_y, dtype=int)

        score_matrix = 1 - norm_dis_matrix.detach().clone()

        # dis_matrix, matches = augmented_hungarian_matching(fea1, fea2, cost_threshold=0.5)
        # matches_x, matches_y = matches

        best_dis = dis_matrix[matches_x, matches_y].sum() / len(matches_x)
        return best_dis, (matches_x, matches_y)

        if adaptive:
            scores = score_matrix[matches_x, matches_y]
            best_dis = (dis_matrix[matches_x, matches_y] * scores).sum() / len(matches_x)
            return best_dis, (matches_x, matches_y)
        
        if mode == 'iou':
            mask = (dis_matrix[matches_x, matches_y] != 1).cpu().detach().numpy()
        else:
            mask = (score_matrix[matches_x, matches_y] >= thresh).cpu().detach().numpy()
        
        if len(matches_x[mask]) == 0:
            # best_dis = dis_matrix[matches_x, matches_y].sum() / len(matches_x)
            # return best_dis, (matches_x, matches_y)
            return 0, ([], [])
        else:
            best_dis = dis_matrix[matches_x[mask], matches_y[mask]].sum() / len(matches_x[mask])
            return best_dis, (matches_x[mask], matches_y[mask])
    else:
        return 0, 0, 0, 0


def augmented_hungarian_matching(
    fea1: torch.Tensor,
    fea2: torch.Tensor,
    reid_fea1: torch.Tensor = None,
    reid_fea2: torch.Tensor = None,
    cost_threshold: float = 0.5,
    mode: float = 'giou',
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Performs augmented Hungarian matching between two sets of embeddings.

    Allows items to remain unmatched if the minimum cost exceeds the threshold.
    Uses cosine similarity and converts it to cost (1 - similarity).

    Args:
        fea1: Tensor of shape (N, D) for N items in set A.
        fea2: Tensor of shape (M, D) for M items in set B.
        cost_threshold: The maximum allowable cost for a valid match.
                        Corresponds to a minimum similarity of (1 - cost_threshold).
                        E.g., cost_threshold=0.5 means similarity >= 0.5 is required.

    Returns:
        A tuple containing:
        - matched_pairs: List of tuples (index_a, index_b) for matched items.
        - unmatched_a: List of indices from set A that were not matched.
        - unmatched_b: List of indices from set B that were not matched.
    """
    N = fea1.shape[0]
    M = fea2.shape[0]

    # Handle empty inputs
    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))

    # --- 1. Calculate the cost matrix between real items using the specified metric ---
    cost_matrix = get_dis_matrix(fea1, fea2, mode) # Shape (N, M)

    # --- 2. Construct the Augmented Cost Matrix ---
    # Size (N + M) x (N + M)
    augmented_size = N + M
    # Use np.inf for disallowed assignments, handled by linear_sum_assignment
    augmented_cost_matrix = np.full((augmented_size, augmented_size), np.inf)

    # Fill Top-Left (N x M): Real costs
    augmented_cost_matrix[:N, :M] = cost_matrix.cpu().detach().numpy()

    # Fill Top-Right (N x N): Cost for A items being unmatched
    # Diagonal elements represent the cost of item 'i' from A being unmatched
    np.fill_diagonal(augmented_cost_matrix[:N, M:], cost_threshold)

    # Fill Bottom-Left (M x M): Cost for B items being unmatched
    # Diagonal elements represent the cost of item 'j' from B being unmatched
    np.fill_diagonal(augmented_cost_matrix[N:, :M], cost_threshold)

    # Fill Bottom-Right (M x N): Dummy-to-dummy matches (can be low cost, e.g., 0)
    # These are needed for the square matrix but don't represent real assignments
    augmented_cost_matrix[N:, M:] = 0 # Or some other low cost

    # --- 3. Run the Hungarian Algorithm (scipy.optimize.linear_sum_assignment) ---
    # Finds the assignment with the minimum total cost
    try:
        row_ind, col_ind = linear_sum_assignment(augmented_cost_matrix)
    except ValueError as e:
        print(f"Error during linear_sum_assignment: {e}")
        print("This might happen if the cost matrix contains NaN or infinite values"
              " in unexpected places, or if dimensions are inconsistent.")
        return [], list(range(N)), list(range(M))


    # --- 4. Interpret the Results ---
    matches_x = []
    matches_y = []
    unmatched_a = list(range(N)) # Assume all are unmatched initially
    unmatched_b = list(range(M))

    for r, c in zip(row_ind, col_ind):
        cost = augmented_cost_matrix[r, c]

        if r < N and c < M: # Real A item matched with Real B item
            # Check if the cost is acceptable (redundant check if threshold logic is correct, but safe)
            # Use a tolerance for floating point comparison
            if cost <= cost_threshold + 1e-6:
                 # Check if this cost is not infinity (meaning it was a possible match)
                 if np.isfinite(cost):
                    matches_x.append(r)
                    matches_y.append(c)
                    # If matched, remove from unmatched lists
                    if r in unmatched_a: unmatched_a.remove(r)
                    if c in unmatched_b: unmatched_b.remove(c)
            # Else: Cost was > threshold, Hungarian chose it only because dummy match was even worse (shouldn't happen with correct setup)
            # Or cost was np.inf, meaning it was disallowed.
            # In either case, they remain unmatched.

        elif r < N and c >= M: # Real A item matched with Dummy B (i.e., A is unmatched)
            # No action needed, 'r' remains in unmatched_a
            pass # Ensure r stays in unmatched_a

        elif r >= N and c < M: # Dummy A matched with Real B item (i.e., B is unmatched)
            # No action needed, 'c' remains in unmatched_b
            pass # Ensure c stays in unmatched_b

        # else: r >= N and c >= M: Dummy-dummy match, ignore.

    # Consistency check (optional but good): Ensure indices removed were actually present
    # In theory, the logic above handles this, but for complex cases:
    final_unmatched_a = [idx for idx in range(N) if not any(idx == i for i in matches_x)]
    final_unmatched_b = [idx for idx in range(M) if not any(idx == i for i in matches_y)]

    matches_x = np.asarray(matches_x, dtype=int)
    matches_y = np.asarray(matches_y, dtype=int)

    return cost_matrix, (matches_x, matches_y)
    scores = cost_matrix_real[matches_x, matches_y].sum() / len(matches_x)
    return scores, (matches_x, matches_y)

    # return matched_pairs, unmatched_a, unmatched_b # Return the lists derived during iteration
    return matched_pairs, final_unmatched_a, final_unmatched_b # Return the lists derived via filtering


def get_center_prompts(pts1, pts2, matches_x, matches_y, mode='l2'):
    match_num = len(matches_x)
    if match_num < 2:
        return [], []
    matches_x = matches_x[..., None]
    matches_y = matches_y[..., None]
    x_idx1 = matches_x.repeat(len(matches_x), 0)
    x_idx2 = np.tile(matches_x, (len(matches_x), 1))
    x_prompts = pts1[x_idx1.squeeze(-1), x_idx2.squeeze(-1)]

    y_idx1 = matches_y.repeat(len(matches_y), 0)
    y_idx2 = np.tile(matches_y, (len(matches_y), 1))
    y_prompts = pts2[y_idx1.squeeze(-1), y_idx2.squeeze(-1)]

    return x_prompts, y_prompts

def get_center_prompts_labels(center_prompts, preds, mode='l2'):
    idx_row, idx_col = torch.triu_indices(center_prompts.size(0), center_prompts.size(1), offset=1)
    selected_prompts = center_prompts[idx_row, idx_col]
    # row_preds.shape: [selected num of instances, num of view, 4]
    row_preds = preds[idx_row]
    col_preds = preds[idx_col]
    selected_labels = (row_preds + col_preds) * 0.5
    # selected_labels = torch.clamp(selected_labels, min=-1e10, max=1e10)
    return selected_prompts, selected_labels
    

def FPR_95(labels, scores):
    """
    compute FPR@95
    """
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    indices = np.argsort(scores)[::-1]
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    n_match = sum(sorted_labels)
    n_thresh = recall_point * n_match
    thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
    FP = np.sum(sorted_labels[:thresh_index] == 0)
    TN = np.sum(sorted_labels[thresh_index:] == 0)
    return float(FP) / float(FP + TN)


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_logger(cfg, cfg_name, phase='train'):
    this_dir = Path(os.path.dirname(__file__))  ##
    root_output_dir = (this_dir / '..' / cfg.OUTPUT_DIR).resolve()  ##
    tensorboard_log_dir = (this_dir / '..' / cfg.LOG_DIR).resolve()
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / Path(cfg_name)

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = final_output_dir / "tb_logs"
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'model' in states:
        torch.save(states, os.path.join(output_dir, 'model_epoch_'+ str(states['epoch']) + '.pth.tar'))
        torch.save(states,
                   os.path.join(output_dir, 'model_best.pth.tar'))

def convert_rvec_tvec_to_rt(rvec):
    """
    Convert rvec (rotation vector) and tvec (translation vector) to
    rotation matrix and translation vector.

    Parameters:
        rvec (numpy.ndarray): Rotation vector (3x1 or 1x3).

    Returns:
        R (numpy.ndarray): Rotation matrix (3x3).
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    return R


def epipolar_soft_constraint(
        bbox_list1, bbox_list2, intrin1, intrin2, extrin1, extrin2, shape):
    """
    inputs:
    bbox list []
    instrin : (3,3) numpy
    extr: list of len 6
    """
    
    def ext_a2b(ext_a, ext_b):
        T_a2r = ext_a
        T_b2r = ext_b
        
        T_a2b = np.matmul(T_b2r, np.linalg.inv(T_a2r))

        return T_a2b

    def find_line(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        d = (y2 - y1) / (x2 - x1)
        e = y1 - x1 * d
        return [-d, 1, -e]

    def find_foot(a, b, c, pt):
        x1, y1 = pt
        temp = (-1 * (a * x1 + b * y1 + c) / (a * a + b * b))
        x = temp * a + x1
        y = temp * b + y1
        return [x, y]

    def find_dist(pt1, pt2):
        return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

    T_a2b = ext_a2b(extrin1, extrin2)

    dist_matrix = np.zeros((len(bbox_list1), len(bbox_list2)))

    for i in range(len(bbox_list1)):
        for j in range(len(bbox_list2)):
            b1x1, b1y1, b1x2, b1y2 = bbox_list1[i]
            b2x1, b2y1, b2x2, b2y2 = bbox_list2[j]
            bbox1_2dpt = ((b1x1 + b1x2) / 2, (b1y1 + b1y2) / 2)
            bbox2_2dpt = ((b2x1 + b2x2) / 2, (b2y1 + b2y2) / 2)

            # bbox 1 in camera 2
            bbox1_3dpt = np.matmul(np.linalg.inv(intrin1), np.array([*bbox1_2dpt, 1]))
            bbox1_3dpt = np.array([*bbox1_3dpt.tolist(), 1])

            bbox1_in2_3dpt = np.matmul(T_a2b, bbox1_3dpt)[:3]
            bbox1_in2_2dpt = np.matmul(intrin2, bbox1_in2_3dpt)
            bbox1_in2_2dpt = bbox1_in2_2dpt[:2] / bbox1_in2_2dpt[2]

            # camera 1 epipole in camera 2
            epipole1_3dpt = np.array([0, 0, 0, 1])
            epipole1_in2_3dpt = np.matmul(T_a2b, epipole1_3dpt)[:3]
            epipole1_in2_2dpt = np.matmul(intrin2, epipole1_in2_3dpt)
            epipole1_in2_2dpt = epipole1_in2_2dpt[:2] / epipole1_in2_2dpt[2]

            # find epipolar line
            a, b, c = find_line(bbox1_in2_2dpt, epipole1_in2_2dpt)

            foot = find_foot(a, b, c, bbox2_2dpt)
            dist = find_dist(bbox2_2dpt, foot)

            # measure distance
            dist_matrix[i, j] = dist

    # normalize by diagonal line
    diag =  np.sqrt(shape[0]**2 + shape[1]**2)
    dist_matrix = dist_matrix / diag
    return dist_matrix


def cross_view_matching_loss(pred_bboxes, gt_bboxes):
    box1 = pred_bboxes.repeat_interleave(len(gt_bboxes), 0)
    box2 = gt_bboxes.repeat(len(pred_bboxes), 1)
    giou_loss_matrix = 1 - bbox_iou(box1.T, box2.T, x1y1x2y2=True, DIoU=True).reshape(len(pred_bboxes), len(gt_bboxes))
    matches_x, matches_y = linear_sum_assignment(giou_loss_matrix.cpu().detach().numpy())
    giou_loss = giou_loss_matrix[matches_x, matches_y].sum() / len(pred_bboxes)
    return giou_loss

def cross_view_point_matching_loss(pred_pts, gt_bboxes):
    pts1 = pred_pts.repeat_interleave(len(gt_bboxes), 0)
    box2 = gt_bboxes.repeat(len(pred_pts), 1)
    x_dis = ((pts1[:, 0] - box2[:, 0]).abs() + (pts1[:, 0] - box2[:, 2]).abs() - (box2[:, 2] - box2[:, 0])) / (box2[:, 2] - box2[:, 0]) * 0.5
    y_dis = ((pts1[:, 1] - box2[:, 1]).abs() + (pts1[:, 1] - box2[:, 3]).abs() - (box2[:, 3] - box2[:, 1])) / (box2[:, 3] - box2[:, 1]) * 0.5
    dis_matrix = (x_dis + y_dis).reshape(len(pred_pts), len(gt_bboxes))
    matches_x, matches_y = linear_sum_assignment(dis_matrix.cpu().detach().numpy())
    dis_loss = dis_matrix[matches_x, matches_y].sum() / len(pred_pts)
    return dis_loss

def generate_neg_bboxes(bboxes, point=False, in_boundry=False):
    # wh_scale = 0.5
    wh_scale = 0.2
    # xy_scale = 1.
    xy_scale = 0.6
    while 1:
        neg_bboxes = torch.zeros_like(bboxes)
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        center_x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        center_y = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
        if random.random() > 0.5:
            neg_w = w + w * torch.rand(len(bboxes), device=bboxes.device) * wh_scale
            neg_h = h + h * torch.rand(len(bboxes), device=bboxes.device) * wh_scale
        else:
            neg_w = w - w * torch.rand(len(bboxes), device=bboxes.device) * wh_scale
            neg_h = h - h * torch.rand(len(bboxes), device=bboxes.device) * wh_scale
        if random.random() > 0.5:
            neg_center_x = center_x + w * (torch.rand(len(bboxes), device=bboxes.device) * xy_scale)
        else:
            neg_center_x = center_x - w * (torch.rand(len(bboxes), device=bboxes.device) * xy_scale)
        if random.random() > 0.5:
            neg_center_y = center_y + h * (torch.rand(len(bboxes), device=bboxes.device) * xy_scale)
        else:
            neg_center_y = center_y - h * (torch.rand(len(bboxes), device=bboxes.device) * xy_scale)
        neg_bboxes[:, 0] = neg_center_x - neg_w * 0.5
        neg_bboxes[:, 2] = neg_center_x + neg_w * 0.5
        neg_bboxes[:, 1] = neg_center_y - neg_h * 0.5
        neg_bboxes[:, 3] = neg_center_y + neg_h * 0.5
        mask = torch.where((neg_bboxes[:, 0] > 1.) | (neg_bboxes[:, 1] > 1.) | (neg_bboxes[:, 2] < 0) | (neg_bboxes[:, 3] < 0), False, True)
        if in_boundry:
            if mask.sum() != len(neg_bboxes):
                continue
            else:
                break
        else:
            break
    neg_bboxes = neg_bboxes[mask]
    neg_bboxes = torch.clamp(neg_bboxes, min=0.0, max=1.0)

    if point:
        x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        y = bboxes[:, 3]
        pts = torch.stack([x, y], dim=1)
        pt_loss = cross_view_point_matching_loss(pts, neg_bboxes)
        return neg_bboxes, pt_loss * 0.8
    else:
        giou_loss = cross_view_matching_loss(bboxes, neg_bboxes)
        return neg_bboxes, giou_loss * 0.8



def merge_cross_view_associations_with_constraints(pairs):
    """
    Merges cross-view associations into multi-view associations, enforcing constraints
    that each instance can only be associated with one instance per other view.
    
    Args:
        pairs (list of tuples): A list of tuples where each tuple contains two lists.
                                Each list represents the indices of instances that correspond
                                in two different images/views.
    
    Returns:
        list of sets: A list where each set represents a multi-view association.
    """
    # Graph representation with view tracking
    graph = defaultdict(lambda: defaultdict(dict))  # graph[view][instance][other_view] = {instances}
    
    # Build the graph with constraints
    for x, y in pairs:
        if len(x) != len(y):
            raise ValueError("Mismatched pair lengths in associations")
        
        for a, b in zip(x, y):
            view_a, instance_a = a
            view_b, instance_b = b
            
            # Ensure each instance is only connected to one per view
            if view_b in graph[view_a][instance_a]:
                if graph[view_a][instance_a][view_b] != {instance_b}:
                    continue  # Ignore noisy conflicts
            
            if view_a in graph[view_b][instance_b]:
                if graph[view_b][instance_b][view_a] != {instance_a}:
                    continue  # Ignore noisy conflicts
            
            # Add the association
            graph[view_a][instance_a].setdefault(view_b, set()).add(instance_b)
            graph[view_b][instance_b].setdefault(view_a, set()).add(instance_a)
    
    # Convert graph to a multi-view association
    def dfs(node, visited, component):
        visited.add(node)
        component.add(node)
        view, instance = node
        
        for neighbor_view, neighbors in graph[view][instance].items():
            for neighbor in neighbors:
                neighbor_node = (neighbor_view, neighbor)
                if neighbor_node not in visited:
                    dfs(neighbor_node, visited, component)
    
    # Find all connected components
    visited = set()
    multi_view_associations = []
    
    for view in graph:
        for instance in graph[view]:
            node = (view, instance)
            if node not in visited:
                component = set()
                dfs(node, visited, component)
                multi_view_associations.append(component)
    
    return multi_view_associations

def visualize_multi_view_associations(img_paths, multi_view_associations, bboxes, save_dir='vis'):
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        imgs.append(img)
    for k in range(len(multi_view_associations)):
        ins_set = multi_view_associations[k]
        for cam_id, idx in ins_set:
            bbox = bboxes[cam_id][idx]
            color = get_color(k)
            h, w, _ = imgs[cam_id].shape
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            cv2.rectangle(imgs[cam_id], [x1, y1], [x2, y2], color=color, thickness=3)
            label_color = (0, 255, 0)
            cv2.putText(imgs[cam_id], str(k), [x1, y2], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)
    for i, img in enumerate(imgs):
        name = img_paths[i].rsplit('/', 1)[1].rsplit('.', 1)[0]
        cv2.imwrite(os.path.join(save_dir, name + '.jpg'), img)


def normalize_extrinsics_numpy(extrinsics, ref_index):
    """
    Normalize a list of extrinsic (4x4) matrices so that they are expressed in the coordinate system of
    the reference camera (specified by ref_index).
    
    Args:
        extrinsics (list of np.ndarray): List of 4x4 extrinsic matrices.
        ref_index (int): Index of the reference camera.
        
    Returns:
        list of np.ndarray: Normalized extrinsic matrices.
    """
    # Compute the inverse of the reference extrinsic.
    T_ref_inv = np.linalg.inv(extrinsics[ref_index])
    # Normalize: transform each extrinsic E as T_ref_inv @ E.
    normalized = [T_ref_inv @ E for E in extrinsics]
    return normalized

def align_extrinsics_numpy(extrinsics_est, extrinsics_gt, ref_est=0, ref_gt=0):
    """
    Align the estimated extrinsics to the ground-truth coordinate system.
    We compute the transformation T that maps the estimated reference camera to the 
    ground-truth reference camera, then apply T to all estimated extrinsics.
    
    T = E_gt_ref * inv(E_est_ref)
    
    Args:
        extrinsics_est (list of np.ndarray): Estimated 4x4 extrinsic matrices.
        extrinsics_gt (list of np.ndarray): Ground-truth 4x4 extrinsic matrices.
        ref_est (int): Reference index in estimated extrinsics.
        ref_gt (int): Reference index in GT extrinsics.
        
    Returns:
        list of np.ndarray: Aligned estimated extrinsic matrices.
    """
    E_est_ref = extrinsics_est[ref_est]
    E_gt_ref  = extrinsics_gt[ref_gt]
    T = E_gt_ref @ np.linalg.inv(E_est_ref)
    aligned = [T @ E for E in extrinsics_est]
    return aligned

def numpy_rotation_error(R_est, R_gt):
    """
    Computes the angular error in degrees between two rotation matrices using the trace formula.
    
    Args:
        R_est (np.ndarray): Estimated rotation matrix (3x3).
        R_gt  (np.ndarray): Ground-truth rotation matrix (3x3).
        
    Returns:
        float: The rotation error in degrees.
    """
    # Relative rotation: R_diff = R_est * R_gt^T
    R_diff = R_est @ R_gt.T
    # Compute the trace. Clip to avoid numerical issues.
    trace_val = np.clip(np.trace(R_diff), -1.0, 3.0)
    # Compute the angle error in radians.
    angle_rad = np.arccos((trace_val - 1) / 2)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def numpy_intrinsic_scale_error(K_est, K_gt):
    """
    Computes the average relative error between the estimated and ground-truth focal lengths.
    
    Args:
        K_est (np.ndarray): Estimated intrinsic matrix (3x3).
        K_gt  (np.ndarray): Ground-truth intrinsic matrix (3x3).
        
    Returns:
        float: The average relative error (for fx and fy).
    """
    fx_est, fy_est = K_est[0, 0], K_est[1, 1]
    fx_gt,  fy_gt  = K_gt[0, 0], K_gt[1, 1]
    err_fx = np.abs(fx_est - fx_gt) / fx_gt
    err_fy = np.abs(fy_est - fy_gt) / fy_gt
    return (err_fx + err_fy) / 2

def evaluate_multi_view_numpy(extrinsics_est, extrinsics_gt, intrinsics_est, intrinsics_gt, ref_est=0, ref_gt=0):
    """
    Evaluate the errors for multiple cameras, after aligning the estimated extrinsics to the GT coordinate system.
    
    Args:
        extrinsics_est (list of np.ndarray): List of estimated 4x4 extrinsic matrices.
        extrinsics_gt  (list of np.ndarray): List of ground-truth 4x4 extrinsic matrices.
        intrinsics_est (list of np.ndarray): List of estimated 3x3 intrinsic matrices.
        intrinsics_gt  (list of np.ndarray): List of ground-truth 3x3 intrinsic matrices.
        ref_est (int): Reference index for estimated extrinsics.
        ref_gt (int): Reference index for GT extrinsics.
        
    Returns:
        dict: Dictionary containing per-camera rotation errors (in degrees) and intrinsic scale errors,
              along with aggregated summary statistics.
    """
    # First, align the estimated extrinsics to the GT coordinate system.
    aligned_est = align_extrinsics_numpy(extrinsics_est, extrinsics_gt, ref_est, ref_gt)
    
    num_cameras = len(aligned_est)
    rotation_errs = []
    scale_errs = []
    
    for i in range(num_cameras):
        # Extract rotation parts from the 4x4 matrices.
        R_est = aligned_est[i][:3, :3]
        R_gt  = extrinsics_gt[i][:3, :3]
        rot_err = numpy_rotation_error(R_est, R_gt)
        rotation_errs.append(rot_err)
        
        # Process intrinsic errors.
        K_est = intrinsics_est[i]
        K_gt  = intrinsics_gt[i]
        scale_err = numpy_intrinsic_scale_error(K_est, K_gt)
        scale_errs.append(scale_err)
    
    rotation_errs = np.array(rotation_errs)
    scale_errs = np.array(scale_errs)
    
    stats = {
        'rotation_errors_degrees': rotation_errs,
        'intrinsic_scale_errors': scale_errs,
        'mean_rotation_error_deg': float(np.mean(rotation_errs)),
        'median_rotation_error_deg': float(np.median(rotation_errs)),
        'std_rotation_error_deg': float(np.std(rotation_errs)),
        'mean_scale_error': float(np.mean(scale_errs)),
        'median_scale_error': float(np.median(scale_errs)),
        'std_scale_error': float(np.std(scale_errs))
    }
    return stats