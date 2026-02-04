import argparse
import pickle
import lmdb
import json
import numpy as np
from sklearn.metrics import fbeta_score, roc_curve, average_precision_score, auc
import torch
from torchvision.ops import nms
import matplotlib.pyplot as plt
import os

# Metrics: precision, recall, F1, AP, AR (AR is not computable due to the inaccurate annotations)
# We consider a prediction matched if it has overlapping region with a label

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    # 005_CG_JPG_temp_face_partial.lmdb
    # 005_CG_JPG_iter1_mva_tracking_bi_face.lmdb
    parser.add_argument(
        "--pred_path", default="/media/camma-monitor/Storage_postprocessing2/pose_detection/face/005_CG_JPG_iter1_mva_tracking_bi_face_partial.lmdb", nargs='+', help="path to detections"
    )
    parser.add_argument(
        "--anno_path", default='/media/camma-monitor/Storage_postprocessing2/pose_detection_utils/code/anonymization/evaluation/annotations/005_CG.json', nargs='+', help="path to images"
    )
    parser.add_argument(
        "--img_dicts", default='/home2020/home/icube/keqichen/code/4D-OR/datasets/4D-OR/export_holistic_take2_processed/img_dicts.pkl', nargs='+', help="path to images"
    )
    parser.add_argument(
        '--kpt_thresh',
        type=float,
        default=0.0,
        help='Bounding box score threshold')
    parser.add_argument(
        '--vis_thresh',
        type=float,
        default=0.0,
        help='Bounding box score threshold')
    parser.add_argument(
        '--max_num_person',
        type=int,
        default=100,
        help='Bounding box score threshold')
    parser.add_argument('--out_of_body', action='store_true')

    args, rest = parser.parse_known_args()

    return args

def plot_pr_curve(precision_curve, recall_curve, label=None, ap_score=None):
    """
    Plots a Precision-Recall curve.

    Args:
        precision_curve (list or np.ndarray): A list of precision values.
        recall_curve (list or np.ndarray): A list of recall values.
        label (str, optional): The label for the curve in the plot legend.
        ap_score (float, optional): The Average Precision score to display.
    """
    # Ensure inputs are NumPy arrays
    precision = np.array(precision_curve)
    recall = np.array(recall_curve)

    # --- Plotting ---
    # It's standard to plot recall on the x-axis and precision on the y-axis.
    
    # Method 1: A smooth line plot (good for visualization)
    # plt.plot(recall, precision, marker='.')

    # Method 2: A step plot (technically more accurate representation)
    # The 'where='post'' argument creates the classic staircase shape.
    # We add a starting point to make the plot begin at recall=0.
    plt.step(np.insert(recall, 0, 0), np.insert(precision, 0, 1), where='post')
    
    # Fill the area under the curve for better visualization
    plt.fill_between(np.insert(recall, 0, 0), np.insert(precision, 0, 1), step='post', alpha=0.2)

    # --- Styling the Plot ---
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    # Set the limits for a standard P-R curve plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Add a grid for easier reading
    plt.grid(True)
    
    # Create the legend
    if label is not None or ap_score is not None:
        legend_label = f'{label}' if label else ''
        if ap_score is not None:
            # Use the provided AP score or calculate it if not provided
            legend_label += f' (AP = {ap_score:.3f})'
        plt.legend([legend_label.strip()], loc='lower left')


def compute_iou_matrix(pred_boxes, gt_boxes):
    """Compute IoU matrix between all predicted boxes and ground-truth boxes."""
    pred_boxes = np.array(pred_boxes)
    gt_boxes = np.array(gt_boxes)

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)))

    # Compute intersection
    x1 = np.maximum(pred_boxes[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(pred_boxes[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(pred_boxes[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(pred_boxes[:, None, 3], gt_boxes[None, :, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Compute union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    union_area = pred_area[:, None] + gt_area[None, :] - inter_area
    iou_matrix = inter_area / np.clip(union_area, 1e-6, None)  # Avoid division by zero

    return iou_matrix

def compute_tpr_at_fpr(y_true, y_pred_scores, fixed_fpr=0.05):
    fpr, tpr, _ = roc_curve(y_true, y_pred_scores)
    closest_index = np.argmin(np.abs(fpr - fixed_fpr))
    return tpr[closest_index]

def compute_interpolated_ap(precision_curve, recall_curve):
    """
    Computes the 11-point interpolated Average Precision.
    This is the standard metric from Pascal VOC.

    Args:
        precision_curve (np.ndarray): Array of precision values.
        recall_curve (np.ndarray): Array of recall values, assumed to be sorted.
    """
    # 11-point recall levels
    recall_levels = np.linspace(0, 1.0, 11)
    interpolated_precisions = []

    for r in recall_levels:
        # Find all precision values where the recall is >= r
        possible_precisions = precision_curve[recall_curve >= r]
        
        # The interpolated precision is the maximum of these values.
        # If there are none, the precision is 0.
        if possible_precisions.size == 0:
            interpolated_p = 0.0
        else:
            interpolated_p = np.max(possible_precisions)
        
        interpolated_precisions.append(interpolated_p)
    
    # AP is the average of the interpolated precisions
    ap = np.mean(interpolated_precisions)
    return ap


def evaluate_multiple_images(gt_data, pred_data, iou_threshold=0.3):
    """
    Compute precision, recall, and AP for multiple images.
    
    Arguments:
    - gt_data: dict {image_id: [list of gt_boxes]}
    - pred_data: dict {image_id: [(pred_box, confidence)]}
    
    Returns:
    - AP, Precision-Recall Curve
    """

    all_confidences = []
    all_tp = []
    all_fp = []
    total_gt_boxes = 0  # Total GT boxes across all images
    total_gt_boxes_hard = 0
    tp_hard = 0

    for img_id in gt_data:
        gt_boxes = gt_data[img_id]
        hard_mask = gt_boxes[:, -1] > 0
        gt_boxes_hard = gt_boxes[hard_mask]
        pred_entries = pred_data.get(img_id, [])

        if len(pred_entries) == 0:
            # No predictions → all GTs are false negatives
            total_gt_boxes += len(gt_boxes)
            total_gt_boxes_hard += len(gt_boxes_hard)
            continue

        pred_boxes, conf_scores = pred_entries
        pred_boxes = np.array(pred_boxes)
        conf_scores = np.array(conf_scores)

        total_gt_boxes += len(gt_boxes)
        total_gt_boxes_hard += len(gt_boxes_hard)
        sorted_indices = np.argsort(conf_scores)[::-1]  # Sort by confidence (high to low)
        pred_boxes = pred_boxes[sorted_indices]
        conf_scores = conf_scores[sorted_indices]

        # Compute IoU matrix for this image
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
        
        # Match predictions to ground truth
        matched_gt = set()
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))

        for i, pred_box in enumerate(pred_boxes):
            ious = iou_matrix[i]
            best_match = np.argmax(ious)
            max_iou = ious[best_match]

            if max_iou >= iou_threshold and best_match not in matched_gt:
                tp[i] = 1
                matched_gt.add(best_match)
            else:
                fp[i] = 1

        all_confidences.extend(conf_scores)
        all_tp.extend(tp)
        all_fp.extend(fp)
        
        if len(gt_boxes_hard):
            # Compute IoU matrix of hard cases for this image
            iou_matrix_hard = compute_iou_matrix(pred_boxes, gt_boxes_hard)
            
            # Match predictions to ground truth
            matched_gt_hard = set()
            for i, pred_box in enumerate(pred_boxes):
                ious = iou_matrix_hard[i]
                best_match = np.argmax(ious)
                max_iou = ious[best_match]

                if max_iou >= iou_threshold and best_match not in matched_gt_hard:
                    tp_hard += 1
                    matched_gt_hard.add(best_match)

    if total_gt_boxes == 0:
        return 0.0, [], []  # No GT boxes in dataset

    # Sort across all images based on confidence scores
    sorted_indices = np.argsort(-np.array(all_confidences))
    all_tp = np.array(all_tp)[sorted_indices]
    all_fp = np.array(all_fp)[sorted_indices]
    all_confidences = np.array(all_confidences)[sorted_indices]

    # Compute cumulative sums
    tp_cumsum = np.cumsum(all_tp)
    fp_cumsum = np.cumsum(all_fp)

    recall_curve = tp_cumsum / total_gt_boxes
    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP using sklearn's average_precision_score
    # ap = average_precision_score(all_tp, all_confidences)
    ap = compute_interpolated_ap(precision_curve, recall_curve)

    valid_indices = np.where(precision_curve >= 0.6)[0]
    recall_at_60_precision = max(recall_curve[valid_indices]) if len(valid_indices) > 0 else 0
    
    valid_indices = np.where(precision_curve >= 0.9)[0]
    recall_at_90_precision = max(recall_curve[valid_indices]) if len(valid_indices) > 0 else 0
    
    hard_recall = tp_hard / total_gt_boxes_hard

    return ap, precision_curve, recall_curve, recall_at_60_precision, recall_at_90_precision, hard_recall




def compute_iou_vectorized(box, boxes):
    """
    Calculates the IoU of a single box with an array of boxes.
    This is a vectorized implementation.

    Args:
        box (np.ndarray): A single bounding box, shape (4,) [x1, y1, x2, y2].
        boxes (np.ndarray): An array of bounding boxes, shape (N, 4).

    Returns:
        np.ndarray: An array of IoU values, shape (N,).
    """
    # Coordinates of the intersection rectangles
    x1_inter = np.maximum(box[0], boxes[:, 0])
    y1_inter = np.maximum(box[1], boxes[:, 1])
    x2_inter = np.minimum(box[2], boxes[:, 2])
    y2_inter = np.minimum(box[3], boxes[:, 3])

    # Width and height of intersection
    width_inter = np.maximum(0, x2_inter - x1_inter)
    height_inter = np.maximum(0, y2_inter - y1_inter)
    intersection_area = width_inter * height_inter

    # Area of the individual boxes
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union_area = box_area + boxes_area - intersection_area
    
    # Handle division by zero
    iou = intersection_area / np.maximum(union_area, 1e-8)
    return iou


def compute_multiview_recall_curve_vectorized(gt_data, pred_data, iou_threshold=0.5):
    """
    Computes a recall vs. confidence curve using a vectorized IoU calculation
    for improved performance.
    """
    successful_detection_scores = []
    total_gt_objects = 0

    for set_id, gt_set in gt_data.items():
        total_gt_objects += len(gt_set)
        pred_set = pred_data.get(set_id, {})

        for gt_idx, gt_obj_views in gt_set.items():
            is_fully_detected = True
            confidences_for_this_gt = []

            for cam_name, gt_box in gt_obj_views.items():
                cam_preds = pred_set.get(cam_name, [])
                
                if not cam_preds:
                    is_fully_detected = False
                    break

                # --- Vectorized IoU Calculation ---
                # 1. Prepare arrays for vectorized computation
                pred_boxes_np, pred_confs_np = cam_preds
                gt_box_np = np.array(gt_box)

                # 2. Compute all IoUs for this camera view at once
                ious = compute_iou_vectorized(gt_box_np, pred_boxes_np)
                
                # 3. Find the best match
                best_match_idx = np.argmax(ious)
                best_iou = ious[best_match_idx]
                
                if best_iou >= iou_threshold:
                    # Get the confidence of the best matching prediction
                    best_pred_conf = pred_confs_np[best_match_idx]
                    confidences_for_this_gt.append(best_pred_conf)
                else:
                    is_fully_detected = False
                    break
            
            if is_fully_detected:
                overall_score = min(confidences_for_this_gt)
                successful_detection_scores.append(overall_score)

    if total_gt_objects == 0:
        return np.array([1.0]), np.array([0.0])
    if not successful_detection_scores:
        return np.array([1.0]), np.array([0.0])

    successful_detection_scores.sort(reverse=True)
    
    confidence_thresholds = np.array(successful_detection_scores)
    tp_counts = np.arange(1, len(successful_detection_scores) + 1)
    recall_curve = tp_counts / total_gt_objects
    
    confidence_thresholds = np.insert(confidence_thresholds, 0, 1.0)
    recall_curve = np.insert(recall_curve, 0, 0.0)

    return confidence_thresholds, recall_curve


def evaluate(args):
    kpts_score_threshold = args.kpt_thresh
    vis_score_threshold = args.vis_thresh
    max_num_person = args.max_num_person
    
    # prepare ground-truth data
    if isinstance(args.anno_path, str):
        anno_paths = [args.anno_path]
    else:
        anno_paths = args.anno_path
    
    if isinstance(args.pred_path, str):
        pred_paths = [args.pred_path]
    else:
        pred_paths = args.pred_path
    
    if isinstance(args.img_dicts, str):
        img_dicts_paths = [args.img_dicts]
    else:
        img_dicts_paths = args.img_dicts
    
    file_num = len(anno_paths)
    
    gt_face_boxes_dict = {}
    gt_eyes_boxes_dict = {}
    pred_face_boxes_dict = {}
    pred_eyes_boxes_dict = {}
    pseudo_face_height = 160
    pseudo_face_width = 160
    pseudo_eyes_height = 160
    pseudo_eyes_width = 160
    
    mv_gt_face_boxes_dict = {}
    mv_gt_eyes_boxes_dict = {}
    mv_pred_face_boxes_dict = {}
    mv_pred_eyes_boxes_dict = {}
    
    gt_fullbody_boxes_dict = {}
    gt_boxes_with_face_dict = {}
    gt_boxes_with_eyes_dict = {}
    pred_fullbody_boxes_dict = {}
    for f_n in range(file_num):
        anno_path = anno_paths[f_n]
        pred_path = pred_paths[f_n]
        img_dicts_path = img_dicts_paths[f_n]
        img_dicts_pkl = os.path.join(img_dicts_path)
        with open(img_dicts_pkl, mode='rb') as f:
            img_dicts = pickle.load(f)
        cam_num = len(img_dicts)
        frame_num = len(img_dicts[0])
        for c_id in range(1, cam_num):
            if len(img_dicts[c_id]) < frame_num:
                frame_num = len(img_dicts[c_id])
    
        with open(anno_path, 'r') as fp:
            gt = json.load(fp)
            
        # Step 1: process gt labels
        for f_id in range(frame_num):
            for c_id in range(cam_num):
                img_key = img_dicts[c_id][f_id]
                if img_key not in gt:
                    continue
                boxes = []
                boxes_face_vis = []
                boxes_eyes_vis = []
                face_boxes = []
                eyes_boxes = []
                mv_img_key = str(f_n) + '_' + str(f_id)
                if mv_img_key not in mv_gt_face_boxes_dict:
                    mv_gt_face_boxes_dict[mv_img_key] = {}
                if mv_img_key not in mv_gt_eyes_boxes_dict:
                    mv_gt_eyes_boxes_dict[mv_img_key] = {}
                    
                annos = gt[img_key]
                for anno in annos:
                    box, eye1, eye2, chin, idx, is_hard = anno
                    if not len(box):
                        print(str(f_n) + '_' + img_key)
                    boxes.append(box)
                    boxes_face_vis.append(True)
                    boxes_eyes_vis.append(True)
                    
                    if len(eye1) == 0 and len(eye2) == 0:
                        boxes_face_vis[-1] = False
                        boxes_eyes_vis[-1] = False
                        continue
                    face_kpts = []
                    eyes_kpts = []
                    if len(eye1):
                        face_kpts.append(eye1)
                        eyes_kpts.append(eye1)
                    if len(eye2):
                        face_kpts.append(eye2)
                        eyes_kpts.append(eye2)
                    if len(chin):
                        face_kpts.append(chin)
                    
                    if len(face_kpts):
                        face_kpts = np.array(face_kpts)
                        min_xy = np.min(face_kpts, axis=0)
                        max_xy = np.max(face_kpts, axis=0)
                        center_x = (min_xy[0] + max_xy[0]) * 0.5
                        center_y = (min_xy[1] + max_xy[1]) * 0.5
                        x1 = center_x - pseudo_face_width * 0.5
                        y1 = center_y - pseudo_face_height * 0.5
                        x2 = center_x + pseudo_face_width * 0.5
                        y2 = center_y + pseudo_face_height * 0.5
                        face_box = np.array([x1, y1, x2, y2, is_hard])
                        face_boxes.append(face_box)
                        
                        if idx not in mv_gt_face_boxes_dict[mv_img_key]:
                            mv_gt_face_boxes_dict[mv_img_key][idx] = {}
                        if c_id not in mv_gt_face_boxes_dict[mv_img_key][idx]:
                            mv_gt_face_boxes_dict[mv_img_key][idx][c_id] = []
                        mv_gt_face_boxes_dict[mv_img_key][idx][c_id] = face_box
                    else:
                        boxes_face_vis[-1] = False
                    
                    if len(eyes_kpts):
                        eyes_kpts = np.array(eyes_kpts)
                        min_xy = np.min(eyes_kpts, axis=0)
                        max_xy = np.max(eyes_kpts, axis=0)
                        center_x = (min_xy[0] + max_xy[0]) * 0.5
                        center_y = (min_xy[1] + max_xy[1]) * 0.5
                        x1 = center_x - pseudo_eyes_width * 0.5
                        y1 = center_y - pseudo_eyes_height * 0.5
                        x2 = center_x + pseudo_eyes_width * 0.5
                        y2 = center_y + pseudo_eyes_height * 0.5
                        eyes_box = np.array([x1, y1, x2, y2, is_hard])
                        eyes_boxes.append(eyes_box)
                        
                        if idx not in mv_gt_eyes_boxes_dict[mv_img_key]:
                            mv_gt_eyes_boxes_dict[mv_img_key][idx] = {}
                        if c_id not in mv_gt_eyes_boxes_dict[mv_img_key][idx]:
                            mv_gt_eyes_boxes_dict[mv_img_key][idx][c_id] = []
                        mv_gt_eyes_boxes_dict[mv_img_key][idx][c_id] = eyes_box
                    else:
                        boxes_eyes_vis[-1] = False
                    
                if len(boxes):
                    new_img_key = str(f_n) + '_' + str(f_id) + '_' + str(c_id)
                    # print(boxes)
                    boxes = np.stack(boxes, axis=0)
                    boxes_face_vis = np.stack(boxes_face_vis, axis=0)
                    boxes_eyes_vis = np.stack(boxes_eyes_vis, axis=0)
                    boxes_face = boxes[boxes_face_vis]
                    boxes_eyes = boxes[boxes_eyes_vis]
                    
                    if new_img_key in gt_fullbody_boxes_dict:
                        gt_fullbody_boxes_dict[new_img_key] += boxes
                    else:
                        gt_fullbody_boxes_dict[new_img_key] = boxes
                        
                    if len(boxes_face):
                        if new_img_key in gt_boxes_with_face_dict:
                            gt_boxes_with_face_dict[new_img_key] += boxes_face
                        else:
                            gt_boxes_with_face_dict[new_img_key] = boxes_face
                        
                    if len(boxes_eyes):
                        if new_img_key in gt_boxes_with_eyes_dict:
                            gt_boxes_with_eyes_dict[new_img_key] += boxes_eyes
                        else:
                            gt_boxes_with_eyes_dict[new_img_key] = boxes_eyes
                
                if len(face_boxes):
                    new_img_key = str(f_n) + '_' + str(f_id) + '_' + str(c_id)
                    face_boxes = np.stack(face_boxes, axis=0)
                    # print(face_boxes)
                    if new_img_key in gt_face_boxes_dict:
                        gt_face_boxes_dict[new_img_key] += face_boxes
                    else:
                        gt_face_boxes_dict[new_img_key] = face_boxes
                
                if len(eyes_boxes):
                    new_img_key = str(f_n) + '_' + str(f_id) + '_' + str(c_id)
                    eyes_boxes = np.stack(eyes_boxes, axis=0)
                    # print(eyes_boxes)
                    if new_img_key in gt_eyes_boxes_dict:
                        gt_eyes_boxes_dict[new_img_key] += eyes_boxes
                    else:
                        gt_eyes_boxes_dict[new_img_key] = eyes_boxes
    
        # prepare predictions
        using_nms = True
        env = lmdb.open(pred_path, readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            # Use a cursor for efficient iteration if desired.
            cursor = txn.cursor()
            for f_id in range(frame_num):
                for c_id in range(cam_num):
                    img_key = img_dicts[c_id][f_id]
                    if img_key not in gt:
                        continue
                    new_img_key = str(f_n) + '_' + str(f_id) + '_' + str(c_id)
            
                    mv_img_key = str(f_n) + '_' + str(f_id)
                    if mv_img_key not in mv_pred_face_boxes_dict:
                        mv_pred_face_boxes_dict[mv_img_key] = {}
                    if mv_img_key not in mv_pred_eyes_boxes_dict:
                        mv_pred_eyes_boxes_dict[mv_img_key] = {}
                    
                    if new_img_key in gt_boxes_with_face_dict:
                        gt_face_boxes = gt_boxes_with_face_dict[new_img_key]
                    else:
                        gt_face_boxes = []
                    if new_img_key in gt_boxes_with_eyes_dict:
                        gt_eyes_boxes = gt_boxes_with_eyes_dict[new_img_key]
                    else:
                        gt_eyes_boxes = []
                    
                    anno_key = str(f_id) + '_' + str(c_id)
                    key_kpts = f"{anno_key}/kpts".encode('utf-8')
                    key_kpts_scores = f"{anno_key}/kpts_scores".encode('utf-8')
                    key_kpts_vis = f"{anno_key}/kpts_vis".encode('utf-8')
                    key_boxes = f"{anno_key}/boxes".encode('utf-8')
                    key_boxes_scores = f"{anno_key}/boxes_scores".encode('utf-8')
                    # key_kpts_scores = f"{anno_key}/scores".encode('utf-8')
                    key_matching_scores = f"{anno_key}/matching_scores".encode('utf-8')
                    
                    kpts_data = txn.get(key_kpts)
                    kpts_scores_data = txn.get(key_kpts_scores)
                    kpts_vis_data = txn.get(key_kpts_vis)
                    boxes_data = txn.get(key_boxes)
                    boxes_scores_data = txn.get(key_boxes_scores)
                    matching_scores_data = txn.get(key_matching_scores)
                    if boxes_data is not None:
                        # Deserialize using pickle.
                        kpts = pickle.loads(kpts_data)
                        kpts_scores = pickle.loads(kpts_scores_data)
                        kpts_vis = pickle.loads(kpts_vis_data)
                        boxes = pickle.loads(boxes_data)
                        boxes_scores = pickle.loads(boxes_scores_data)
                        # matching_scores = pickle.loads(matching_scores_data)
                        
                        if args.out_of_body:
                            for k in range(len(boxes)):
                                for q in range(len(kpts[0])):
                                    if kpts[k][q][0] < boxes[k][0] or kpts[k][q][0] > boxes[k][2] or kpts[k][q][1] < boxes[k][1] or kpts[k][q][1] > boxes[k][3]:
                                        kpts_scores[k][q] = 0.
                        
                        sorted_indices = np.argsort(boxes_scores)[::-1]  # Sort by confidence (high to low)
                        kpts = kpts[sorted_indices]
                        kpts_scores = kpts_scores[sorted_indices]
                        kpts_vis = kpts_vis[sorted_indices]
                        boxes = boxes[sorted_indices]
                        boxes_scores = boxes_scores[sorted_indices]
                        
                        if len(boxes):
                            if using_nms:
                                # indices = face_score > kpts_score_threshold
                                torch_boxes = torch.from_numpy(boxes).float()
                                torch_boxes_scores = torch.from_numpy(boxes_scores).float()
                                indices = nms(torch_boxes, torch_boxes_scores, 0.7)[:max_num_person]
                                nms_boxes = torch_boxes[indices].numpy()
                                nms_boxes_scores = torch_boxes_scores[indices].numpy()
                            # indices = face_score > kpts_score_threshold
                            # face_boxes = face_boxes[indices]
                            # face_score = face_score[indices]
                            if len(nms_boxes):
                                pred_fullbody_boxes_dict[new_img_key] = [nms_boxes, nms_boxes_scores]
                        
                        if len(gt_face_boxes):
                            # Compute IoU matrix for this image
                            iou_matrix = compute_iou_matrix(boxes, gt_face_boxes)
                            # Match predictions to ground truth
                            matched_gt = set()
                            face_mask = np.zeros(len(boxes), dtype=bool)
                            for i, box in enumerate(boxes):
                                ious = iou_matrix[i]
                                best_match = np.argmax(ious)
                                max_iou = ious[best_match]

                                if max_iou >= 0.1:# and best_match not in matched_gt:
                                    face_mask[i] = True
                                    matched_gt.add(best_match)
                            
                            boxes_with_face = boxes[face_mask]
                            kpts_with_face = kpts[face_mask]
                            kpts_scores_with_face = kpts_scores[face_mask]
                            kpts_vis_with_face = kpts_vis[face_mask]
                            
                            face_boxes = []
                            face_scores = []
                            for one_face_kpts, one_face_kpts_scores, one_face_kpts_vis in zip(kpts_with_face, kpts_scores_with_face, kpts_vis_with_face):
                                mask = (one_face_kpts_scores > kpts_score_threshold) & (one_face_kpts_vis > vis_score_threshold)
                                new_face_kpts = one_face_kpts[mask]
                                new_face_kpts_scores = one_face_kpts_scores[mask]
                                if len(new_face_kpts):
                                    min_xy = np.min(new_face_kpts, axis=0)
                                    max_xy = np.max(new_face_kpts, axis=0)
                                    center_x = (min_xy[0] + max_xy[0]) * 0.5
                                    center_y = (min_xy[1] + max_xy[1]) * 0.5
                                    x1 = center_x - pseudo_face_width * 0.5
                                    y1 = center_y - pseudo_face_height * 0.5
                                    x2 = center_x + pseudo_face_width * 0.5
                                    y2 = center_y + pseudo_face_height * 0.5
                                    face_box = np.array([x1, y1, x2, y2])
                                    face_score = np.mean(new_face_kpts_scores, axis=0)
                                    face_boxes.append(face_box)
                                    face_scores.append(face_score)
                            
                            if len(face_boxes):
                                face_boxes = np.stack(face_boxes, axis=0)
                                face_scores = np.array(face_scores)
                                if using_nms:
                                    face_boxes = torch.from_numpy(face_boxes).float()
                                    face_scores = torch.from_numpy(face_scores).float()
                                    indices = nms(face_boxes, face_scores, 0.6)[:max_num_person]
                                    face_boxes = face_boxes[indices].numpy()
                                    face_scores = face_scores[indices].numpy()
                                if len(face_boxes):
                                    pred_face_boxes_dict[new_img_key] = [face_boxes, face_scores]
                                    mv_pred_face_boxes_dict[mv_img_key][c_id] = [face_boxes, face_scores]
                        
                        if len(gt_eyes_boxes):
                            # Compute IoU matrix for this image
                            iou_matrix = compute_iou_matrix(boxes, gt_eyes_boxes)
                            # Match predictions to ground truth
                            matched_gt = set()
                            eyes_mask = np.zeros(len(boxes), dtype=bool)
                            for i, box in enumerate(boxes):
                                ious = iou_matrix[i]
                                best_match = np.argmax(ious)
                                max_iou = ious[best_match]

                                if max_iou >= 0.1:# and best_match not in matched_gt:
                                    eyes_mask[i] = True
                                    matched_gt.add(best_match)
                            
                            boxes_with_eyes = boxes[eyes_mask]
                            kpts_with_eyes = kpts[eyes_mask]
                            kpts_scores_with_eyes = kpts_scores[eyes_mask]
                            kpts_vis_with_eyes = kpts_vis[eyes_mask]
                            
                            eyes_boxes = []
                            eyes_scores = []
                            for one_eyes_kpts, one_eyes_kpts_scores, one_eyes_kpts_vis in zip(kpts_with_eyes[:, :2], kpts_scores_with_eyes[:, :2], kpts_vis_with_eyes[:, :2]):
                                mask = (one_eyes_kpts_scores > kpts_score_threshold) & (one_eyes_kpts_vis > vis_score_threshold)
                                new_eyes_kpts = one_eyes_kpts[mask]
                                new_eyes_kpts_scores = one_eyes_kpts_scores[mask]
                                if len(new_eyes_kpts):
                                    min_xy = np.min(new_eyes_kpts, axis=0)
                                    max_xy = np.max(new_eyes_kpts, axis=0)
                                    center_x = (min_xy[0] + max_xy[0]) * 0.5
                                    center_y = (min_xy[1] + max_xy[1]) * 0.5
                                    x1 = center_x - pseudo_eyes_width * 0.5
                                    y1 = center_y - pseudo_eyes_height * 0.5
                                    x2 = center_x + pseudo_eyes_width * 0.5
                                    y2 = center_y + pseudo_eyes_height * 0.5
                                    eyes_box = np.array([x1, y1, x2, y2])
                                    eyes_score = np.mean(new_eyes_kpts_scores, axis=0)
                                    eyes_boxes.append(eyes_box)
                                    eyes_scores.append(eyes_score)
                            
                            if len(eyes_boxes):
                                eyes_boxes = np.stack(eyes_boxes, axis=0)
                                eyes_scores = np.array(eyes_scores)
                                if using_nms:
                                    eyes_boxes = torch.from_numpy(eyes_boxes).float()
                                    eyes_scores = torch.from_numpy(eyes_scores).float()
                                    indices = nms(eyes_boxes, eyes_scores, 0.6)[:max_num_person]
                                    eyes_boxes = eyes_boxes[indices].numpy()
                                    eyes_scores = eyes_scores[indices].numpy()
                                if len(eyes_boxes):
                                    pred_eyes_boxes_dict[new_img_key] = [eyes_boxes, eyes_scores]
                                    mv_pred_eyes_boxes_dict[mv_img_key][c_id] = [eyes_boxes, eyes_scores]
        env.close()
    
    ap, precision_curve, recall_curve, recall_at_60_precision, recall_at_90_precision, hard_recall = evaluate_multiple_images(gt_fullbody_boxes_dict, pred_fullbody_boxes_dict, 0.5)
    print('Full body')
    print(f'Precision: {precision_curve[-1]}')
    print(f'Recall: {recall_curve[-1]}')
    print(f'Average Precision: {ap}')
    print(f'Hard Case Recall: {hard_recall}')
    # print(f'Recall at 60 precision: {recall_at_60_precision}')
    # print(f'Recall at 90 precision: {recall_at_90_precision}')
    
    ap, precision_curve, recall_curve, recall_at_60_precision, recall_at_90_precision, hard_recall = evaluate_multiple_images(gt_face_boxes_dict, pred_face_boxes_dict)
    mv_thresholds, mv_recall = compute_multiview_recall_curve_vectorized(mv_gt_face_boxes_dict, mv_pred_face_boxes_dict, iou_threshold=0.3)
    print('Face')
    print(f'Precision: {precision_curve[-1]}')
    print(f'Recall: {recall_curve[-1]}')
    print(f'Average Precision: {ap}')
    print(f'Hard Case Recall: {hard_recall}')
    # print(f'Recall at 60 precision: {recall_at_60_precision}')
    # print(f'Recall at 90 precision: {recall_at_90_precision}')
    print(f'Multi-view recall: {mv_recall[-1]}')

    ap, precision_curve, recall_curve, recall_at_60_precision, recall_at_90_precision, hard_recall = evaluate_multiple_images(gt_eyes_boxes_dict, pred_eyes_boxes_dict)
    mv_thresholds, mv_recall = compute_multiview_recall_curve_vectorized(mv_gt_eyes_boxes_dict, mv_pred_eyes_boxes_dict, iou_threshold=0.3)
    print('Eyes')
    print(f'Precision: {precision_curve[-1]}')
    print(f'Recall: {recall_curve[-1]}')
    print(f'Average Precision: {ap}')
    print(f'Hard Case Recall: {hard_recall}')
    # print(f'Recall at 60 precision: {recall_at_60_precision}')
    # print(f'Recall at 90 precision: {recall_at_90_precision}')
    print(f'Multi-view recall: {mv_recall[-1]}')


if __name__ == '__main__':
    args = parse_args()

    evaluate(args)
