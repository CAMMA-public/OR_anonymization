import argparse
import pickle
import lmdb
import json
import numpy as np
from sklearn.metrics import fbeta_score, roc_curve, average_precision_score, auc
import torch
from torchvision.ops import nms
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--pred_path", default="pose_detection/face/005_CG_JPG_iter1_mva_tracking_bi_face_partial.lmdb", nargs='+', help="path to detections"
    )
    parser.add_argument(
        "--anno_path", default='pose_detection_utils/code/anonymization/evaluation/annotations/005_CG.json', nargs='+', help="path to images"
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

def compute_iou_matrix(pred_boxes, gt_boxes):
    pred_boxes = np.array(pred_boxes)
    gt_boxes = np.array(gt_boxes)

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)))

    x1 = np.maximum(pred_boxes[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(pred_boxes[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(pred_boxes[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(pred_boxes[:, None, 3], gt_boxes[None, :, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    union_area = pred_area[:, None] + gt_area[None, :] - inter_area
    iou_matrix = inter_area / np.clip(union_area, 1e-6, None)

    return iou_matrix

def compute_interpolated_ap(precision_curve, recall_curve):
    recall_levels = np.linspace(0, 1.0, 11)
    interpolated_precisions = []

    for r in recall_levels:
        possible_precisions = precision_curve[recall_curve >= r]
        if possible_precisions.size == 0:
            interpolated_p = 0.0
        else:
            interpolated_p = np.max(possible_precisions)
        interpolated_precisions.append(interpolated_p)
    
    ap = np.mean(interpolated_precisions)
    return ap

def compute_partial_ap(precision_curve, recall_curve, min_recall=0.9):
    """ Calculate the Area Under the PR Curve stricly from min_recall to 1.0 """
    valid_mask = recall_curve >= min_recall
    if not np.any(valid_mask):
        return 0.0
    
    p_valid = precision_curve[valid_mask]
    r_valid = recall_curve[valid_mask]
    
    if r_valid[0] > min_recall:
        idx = np.where(recall_curve < min_recall)[0]
        if len(idx) > 0:
            p_boundary = precision_curve[idx[-1]]
            r_valid = np.insert(r_valid, 0, min_recall)
            p_valid = np.insert(p_valid, 0, p_boundary)
        else:
            r_valid = np.insert(r_valid, 0, min_recall)
            p_valid = np.insert(p_valid, 0, p_valid[0])

    pap = np.trapz(p_valid, r_valid) / (1.0 - min_recall)
    return pap

def get_optimal_f_metrics(precision_curve, recall_curve, confidences, tp_hard_cumsum, total_gt_boxes_hard, beta=2.0):
    """ Find the threshold that maximizes F-beta, and return the metrics at that point """
    if len(precision_curve) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    f_beta_curve = (1 + beta**2) * (precision_curve * recall_curve) / ((beta**2 * precision_curve) + recall_curve + 1e-8)
    best_idx = np.argmax(f_beta_curve)
    
    best_f_beta = f_beta_curve[best_idx]
    best_thresh = confidences[best_idx]
    best_p = precision_curve[best_idx]
    best_r = recall_curve[best_idx]
    
    best_hard_r = tp_hard_cumsum[best_idx] / total_gt_boxes_hard if total_gt_boxes_hard > 0 else 0.0
    
    return best_f_beta, best_thresh, best_p, best_r, best_hard_r


def evaluate_multiple_images(gt_data, pred_data, iou_threshold=0.3):
    all_confidences = []
    all_tp = []
    all_fp = []
    all_tp_hard = []
    
    total_gt_boxes = 0 
    total_gt_boxes_hard = 0

    for img_id in gt_data:
        gt_boxes = gt_data[img_id]
        hard_mask = gt_boxes[:, -1] > 0
        pred_entries = pred_data.get(img_id, [])

        if len(pred_entries) == 0:
            total_gt_boxes += len(gt_boxes)
            total_gt_boxes_hard += np.sum(hard_mask)
            continue

        pred_boxes, conf_scores = pred_entries
        pred_boxes = np.array(pred_boxes)
        conf_scores = np.array(conf_scores)

        total_gt_boxes += len(gt_boxes)
        total_gt_boxes_hard += np.sum(hard_mask)
        
        sorted_indices = np.argsort(conf_scores)[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        conf_scores = conf_scores[sorted_indices]

        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
        
        matched_gt = set()
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        tp_h = np.zeros(len(pred_boxes))

        for i, pred_box in enumerate(pred_boxes):
            ious = iou_matrix[i]
            best_match = np.argmax(ious)
            max_iou = ious[best_match]

            if max_iou >= iou_threshold and best_match not in matched_gt:
                tp[i] = 1
                matched_gt.add(best_match)
                if hard_mask[best_match]:
                    tp_h[i] = 1
            else:
                fp[i] = 1

        all_confidences.extend(conf_scores)
        all_tp.extend(tp)
        all_fp.extend(fp)
        all_tp_hard.extend(tp_h)

    empty_metrics = {
        'AP': 0.0, 'pAP_90': 0.0, 'R_max': 0.0, 'R_hard_max': 0.0,
        'F2': {'score': 0, 'threshold': 0, 'precision': 0, 'recall': 0, 'hard_recall': 0},
        'F3': {'score': 0, 'threshold': 0, 'precision': 0, 'recall': 0, 'hard_recall': 0}
    }

    if total_gt_boxes == 0 or len(all_confidences) == 0:
        return empty_metrics

    sorted_indices = np.argsort(-np.array(all_confidences))
    all_tp = np.array(all_tp)[sorted_indices]
    all_fp = np.array(all_fp)[sorted_indices]
    all_tp_hard = np.array(all_tp_hard)[sorted_indices]
    all_confidences = np.array(all_confidences)[sorted_indices]

    tp_cumsum = np.cumsum(all_tp)
    fp_cumsum = np.cumsum(all_fp)
    tp_hard_cumsum = np.cumsum(all_tp_hard)

    recall_curve = tp_cumsum / total_gt_boxes
    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = compute_interpolated_ap(precision_curve, recall_curve)
    
    max_recall = recall_curve[-1] if len(recall_curve) else 0.0
    max_hard_recall = tp_hard_cumsum[-1] / total_gt_boxes_hard if total_gt_boxes_hard > 0 else 0.0

    pap_90 = compute_partial_ap(precision_curve, recall_curve, min_recall=0.90)

    f2_score, f2_thresh, f2_p, f2_r, f2_hr = get_optimal_f_metrics(
        precision_curve, recall_curve, all_confidences, tp_hard_cumsum, total_gt_boxes_hard, beta=2.0)
        
    f3_score, f3_thresh, f3_p, f3_r, f3_hr = get_optimal_f_metrics(
        precision_curve, recall_curve, all_confidences, tp_hard_cumsum, total_gt_boxes_hard, beta=3.0)

    return {
        'AP': ap,
        'pAP_90': pap_90,
        'R_max': max_recall,
        'R_hard_max': max_hard_recall,
        'F2': {'score': f2_score, 'threshold': f2_thresh, 'precision': f2_p, 'recall': f2_r, 'hard_recall': f2_hr},
        'F3': {'score': f3_score, 'threshold': f3_thresh, 'precision': f3_p, 'recall': f3_r, 'hard_recall': f3_hr}
    }


def compute_iou_vectorized(box, boxes):
    x1_inter = np.maximum(box[0], boxes[:, 0])
    y1_inter = np.maximum(box[1], boxes[:, 1])
    x2_inter = np.minimum(box[2], boxes[:, 2])
    y2_inter = np.minimum(box[3], boxes[:, 3])

    width_inter = np.maximum(0, x2_inter - x1_inter)
    height_inter = np.maximum(0, y2_inter - y1_inter)
    intersection_area = width_inter * height_inter

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union_area = box_area + boxes_area - intersection_area
    iou = intersection_area / np.maximum(union_area, 1e-8)
    return iou


def compute_multiview_recall_curve_vectorized(gt_data, pred_data, iou_threshold=0.5):
    successful_detection_scores = []
    total_gt_objects = 0

    for set_id, gt_set in gt_data.items():
        total_gt_objects += len(gt_set)
        pred_set = pred_data.get(set_id, {})

        for gt_idx, gt_obj_views in gt_set.items():
            is_fully_detected = True
            confidences_for_this_gt = []

            for cam_name, gt_boxes in gt_obj_views.items():
                cam_preds = pred_set.get(cam_name, [])
                
                if not cam_preds:
                    is_fully_detected = False
                    break

                pred_boxes_np, pred_confs_np = cam_preds
                gt_boxes_np = np.array(gt_boxes)
                if gt_boxes_np.ndim == 1:
                    gt_boxes_np = gt_boxes_np[None, :]

                num_gts = len(gt_boxes_np)
                num_preds = len(pred_boxes_np)

                if num_preds < num_gts:
                    is_fully_detected = False
                    break

                ious = compute_iou_matrix(pred_boxes_np, gt_boxes_np)
                
                matched_count = 0
                for _ in range(num_gts):
                    best_p, best_g = np.unravel_index(np.argmax(ious), ious.shape)
                    max_iou = ious[best_p, best_g]
                    
                    if max_iou >= iou_threshold:
                        best_pred_conf = pred_confs_np[best_p]
                        confidences_for_this_gt.append(best_pred_conf)
                        
                        ious[best_p, :] = -1.0
                        ious[:, best_g] = -1.0
                        matched_count += 1
                    else:
                        break 
                
                if matched_count < num_gts:
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

def get_mv_recall_at_thresh(mv_thresh, mv_rec, target_thresh):
    """ Helper to find Multi-View Recall at a specific confidence threshold """
    if len(mv_thresh) == 0:
        return 0.0
    # Thresholds are sorted descending. We want where thresh >= target_thresh
    valid_idx = np.where(mv_thresh >= target_thresh)[0]
    if len(valid_idx) == 0:
        return 0.0
    # The lowest valid threshold (closest to target) gives the max recall for this criteria
    return mv_rec[valid_idx[-1]]


def print_metrics_report(title, metrics, mv_data=None):
    print(f'\n================ {title} EVALUATION ================')
    print(f"Standard AP                : {metrics['AP']:.4f}")
    print(f"Partial AP (Recall > 90%)  : {metrics['pAP_90']:.4f}")
    print(f"Absolute Max Recall        : {metrics['R_max']:.4f}")
    print(f"Absolute Max Hard Recall   : {metrics['R_hard_max']:.4f}")
    
    if mv_data is not None:
        mv_thresh, mv_rec = mv_data
        abs_mv_recall = mv_rec[-1] if len(mv_rec) > 0 else 0.0
        print(f"Absolute Max MV Recall     : {abs_mv_recall:.4f}")

    print("\n  [ Optimal Point: Max F2 Score (Recall x2 weight) ]")
    f2 = metrics['F2']
    print(f"  -> Best F2 Score : {f2['score']:.4f}  (Threshold: {f2['threshold']:.4f})")
    print(f"  -> Precision     : {f2['precision']:.4f}")
    print(f"  -> Recall        : {f2['recall']:.4f}")
    print(f"  -> Hard Recall   : {f2['hard_recall']:.4f}")
    if mv_data is not None:
        mv_f2 = get_mv_recall_at_thresh(mv_thresh, mv_rec, f2['threshold'])
        print(f"  -> MV Recall     : {mv_f2:.4f}")

    print("\n  [ Optimal Point: Max F3 Score (Recall x3 weight) ]")
    f3 = metrics['F3']
    print(f"  -> Best F3 Score : {f3['score']:.4f}  (Threshold: {f3['threshold']:.4f})")
    print(f"  -> Precision     : {f3['precision']:.4f}")
    print(f"  -> Recall        : {f3['recall']:.4f}")
    print(f"  -> Hard Recall   : {f3['hard_recall']:.4f}")
    if mv_data is not None:
        mv_f3 = get_mv_recall_at_thresh(mv_thresh, mv_rec, f3['threshold'])
        print(f"  -> MV Recall     : {mv_f3:.4f}")
    print('====================================================\n')


def evaluate(args):
    kpts_score_threshold = args.kpt_thresh
    vis_score_threshold = args.vis_thresh
    max_num_person = args.max_num_person
    
    if isinstance(args.anno_path, str):
        anno_paths = [args.anno_path]
    else:
        anno_paths = args.anno_path
    
    if isinstance(args.pred_path, str):
        pred_paths = [args.pred_path]
    else:
        pred_paths = args.pred_path
    
    file_num = len(anno_paths)
    
    gt_face_boxes_dict = {}
    gt_eyes_boxes_dict = {}
    pred_face_boxes_dict = {}
    pred_eyes_boxes_dict = {}
    pseudo_face_height = 40
    pseudo_face_width = 40
    pseudo_eyes_height = 40
    pseudo_eyes_width = 40
    
    mv_gt_face_boxes_dict = {}
    mv_gt_eyes_boxes_dict = {}
    mv_pred_face_boxes_dict = {}
    mv_pred_eyes_boxes_dict = {}
    
    gt_fullbody_boxes_dict = {}
    gt_boxes_with_face_dict = {}
    gt_boxes_with_eyes_dict = {}
    
    # NEW: Dictionaries to store the "Ignore Regions" (bodies without faces/eyes)
    gt_boxes_without_face_dict = {}
    gt_boxes_without_eyes_dict = {}
    
    pred_fullbody_boxes_dict = {}
    
    for f_n in range(file_num):
        anno_path = anno_paths[f_n]
        pred_path = pred_paths[f_n]
    
        with open(anno_path, 'r') as fp:
            gt = json.load(fp)
            
        # Step 1: process gt labels
        for img_key, annos in gt.items():
            boxes = []
            boxes_face_vis = []
            boxes_eyes_vis = []
            face_boxes = []
            eyes_boxes = []
            cam_name, img_name = img_key.split('/')
            mv_img_key = str(f_n) + '_' + img_name
            if mv_img_key not in mv_gt_face_boxes_dict:
                mv_gt_face_boxes_dict[mv_img_key] = {}
            if mv_img_key not in mv_gt_eyes_boxes_dict:
                mv_gt_eyes_boxes_dict[mv_img_key] = {}
                
            for anno in annos:
                box, eye1, eye2, chin, idx, is_hard = anno
                if not len(box):
                    continue
                boxes.append(box + [is_hard])
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
                    if cam_name not in mv_gt_face_boxes_dict[mv_img_key][idx]:
                        mv_gt_face_boxes_dict[mv_img_key][idx][cam_name] = []
                    mv_gt_face_boxes_dict[mv_img_key][idx][cam_name].append(face_box)
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
                    if cam_name not in mv_gt_eyes_boxes_dict[mv_img_key][idx]:
                        mv_gt_eyes_boxes_dict[mv_img_key][idx][cam_name] = []
                    mv_gt_eyes_boxes_dict[mv_img_key][idx][cam_name].append(eyes_box)
                else:
                    boxes_eyes_vis[-1] = False
                
            if len(boxes):
                new_img_key = str(f_n) + '_' + img_key
                boxes = np.stack(boxes, axis=0)
                boxes_face_vis = np.stack(boxes_face_vis, axis=0)
                boxes_eyes_vis = np.stack(boxes_eyes_vis, axis=0)
                
                boxes_face = boxes[boxes_face_vis]
                boxes_eyes = boxes[boxes_eyes_vis]
                
                # Extract the IGNORE bodies (backs of heads)
                boxes_no_face = boxes[~boxes_face_vis]
                boxes_no_eyes = boxes[~boxes_eyes_vis]
                
                if new_img_key in gt_fullbody_boxes_dict:
                    gt_fullbody_boxes_dict[new_img_key] = np.concatenate([gt_fullbody_boxes_dict[new_img_key], boxes], axis=0)
                else:
                    gt_fullbody_boxes_dict[new_img_key] = boxes
                    
                if len(boxes_face):
                    if new_img_key in gt_boxes_with_face_dict:
                        gt_boxes_with_face_dict[new_img_key] = np.concatenate([gt_boxes_with_face_dict[new_img_key], boxes_face], axis=0)
                    else:
                        gt_boxes_with_face_dict[new_img_key] = boxes_face
                        
                # Store IGNORE face bodies
                if len(boxes_no_face):
                    if new_img_key in gt_boxes_without_face_dict:
                        gt_boxes_without_face_dict[new_img_key] = np.concatenate([gt_boxes_without_face_dict[new_img_key], boxes_no_face], axis=0)
                    else:
                        gt_boxes_without_face_dict[new_img_key] = boxes_no_face
                    
                if len(boxes_eyes):
                    if new_img_key in gt_boxes_with_eyes_dict:
                        gt_boxes_with_eyes_dict[new_img_key] = np.concatenate([gt_boxes_with_eyes_dict[new_img_key], boxes_eyes], axis=0)
                    else:
                        gt_boxes_with_eyes_dict[new_img_key] = boxes_eyes
                        
                # Store IGNORE eye bodies
                if len(boxes_no_eyes):
                    if new_img_key in gt_boxes_without_eyes_dict:
                        gt_boxes_without_eyes_dict[new_img_key] = np.concatenate([gt_boxes_without_eyes_dict[new_img_key], boxes_no_eyes], axis=0)
                    else:
                        gt_boxes_without_eyes_dict[new_img_key] = boxes_no_eyes
            
            if len(face_boxes):
                face_boxes = np.stack(face_boxes, axis=0)
                new_img_key = str(f_n) + '_' + img_key
                if new_img_key in gt_face_boxes_dict:
                    gt_face_boxes_dict[new_img_key] = np.concatenate([gt_face_boxes_dict[new_img_key], face_boxes], axis=0)
                else:
                    gt_face_boxes_dict[new_img_key] = face_boxes
            
            if len(eyes_boxes):
                eyes_boxes = np.stack(eyes_boxes, axis=0)
                new_img_key = str(f_n) + '_' + img_key
                if new_img_key in gt_eyes_boxes_dict:
                    gt_eyes_boxes_dict[new_img_key] = np.concatenate([gt_eyes_boxes_dict[new_img_key], eyes_boxes], axis=0)
                else:
                    gt_eyes_boxes_dict[new_img_key] = eyes_boxes
    
        # prepare predictions
        using_nms = True
        env = lmdb.open(pred_path, readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for img_key, annos in gt.items():
                new_img_key = str(f_n) + '_' + img_key
                cam_name, img_name = img_key.split('/')
                mv_img_key = str(f_n) + '_' + img_name
                
                if mv_img_key not in mv_pred_face_boxes_dict:
                    mv_pred_face_boxes_dict[mv_img_key] = {}
                if mv_img_key not in mv_pred_eyes_boxes_dict:
                    mv_pred_eyes_boxes_dict[mv_img_key] = {}
                    
                gt_face_bodies = gt_boxes_with_face_dict.get(new_img_key, [])
                gt_no_face_bodies = gt_boxes_without_face_dict.get(new_img_key, [])
                
                gt_eyes_bodies = gt_boxes_with_eyes_dict.get(new_img_key, [])
                gt_no_eyes_bodies = gt_boxes_without_eyes_dict.get(new_img_key, [])
                
                key_kpts = f"{img_key}/kpts".encode('utf-8')
                key_kpts_scores = f"{img_key}/kpts_scores".encode('utf-8')
                key_kpts_vis = f"{img_key}/kpts_vis".encode('utf-8')
                key_boxes = f"{img_key}/boxes".encode('utf-8')
                key_boxes_scores = f"{img_key}/boxes_scores".encode('utf-8')
                
                kpts_data = txn.get(key_kpts)
                kpts_scores_data = txn.get(key_kpts_scores)
                kpts_vis_data = txn.get(key_kpts_vis)
                boxes_data = txn.get(key_boxes)
                boxes_scores_data = txn.get(key_boxes_scores)
                
                if boxes_data is not None:
                    kpts = pickle.loads(kpts_data)
                    kpts_scores = pickle.loads(kpts_scores_data)
                    kpts_vis = pickle.loads(kpts_vis_data)
                    boxes = pickle.loads(boxes_data)
                    boxes_scores = pickle.loads(boxes_scores_data)
                    
                    if args.out_of_body:
                        for k in range(len(boxes)):
                            for q in range(len(kpts[0])):
                                if kpts[k][q][0] < boxes[k][0] or kpts[k][q][0] > boxes[k][2] or kpts[k][q][1] < boxes[k][1] or kpts[k][q][1] > boxes[k][3]:
                                    kpts_scores[k][q] = 0.
                    
                    # mask = boxes_scores > 0.2
                    # kpts = kpts[mask]
                    # kpts_scores = kpts_scores[mask]
                    # kpts_vis = kpts_vis[mask]
                    # boxes = boxes[mask]
                    # boxes_scores = boxes_scores[mask]
                    # if not len(kpts):
                    #     continue
                    
                    sorted_indices = np.argsort(boxes_scores)[::-1] 
                    kpts = kpts[sorted_indices]
                    kpts_scores = kpts_scores[sorted_indices]
                    kpts_vis = kpts_vis[sorted_indices]
                    boxes = boxes[sorted_indices]
                    boxes_scores = boxes_scores[sorted_indices]
                    
                    if len(boxes):
                        if using_nms:
                            torch_boxes = torch.from_numpy(boxes).float()
                            torch_boxes_scores = torch.from_numpy(boxes_scores).float()
                            indices = nms(torch_boxes, torch_boxes_scores, 0.7)[:max_num_person].numpy()
                            
                            boxes = torch_boxes[indices].numpy()
                            boxes_scores = torch_boxes_scores[indices].numpy()
                            
                            # CRITICAL FIX: kpts must also be filtered by NMS!
                            # If not, duplicate body keypoints completely bypass the body suppression.
                            kpts = kpts[indices]
                            kpts_scores = kpts_scores[indices]
                            kpts_vis = kpts_vis[indices]
                            
                        if len(boxes):
                            pred_fullbody_boxes_dict[new_img_key] = [boxes, boxes_scores]
                    
                    if len(kpts) > 0:
                        face_boxes = []
                        face_scores = []
                        
                        # Step 1: Generate ALL candidate faces
                        for one_face_kpts, one_face_kpts_scores, one_face_kpts_vis in zip(kpts, kpts_scores, kpts_vis):
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
                            
                            # Step 2: NMS (Clean up duplicates first!)
                            if using_nms:
                                face_boxes_t = torch.from_numpy(face_boxes).float()
                                face_scores_t = torch.from_numpy(face_scores).float()
                                indices = nms(face_boxes_t, face_scores_t, 0.6)[:max_num_person].numpy()
                                face_boxes = face_boxes[indices]
                                face_scores = face_scores[indices]
                                
                            # Step 3: Apply the 1-to-1 Ignore Filter AFTER NMS
                            final_face_boxes = []
                            final_face_scores = []
                            claimed_ignore_face_bodies = set() 
                            
                            for i, face_box in enumerate(face_boxes):
                                center_x = (face_box[0] + face_box[2]) * 0.5
                                center_y = (face_box[1] + face_box[3]) * 0.5
                                
                                in_valid_body = False
                                in_ignore_body = False
                                
                                for gt_b in gt_face_bodies:
                                    if center_x >= gt_b[0]-10 and center_x <= gt_b[2]+10 and center_y >= gt_b[1]-10 and center_y <= gt_b[3]+10:
                                        in_valid_body = True
                                        break
                                
                                if not in_valid_body:
                                    for idx, gt_b in enumerate(gt_no_face_bodies):
                                        if center_x >= gt_b[0]-10 and center_x <= gt_b[2]+10 and center_y >= gt_b[1]-10 and center_y <= gt_b[3]+10:
                                            # Only forgive if this body hasn't forgiven a prediction yet
                                            if idx not in claimed_ignore_face_bodies:
                                                in_ignore_body = True
                                                claimed_ignore_face_bodies.add(idx)
                                                break
                                                
                                if not in_ignore_body:
                                    final_face_boxes.append(face_box)
                                    final_face_scores.append(face_scores[i])
                                    
                            if len(final_face_boxes):
                                final_face_boxes = np.array(final_face_boxes)
                                final_face_scores = np.array(final_face_scores)
                                pred_face_boxes_dict[new_img_key] = [final_face_boxes, final_face_scores]
                                mv_pred_face_boxes_dict[mv_img_key][cam_name] = [final_face_boxes, final_face_scores]
                    
                    if len(kpts) > 0:
                        eyes_boxes = []
                        eyes_scores = []
                        
                        # Step 1: Generate ALL candidate eyes
                        for one_eyes_kpts, one_eyes_kpts_scores, one_eyes_kpts_vis in zip(kpts[:, :2], kpts_scores[:, :2], kpts_vis[:, :2]):
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
                            
                            # Step 2: NMS (Clean up duplicates first!)
                            if using_nms:
                                eyes_boxes_t = torch.from_numpy(eyes_boxes).float()
                                eyes_scores_t = torch.from_numpy(eyes_scores).float()
                                indices = nms(eyes_boxes_t, eyes_scores_t, 0.6)[:max_num_person].numpy()
                                eyes_boxes = eyes_boxes[indices]
                                eyes_scores = eyes_scores[indices]
                                
                            # Step 3: Apply the 1-to-1 Ignore Filter AFTER NMS
                            final_eyes_boxes = []
                            final_eyes_scores = []
                            claimed_ignore_eyes_bodies = set()
                            
                            for i, eyes_box in enumerate(eyes_boxes):
                                center_x = (eyes_box[0] + eyes_box[2]) * 0.5
                                center_y = (eyes_box[1] + eyes_box[3]) * 0.5
                                
                                in_valid_body = False
                                in_ignore_body = False
                                
                                for gt_b in gt_eyes_bodies:
                                    if center_x >= gt_b[0]-10 and center_x <= gt_b[2]+10 and center_y >= gt_b[1]-10 and center_y <= gt_b[3]+10:
                                        in_valid_body = True
                                        break
                                        
                                if not in_valid_body:
                                    for idx, gt_b in enumerate(gt_no_eyes_bodies):
                                        if center_x >= gt_b[0]-10 and center_x <= gt_b[2]+10 and center_y >= gt_b[1]-10 and center_y <= gt_b[3]+10:
                                            # Only forgive if this body hasn't forgiven a prediction yet
                                            if idx not in claimed_ignore_eyes_bodies:
                                                in_ignore_body = True
                                                claimed_ignore_eyes_bodies.add(idx)
                                                break
                                                
                                if not in_ignore_body:
                                    final_eyes_boxes.append(eyes_box)
                                    final_eyes_scores.append(eyes_scores[i])
                                    
                            if len(final_eyes_boxes):
                                final_eyes_boxes = np.array(final_eyes_boxes)
                                final_eyes_scores = np.array(final_eyes_scores)
                                pred_eyes_boxes_dict[new_img_key] = [final_eyes_boxes, final_eyes_scores]
                                mv_pred_eyes_boxes_dict[mv_img_key][cam_name] = [final_eyes_boxes, final_eyes_scores]
        env.close()
    
    # ---------------- EVALUATION & PRINTING ----------------

    fullbody_metrics = evaluate_multiple_images(gt_fullbody_boxes_dict, pred_fullbody_boxes_dict, 0.5)
    print_metrics_report('FULL BODY', fullbody_metrics)

    face_metrics = evaluate_multiple_images(gt_face_boxes_dict, pred_face_boxes_dict)
    mv_face_data = compute_multiview_recall_curve_vectorized(mv_gt_face_boxes_dict, mv_pred_face_boxes_dict, iou_threshold=0.3)
    print_metrics_report('FACE', face_metrics, mv_face_data)

    eyes_metrics = evaluate_multiple_images(gt_eyes_boxes_dict, pred_eyes_boxes_dict)
    mv_eyes_data = compute_multiview_recall_curve_vectorized(mv_gt_eyes_boxes_dict, mv_pred_eyes_boxes_dict, iou_threshold=0.3)
    print_metrics_report('EYES', eyes_metrics, mv_eyes_data)

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
