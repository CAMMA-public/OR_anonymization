import argparse
import pickle
import lmdb
import json
import numpy as np
from sklearn.metrics import fbeta_score, roc_curve, average_precision_score, auc
import torch
from torchvision.ops import nms
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

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
        '--kpt_thresh',
        type=float,
        default=0.0,
        help='Bounding box score threshold')
    parser.add_argument(
        '--vis_thresh',
        type=float,
        default=0.0,
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


def compute_iou_matrix(pred_boxes, gt_kpts):
    """Compute IoU matrix between all predicted boxes and ground-truth boxes."""
    dis_matrix = np.zeros((len(pred_boxes), len(gt_kpts)))
    for i in range(len(pred_boxes)):
        box = pred_boxes[i]
        for j in range(len(gt_kpts)):
            kpts = gt_kpts[j]
            dis = 0
            for kpt in kpts:
                if kpt[0] < box[0] or kpt[0] > box[2] or kpt[1] < box[1] or kpt[1] > box[3]:
                    dis = 1
                    break
            dis_matrix[i, j] = dis
    return dis_matrix
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
    total_recall = 0

    for img_id in gt_data:
        gt_kpts = gt_data[img_id][0]
        pred_entries = pred_data.get(img_id, [])
        # print(gt_kpts)
        # print(pred_entries)

        if len(pred_entries) == 0:
            # No predictions → all GTs are false negatives
            total_gt_boxes += len(gt_kpts)
            continue

        pred_boxes, conf_scores = pred_entries
        pred_boxes = np.array(pred_boxes)
        conf_scores = np.array(conf_scores)

        total_gt_boxes += len(gt_kpts)
        sorted_indices = np.argsort(conf_scores)[::-1]  # Sort by confidence (high to low)
        pred_boxes = pred_boxes[sorted_indices]
        conf_scores = conf_scores[sorted_indices]
        
        # Compute IoU matrix for this image
        iou_matrix = compute_iou_matrix(pred_boxes, gt_kpts)
        
        # print(iou_matrix)
        # exit(1)
        
        # Match predictions to ground truth
        matched_gt = set()
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        
        matches_x, matches_y = linear_sum_assignment(iou_matrix)
        dis_sum = iou_matrix[matches_x, matches_y].sum()
        total_recall += (len(gt_kpts) - dis_sum)

    return total_recall / total_gt_boxes

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

    return ap, precision_curve, recall_curve, recall_at_60_precision, recall_at_90_precision


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
    
    # prepare ground-truth data
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
    for f_n in range(file_num):
        anno_path = anno_paths[f_n]
        pred_path = pred_paths[f_n]
    
        with open(anno_path, 'r') as fp:
            gt = json.load(fp)
            
        # Step 1: process gt faces
        for img_key, annos in gt.items():
            face_kpts = []
            cam_name, img_name = img_key.split('/')
            mv_img_key = str(f_n) + '_' + img_name
            if mv_img_key not in mv_gt_face_boxes_dict:
                mv_gt_face_boxes_dict[mv_img_key] = {}
            for anno in annos:
                eye1 = anno[0]
                eye2 = anno[1]
                chin = anno[2]
                idx = anno[3]
                if len(eye1) == 0 and len(eye2) == 0:
                    continue
                kpts = []
                if len(eye1):
                    kpts.append(eye1)
                if len(eye2):
                    kpts.append(eye2)
                if len(chin):
                    kpts.append(chin)
                if len(kpts) == 0:
                    continue
                face_kpts.append(kpts)
                
            if len(face_kpts):
                new_img_key = str(f_n) + '_' + img_key
                if new_img_key in gt_face_boxes_dict:
                    gt_face_boxes_dict[new_img_key].append(face_kpts)
                else:
                    gt_face_boxes_dict[new_img_key] = [face_kpts]
    
        # Step 2: process gt eyes
        for img_key, annos in gt.items():
            eyes_kpts = []
            cam_name, img_name = img_key.split('/')
            mv_img_key = str(f_n) + '_' + img_name
            if mv_img_key not in mv_gt_eyes_boxes_dict:
                mv_gt_eyes_boxes_dict[mv_img_key] = {}
            for anno in annos:
                eye1 = anno[0]
                eye2 = anno[1]
                chin = anno[2]
                idx = anno[3]
                if len(eye1) == 0 and len(eye2) == 0:
                    continue
                kpts = []
                if len(eye1):
                    kpts.append(eye1)
                if len(eye2):
                    kpts.append(eye2)
                if len(kpts) == 0:
                    continue
                eyes_kpts.append(kpts)
                
            if len(eyes_kpts):
                new_img_key = str(f_n) + '_' + img_key
                if new_img_key in gt_eyes_boxes_dict:
                    gt_eyes_boxes_dict[new_img_key].append(eyes_kpts)
                else:
                    gt_eyes_boxes_dict[new_img_key] = [eyes_kpts]
    
        # prepare predictions
        using_nms = True
        env = lmdb.open(pred_path, readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            # Use a cursor for efficient iteration if desired.
            cursor = txn.cursor()
            for img_key, annos in gt.items():
                new_img_key = str(f_n) + '_' + img_key
                cam_name, img_name = img_key.split('/')
                mv_img_key = str(f_n) + '_' + img_name
                if mv_img_key not in mv_pred_face_boxes_dict:
                    mv_pred_face_boxes_dict[mv_img_key] = {}
                if mv_img_key not in mv_pred_eyes_boxes_dict:
                    mv_pred_eyes_boxes_dict[mv_img_key] = {}
                
                key_kpts = f"{img_key}/kpts".encode('utf-8')
                key_kpts_scores = f"{img_key}/kpts_scores".encode('utf-8')
                key_kpts_vis = f"{img_key}/kpts_vis".encode('utf-8')
                key_boxes = f"{img_key}/boxes".encode('utf-8')
                key_boxes_scores = f"{img_key}/boxes_scores".encode('utf-8')
                # key_kpts_scores = f"{img_key}/scores".encode('utf-8')
                key_matching_scores = f"{img_key}/matching_scores".encode('utf-8')
                
                kpts_data = txn.get(key_kpts)
                kpts_scores_data = txn.get(key_kpts_scores)
                kpts_vis_data = txn.get(key_kpts_vis)
                boxes_data = txn.get(key_boxes)
                boxes_scores_data = txn.get(key_boxes_scores)
                matching_scores_data = txn.get(key_matching_scores)
                if kpts_data is not None:
                    # Deserialize using pickle.
                    # kpts = pickle.loads(kpts_data)
                    # kpts_scores = pickle.loads(kpts_scores_data)
                    # kpts_vis = pickle.loads(kpts_vis_data)
                    boxes = pickle.loads(boxes_data)
                    boxes_scores = pickle.loads(boxes_scores_data)
                    # matching_scores = pickle.loads(matching_scores_data)
                    
                    # mask = (matching_scores < 0) | (matching_scores >= 0.1)
                    # kpts = kpts[mask]
                    # kpts_scores = kpts_scores[mask]
                    # kpts_vis = kpts_vis[mask]
                    # boxes = boxes[mask]
                    # boxes_scores = boxes_scores[mask]
                    # matching_scores = matching_scores[mask]
                    
                    # mask = boxes_scores > 0.6
                    # kpts = kpts[mask]
                    # kpts_scores = kpts_scores[mask]
                    # if not len(kpts):
                    #     continue
                    
                    # if args.out_of_body:
                    #     for k in range(len(boxes)):
                    #         for q in range(len(kpts[0])):
                    #             if kpts[k][q][0] < boxes[k][0] or kpts[k][q][0] > boxes[k][2] or kpts[k][q][1] < boxes[k][1] or kpts[k][q][1] > boxes[k][3]:
                    #                 kpts_scores[k][q] = 0.
                            
                    # face_kpts = np.copy(kpts)
                    # face_kpts_scores = np.copy(kpts_scores)
                    # face_kpts_vis = np.copy(kpts_vis)
                    # min_xy = np.min(face_kpts, axis=1)
                    # max_xy = np.max(face_kpts, axis=1)
                    
                    # center_x = (min_xy[:, 0] + max_xy[:, 0]) * 0.5
                    # center_y = (min_xy[:, 1] + max_xy[:, 1]) * 0.5
                    # x1 = center_x - pseudo_face_width * 0.5
                    # y1 = center_y - pseudo_face_height * 0.5
                    # x2 = center_x + pseudo_face_width * 0.5
                    # y2 = center_y + pseudo_face_height * 0.5
                    # face_boxes = np.stack([x1, y1, x2, y2], axis=1)
                    # face_score = np.mean(face_kpts_scores, axis=1)
                    
                    # face_kpts_vis_indices = np.any(face_kpts_vis>vis_score_threshold, axis=1)
                    # face_boxes = face_boxes[face_kpts_vis_indices]
                    # face_score = face_score[face_kpts_vis_indices]
                    
                    # width = max_xy[:, 0] - min_xy[:, 0]
                    # height = max_xy[:, 1] - min_xy[:, 1]
                    # if len(face_boxes):
                    #     filter1 = np.logical_and(width < 70, height < 70)
                    #     face_boxes = face_boxes[filter1]
                    #     face_score = face_score[filter1]
                    
                    if len(boxes):
                        if using_nms:
                            # indices = face_score > kpts_score_threshold
                            boxes = torch.from_numpy(boxes).float()
                            boxes_scores = torch.from_numpy(boxes_scores).float()
                            indices = nms(boxes, boxes_scores, 0.6)
                            boxes = boxes[indices].numpy()
                            boxes_scores = boxes_scores[indices].numpy()
                        # indices = face_score > kpts_score_threshold
                        # face_boxes = face_boxes[indices]
                        # face_score = face_score[indices]
                        if len(boxes):
                            pred_face_boxes_dict[new_img_key] = [boxes, boxes_scores]
                            mv_pred_face_boxes_dict[mv_img_key][cam_name] = [boxes, boxes_scores]
        env.close()
    
    recall = evaluate_multiple_images(gt_face_boxes_dict, pred_face_boxes_dict)
    # mv_thresholds, mv_recall = compute_multiview_recall_curve_vectorized(mv_gt_face_boxes_dict, mv_pred_face_boxes_dict, iou_threshold=0.3)
    print('Face')
    print(f'Recall: {recall}')
    # print(f'Multi-view recall: {mv_recall[-1]}')

    recall = evaluate_multiple_images(gt_eyes_boxes_dict, pred_face_boxes_dict)
    # mv_thresholds, mv_recall = compute_multiview_recall_curve_vectorized(mv_gt_eyes_boxes_dict, mv_pred_eyes_boxes_dict, iou_threshold=0.3)
    print('Eyes')
    print(f'Recall: {recall}')
    # print(f'Multi-view recall: {mv_recall[-1]}')


if __name__ == '__main__':
    args = parse_args()

    evaluate(args)
