import argparse
import h5py
import json
import numpy as np
from sklearn.metrics import fbeta_score, roc_curve, average_precision_score
import torch
from torchvision.ops import nms
import cv2
import os

# Metrics: precision, recall, F1, AP, AR (AR is not computable due to the inaccurate annotations)
# We consider a prediction matched if it has overlapping region with a label

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--pred_path", default="../mmpose/001_OF_JPG_baseline.hdf5", nargs='+', help="path to detections"
    )
    parser.add_argument(
        "--anno_path", default='/home/kchen/code/eyes_annotation/face_annotations_v2/001_OF_JPG.json', nargs='+', help="path to images"
    )
    parser.add_argument(
        "--root_dir", default='/media/camma-monitor/Storage_backup/001_OF_JPG', type=str, help="path to images"
    )
    parser.add_argument(
        "--sub_dirs", default=['117222250956', '309622300656', '309622301491'], type=list, help="path to images"
    )
    parser.add_argument(
        '--kpt_thresh',
        type=float,
        default=0.5,
        help='Bounding box score threshold')

    args, rest = parser.parse_known_args()

    return args


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

    for img_id in gt_data:
        gt_boxes = gt_data[img_id]
        pred_entries = pred_data.get(img_id, [])

        if len(pred_entries) == 0:
            # No predictions → all GTs are false negatives
            total_gt_boxes += len(gt_boxes)
            continue

        pred_boxes, conf_scores = pred_entries
        pred_boxes = np.array(pred_boxes)
        conf_scores = np.array(conf_scores)

        total_gt_boxes += len(gt_boxes)
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

    simplified_recall_curve = []
    simplified_precision_curve = []
    for i in range(1, len(recall_curve)):
        if recall_curve[i] > 0.5 and recall_curve[i] != recall_curve[i - 1]:
            simplified_recall_curve.append(round(recall_curve[i], 4))
            simplified_precision_curve.append(round(precision_curve[i], 4))

    # Compute AP using sklearn's average_precision_score
    ap = average_precision_score(all_tp, all_confidences)

    valid_indices = np.where(precision_curve >= 0.6)[0]
    recall_at_60_precision = max(recall_curve[valid_indices]) if len(valid_indices) > 0 else 0
    
    valid_indices = np.where(precision_curve >= 0.9)[0]
    recall_at_90_precision = max(recall_curve[valid_indices]) if len(valid_indices) > 0 else 0

    return ap, simplified_precision_curve, simplified_recall_curve, recall_at_60_precision, recall_at_90_precision

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def evaluate(args):
    if isinstance(args.pred_path, str):
        pred_paths = [args.pred_path]
    else:
        pred_paths = args.pred_path
    
    file_num = len(pred_paths)
    
    anno_paths = ['/home2020/home/icube/keqichen/datasets/4D-OR/face_annotations/export_holistic_take2_processed.json', '/home2020/home/icube/keqichen/datasets/4D-OR/face_annotations/export_holistic_take6_processed.json']
    
    pred_paths = ['../mmpose/4dor_2_tracking_prompted.hdf5', '../mmpose/4dor_6_tracking_prompted.hdf5']
    
    root_dirs = ['/home2020/home/icube/keqichen/datasets/4D-OR/export_holistic_take2_processed/colorimage', '/home2020/home/icube/keqichen/datasets/4D-OR/export_holistic_take6_processed/colorimage']
    
    file_num = len(pred_paths)
    
    pred_face_boxes_dict = {}
    pred_eyes_boxes_dict = {}
    for f_n in range(file_num):
        if f_n == 0:
            continue
        anno_path = anno_paths[f_n]
        pred_path = pred_paths[f_n]
    
        with open(anno_path, 'r') as fp:
            gt = json.load(fp)
            
        root_dir = root_dirs[f_n]
        # prepare predictions
        pred_file = h5py.File(pred_path, 'r')
    
        # indices of eyes kpts: [59-65] [65-71]
        # indices of chin kpts: [23-40]
        # indices of face kpts: [23-91]
        score_threshold = 0.0
        using_nms = True
        for img_key, annos in gt.items():
            if img_key in pred_file:
                new_img_key = str(f_n) + '_' + img_key
                # kpts.shape: [num_person, 133, 2]
                # kpts_score.shape: [num_person, 133]
                boxes = pred_file.get(img_key + '/boxes')
                
                kpts = pred_file.get(img_key + '/kpts')
                kpts_score = pred_file.get(img_key + '/kpts_score')
                
                chin_kpts = kpts[:, 23:40]
                chin_kpts_score = kpts_score[:, 23:40]
                # min_xy = np.min(chin_kpts, axis=1)
                # max_xy = np.max(chin_kpts, axis=1)
                
                # center_x = (min_xy[:, 0] + max_xy[:, 0]) * 0.5
                # center_y = (min_xy[:, 1] + max_xy[:, 1]) * 0.5
                # x1 = center_x - pseudo_face_width * 0.5
                # y1 = center_y - pseudo_face_height * 0.5
                # x2 = center_x + pseudo_face_width * 0.5
                # y2 = center_y + pseudo_face_height * 0.5
                # face_boxes = np.stack([x1, y1, x2, y2], axis=1)
                
                # face_score = np.mean(face_kpts_score, axis=1)
                # if using_nms:
                #     # indices = face_score > score_threshold
                #     face_boxes = torch.from_numpy(face_boxes).float()
                #     face_score = torch.from_numpy(face_score).float()
                #     indices = nms(face_boxes, face_score, 0.6)
                #     face_boxes = face_boxes[indices].numpy()
                #     face_score = face_score[indices].numpy()
                
                # pred_face_boxes_dict[new_img_key] = [face_boxes, face_score]
                    

                eyes_kpts = kpts[:, 59:71]
                eyes_kpts_score = kpts_score[:, 59:71]
                # min_xy = np.min(eyes_kpts, axis=1)
                # max_xy = np.max(eyes_kpts, axis=1)
                
                # center_x = (min_xy[:, 0] + max_xy[:, 0]) * 0.5
                # center_y = (min_xy[:, 1] + max_xy[:, 1]) * 0.5
                # x1 = center_x - pseudo_eyes_width * 0.5
                # y1 = center_y - pseudo_eyes_height * 0.5
                # x2 = center_x + pseudo_eyes_width * 0.5
                # y2 = center_y + pseudo_eyes_height * 0.5
                # eyes_boxes = np.stack([x1, y1, x2, y2], axis=1)
                
                # eyes_score = np.mean(eyes_kpts_score, axis=1)
                # if using_nms:
                #     # indices = eyes_score > score_threshold
                #     eyes_boxes = torch.from_numpy(eyes_boxes).float()
                #     eyes_score = torch.from_numpy(eyes_score).float()
                #     indices = nms(eyes_boxes, eyes_score, 0.6)
                #     eyes_boxes = eyes_boxes[indices].numpy()
                #     eyes_score = eyes_score[indices].numpy()
                # pred_eyes_boxes_dict[new_img_key] = [eyes_boxes, eyes_score]
                
                num_person = len(boxes)
                img_path = os.path.join(root_dir, img_key)
                img = cv2.imread(img_path)
                for n in range(num_person):
                    color = get_color(n)
                    cv2.rectangle(img, (int(boxes[n][0]), int(boxes[n][1])), (int(boxes[n][2]), int(boxes[n][3])), color=(0, 255, 0), thickness=3)
                    for i in range(12):
                        kpt_x = int(eyes_kpts[n][i][0])
                        kpt_y = int(eyes_kpts[n][i][1])
                        cv2.circle(img, (kpt_x, kpt_y), radius=3, color=(0, 0, 255), thickness=-1)
                    for i in range(17):
                        kpt_x = int(chin_kpts[n][i][0])
                        kpt_y = int(chin_kpts[n][i][1])
                        cv2.circle(img, (kpt_x, kpt_y), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.imwrite(os.path.join('vis_results', str(f_n) + img_key), img)
        
        pred_file.close()

if __name__ == '__main__':
    args = parse_args()

    evaluate(args)