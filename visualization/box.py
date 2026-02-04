import argparse
import lmdb
import pickle
import json
import numpy as np
from sklearn.metrics import fbeta_score, roc_curve, average_precision_score
import torch
from torchvision.ops import nms
import cv2
import os
from tqdm import tqdm

COLORS = [
    (0, 255, 255),  # 1. Bright Yellow
    (255, 0, 255),  # 2. Vibrant Magenta / Fuchsia
    (0, 165, 255),  # 3. Bright Orange
    (0, 255, 0),    # 4. Bright Lime Green
    (255, 255, 0),  # 5. Bright Cyan
    (128, 0, 128),  # 6. Deep Purple
    (255, 191, 0),  # 7. Deep Sky Blue (if needed)
    (0, 128, 255),  # 8. Orange-Red
    (203, 192, 255),# 9. Pink
    (60, 20, 220),  # 10. Crimson Red
]

# Metrics: precision, recall, F1, AP, AR (AR is not computable due to the inaccurate annotations)
# We consider a prediction matched if it has overlapping region with a label

def remove_redundant_results(kpts, scores):
    pseudo_face_height = 40
    pseudo_face_width = 40
    min_xy = np.min(kpts, axis=1)
    max_xy = np.max(kpts, axis=1)
    center_x = (min_xy[:, 0] + max_xy[:, 0]) * 0.5
    center_y = (min_xy[:, 1] + max_xy[:, 1]) * 0.5
    x1 = center_x - pseudo_face_width * 0.5
    y1 = center_y - pseudo_face_height * 0.5
    x2 = center_x + pseudo_face_width * 0.5
    y2 = center_y + pseudo_face_height * 0.5
    face_boxes = np.stack([x1, y1, x2, y2], axis=1)
    face_scores = np.mean(scores, axis=1)
    face_boxes = torch.from_numpy(face_boxes).float()
    face_scores = torch.from_numpy(face_scores).float()
    indices = nms(face_boxes, face_scores, 0.6)
    indices = indices.numpy()
    return indices

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
    
    # anno_paths = ['/home/kchen/code/eyes_annotation/face_annotations_v2/001_OF_JPG.json', '/home/kchen/code/eyes_annotation/face_annotations_v2/002_PH_JPG.json']
    
    # pred_paths = ['../mmpose/001_OF_JPG_tracking_prompted_07_003_004_521_32.hdf5', '../mmpose/002_PH_JPG_tracking_prompted_07_003_004_521_32.hdf5']
    
    # root_dirs = ['/media/camma-monitor/Storage_backup/001_OF_JPG', '/media/camma-monitor/Storage_backup/002_PH_JPG']
    
    # anno_paths = ['/media/camma-monitor/Storage_postprocessing2/pose_detection_utils/code/anonymization/evaluation/annotations/008_OT_JPG.json']
    
    # pred_paths = ['/media/camma-monitor/Storage_postprocessing2/pose_detection/face/005_CG_JPG_mva_tracking_bi_test_face_partial.lmdb']
    pred_paths = ['/media/camma-monitor/Storage_postprocessing2/pose_detection/face/006_RS_JPG_iter2_mva_tracking_reid_face.lmdb']
    
    # root_dirs = ['/media/camma-monitor/Storage_postprocessing2/005_CG_JPG/005_CG']
    # root_dirs = ['/media/camma-monitor/Storage_postprocessing3/008_OT_JPG']
    root_dirs = ['/media/camma-monitor/Storage_postprocessing2/006_RS_JPG/006_RS']
        
    file_num = len(pred_paths)
    
    pred_face_boxes_dict = {}
    pred_eyes_boxes_dict = {}
    for f_n in range(file_num):
        # anno_path = anno_paths[f_n]
        pred_path = pred_paths[f_n]
        root_dir = root_dirs[f_n]
        save_dir_name = pred_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        
        sub_dirs = []
        for dir_name in os.listdir(root_dir):
            if dir_name == '547758':
                continue
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                sub_dirs.append(dir_name)
        sub_dirs.sort()
        
        img_dir = os.path.join(root_dir, sub_dirs[0])
        img_names = os.listdir(img_dir)
        img_names = [x for x in img_names if x[-3:] == 'jpg' and x[0] == 'c']
        img_names.sort()
        
        save_interval = 150
        count_seed = 0
        
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join('vis', save_dir_name, sub_dir), exist_ok=True)
    
        # with open(anno_path, 'r') as fp:
        #     gt = json.load(fp)
            
        # prepare predictions
        env = lmdb.open(pred_path, readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            # Use a cursor for efficient iteration if desired.
            cursor = txn.cursor()
            count = count_seed
            for img_name in tqdm(img_names):
                count += 1
                if count % save_interval != 0:
                    continue
                for c_id, sub_dir in enumerate(sub_dirs):
                    img_key = os.path.join(sub_dir, img_name)
            # for img_key, annos in gt.items():
                    key_kpts = f"{img_key}/kpts".encode('utf-8')
                    key_kpts_scores = f"{img_key}/kpts_scores".encode('utf-8')
                    key_boxes = f"{img_key}/boxes".encode('utf-8')
                    key_boxes_scores = f"{img_key}/boxes_scores".encode('utf-8')
                    kpts_data = txn.get(key_kpts)
                    kpts_scores_data = txn.get(key_kpts_scores)
                    boxes_data = txn.get(key_boxes)
                    boxes_scores_data = txn.get(key_boxes_scores)
                    num_person = 0
                    if kpts_data is not None and kpts_scores_data is not None:
                        # Deserialize using pickle.
                        kpts = pickle.loads(kpts_data)
                        kpts_scores = pickle.loads(kpts_scores_data)
                        boxes = pickle.loads(boxes_data)
                        boxes_scores = pickle.loads(boxes_scores_data)
                        if len(kpts):
                            indices = remove_redundant_results(kpts, kpts_scores)
                            kpts = kpts[indices]
                            kpts_scores = kpts_scores[indices]
                            boxes = boxes[indices]
                            boxes_scores = boxes_scores[indices]
                            num_person = len(kpts)
                            
                    img_path = os.path.join(root_dir, img_key)
                    img = cv2.imread(img_path)
                    for n in range(num_person):
                        color = get_color(n)
                        vis_scores = []
                        for i in range(3):
                            kpt_x = int(kpts[n][i][0])
                            kpt_y = int(kpts[n][i][1])
                            kpts_score = round(kpts_scores[n][i], 2)
                            vis_scores.append(str(kpts_score))
                            cv2.circle(img, (kpt_x, kpt_y), radius=3+i, color=COLORS[n%10], thickness=-1)
                        box_x1 = int(boxes[n][0])
                        box_y1 = int(boxes[n][1])
                        box_x2 = int(boxes[n][2])
                        box_y2 = int(boxes[n][3])
                        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), color=COLORS[n%10], thickness=2)
                        score_txt = ' '.join(vis_scores)
                        cv2.putText(img, score_txt, (box_x1, box_y1), cv2.FONT_HERSHEY_PLAIN, 2, COLORS[n%10], thickness=2)
                    cv2.imwrite(os.path.join('vis', save_dir_name, img_key), img)
        env.close()
      

if __name__ == '__main__':
    args = parse_args()

    evaluate(args)