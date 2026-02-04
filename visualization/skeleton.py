import lmdb
import pickle
import numpy as np
import torch
from torchvision.ops import nms
import os
import cv2
from tqdm import tqdm
import argparse
import json
import datetime
import shutil
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Pseudu labels")
    # /media/camma-monitor/Storage_postprocessing2/pose_detection/face/006_RS_JPG_mva_tracking_bi_face.lmdb
    # /media/camma-monitor/Storage_postprocessing2/pose_detection/detections/006_RS_JPG.lmdb
    # /media/camma-monitor/Storage_postprocessing2/pose_detection/tracking/006_RS_JPG_tracking.lmdb
    parser.add_argument(
        "--pred_path", default="/media/camma-monitor/Storage_postprocessing2/pose_detection/pose/006_RS_JPG_pose_ft.lmdb", nargs='+', help="path to detections"
    )
    # /media/camma-monitor/Storage_postprocessing2/005_CG_JPG/005_CG
    # /media/camma-monitor/Storage_postprocessing2/006_RS_JPG/006_RS
    # /media/camma-monitor/Storage_postprocessing3/007_CL_JPG
    # /media/camma-monitor/Storage_postprocessing3/009_SV_JPG
    parser.add_argument(
        "--root_dir", default="/media/camma-monitor/Storage_postprocessing2/006_RS_JPG/006_RS", nargs='+', help="path to detections"
    )
    parser.add_argument(
        "--save_root_dir", default="006", nargs='+', help="path to detections"
    )
    parser.add_argument(
        '--kpt_thresh',
        type=float,
        default=0.0,
        help='Bounding box score threshold')

    args, rest = parser.parse_known_args()

    return args


# Order of keypoints in the (17, 2) array
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Pairs of keypoint indices to form the skeleton's limbs
# Each tuple represents a connection between two keypoints
SKELETON_CONNECTIONS = [
    # Torso
    (5, 6),   # Left Shoulder -> Right Shoulder
    (5, 11),  # Left Shoulder -> Left Hip
    (6, 12),  # Right Shoulder -> Right Hip
    (11, 12), # Left Hip -> Right Hip
    # Left Arm
    (5, 7),   # Left Shoulder -> Left Elbow
    (7, 9),   # Left Elbow -> Left Wrist
    # Right Arm
    (6, 8),   # Right Shoulder -> Right Elbow
    (8, 10),  # Right Elbow -> Right Wrist
    # Left Leg
    (11, 13), # Left Hip -> Left Knee
    (13, 15), # Left Knee -> Left Ankle
    # Right Leg
    (12, 14), # Right Hip -> Right Knee
    (14, 16), # Right Knee -> Right Ankle
    # Head
    (0, 1),   # Nose -> Left Eye
    (0, 2),   # Nose -> Right Eye
    (1, 3),   # Left Eye -> Left Ear
    (2, 4),   # Right Eye -> Right Ear
    (0, 5),   # Nose -> Left Shoulder (added for better visualization)
    (0, 6),   # Nose -> Right Shoulder (added for better visualization)
]


# --- Create a list of visually distinct colors ---
# We use a colormap from matplotlib to get a nice set of colors
# Using 'tab20' which has 20 distinct colors. We'll get them in BGR format for OpenCV.
cmap = plt.get_cmap('tab20')
# COLORS = [tuple(int(c * 255) for c in cmap(i)[:3][::-1]) for i in range(20)]
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


def visualize_skeletons_advanced(image, all_keypoints, all_boxes, all_confidences=None, confidence_threshold=0.5):
    """
    Draws skeletons for multiple people on an image with unique colors.

    Args:
        image (np.ndarray): The image to draw on (in BGR format).
        all_keypoints (list of np.ndarray): A list where each element is a (17, 2) array of keypoints for one person.
        all_confidences (list of np.ndarray, optional): A list where each element is a (17,) array of confidences.
        confidence_threshold (float): The threshold for drawing keypoints and limbs.
    
    Returns:
        np.ndarray: The image with skeletons visualized.
    """
    vis_image = image.copy()
    num_people = len(all_keypoints)

    for i in range(num_people):
        keypoints = all_keypoints[i]
        confidences = all_confidences[i] if all_confidences is not None else np.ones(17) # Assume full confidence if not provided
        
        # --- Pick a unique color for this person ---
        person_color = COLORS[i % len(COLORS)] # Cycle through colors if more people than colors
        
        x1 = int(all_boxes[i][0])
        y1 = int(all_boxes[i][1])
        x2 = int(all_boxes[i][2])
        y2 = int(all_boxes[i][3])
        cv2.rectangle(vis_image, [x1, y1], [x2, y2], color=person_color, thickness=2)

        # --- 1. Draw the skeleton connections (limbs) ---
        for connection in SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            
            # Check if both joints are confident enough to draw a limb
            if confidences[start_idx] > confidence_threshold and confidences[end_idx] > confidence_threshold:
                start_point = tuple(np.round(keypoints[start_idx]).astype(int))
                end_point = tuple(np.round(keypoints[end_idx]).astype(int))
                
                # A limb should not be drawn if a point is at (0,0) (often indicates not detected)
                if start_point == (0, 0) or end_point == (0, 0):
                    continue
                
                cv2.line(vis_image, start_point, end_point, color=person_color, thickness=2)

        # --- 2. Draw the keypoints (joints) ---
        for j in range(len(keypoints)):
            if confidences[j] > confidence_threshold:
                point = tuple(np.round(keypoints[j]).astype(int))
                
                if point == (0, 0):
                    continue

                # Draw a filled circle for the joint
                cv2.circle(vis_image, point, radius=4, color=person_color, thickness=-1)
                # Draw a thin black border around the circle for better visibility
                cv2.circle(vis_image, point, radius=4, color=(0, 0, 0), thickness=1)

    return vis_image



def calculate_iou(box, boxes):
    """
    Calculates IoU between a single box and an array of boxes.
    Assumes box format [x1, y1, x2, y2].

    Args:
        box (np.ndarray): A single bounding box, shape (4,).
        boxes (np.ndarray): An array of bounding boxes, shape (N, 4).

    Returns:
        np.ndarray: IoU values, shape (N,).
    """
    # Coordinates of the intersection box
    x1_inter = np.maximum(box[0], boxes[:, 0])
    y1_inter = np.maximum(box[1], boxes[:, 1])
    x2_inter = np.minimum(box[2], boxes[:, 2])
    y2_inter = np.minimum(box[3], boxes[:, 3])

    # Width and height of the intersection box
    # Add 1 if your coordinates are inclusive (e.g., pixel indices)
    # For simplicity here, assuming standard geometric interpretation (no +1)
    # If you need pixel-inclusive, use:
    # width_inter = np.maximum(0, x2_inter - x1_inter + 1)
    # height_inter = np.maximum(0, y2_inter - y1_inter + 1)
    width_inter = np.maximum(0, x2_inter - x1_inter)
    height_inter = np.maximum(0, y2_inter - y1_inter)

    intersection_area = width_inter * height_inter

    # Area of the individual boxes
    # Again, adjust with +1 if using inclusive pixel coordinates
    # box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    # boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # union_area = box_area + boxes_area - intersection_area
    union_area1 = box_area
    union_area2 = boxes_area

    # Avoid division by zero
    iou1 = intersection_area / np.maximum(union_area1, 1e-7) # Add a small epsilon
    iou2 = intersection_area / np.maximum(union_area2, 1e-7) # Add a small epsilon
    iou = np.max(np.stack([iou1, iou2], axis=1), axis=1)
    
    return iou


def non_max_suppression_numpy(boxes, scores, iou_threshold=0.8):
    """
    Performs Non-Maximum Suppression (NMS) on an array of bounding boxes and scores.

    Args:
        boxes_scores (np.ndarray): Array of shape (N, 5) where each row is
                                   [x1, y1, x2, y2, score].
                                   (x1, y1) = top-left, (x2, y2) = bottom-right.
        iou_threshold (float): IoU threshold for suppressing overlapping boxes.

    Returns:
        list: A list of indices of the boxes to keep.
    """
    if len(boxes) == 0:
        return []

    # Extract coordinates and scores
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Sort by score in descending order
    # `order` will contain the original indices of the boxes, sorted by score
    order = scores.argsort()[::-1]

    keep_indices = []
    while order.size > 0:
        # Index of the current highest-scoring box (in the original `boxes_scores` array)
        current_idx = order[0]
        keep_indices.append(current_idx)

        # If only one box left in `order`, no need to compare
        if order.size == 1:
            break

        # Current box details
        current_box = np.array([x1[current_idx], y1[current_idx], x2[current_idx], y2[current_idx]])

        # Boxes to compare against (all remaining boxes in `order` except the current one)
        # `order[1:]` gives the original indices of the remaining boxes
        remaining_indices = order[1:]
        remaining_boxes_coords = np.column_stack((x1[remaining_indices],
                                                  y1[remaining_indices],
                                                  x2[remaining_indices],
                                                  y2[remaining_indices]))

        # Calculate IoU of the current box with all remaining boxes
        ious = calculate_iou(current_box, remaining_boxes_coords)

        # Find indices of boxes in `remaining_indices` that have IoU <= threshold
        # These are the boxes to keep for the next iteration relative to `remaining_indices`
        indices_to_keep_in_remaining = np.where(ious <= iou_threshold)[0]

        # Update `order`: keep only those boxes from `remaining_indices` that passed the IoU check
        # `order[1:][indices_to_keep_in_remaining]` selects the original indices
        order = order[1:][indices_to_keep_in_remaining]

    return keep_indices


def fix_boxes(boxes, scores):
    if len(boxes):
        filter1 = boxes[:, 1] < boxes[:, 3]
        boxes = boxes[filter1]
        scores = scores[filter1]
    if len(boxes):
        filter2 = boxes[:, 0] < boxes[:, 2]
        boxes = boxes[filter2]
        scores = scores[filter2]
    if len(boxes):
        filter3 = scores >= 0.1
        boxes = boxes[filter3]
        scores = scores[filter3]
    if len(boxes):
        filter4 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 500
        boxes = boxes[filter4]
        scores = scores[filter4]
    return boxes, scores


def generate_coco_annotations(output_file_path, all_image_data, category_names):
    """
    Generates a COCO-format JSON annotation file.

    Args:
        output_file_path (str): Path to save the output JSON file.
        all_image_data (list): A list of dictionaries, where each dictionary
                               represents an image and its annotations.
                               Expected structure for each item:
                               {
                                   "file_name": "image_name.jpg",
                                   "height": H,
                                   "width": W,
                                   "annotations": [
                                       {"bbox": [x, y, w, h], "category_name": "name"},
                                       ...
                                   ]
                               }
                               The bbox is [x_min, y_min, width, height].
        category_names (list): A list of unique string category names.
    """

    coco_output = {
        "info": {
            "description": "Monitor",
            "url": "",
            "version": "1.0",
            "year": datetime.date.today().year,
            "contributor": "Keqi Chen",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 1. Process Categories
    category_name_to_id = {}
    for i, name in enumerate(category_names):
        category_id = i + 1  # COCO category IDs typically start from 1
        coco_output["categories"].append({
            "id": category_id,
            "name": name,
            "supercategory": "object" # Or be more specific if you have supercategories
        })
        category_name_to_id[name] = category_id

    # 2. Process Images and their Annotations
    image_id_counter = 1
    annotation_id_counter = 1

    for image_data in all_image_data:
        # if not os.path.exists(image_data["file_name"]):
        #     print(f"Warning: Image file not found: {image_data['file_name']}. Skipping image entry if dimensions are missing.")
        #     # Decide if you want to skip or handle differently
        #     # For now, we assume height/width are provided. If not, you'd get them here:
        #     # try:
        #     #     with Image.open(image_data["file_name"]) as img:
        #     #         width, height = img.size
        #     # except FileNotFoundError:
        #     #     print(f"Error: Image file not found and dimensions not provided: {image_data['file_name']}")
        #     #     continue # Skip this image
        #     # image_data["width"] = width
        #     # image_data["height"] = height
        #     pass # Assuming height/width are always provided in image_data

        image_entry = {
            "id": image_id_counter,
            "file_name": image_data["file_name"], # Store only basename
            "height": image_data["height"],
            "width": image_data["width"],
        }
        coco_output["images"].append(image_entry)

        for ann_data in image_data.get("annotations", []):
            bbox = ann_data["bbox"]
            category_name = ann_data["category_name"]

            if category_name not in category_name_to_id:
                print(f"Warning: Category '{category_name}' in annotation for image "
                      f"'{image_data['file_name']}' not found in category_names. Skipping annotation.")
                continue

            category_id = category_name_to_id[category_name]

            annotation_entry = {
                "id": annotation_id_counter,
                "image_id": image_id_counter,
                "category_id": category_id,
                "area": bbox[2] * bbox[3], # width * height
                "bbox": bbox, # Ensure float
                "iscrowd": 0 # 0 for single instances, 1 for crowds
            }
            coco_output["annotations"].append(annotation_entry)
            annotation_id_counter += 1

        image_id_counter += 1

    # 3. Save to JSON file
    with open(output_file_path, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"COCO annotation file saved to: {output_file_path}")
    print(f"Processed {len(coco_output['images'])} images and {len(coco_output['annotations'])} annotations.")
    print(f"Categories: {coco_output['categories']}")


def main(args):
    # prepare predictions
    if isinstance(args.pred_path, str):
        pred_paths = [args.pred_path]
    else:
        pred_paths = args.pred_path
    
    if isinstance(args.root_dir, str):
        root_dirs = [args.root_dir]
    else:
        root_dirs = args.root_dir
    
    file_num = len(pred_paths)
    for f_n in range(file_num):
        pred_path = pred_paths[f_n]
        root_dir = root_dirs[f_n]
        
        sub_dirs = []
        for dir_name in os.listdir(root_dir):
            if dir_name == '547758':
                continue
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                sub_dirs.append(dir_name)
        sub_dirs.sort()
        
        visualize = True
        if visualize:
            base_name = pred_path.rsplit(os.sep, 1)[1].rsplit('.', 1)[0]
            save_dir = os.path.join('pose', base_name)
            for c_id, sub_dir in enumerate(sub_dirs):
                os.makedirs(os.path.join(save_dir, sub_dir), exist_ok=True)
        
        img_dir = os.path.join(root_dir, sub_dirs[0])
        img_names = os.listdir(img_dir)
        img_names = [x for x in img_names if x[-3:] == 'jpg' and x[0] == 'c']
        img_names.sort()
        
        demo_img_path = os.path.join(root_dir, sub_dirs[0], img_names[0])
        demo_img = cv2.imread(demo_img_path)
        height, width, _ = demo_img.shape
        
        using_mva = False
        using_pose = False
        
        save_interval = 15
        count_seed = 0
        
        category_names = ["person"]
        all_image_data = []
        
        env = lmdb.open(pred_path, readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            # Use a cursor for efficient iteration if desired.
            cursor = txn.cursor()
            count = count_seed
            for img_name in tqdm(img_names):
                count += 1
                if count % save_interval != 0:
                    continue
                # if img_name != 'color-00012143.jpg':
                #     continue
                for c_id, sub_dir in enumerate(sub_dirs):
                    img_key = os.path.join(sub_dir, img_name)
                    key_boxes = f"{img_key}/boxes".encode('utf-8')
                    key_kpts = f"{img_key}/kpts".encode('utf-8')
                    key_kpts_scores = f"{img_key}/kpts_scores".encode('utf-8')
                    key_boxes_scores = f"{img_key}/boxes_scores".encode('utf-8')
                    
                    boxes_data = txn.get(key_boxes)
                    boxes_scores_data = txn.get(key_boxes_scores)
                    kpts_data = txn.get(key_kpts)
                    kpts_scores_data = txn.get(key_kpts_scores)
                        
                    if boxes_data is not None:
                        # Deserialize using pickle.
                        boxes = pickle.loads(boxes_data)
                        boxes_scores = pickle.loads(boxes_scores_data)
                        boxes, boxes_scores = fix_boxes(boxes, boxes_scores)
                        kpts = pickle.loads(kpts_data)
                        kpts_scores = pickle.loads(kpts_scores_data)
                        
                        indices = non_max_suppression_numpy(boxes, boxes_scores)
                        boxes = boxes[indices]
                        boxes_scores = boxes_scores[indices]
                        kpts = kpts[indices]
                        kpts_scores = kpts_scores[indices]
                        
                        torch_boxes = torch.from_numpy(boxes).float()
                        torch_score = torch.from_numpy(boxes_scores).float()
                        indices = nms(torch_boxes, torch_score, 0.5)
                        indices = indices.numpy()
                        
                        new_boxes = boxes[indices]
                        new_boxes_score = boxes_scores[indices]
                        new_kpts = kpts[indices]
                        new_kpts_scores = kpts_scores[indices]
                        avg_kpts_scores = np.mean(new_kpts_scores, axis=1)
                        
                        if visualize:
                            img_path = os.path.join(root_dir, img_key)
                            img = cv2.imread(img_path)
                            img = visualize_skeletons_advanced(img, new_kpts, new_boxes)
                            save_path = os.path.join(save_dir, img_key)
                            cv2.imwrite(save_path, img)

        env.close()
        # output_json_path = args.save_root_dir + ".json"
        # generate_coco_annotations(output_json_path, all_image_data, category_names)
                    


if __name__ == '__main__':
    args = parse_args()

    main(args)
