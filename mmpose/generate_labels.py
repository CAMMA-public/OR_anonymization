import os
import json
import numpy as np
import torch
from torchvision.ops import nms
from PIL import Image # Used to get image dimensions
from tqdm import tqdm # A nice progress bar
import cv2
import lmdb
import pickle
import shutil

def remove_redundant_results(kpts, kpts_scores, boxes, kpts_score_threshold):
    for k in range(len(boxes)):
        for q in range(len(kpts[0])):
            if kpts[k][q][0] < boxes[k][0] or kpts[k][q][0] > boxes[k][2] or kpts[k][q][1] < boxes[k][1] or kpts[k][q][1] > boxes[k][3]:
                kpts_scores[k][q] = 0.
    new_kpts = []
    new_kpts_scores = []
    new_boxes = []
    min_xy = []
    max_xy = []
    for one_face_kpts, one_face_kpts_scores, one_box in zip(kpts, kpts_scores, boxes):
        mask = (one_face_kpts_scores > kpts_score_threshold)
        new_face_kpts = one_face_kpts[mask]
        # if len(new_face_kpts) == len(one_face_kpts):
        #     new_kpts.append(new_face_kpts)
        #     new_kpts_scores.append(one_face_kpts_scores)
        #     new_boxes.append(one_box)
        if len(new_face_kpts) > 0:
            min_xy.append(np.min(new_face_kpts, axis=0))
            max_xy.append(np.max(new_face_kpts, axis=0))
            new_kpts.append(one_face_kpts * np.repeat(mask[:, np.newaxis], 2, axis=1))
            new_kpts_scores.append(one_face_kpts_scores * mask)
            new_boxes.append(one_box)
    if len(new_kpts) == 0:
        return [], [], []
    new_kpts = np.stack(new_kpts, axis=0)
    new_kpts_scores = np.array(new_kpts_scores)
    new_boxes = np.stack(new_boxes, axis=0)
    min_xy = np.stack(min_xy, axis=0)
    max_xy = np.stack(max_xy, axis=0)
    
    pseudo_face_height = 40
    pseudo_face_width = 40
    # min_xy = np.min(new_kpts, axis=1)
    # max_xy = np.max(new_kpts, axis=1)
    center_x = (min_xy[:, 0] + max_xy[:, 0]) * 0.5
    center_y = (min_xy[:, 1] + max_xy[:, 1]) * 0.5
    x1 = center_x - pseudo_face_width * 0.5
    y1 = center_y - pseudo_face_height * 0.5
    x2 = center_x + pseudo_face_width * 0.5
    y2 = center_y + pseudo_face_height * 0.5
    face_boxes = np.stack([x1, y1, x2, y2], axis=1)
    face_scores = np.mean(new_kpts_scores, axis=1)
    face_boxes = torch.from_numpy(face_boxes).float()
    face_scores = torch.from_numpy(face_scores).float()
    indices = nms(face_boxes, face_scores, 0.6)
    indices = indices.numpy()
    
    return new_kpts[indices], new_kpts_scores[indices], new_boxes[indices]

# --- [STEP 1: YOU MUST MODIFY THIS FUNCTION] ---
def load_pseudo_labels(pseudo_label_path):
    """
    Loads pseudo-labels for a single image from a given path.

    This is the ONLY function you need to modify. It should read your
    pseudo-label file (e.g., a JSON, TXT, or .npz file) and return
    the data in a specific format.

    Args:
        pseudo_label_path (str): The full path to the pseudo-label file
                                 corresponding to one image.

    Returns:
        list[np.ndarray]: A list of detected person instances.
                          Each instance is a NumPy array of shape (133, 3),
                          representing the 133 keypoints for one person.
                          The columns for each keypoint are [x, y, confidence_score].
                          Return an empty list if no persons are detected.
    """
    # --- EXAMPLE IMPLEMENTATION (assuming labels are in a simple JSON file) ---
    #
    # Replace this example with your actual file reading logic.
    #
    # Let's assume your JSON file looks like this:
    # {
    #   "predictions": [
    #     { "keypoints": [x1, y1, score1, x2, y2, score2, ...] }, // Person 1
    #     { "keypoints": [x1, y1, score1, x2, y2, score2, ...] }  // Person 2
    #   ]
    # }
    
    if not os.path.exists(pseudo_label_path):
        return [] # Return empty list if no label file exists for this image

    instances = []
    with open(pseudo_label_path, 'r') as f:
        data = json.load(f)

    for person_data in data.get("predictions", []):
        # Reshape the flat list of keypoints into a (133, 3) array
        keypoints = np.array(person_data["keypoints"]).reshape((133, 3))
        instances.append(keypoints)
        
    return instances
    # --- END OF EXAMPLE IMPLEMENTATION ---


# --- [STEP 2: CONFIGURE YOUR PATHS AND THRESHOLD] ---
def generate_coco_annotations():
    """
    Main function to generate the COCO-WholeBody annotation file.
    """
    # Configure your paths here
    pred_paths = [
        '/media/camma-monitor/Storage_postprocessing2/pose_detection/face/006_RS_JPG_iter2_mva_tracking_reid_face.lmdb', 
        '/media/camma-monitor/Storage_postprocessing2/pose_detection/face/007_CL_JPG_iter2_mva_tracking_reid_face.lmdb', 
        '/media/camma-monitor/Storage_postprocessing2/pose_detection/face/009_SV_JPG_iter2_mva_tracking_reid_face.lmdb'
    ]
    root_dirs = [
        '/media/camma-monitor/Storage_postprocessing2/006_RS_JPG/006_RS', 
        '/media/camma-monitor/Storage_postprocessing3/007_CL_JPG', 
        '/media/camma-monitor/Storage_postprocessing3/009_SV_JPG'
    ]
    save_dirs = ['006', '007', '009']
    file_num = len(pred_paths)
    OUTPUT_JSON_PATH = "monitor_kpts_full.json"

    # Set the confidence score threshold for your pseudo-labels
    kpts_score_threshold = 2.5

    # --- Main processing loop ---
    coco_output = {
        "info": {"description": "My Custom Pseudo-Labeled Dataset"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "supercategory": "person",
                "id": 1,
                "name": "person",
                "keypoints": [
                    "nose",
                    "left_eye",
                    "right_eye",
                    "left_ear",
                    "right_ear",
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle"
                ],
                "skeleton": [
                    [
                        16,
                        14
                    ],
                    [
                        14,
                        12
                    ],
                    [
                        17,
                        15
                    ],
                    [
                        15,
                        13
                    ],
                    [
                        12,
                        13
                    ],
                    [
                        6,
                        12
                    ],
                    [
                        7,
                        13
                    ],
                    [
                        6,
                        7
                    ],
                    [
                        6,
                        8
                    ],
                    [
                        7,
                        9
                    ],
                    [
                        8,
                        10
                    ],
                    [
                        9,
                        11
                    ],
                    [
                        2,
                        3
                    ],
                    [
                        1,
                        2
                    ],
                    [
                        1,
                        3
                    ],
                    [
                        2,
                        4
                    ],
                    [
                        3,
                        5
                    ],
                    [
                        4,
                        6
                    ],
                    [
                        5,
                        7
                    ]
                ]
            }
        ]
    }
    
    image_id_counter = 1
    annotation_id_counter = 1
    
    for f_n in range(file_num):
        pred_path = pred_paths[f_n]
        root_dir = root_dirs[f_n]
        save_dir = save_dirs[f_n]
        
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
        
        demo_img_path = os.path.join(root_dir, sub_dirs[0], img_names[0])
        demo_img = cv2.imread(demo_img_path)
        height, width, _ = demo_img.shape
        
        save_interval = 150
        count_seed = 0
        
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
                    img_path = os.path.join(root_dir, img_key)
                    
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
                            kpts, kpts_scores, boxes = remove_redundant_results(kpts, kpts_scores, boxes, kpts_score_threshold)
                            num_person = len(kpts)
                    
                    if num_person == 0:
                        continue
                    
                    # Create the image entry
                    image_info = {
                        "id": image_id_counter,
                        "file_name": os.path.join(save_dir, sub_dir, img_name),
                        "width": width,
                        "height": height
                    }
                    coco_output["images"].append(image_info)
                    
                    old_path = os.path.join(root_dir, sub_dir, img_name)
                    new_path = os.path.join('/media/camma-monitor/Storage_postprocessing2/pose_detection/CrowdHuman/Images', save_dir, sub_dir, img_name)
                    shutil.copy(old_path, new_path)
                    
                    for n in range(num_person):
                        x1 = round(boxes[n][0], 2)
                        y1 = round(boxes[n][1], 2)
                        w = round(boxes[n][2] - boxes[n][0], 2)
                        h = round(boxes[n][3] - boxes[n][1], 2)
                        keypoints = [0 for _ in range(51)]
                        face_kpts = [0.0 for _ in range(204)]
                        face_valid = False
                        if kpts[n][0][0] != 0. and kpts[n][0][1] != 0.:
                            keypoints[3] = int(kpts[n][0][0])
                            keypoints[4] = int(kpts[n][0][1])
                            keypoints[5] = 2
                        if kpts[n][1][0] != 0. and kpts[n][1][1] != 0.:
                            keypoints[6] = int(kpts[n][1][0])
                            keypoints[7] = int(kpts[n][1][1])
                            keypoints[8] = 2
                        if kpts[n][2][0] != 0. and kpts[n][2][1] != 0.:
                            face_kpts[24] = kpts[n][2][0]
                            face_kpts[25] = kpts[n][2][1]
                            face_kpts[26] = 1.0
                            face_valid = True
                        # Create the annotation entry
                        annotation_info = {
                            "segmentation": [],
                            "id": annotation_id_counter,
                            "image_id": image_id_counter,
                            "category_id": 1, # "person" category
                            "bbox": [x1, y1, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                            "num_keypoints": 2,
                            "keypoints": keypoints, 
                            "face_box": [0.0 for _ in range(4)], 
                            "lefthand_box": [0.0 for _ in range(4)], 
                            "righthand_box": [0.0 for _ in range(4)], 
                            "lefthand_kpts": [0.0 for _ in range(63)], 
                            "righthand_kpts": [0.0 for _ in range(63)], 
                            "face_kpts": face_kpts, 
                            "face_valid": face_valid, 
                            "lefthand_valid": False, 
                            "righthand_valid": False, 
                            "foot_valid": False, 
                            "foot_kpts": [0.0 for _ in range(18)]
                        }
                        coco_output["annotations"].append(annotation_info)
                        annotation_id_counter += 1
                    image_id_counter += 1
        
    # Write the final JSON file
    print(f"\nSaving annotations to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print("Done!")

if __name__ == '__main__':
    generate_coco_annotations()