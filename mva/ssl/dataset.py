from torch.utils.data import Dataset
from collections import defaultdict
import os
import numpy as np
import cv2
import random
import re
import config as C
from torchvision import transforms as T
import torch
import json
import pickle
import lmdb
from utils import bbox_iou, generate_neg_bboxes


class OR_Loader(Dataset):
    def __init__(self, config, mode='train', inference=False, box_thresh=0.6):
        self.img_size = config.IMG_SIZE
        self.sampling_range = config.DATASET.SAMPLING_RANGE
        self.mode = mode
        self.inference = inference
        self.dataset = config.DATASET.TRAIN_DATASET
        self.train_mode = config.TRAIN.TRAIN_MODE
        self.crop_box = config.CROP_BOX
        self.zoom_out_ratio = config.ZOOM_OUT_RATIO
        self.intra_loss = config.TRAIN.INTRA_LOSS
        self.depth = config.DEPTH
        self.fix_view_id = config.FIX_VIEW_ID
        self.monopair = config.TRAIN.MONOPAIR
        self.pseudo_neg = config.TRAIN.PSEUDO_NEG
        self.view_ls = config.DATASET.VIEW_IDS
        self.tracking = config.TRAIN.TRACKING
        self.box_thresh = box_thresh

        self.suffix = '.jpg'
        self.prompt_mode = config.DATASET.PROMPT_MODE
        self.crop_img = False
        if self.prompt_mode == 'box':
            if config.TRAIN.REID or config.TEST.REID:
                self.crop_img = True

        # self.root_dir = '/media/camma-monitor/Storage_backup/'
        # self.vid_names = ['001_OF_JPG']
        # self.cam_names = ['117222250956', '309622300656', '309622301491']
        # self.anno_dir = '/media/camma-monitor/Storage_postprocessing2/pose_detection_utils/code/anonymization/detector'
        # self.anno_dicts, self.img_dict = self.gen_anno_path_dict()
        
        self.image_dirs = []
        self.anno_paths = []
        self.sample_intervals = []
        self.frame_ranges = []
        if self.tracking:
            self.tracking_paths = []
        if self.dataset == 'monitor':
            self.cam_names = ['117222250956', '239222302045','309622300656', '309622301491']
            if mode == 'train':
                self.register_dataset('/media/camma-monitor/Storage_postprocessing2/006_RS_JPG/006_RS', '/media/camma-monitor/Storage_postprocessing2/pose_detection/detections/006_RS_JPG_iter2.lmdb', 60)
                self.register_dataset('/media/camma-monitor/Storage_postprocessing3/007_CL_JPG', '/media/camma-monitor/Storage_postprocessing2/pose_detection/detections/007_CL_JPG_iter2.lmdb', 60)
                self.register_dataset('/media/camma-monitor/Storage_postprocessing3/009_SV_JPG', '/media/camma-monitor/Storage_postprocessing2/pose_detection/detections/009_SV_JPG_iter2.lmdb', 60)
            else:
                self.register_dataset('/media/camma-monitor/Storage_postprocessing2/005_CG_JPG/005_CG', '/media/camma-monitor/Storage_postprocessing2/pose_detection/detections/005_CG_JPG_iter2.lmdb', 60)
                self.register_dataset('/media/camma-monitor/Storage_postprocessing3/008_OT_JPG', '/media/camma-monitor/Storage_postprocessing2/pose_detection/detections/008_OT_JPG_iter2.lmdb', 60)
            if self.tracking:
                if mode == 'train':
                    self.tracking_paths.append('/media/camma-monitor/Storage_postprocessing2/pose_detection/tracking/006_RS_JPG_iter2_tracking.lmdb')
                    self.tracking_paths.append('/media/camma-monitor/Storage_postprocessing2/pose_detection/tracking/007_CL_JPG_iter2_tracking.lmdb')
                    self.tracking_paths.append('/media/camma-monitor/Storage_postprocessing2/pose_detection/tracking/009_SV_JPG_iter2_tracking.lmdb')
                else:
                    self.tracking_paths.append('/media/camma-monitor/Storage_postprocessing2/pose_detection/tracking/005_CG_JPG_iter2_tracking.lmdb')
                    self.tracking_paths.append('/media/camma-monitor/Storage_postprocessing2/pose_detection/tracking/008_OT_JPG_iter2_tracking.lmdb')
        elif self.dataset == '4dor':
            if mode == 'train':
                dataset_ids = [1, 3, 4, 5, 7, 8, 9, 10]
            else:
                dataset_ids = [2, 6]
            self.img_dicts = []
            for i in dataset_ids:
                self.register_dataset(f'/home2020/home/icube/keqichen/code/4D-OR/datasets/4D-OR/export_holistic_take{i}_processed/colorimage', f'/home2020/home/icube/keqichen/code/4D-OR/datasets/4D-OR/detections/export_holistic_take{i}_processed_iter3.lmdb', 1)
                img_dicts_pkl = os.path.join(f'/home2020/home/icube/keqichen/code/4D-OR/datasets/4D-OR/export_holistic_take{i}_processed', 'img_dicts.pkl')
                with open(img_dicts_pkl, mode='rb') as f:
                    img_dicts = pickle.load(f)
                self.img_dicts.append(img_dicts)
            if self.tracking:
                for i in dataset_ids:
                    self.tracking_paths.append(f'/home2020/home/icube/keqichen/code/4D-OR/datasets/4D-OR/tracking/export_holistic_take{i}_processed_iter3_tracking.lmdb')
        elif self.dataset == 'Garden1':
            if mode == 'train':
                self.register_dataset('/home2020/home/icube/keqichen/datasets/CAMPUS/Garden1/output/frames', '/home2020/home/icube/keqichen/datasets/CAMPUS/detections/Garden1.lmdb', 1, [0, 2281])
            else:
                self.register_dataset('/home2020/home/icube/keqichen/datasets/CAMPUS/Garden1/output/frames', '/home2020/home/icube/keqichen/datasets/CAMPUS/detections/Garden1.lmdb', 1, [2281, 2849])
            if self.tracking:
                if mode == 'train':
                    self.tracking_paths.append('/home2020/home/icube/keqichen/datasets/CAMPUS/tracking/Garden1_tracking.lmdb')
                else:
                    self.tracking_paths.append('/home2020/home/icube/keqichen/datasets/CAMPUS/tracking/Garden1_tracking.lmdb')
        elif self.dataset == 'Garden2':
            if mode == 'train':
                self.register_dataset('/home2020/home/icube/keqichen/datasets/CAMPUS/Garden2/output/frames', '/home2020/home/icube/keqichen/datasets/CAMPUS/detections/Garden2.lmdb', 1, [0, 4801])
            else:
                self.register_dataset('/home2020/home/icube/keqichen/datasets/CAMPUS/Garden2/output/frames', '/home2020/home/icube/keqichen/datasets/CAMPUS/detections/Garden2.lmdb', 1, [4801, 6000])
            if self.tracking:
                if mode == 'train':
                    self.tracking_paths.append('/home2020/home/icube/keqichen/datasets/CAMPUS/tracking/Garden2_tracking.lmdb')
                else:
                    self.tracking_paths.append('/home2020/home/icube/keqichen/datasets/CAMPUS/tracking/Garden2_tracking.lmdb')
        elif self.dataset == 'Parkinglot':
            if mode == 'train':
                self.register_dataset('/home2020/home/icube/keqichen/datasets/CAMPUS/Parkinglot/output/frames', '/home2020/home/icube/keqichen/datasets/CAMPUS/detections/Parkinglot.lmdb', 1, [0, 5829])
            else:
                self.register_dataset('/home2020/home/icube/keqichen/datasets/CAMPUS/Parkinglot/output/frames', '/home2020/home/icube/keqichen/datasets/CAMPUS/detections/Parkinglot.lmdb', 1, [5829, 6475])
            if self.tracking:
                if mode == 'train':
                    self.tracking_paths.append('/home2020/home/icube/keqichen/datasets/CAMPUS/tracking/Parkinglot_tracking.lmdb')
                else:
                    self.tracking_paths.append('/home2020/home/icube/keqichen/datasets/CAMPUS/tracking/Parkinglot_tracking.lmdb')
        
        result = self.prepare_data()
        self.anno_dicts = result['anno_dicts']
        self.img_dict = result['path_dict']
        if self.tracking:
            self.tracking_dicts = result['tracking_dicts']

        self.heights = []
        self.widths = []
        for cam_id in range(len(self.view_ls)):
            demo_img_path, _ = self.img_dict[self.view_ls[cam_id]][0]
            demo_img = cv2.imread(demo_img_path)
            height, width, _ = demo_img.shape
            self.heights.append(height)
            self.widths.append(width)
    
    def get_width_height(self):
        return self.widths, self.heights
    
    def register_dataset(self, img_dir, anno_path, sample_interval=1, frame_range=None):
        self.image_dirs.append(img_dir)
        self.anno_paths.append(anno_path)
        self.sample_intervals.append(sample_interval)
        if frame_range is not None:
            self.frame_ranges.append(frame_range)
    
    def prepare_data(self):
        anno_dicts = []
        path_dict = defaultdict(list)
        if self.tracking:
            tracking_dicts = []
        
        for vid_id, anno_path in enumerate(self.anno_paths):
            root_dir = self.image_dirs[vid_id]
            sample_interval = self.sample_intervals[vid_id]
            
            img_names_dict = {}
            if self.dataset == 'monitor':
                for c_id, sub_dir in enumerate(self.cam_names):
                    img_dir = os.path.join(root_dir, sub_dir)
                    img_names = os.listdir(img_dir)
                    img_names = [x for x in img_names if x[-3:] == 'jpg' and x[0] == 'c']
                    img_names.sort()
                    img_names = img_names[::sample_interval]
                    img_names_dict[c_id] = img_names
                    
                    for img_name in img_names:
                        img_path = os.path.join(img_dir, img_name)
                        path_dict[c_id].append((img_path, vid_id))
            elif self.dataset == '4dor':
                img_dicts = self.img_dicts[vid_id]
                for c_id, img_dict in enumerate(img_dicts):
                    for f_id, img_name in img_dict.items():
                        img_path = os.path.join(root_dir, img_name)
                        path_dict[c_id].append((img_path, vid_id))
            elif self.dataset in ['Garden1', 'Garden2', 'Parkinglot']:
                start_f_id, end_f_id = self.frame_ranges[vid_id]
                for c_id in range(4):
                    for f_id in range(start_f_id, end_f_id, sample_interval):
                        img_name = str(f_id) + '_' + str(c_id) + '.jpg'
                        img_path = os.path.join(root_dir, img_name)
                        path_dict[c_id].append((img_path, vid_id))
                    
            anno_dict = {}
            
            env = lmdb.open(anno_path, readonly=True, lock=False, subdir=False)
            with env.begin() as txn:
                # Use a cursor for efficient iteration if desired.
                cursor = txn.cursor()
                
                if self.dataset == 'monitor':
                    for c_id, sub_dir in enumerate(self.cam_names):
                        img_names = img_names_dict[c_id]
                        for img_name in img_names:
                            f_id = int(img_name.rsplit('.', 1)[0].rsplit('-', 1)[1])
                            
                            img_key = os.path.join(sub_dir, img_name)
                            key_boxes = f"{img_key}/boxes".encode('utf-8')
                            key_scores = f"{img_key}/scores".encode('utf-8')
                            
                            boxes_data = txn.get(key_boxes)
                            scores_data = txn.get(key_scores)
                            if boxes_data is not None and scores_data is not None:
                                # Deserialize using pickle.
                                boxes = pickle.loads(boxes_data)
                                scores = pickle.loads(scores_data)
                                
                                if len(boxes):
                                    mask = scores > self.box_thresh
                                    left_boxes = boxes[mask]
                                    if len(left_boxes):
                                        f_ids = np.ones((len(left_boxes), 1), dtype=int) * f_id
                                        t_ids = np.ones((len(left_boxes), 1), dtype=int) * -1
                                        left_scores = np.expand_dims(scores[mask], axis=1)
                                        save_data = np.concatenate([f_ids, t_ids, left_boxes, left_scores], axis=1)
                                        if c_id not in anno_dict:
                                            anno_dict[c_id] = defaultdict(list)
                                        anno_dict[c_id][f_id] = save_data
                elif self.dataset == '4dor':
                    img_dicts = self.img_dicts[vid_id]
                    for c_id, img_dict in enumerate(img_dicts):
                        if c_id not in anno_dict:
                            anno_dict[c_id] = defaultdict(list)
                        for f_id, img_name in img_dict.items():
                            img_key = str(f_id) + '_' + str(c_id)
                            key_boxes = f"{img_key}/boxes".encode('utf-8')
                            key_scores = f"{img_key}/scores".encode('utf-8')
                            
                            boxes_data = txn.get(key_boxes)
                            scores_data = txn.get(key_scores)
                            if boxes_data is not None and scores_data is not None:
                                # Deserialize using pickle.
                                boxes = pickle.loads(boxes_data)
                                scores = pickle.loads(scores_data)
                                
                                if len(boxes):
                                    mask = scores > self.box_thresh
                                    left_boxes = boxes[mask]
                                    if len(left_boxes):
                                        f_ids = np.ones((len(left_boxes), 1), dtype=int) * f_id
                                        t_ids = np.ones((len(left_boxes), 1), dtype=int) * -1
                                        left_scores = np.expand_dims(scores[mask], axis=1)
                                        save_data = np.concatenate([f_ids, t_ids, left_boxes, left_scores], axis=1)
                                        anno_dict[c_id][f_id] = save_data
                                else:
                                    anno_dict[c_id][f_id] = []
                elif self.dataset in ['Garden1', 'Garden2', 'Parkinglot']:
                    start_f_id, end_f_id = self.frame_ranges[vid_id]
                    for c_id in range(4):
                        if c_id not in anno_dict:
                            anno_dict[c_id] = defaultdict(list)
                        for f_id in range(start_f_id, end_f_id, sample_interval):
                            img_key = str(f_id) + '_' + str(c_id) + '.jpg'
                            key_boxes = f"{img_key}/boxes".encode('utf-8')
                            key_scores = f"{img_key}/scores".encode('utf-8')
                            
                            boxes_data = txn.get(key_boxes)
                            scores_data = txn.get(key_scores)
                            if boxes_data is not None and scores_data is not None:
                                # Deserialize using pickle.
                                boxes = pickle.loads(boxes_data)
                                scores = pickle.loads(scores_data)
                                
                                if len(boxes):
                                    mask = scores > self.box_thresh
                                    left_boxes = boxes[mask]
                                    if len(left_boxes):
                                        f_ids = np.ones((len(left_boxes), 1), dtype=int) * f_id
                                        t_ids = np.ones((len(left_boxes), 1), dtype=int) * -1
                                        left_scores = np.expand_dims(scores[mask], axis=1)
                                        save_data = np.concatenate([f_ids, t_ids, left_boxes, left_scores], axis=1)
                                        anno_dict[c_id][f_id] = save_data
                                else:
                                    anno_dict[c_id][f_id] = []
            env.close()
            anno_dicts.append(anno_dict)
            
            if self.tracking:
                tracking_dict = {}
                tracking_path = self.tracking_paths[vid_id]
                tracking_env = lmdb.open(tracking_path, readonly=True, lock=False, subdir=False)
                with tracking_env.begin() as txn:
                    # Use a cursor for efficient iteration if desired.
                    cursor = txn.cursor()
                    
                    if self.dataset == 'monitor':
                        for c_id, sub_dir in enumerate(self.cam_names):
                            img_names = img_names_dict[c_id]
                            for img_name in img_names:
                                f_id = int(img_name.rsplit('.', 1)[0].rsplit('-', 1)[1])
                                
                                img_key = os.path.join(sub_dir, img_name)
                                key_boxes = f"{img_key}/boxes".encode('utf-8')
                                key_scores = f"{img_key}/scores".encode('utf-8')
                                
                                boxes_data = txn.get(key_boxes)
                                scores_data = txn.get(key_scores)
                                if boxes_data is not None and scores_data is not None:
                                    # Deserialize using pickle.
                                    boxes = pickle.loads(boxes_data)
                                    scores = pickle.loads(scores_data)
                                    
                                    if len(boxes):
                                        mask = scores > self.box_thresh
                                        left_boxes = boxes[mask]
                                        if len(left_boxes):
                                            f_ids = np.ones((len(left_boxes), 1), dtype=int) * f_id
                                            t_ids = np.ones((len(left_boxes), 1), dtype=int) * -1
                                            left_scores = np.expand_dims(scores[mask], axis=1)
                                            save_data = np.concatenate([f_ids, t_ids, left_boxes, left_scores], axis=1)
                                            if c_id not in tracking_dict:
                                                tracking_dict[c_id] = defaultdict(list)
                                            tracking_dict[c_id][f_id] = save_data
                    elif self.dataset == '4dor':
                        img_dicts = self.img_dicts[vid_id]
                        for c_id, img_dict in enumerate(img_dicts):
                            if c_id not in tracking_dict:
                                tracking_dict[c_id] = defaultdict(list)
                            for f_id, img_name in img_dict.items():
                                img_key = str(f_id) + '_' + str(c_id)
                                key_boxes = f"{img_key}/boxes".encode('utf-8')
                                key_scores = f"{img_key}/scores".encode('utf-8')
                                
                                boxes_data = txn.get(key_boxes)
                                scores_data = txn.get(key_scores)
                                if boxes_data is not None and scores_data is not None:
                                    # Deserialize using pickle.
                                    boxes = pickle.loads(boxes_data)
                                    scores = pickle.loads(scores_data)
                                    
                                    if len(boxes):
                                        mask = scores > self.box_thresh
                                        left_boxes = boxes[mask]
                                        if len(left_boxes):
                                            f_ids = np.ones((len(left_boxes), 1), dtype=int) * f_id
                                            t_ids = np.ones((len(left_boxes), 1), dtype=int) * -1
                                            left_scores = np.expand_dims(scores[mask], axis=1)
                                            save_data = np.concatenate([f_ids, t_ids, left_boxes, left_scores], axis=1)
                                            tracking_dict[c_id][f_id] = save_data
                                    else:
                                        tracking_dict[c_id][f_id] = []
                    elif self.dataset in ['Garden1', 'Garden2', 'Parkinglot']:
                        start_f_id, end_f_id = self.frame_ranges[vid_id]
                        for c_id in range(4):
                            if c_id not in tracking_dict:
                                tracking_dict[c_id] = defaultdict(list)
                            for f_id in range(start_f_id, end_f_id, sample_interval):
                                img_key = str(f_id) + '_' + str(c_id) + '.jpg'
                                key_boxes = f"{img_key}/boxes".encode('utf-8')
                                key_scores = f"{img_key}/scores".encode('utf-8')
                                
                                boxes_data = txn.get(key_boxes)
                                scores_data = txn.get(key_scores)
                                if boxes_data is not None and scores_data is not None:
                                    # Deserialize using pickle.
                                    boxes = pickle.loads(boxes_data)
                                    scores = pickle.loads(scores_data)
                                    
                                    if len(boxes):
                                        mask = scores > self.box_thresh
                                        left_boxes = boxes[mask]
                                        if len(left_boxes):
                                            f_ids = np.ones((len(left_boxes), 1), dtype=int) * f_id
                                            t_ids = np.ones((len(left_boxes), 1), dtype=int) * -1
                                            left_scores = np.expand_dims(scores[mask], axis=1)
                                            save_data = np.concatenate([f_ids, t_ids, left_boxes, left_scores], axis=1)
                                            tracking_dict[c_id][f_id] = save_data
                                    else:
                                        tracking_dict[c_id][f_id] = []
                tracking_env.close()
                tracking_dicts.append(tracking_dict)
        
        result = {
            'anno_dicts': anno_dicts, 
            'path_dict': path_dict
        }
        if self.tracking:
            result['tracking_dicts'] = tracking_dicts
            
        return result
                                

    def gen_anno_path_dict(self):
        anno_dicts = []
        path_dict = defaultdict(list)
        
        for vid_id, vid_name in enumerate(self.vid_names):
            anno_name = vid_name + '_for_tracking.json'
            anno_path = os.path.join(self.anno_dir, anno_name)
            with open(anno_path, 'r') as fp:
                frames = json.load(fp)
            frames_id = list(map(int, frames.keys())) # [0, 1, ..., frame_num - 1]
            # FPS = 15
            # samples the frames every 4 seconds
            frames_id = frames_id[::60]
            
            for f_id in frames_id:
                for cam_id, cam_name in enumerate(self.cam_names):
                    img_dir = os.path.join(self.root_dir, vid_name, cam_name)
                    img_name = 'color-' + str(f_id).zfill(8) + self.suffix
                    img_path = os.path.join(img_dir, img_name)
                    path_dict[cam_id].append((img_path, vid_id))
            
            anno_dict = {}
            for f_id in frames_id:
                nodes = frames[str(f_id)]
                for node in nodes:
                    x, y, w, h, tid, cid, score = node
                    if score > 0.6:
                        if cid not in anno_dict:
                            anno_dict[cid] = defaultdict(list)
                        anno_dict[cid][f_id].append([f_id, tid, x, y, x+w, y+h])
            anno_dicts.append(anno_dict)

        return anno_dicts, path_dict

    def read_anno(self, path, vid_id, c_id, f_id, tracking=False):
        if self.dataset == 'monitor':
            cam_name, img_name = path.rsplit('/', 2)[1:]
            c_id = self.cam_names.index(cam_name)
            f_id = img_name.rsplit('.', 1)[0].rsplit('-', 1)[1]
        elif self.dataset in ['Garden1', 'Garden2', 'Parkinglot']:
            f_id, c_id = path.rsplit('/', 1)[-1].rsplit('.', 1)[0].rsplit('_', 1)
            c_id = int(c_id)
            f_id = int(f_id)
        if tracking and self.tracking:
            annos = self.tracking_dicts[vid_id][c_id][int(f_id)]
        else:
            annos = self.anno_dicts[vid_id][c_id][int(f_id)]

        bbox_dict = {}
        for idx, anno in enumerate(annos):
            bbox = anno[2:6]
            bbox = [int((float(i))) for i in bbox]
            if anno[1] == -1:
                dict_key = idx
            else:
                dict_key = anno[1]
            bbox_dict[dict_key] = (bbox, float(anno[6]))
        return bbox_dict
    
    def get_intra_loss_pair(self, bbox_dict):
        bboxes = []
        for key in bbox_dict:
            bbox = bbox_dict[key]
            bboxes.append(bbox)
        bboxes = np.asarray(bboxes)
        box_num = len(bboxes)
        box1 = bboxes.repeat(box_num, 0)
        box2 = np.tile(bboxes, (box_num, 1))
        giou_matrix = bbox_iou(box1.T, box2.T, x1y1x2y2=True, GIoU=True).reshape(box_num, box_num)
        giou_matrix = torch.from_numpy(giou_matrix)
        _, max_indices = torch.topk(giou_matrix, 2)
        _, min_indices = torch.min(giou_matrix, 1)
        return max_indices[:, 1], min_indices

    
    def load_image(self, frame_img, img_size=(224, 224), tensor=True):
        img = cv2.imread(frame_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if tensor:
            img = T.ToTensor()(img)  # (C, H, W), float
        else:
            img = torch.from_numpy(img)
            img = torch.permute(img, (2, 0, 1))  # (C, H, W), uint8
        img = T.Resize(img_size)(img)
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img
    
    def get_bboxes(self, bbox_dict, c_id):
        label_ls = []
        bbox_ls = []
        for key in bbox_dict:
            bbox = bbox_dict[key]
            if self.prompt_mode == 'box':
                bbox[0] /= self.widths[c_id]
                bbox[1] /= self.heights[c_id]
                bbox[2] /= self.widths[c_id]
                bbox[3] /= self.heights[c_id]
            elif self.prompt_mode == 'point':
                bbox[0] /= self.widths[c_id]
                bbox[1] /= self.heights[c_id]
            # bbox = [0 if i < 0 else i for i in bbox]
            # bbox = [1 if i > 1 else i for i in bbox]
            bbox = torch.tensor(bbox, dtype=torch.float32)
            label_ls.append(key)
            bbox_ls.append(bbox)
        if len(bbox_ls):
            bbox_ls = torch.stack(bbox_ls)
        return label_ls, bbox_ls

    
    def get_view_num(self):
        return len(self.view_ls)

    def __len__(self):
        # return self.len
        return min([len(self.img_dict[i]) for i in self.view_ls])

    def __getitem__(self, item):
        if self.inference:
            return self.get_association_item(item)
        if self.fix_view_id == -1:
            anchor_view_id, sample_view_id = random.sample(range(0, len(self.view_ls)), 2)
        else:
            anchor_view_id = self.fix_view_id
            sample_view_id = random.sample(range(0, len(self.view_ls) - 1), 1)[0]
            if sample_view_id >= anchor_view_id:
                sample_view_id += 1

        if self.crop_box:
            return self.get_crop_item(item, anchor_view_id, sample_view_id)

        anchor_img_path, anchor_vid_id = self.img_dict[self.view_ls[anchor_view_id]][item]
        anchor_anno = self.read_anno(anchor_img_path, anchor_vid_id, anchor_view_id, item)
        _, anchor_bboxes = self.get_bboxes(anchor_anno, anchor_view_id)

        pos_img_path, pos_vid_id = self.img_dict[self.view_ls[sample_view_id]][item]
        pos_anno = self.read_anno(pos_img_path, pos_vid_id, sample_view_id, item)
        _, pos_bboxes = self.get_bboxes(pos_anno, sample_view_id)

        anchor_img = self.load_image(anchor_img_path, self.img_size)
        positive_img = self.load_image(pos_img_path, self.img_size)

        anchor_info = {
            'img': anchor_img, 
            'bboxes': anchor_bboxes, 
            'cam_id': anchor_view_id
        }
        pos_info = {
            'img': positive_img, 
            'bboxes': pos_bboxes, 
            'cam_id': sample_view_id
        }

        if self.mode == 'train' and self.train_mode == 'triplet':
            sample_min, sample_max = self.sampling_range
            interval = sample_min + int(random.random() * (sample_max - sample_min))
            previous_sample = False
            if random.random() > 0.5:
                previous_sample = True
            if previous_sample:
                if item - interval >= 0:
                    new_item = item - interval
                else:
                    new_item = item + interval
            else:
                if item + interval < len(self.img_dict[self.view_ls[sample_view_id]]):
                    new_item = item + interval
                else:
                    new_item = item - interval
            
            negative_img = self.load_image(self.img_dict[self.view_ls[sample_view_id]][new_item], self.img_size)

            neg_img_path, neg_vid_id = self.img_dict[self.view_ls[sample_view_id]][new_item]
            neg_anno = self.read_anno(neg_img_path, neg_vid_id, sample_view_id, new_item)
            _, neg_bboxes = self.get_bboxes(neg_anno, sample_view_id)

            loaded_imgs = torch.stack([anchor_img, positive_img, negative_img], axis=0)

            neg_info = {
                'img': negative_img, 
                'bboxes': neg_bboxes, 
                'cam_id': sample_view_id
            }

            if self.mode == 'train' and self.intra_loss:
                anchor_intra_pos, anchor_intra_neg = self.get_intra_loss_pair(anchor_anno)
                anchor_info['intra'] = (anchor_intra_pos, anchor_intra_neg)
                pos_intra_pos, pos_intra_neg = self.get_intra_loss_pair(pos_anno)
                pos_info['intra'] = (pos_intra_pos, pos_intra_neg)
                neg_intra_pos, neg_intra_neg = self.get_intra_loss_pair(neg_anno)
                neg_info['intra'] = (neg_intra_pos, neg_intra_neg)

            return anchor_info, pos_info, neg_info
        
        else:
            if self.mode == 'train' and self.intra_loss:
                anchor_intra_pos, anchor_intra_neg = self.get_intra_loss_pair(anchor_anno)
                anchor_info['intra'] = (anchor_intra_pos, anchor_intra_neg)
                pos_intra_pos, pos_intra_neg = self.get_intra_loss_pair(pos_anno)
                pos_info['intra'] = (pos_intra_pos, pos_intra_neg)

            return anchor_info, pos_info
    

    def load_cropped_bboxes(self, img_path, bbox_dict, c_id, depth_key=None, img_size=(224, 224), tensor=True, bbox_jitter=False, tracking_dict=None):
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if tensor:
            img = T.ToTensor()(img)  # (C, H, W), float
        else:
            img = torch.from_numpy(img)
            img = torch.permute(img, (2, 0, 1))  # (C, H, W), uint8
        
        if self.depth:
            depth = self.depth_dict[depth_key]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            _, depth_h, depth_w = depth.shape
            depth = torch.from_numpy(depth)
        
        if self.crop_img:
            enlarge_ratio = (self.zoom_out_ratio - 1) * 0.5
            cropped_imgs = []
        label_ls = []
        bbox_ls = []
        score_ls = []
        for key in bbox_dict:
            bbox = bbox_dict[key][0].copy()
            score = bbox_dict[key][1]

            if self.crop_img:
                h = bbox[3] - bbox[1]
                w = bbox[2] - bbox[0]
                new_bbox_y1 = int(max(bbox[1] - h * enlarge_ratio, 0))
                new_bbox_y2 = int(min(bbox[3] + h * enlarge_ratio, height))
                new_bbox_x1 = int(max(bbox[0] - w * enlarge_ratio, 0))
                new_bbox_x2 = int(min(bbox[2] + w * enlarge_ratio, width))
                cropped_img = img[:, new_bbox_y1:new_bbox_y2, new_bbox_x1:new_bbox_x2]
                cropped_img = T.Resize(img_size)(cropped_img)
                cropped_img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(cropped_img)

                if self.depth:
                    depth_y1 = int(new_bbox_y1 / height * depth_h)
                    depth_y2 = int(new_bbox_y2 / height * depth_h)
                    depth_x1 = int(new_bbox_x1 / width * depth_w)
                    depth_x2 = int(new_bbox_x2 / width * depth_w)
                    cropped_depth = depth[:, depth_y1:depth_y2, depth_x1:depth_x2]
                    cropped_depth = T.Resize(img_size)(cropped_depth)
                    cropped_img = torch.cat([cropped_img, cropped_depth], dim=0)

                cropped_imgs.append(cropped_img)

            label_ls.append(key)
            if self.prompt_mode == 'box':
                bbox[0] /= self.widths[c_id]
                bbox[1] /= self.heights[c_id]
                bbox[2] /= self.widths[c_id]
                bbox[3] /= self.heights[c_id]
            elif self.prompt_mode == 'point':
                bbox[0] /= self.widths[c_id]
                bbox[1] /= self.heights[c_id]
            # bbox = [0 if i < 0 else i for i in bbox]
            # bbox = [1 if i > 1 else i for i in bbox]
            bbox = torch.tensor(bbox, dtype=torch.float32)
            bbox_ls.append(bbox)
            score_ls.append(score)
        
        if len(bbox_ls):
            bbox_ls = torch.stack(bbox_ls)
            score_ls = torch.tensor(score_ls, dtype=torch.float32)
            if bbox_jitter:
                bbox_ls, _ = generate_neg_bboxes(bbox_ls, in_boundry=True)
        
        if tracking_dict is not None and self.tracking:
            if self.crop_img:
                cropped_tracking_imgs = []
            tracking_ls = []
            for key in tracking_dict:
                bbox = tracking_dict[key][0].copy()
                score = tracking_dict[key][1]

                if self.crop_img:
                    h = bbox[3] - bbox[1]
                    w = bbox[2] - bbox[0]
                    new_bbox_y1 = int(max(bbox[1] - h * enlarge_ratio, 0))
                    new_bbox_y2 = int(min(bbox[3] + h * enlarge_ratio, height))
                    new_bbox_x1 = int(max(bbox[0] - w * enlarge_ratio, 0))
                    new_bbox_x2 = int(min(bbox[2] + w * enlarge_ratio, width))
                    cropped_img = img[:, new_bbox_y1:new_bbox_y2, new_bbox_x1:new_bbox_x2]
                    
                    # save_img = cv2.imread(img_path)
                    # img_name = img_path.rsplit('/', 1)[1]
                    # save_cropped_img = save_img[new_bbox_y1:new_bbox_y2, new_bbox_x1:new_bbox_x2]
                    # cv2.imwrite(os.path.join('debug', str(key) + '_' + img_name), save_cropped_img)
                    
                    cropped_img = T.Resize(img_size)(cropped_img)
                    cropped_img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(cropped_img)

                    if self.depth:
                        depth_y1 = int(new_bbox_y1 / height * depth_h)
                        depth_y2 = int(new_bbox_y2 / height * depth_h)
                        depth_x1 = int(new_bbox_x1 / width * depth_w)
                        depth_x2 = int(new_bbox_x2 / width * depth_w)
                        cropped_depth = depth[:, depth_y1:depth_y2, depth_x1:depth_x2]
                        cropped_depth = T.Resize(img_size)(cropped_depth)
                        cropped_img = torch.cat([cropped_img, cropped_depth], dim=0)

                    cropped_tracking_imgs.append(cropped_img)

                if self.prompt_mode == 'box':
                    bbox[0] /= self.widths[c_id]
                    bbox[1] /= self.heights[c_id]
                    bbox[2] /= self.widths[c_id]
                    bbox[3] /= self.heights[c_id]
                elif self.prompt_mode == 'point':
                    bbox[0] /= self.widths[c_id]
                    bbox[1] /= self.heights[c_id]
                # bbox = [0 if i < 0 else i for i in bbox]
                # bbox = [1 if i > 1 else i for i in bbox]
                bbox = torch.tensor(bbox, dtype=torch.float32)
                tracking_ls.append(bbox)
            if len(tracking_ls):
                tracking_ls = torch.stack(tracking_ls)
        
        full_img = T.Resize(img_size)(img)
        full_img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(full_img)
        
        info = {
            'img': full_img, 
            'labels': label_ls, 
            'bboxes': bbox_ls, 
            'scores': score_ls, 
            'width': self.widths[c_id], 
            'height': self.heights[c_id], 
        }
        if self.crop_img:
            info['cropped_imgs'] = torch.stack(cropped_imgs) if len(cropped_imgs) else []
        if tracking_dict is not None and self.tracking:
            info['tracking_bboxes'] = tracking_ls
            if self.crop_img:
                info['cropped_tracking_imgs'] = torch.stack(cropped_tracking_imgs) if len(cropped_tracking_imgs) else []
        return info
    
    def load_monopair(self, bbox_ls):
        bbox_num = len(bbox_ls)
        if bbox_num == 0:
            return []
        pts1 = bbox_ls.repeat_interleave(len(bbox_ls), 0)
        pts2 = bbox_ls.repeat(len(bbox_ls), 1)
        if self.prompt_mode == 'box':
            w1 = pts1[:, 2] - pts1[:, 0]
            h1 = pts1[:, 3] - pts1[:, 1]
            ground_x1 = (pts1[:, 0] + pts1[:, 2]) * 0.5
            ground_y1 = pts1[:, 3]

            w2 = pts2[:, 2] - pts2[:, 0]
            h2 = pts2[:, 3] - pts2[:, 1]
            ground_x2 = (pts2[:, 0] + pts2[:, 2]) * 0.5
            ground_y2 = pts2[:, 3]

            new_ground_x = (ground_x1 + ground_x2) * 0.5
            new_ground_y = (ground_y1 + ground_y2) * 0.5
            new_w = (w1 + w2) * 0.5
            new_h = (h1 + h2) * 0.5

            new_x1 = new_ground_x - new_w * 0.5
            new_y1 = new_ground_y - new_h
            new_x2 = new_ground_x + new_w * 0.5
            new_y2 = new_ground_y

            center_prompts = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1).reshape(bbox_num, bbox_num, 4)
        elif self.prompt_mode == 'point':
            ground_x1 = pts1[:, 0]
            ground_y1 = pts1[:, 1]

            ground_x2 = pts2[:, 0]
            ground_y2 = pts2[:, 1]

            new_ground_x = (ground_x1 + ground_x2) * 0.5
            new_ground_y = (ground_y1 + ground_y2) * 0.5

            center_prompts = torch.stack([new_ground_x, new_ground_y], dim=1).reshape(bbox_num, bbox_num, 2)
        return center_prompts

    def get_crop_item(self, item, anchor_view_id, sample_view_id):
        anchor_img_path, anchor_vid_id = self.img_dict[self.view_ls[anchor_view_id]][item]
        anchor_anno = self.read_anno(anchor_img_path, anchor_vid_id, anchor_view_id, item)
        anchor_key = anchor_img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        if self.tracking:
            anchor_tracking = self.read_anno(anchor_img_path, anchor_vid_id, anchor_view_id, item, tracking=True)
            anchor_info = self.load_cropped_bboxes(anchor_img_path, anchor_anno, anchor_view_id, anchor_key, tracking_dict=anchor_tracking)
        else:
            anchor_info = self.load_cropped_bboxes(anchor_img_path, anchor_anno, anchor_view_id, anchor_key)

        pos_img_path, pos_vid_id = self.img_dict[self.view_ls[sample_view_id]][item]
        pos_anno = self.read_anno(pos_img_path, pos_vid_id, sample_view_id, item)
        pos_key = pos_img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        if self.tracking:
            pos_tracking = self.read_anno(pos_img_path, pos_vid_id, sample_view_id, item, tracking=True)
            pos_info = self.load_cropped_bboxes(pos_img_path, pos_anno, sample_view_id, pos_key, tracking_dict=pos_tracking)
        else:
            pos_info = self.load_cropped_bboxes(pos_img_path, pos_anno, sample_view_id, pos_key)

        anchor_info['cam_id'] = anchor_view_id
        anchor_info['image_path'] = anchor_img_path
        pos_info['cam_id'] = sample_view_id
        pos_info['image_path'] = pos_img_path

        if self.mode == 'train' and self.train_mode == 'triplet':
            sample_min, sample_max = self.sampling_range
            interval = sample_min + int(random.random() * (sample_max - sample_min))
            previous_sample = False
            if random.random() > 0.5:
                previous_sample = True
            if previous_sample:
                if item - interval >= 0:
                    new_item = item - interval
                else:
                    new_item = item + interval
            else:
                if item + interval < len(self.img_dict[self.view_ls[sample_view_id]]):
                    new_item = item + interval
                else:
                    new_item = item - interval
            
            neg_img_path, neg_vid_id = self.img_dict[self.view_ls[sample_view_id]][new_item]
            neg_anno = self.read_anno(neg_img_path, neg_vid_id, sample_view_id, new_item)
            neg_key = neg_img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
            if self.pseudo_neg:
                neg_info = self.load_cropped_bboxes(self.img_dict[self.view_ls[sample_view_id]][item][0], pos_anno, sample_view_id, pos_key, bbox_jitter=self.pseudo_neg)
                # neg_info = self.load_cropped_bboxes(neg_img_path, neg_anno, sample_view_id, neg_key, bbox_jitter=self.pseudo_neg)
            else:
                if self.tracking:
                    neg_tracking = self.read_anno(neg_img_path, neg_vid_id, sample_view_id, new_item, tracking=True)
                    neg_info = self.load_cropped_bboxes(neg_img_path, neg_anno, sample_view_id, neg_key, tracking_dict=neg_tracking)
                else:
                    neg_info = self.load_cropped_bboxes(neg_img_path, neg_anno, sample_view_id, neg_key)

            neg_info['cam_id'] = sample_view_id
            neg_info['image_path'] = neg_img_path
            if self.mode == 'train' and self.intra_loss:
                anchor_intra_pos, anchor_intra_neg = self.get_intra_loss_pair(anchor_anno)
                anchor_info['intra'] = (anchor_intra_pos, anchor_intra_neg)
                pos_intra_pos, pos_intra_neg = self.get_intra_loss_pair(pos_anno)
                pos_info['intra'] = (pos_intra_pos, pos_intra_neg)
                neg_intra_pos, neg_intra_neg = self.get_intra_loss_pair(neg_anno)
                neg_info['intra'] = (neg_intra_pos, neg_intra_neg)
            if self.mode == 'train' and self.monopair:
                anchor_center_bboxes = self.load_monopair(anchor_info['bboxes'])
                anchor_info['center_bboxes'] = anchor_center_bboxes
                pos_center_bboxes = self.load_monopair(pos_info['bboxes'])
                pos_info['center_bboxes'] = pos_center_bboxes
                neg_center_bboxes = self.load_monopair(neg_info['bboxes'])
                neg_info['center_bboxes'] = neg_center_bboxes
            return anchor_info, pos_info, neg_info
        else:
            if self.mode == 'train' and self.intra_loss:
                anchor_intra_pos, anchor_intra_neg = self.get_intra_loss_pair(anchor_anno)
                anchor_info['intra'] = (anchor_intra_pos, anchor_intra_neg)
                pos_intra_pos, pos_intra_neg = self.get_intra_loss_pair(pos_anno)
                pos_info['intra'] = (pos_intra_pos, pos_intra_neg)
            if self.mode == 'train' and self.monopair:
                anchor_center_bboxes = self.load_monopair(anchor_info['bboxes'])
                anchor_info['center_bboxes'] = anchor_center_bboxes
                pos_center_bboxes = self.load_monopair(pos_info['bboxes'])
                pos_info['center_bboxes'] = pos_center_bboxes
            return anchor_info, pos_info
    

    def load_image_bboxes(self, frame_img, bbox_dict, c_id, img_size=(224, 224)):
        img = self.load_image(frame_img, img_size)
        label_ls = []
        bbox_ls = []
        for key in bbox_dict:
            bbox = bbox_dict[key]
            if self.prompt_mode == 'box':
                bbox[0] /= self.widths[c_id]
                bbox[1] /= self.heights[c_id]
                bbox[2] /= self.widths[c_id]
                bbox[3] /= self.heights[c_id]
            elif self.prompt_mode == 'point':
                bbox[0] /= self.widths[c_id]
                bbox[1] /= self.heights[c_id]
            # bbox = [0 if i < 0 else i for i in bbox]
            # bbox = [1 if i > 1 else i for i in bbox]
            bbox = torch.tensor(bbox, dtype=torch.float32)
            label_ls.append(key)
            bbox_ls.append(bbox)

        info = {
            'img': img, 
            'labels': label_ls, 
            'bboxes': torch.stack(bbox_ls) if len(bbox_ls) else [], 
        }
        return info

    
    def get_association_item(self, item):
        infos = []
        for cam_id, view in enumerate(self.view_ls):
            frame_img, vid_id = self.img_dict[view][item]
            anno = self.read_anno(frame_img, vid_id, cam_id, item)
            depth_key = frame_img.rsplit('/', 1)[1].rsplit('.', 1)[0]
            if self.crop_box:
                info = self.load_cropped_bboxes(frame_img, anno, cam_id, depth_key)
            else:
                info = self.load_image_bboxes(frame_img, anno, cam_id)

            info['cam_id'] = cam_id
            info['image_path'] = frame_img
            info['image_key'] = depth_key
            infos.append(info)

        return infos