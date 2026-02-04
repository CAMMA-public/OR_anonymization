from glob import glob
import argparse
from PIL import Image
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from main import get_args_parser as get_main_args_parser
from detr import build_model
from lib.datasets.coco import make_coco_transforms
from util.misc import nested_tensor_from_tensor_list

import os
import cv2
import json
import lmdb
import numpy as np

def get_inference_arg_parser():
    parser = argparse.ArgumentParser('Inference.')
    parser.add_argument(
        "--root_dir", default='../data/4D-OR/export_holistic_take1_processed', type=str, help="path to images"
    )
    parser.add_argument(
        "--save_dir", default="../data/4D-OR", type=str, help="path to output"
    )
    parser.add_argument(
        "--save_name", default='export_holistic_take1_processed_iter1.lmdb', type=str, help="save name"
    )
    parser.add_argument(
        "--weight", default='weights/4dor_iter1.pth', type=str, help="model weights"
    )
    # parser.add_argument('--pathnames', nargs='*')
    return parser

class GlobDataset(Dataset):
    def __init__(self, img_dir, img_dicts) -> None:
        self.transforms = make_coco_transforms('val')
        self.files = []
        self.cams = []
        self.f_ids = []
        for c_id, img_dict in enumerate(img_dicts):
            for f_id, img_name in img_dict.items():
                img_path = os.path.join(img_dir, img_name)
                self.files.append(img_path)
                self.cams.append(c_id)
                self.f_ids.append(f_id)

    def __getitem__(self, index):
        f_path = self.files[index]
        img = Image.open(f_path)
        w, h = img.size
        transformed_img = self.transforms(img, None)[0]
        return transformed_img, torch.tensor([h, w]), f_path, self.cams[index], self.f_ids[index]

    def __len__(self):
        return len(self.files)


@torch.no_grad()
def main():
    args, _ = get_inference_arg_parser().parse_known_args()
    print(args)
    main_args = get_main_args_parser().parse_args(_)
    model, _, postprocessors = build_model(main_args)
    model.load_state_dict(torch.load(args.weight, 'cpu')["model"])
    model.cuda()
    model.eval()
    save_format = '{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{s:.2f}'
    
    det_dir = os.path.join(args.save_dir, 'detections')
    os.makedirs(det_dir, exist_ok=True)

    thresh = 0.1
    vis = False
    
    det_path = os.path.join(det_dir, args.save_name)
    # maximum 10GB
    map_size = int(10 * 1024 * 1024 * 1024)
    flush_interval = 1000
    
    env = lmdb.open(det_path, map_size=map_size, subdir=False, meminit=False, max_dbs=0)
    txn = env.begin(write=True)
    
    data = {}
    img_dicts_pkl = os.path.join(args.root_dir, 'img_dicts.pkl')
    with open(img_dicts_pkl, mode='rb') as f:
        img_dicts = pickle.load(f)
    cam_num = len(img_dicts)
    img_dir = os.path.join(args.root_dir, 'colorimage')
    dataset = GlobDataset(img_dir, img_dicts)
    dataloader = DataLoader(dataset, 1, num_workers=2)
    count = 0
    if vis:
        save_dir = 'vis'
        os.makedirs(save_dir, exist_ok=True)
    for samples, orig_sizes, f_paths, cam_ids, f_ids in tqdm(dataloader):
        run_flag = False
        for cam_id, frame_id in zip(cam_ids, f_ids):
            img_key = str(frame_id) + '_' + str(cam_id)
            key_boxes = f"{img_key}/boxes".encode('utf-8')
            key_scores = f"{img_key}/scores".encode('utf-8')
            if txn.get(key_boxes) is None or txn.get(key_scores) is None:
                run_flag = True
                break
        if not run_flag:
            continue
        
        samples, orig_sizes = samples.cuda(), orig_sizes.cuda()
        samples = nested_tensor_from_tensor_list(samples)
        outputs = model(samples)
        results = postprocessors['bbox'](outputs, orig_sizes)

        for result, f_path, cam_id, frame_id in zip(results, f_paths, cam_ids, f_ids):
            frame_id = int(frame_id)
            cam_id = int(cam_id)
            
            img_key = str(frame_id) + '_' + str(cam_id)
            key_boxes = f"{img_key}/boxes".encode('utf-8')
            key_scores = f"{img_key}/scores".encode('utf-8')
            
            scores = result['scores']
            boxes = result['boxes']
            keep = scores > thresh
            
            save_boxes = np.array(boxes[keep].cpu().numpy(), dtype=float)
            save_boxes_score = np.array(scores[keep].cpu().numpy(), dtype=float)
            txn.put(key_boxes, pickle.dumps(save_boxes))
            txn.put(key_scores, pickle.dumps(save_boxes_score))

            count += 1
            if count % flush_interval == 0:
                txn.commit()
                txn = env.begin(write=True)

            img = cv2.imread(f_path)
            img_name = f_path.rsplit('/', 1)[1]
            
            frame_id = int(frame_id)
            if frame_id not in data:
                data[frame_id] = []

            for s, b in zip(scores[keep].tolist(), boxes[keep].tolist()):
                x1, y1, x2, y2 = b
                if vis:
                    cv2.putText(img, str(round(s, 2)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    # print(save_format.format(x1=x1, y1=y1, w=x2-x1, h=y2-y1, s=s))
                data[frame_id].append([int(x1), int(y1), int(x2 - x1), int(y2 - y1), -1, int(cam_id), round(s, 2)])
            if vis:
                cv2.imwrite(os.path.join(save_dir, img_name), img)
    
    txn.commit()
    env.close()


if __name__ == '__main__':
    main()