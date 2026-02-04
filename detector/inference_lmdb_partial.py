from glob import glob
import argparse
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from main import get_args_parser as get_main_args_parser
from detr import build_model
from lib.datasets.coco import make_coco_transforms
from util.misc import nested_tensor_from_tensor_list
import numpy as np

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2
import lmdb
import pickle
import json


def get_inference_arg_parser():
    parser = argparse.ArgumentParser('Inference.')
    # /media/camma-monitor/Storage_postprocessing2/005_CG_JPG/005_CG
    # /media/camma-monitor/Storage_postprocessing2/006_RS_JPG/006_RS
    # /media/camma-monitor/Storage_postprocessing3/007_CL_JPG
    # /media/camma-monitor/Storage_postprocessing3/008_OT_JPG
    # /media/camma-monitor/Storage_postprocessing3/009_SV_JPG
    parser.add_argument(
        "--root_dir", default='/media/camma-monitor/Storage_postprocessing2/005_CG_JPG/005_CG', type=str, help="path to images"
    )
    parser.add_argument(
        "--save_dir", default="/media/camma-monitor/Storage_postprocessing2/pose_detection", type=str, help="path to output"
    )
    parser.add_argument(
        "--save_name", default='005_CG_JPG_temp.lmdb', type=str, help="path to images"
    )
    parser.add_argument(
        "--anno_path", default='/media/camma-monitor/Storage_postprocessing2/pose_detection_utils/code/anonymization/evaluation/annotations/005_CG.json', type=str, help="path to images"
    )
    parser.add_argument(
        "--no_sub_dir", action='store_true', help="path to images"
    )
    # parser.add_argument('--pathnames', nargs='*')
    return parser

class GlobDataset(Dataset):
    def __init__(self, img_dir, sub_dir, img_keys=None) -> None:
        self.transforms = make_coco_transforms('val')
        self.files = []
        self.img_dir = img_dir
        if img_keys is not None:
            selected_img_names = [item.rsplit('/', 1)[1] for item in img_keys]
        img_names = os.listdir(img_dir)
        img_names.sort()
        for img_name in img_names:
            if sub_dir is None:
                if img_name[-3:] != 'jpg':
                    continue
            else:
                if img_name[-3:] != 'jpg' or img_name[0] != 'c':
                    continue
            if img_keys is not None:
                if img_name not in selected_img_names:
                    continue

            self.files.append(img_name)

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)
        w, h = img.size
        transformed_img = self.transforms(img, None)[0]
        return transformed_img, torch.tensor([h, w]), img_name

    def __len__(self):
        return len(self.files)


@torch.no_grad()
def main():
    args, _ = get_inference_arg_parser().parse_known_args()
    print(args)
    main_args = get_main_args_parser().parse_args(_)
    model, _, postprocessors = build_model(main_args)
    # model.load_state_dict(torch.load('weights/iter-d-detr-swinl_iter1.pth', 'cpu')["model"])
    model.load_state_dict(torch.load('output/model_dump/aps_swinl_monitor10k_only_score_06/checkpoint-14.pth', 'cpu')["model"])
    model.cuda()
    model.eval()
    save_format = '{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{s:.2f}'
    
    data = {}
    
    thresh = 0.1
    vis = False
    if not args.no_sub_dir:
        sub_dirs = []
        for dir_name in os.listdir(args.root_dir):
            if dir_name == '547758':
                continue
            dir_path = os.path.join(args.root_dir, dir_name)
            if os.path.isdir(dir_path):
                sub_dirs.append(dir_name)
        sub_dirs.sort()
    
    with open(args.anno_path, 'r') as fp:
        gt = json.load(fp)
    img_keys = list(gt.keys())
    
    det_dir = os.path.join(args.save_dir, 'detections')
    os.makedirs(det_dir, exist_ok=True)
    det_path = os.path.join(det_dir, args.save_name)
    
    # maximum 10GB
    map_size = int(10 * 1024 * 1024 * 1024)
    flush_interval = 1000
    
    env = lmdb.open(det_path, map_size=map_size, subdir=False, meminit=False, max_dbs=0)
    txn = env.begin(write=True)
    
    if args.no_sub_dir:
        img_dir = args.root_dir
        dataset = GlobDataset(img_dir, None, img_keys)
        dataloader = DataLoader(dataset, 1, num_workers=2)
        count = 0
        for samples, orig_sizes, img_names in tqdm(dataloader):
            run_flag = False
            for img_name in img_names:
                img_key = img_name
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

            for result, img_name in zip(results, img_names):
                img_key = img_name
                key_boxes = f"{img_key}/boxes".encode('utf-8')
                key_scores = f"{img_key}/scores".encode('utf-8')
                if txn.get(key_boxes) is not None and txn.get(key_scores) is not None:
                    continue
                
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
    else:
        for cam_id, sub_dir in enumerate(sub_dirs):
            if vis:
                save_dir = os.path.join('vis', sub_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            img_dir = os.path.join(args.root_dir, sub_dir)
            dataset = GlobDataset(img_dir, sub_dir, img_keys)
            dataloader = DataLoader(dataset, 1, num_workers=2)
            count = 0
            for samples, orig_sizes, img_names in tqdm(dataloader):
                run_flag = False
                for img_name in img_names:
                    img_key = os.path.join(sub_dir, img_name)
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

                for result, img_name in zip(results, img_names):
                    img_key = os.path.join(sub_dir, img_name)
                    key_boxes = f"{img_key}/boxes".encode('utf-8')
                    key_scores = f"{img_key}/scores".encode('utf-8')
                    
                    scores = result['scores']
                    boxes = result['boxes']
                    keep = scores > thresh

                    if vis:
                        img_path = os.path.join(args.root_dir, sub_dir, img_name)
                        img = cv2.imread(img_path)
                        for s, b in zip(scores[keep].tolist(), boxes[keep].tolist()):
                            x1, y1, x2, y2 = b
                            cv2.putText(img, str(round(s, 2)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            # print(save_format.format(x1=x1, y1=y1, w=x2-x1, h=y2-y1, s=s))
                        cv2.imwrite(os.path.join(save_dir, img_name), img)
                    
                    save_boxes = np.array(boxes[keep].cpu().numpy(), dtype=float)
                    save_boxes_score = np.array(scores[keep].cpu().numpy(), dtype=float)
                    txn.put(key_boxes, pickle.dumps(save_boxes))
                    txn.put(key_scores, pickle.dumps(save_boxes_score))
    
                    count += 1
                    if count % flush_interval == 0:
                        txn.commit()
                        txn = env.begin(write=True)
    txn.commit()
    env.close()


if __name__ == '__main__':
    main()
