from glob import glob
import argparse
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from engine.core import YAMLConfig

import cv2
import lmdb
import pickle


def get_inference_arg_parser():
    parser = argparse.ArgumentParser('Inference.')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument(
        "--root_dir", default='/media/camma-monitor/Storage_postprocessing3/009_SV_JPG', type=str, help="path to images"
    )
    parser.add_argument(
        "--save_dir", default="/media/camma-monitor/Storage_postprocessing2/pose_detection", type=str, help="path to output"
    )
    parser.add_argument(
        "--save_name", default='009_SV_JPG_iter2_deim.lmdb', type=str, help="path to images"
    )
    # parser.add_argument('--pathnames', nargs='*')
    return parser


def make_coco_transforms(image_set):
    
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

class GlobDataset(Dataset):
    def __init__(self, img_dir, img_dicts) -> None:
        self.transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

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
        transformed_img = self.transforms(img)
        return transformed_img, torch.tensor([w, h]), f_path, self.cams[index], self.f_ids[index]

    def __len__(self):
        return len(self.files)


@torch.no_grad()
def main():
    args, _ = get_inference_arg_parser().parse_known_args()
    print(args)
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    
    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    device = 'cuda'
    model = Model().to(device)
    model.eval()
    # save_format = '{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{s:.2f}'
    
    thresh = 0.1
    vis = False
    
    det_dir = os.path.join(args.save_dir, 'detections')
    os.makedirs(det_dir, exist_ok=True)
    det_path = os.path.join(det_dir, args.save_name)
    
    # maximum 10GB
    map_size = int(10 * 1024 * 1024 * 1024)
    flush_interval = 1000
    
    env = lmdb.open(det_path, map_size=map_size, subdir=False, meminit=False, max_dbs=0)
    txn = env.begin(write=True)

    img_dicts_pkl = os.path.join(args.root_dir, 'img_dicts.pkl')
    with open(img_dicts_pkl, mode='rb') as f:
        img_dicts = pickle.load(f)
    cam_num = len(img_dicts)
    img_dir = os.path.join(args.root_dir, 'colorimage')
    dataset = GlobDataset(img_dir, img_dicts)
    dataloader = DataLoader(dataset, 1, num_workers=2)
    count = 0

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
        output = model(samples, orig_sizes)
        labels_batch, boxes_batch, scores_batch = output

        for labels, boxes, scores, f_path, cam_id, frame_id in zip(labels_batch, boxes_batch, scores_batch, f_paths, cam_ids, f_ids):
            frame_id = int(frame_id)
            cam_id = int(cam_id)
            
            img_key = str(frame_id) + '_' + str(cam_id)
            key_boxes = f"{img_key}/boxes".encode('utf-8')
            key_scores = f"{img_key}/scores".encode('utf-8')
            
            keep = scores > thresh
            
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