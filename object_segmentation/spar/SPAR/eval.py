import torch
import os

from mmengine.registry import init_default_scope
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.registry import DATASETS
import cv2

import numpy as np
init_default_scope('mmseg')

import open_clip
import datasets.custom_datasets as custom_datasets
import torchmetrics
from tqdm import tqdm
import argparse

from models import SigLIP2Wrapper, OpenCLIPWrapper, DINOv3txtWrapper
from segmentors import VitForSegmentation

MODEL_WRAPPERS = {
    "OpenCLIPWrapper": OpenCLIPWrapper,
    "SigLIP2Wrapper": SigLIP2Wrapper,
    "DINOv3txtWrapper": DINOv3txtWrapper,
}
SEGMENTORS = {
    "VitForSegmentation": VitForSegmentation,
}

parser = argparse.ArgumentParser(description="Run model segmentation")
parser.add_argument("-gpu", type=int, default=0, help="cuda:")
parser.add_argument("-model", type=str, required=True, help="Model name, lowercase")
parser.add_argument("-dataset", type=str, required=True, help="Dataset name")
parser.add_argument("-checkpoint", type=str, required=False, help="Checkpoint path")
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

cfg_path = f"./configs/{args.model}/{args.dataset}.py"
cfg = Config.fromfile(cfg_path)


dataset = DATASETS.build(cfg.test_dataloader.dataset)
cfg.test_dataloader.dataset = dataset
runner = Runner.build_dataloader(cfg.test_dataloader)


model_wrapper = MODEL_WRAPPERS[cfg.model.wrapper](device=device, **cfg.model).to(device)
segmentor = SEGMENTORS[cfg.model.type](net=model_wrapper, device=device, **cfg.model)


if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if args.checkpoint.endswith("ckpt"):
        checkpoint = checkpoint["state_dict"]
        checkpoint = {k[len("segmentor.net."):]: v for k, v in checkpoint.items() if k.startswith("segmentor.net.") }
    segmentor.net.load_state_dict(checkpoint, strict = True)
os.makedirs("vis_results", exist_ok=True)
with torch.no_grad(): 
    iou_metric = torchmetrics.JaccardIndex(task='multiclass', ignore_index = 255, num_classes=len(dataset._metainfo["classes"]), average= 'none').to(device)

    for idx,img in enumerate(tqdm(runner)):
            pretprocessed_data = segmentor.data_preprocessor(img, training=False)
            device = next(segmentor.parameters()).device
            pretprocessed_data['inputs'] = pretprocessed_data['inputs'].to(device)

            res = segmentor.predict(pretprocessed_data)
            preds =  res[0].pred_sem_seg.data.to(device)
            
            target = res[0].gt_sem_seg.data.to(device)
             
            iou_metric.update(preds, target)
            # Save visual comparison (Ground Truth vs Prediction)
            pred_np = (preds.squeeze().cpu().numpy() * 255).astype(np.uint8)
            gt_np = (target.squeeze().cpu().numpy() * 255).astype(np.uint8)
            orig_img = cv2.imread(res[0].img_path)

            # Convert 1-channel masks to 3-channel BGR (white = animal, black = background)
            gt_vis = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)
            pred_vis = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)

            combined = np.hstack([orig_img, gt_vis, pred_vis])
            cv2.imwrite(f"vis_results/sample_{idx:02d}.png", combined)

    per_class_iou = iou_metric.compute() * 100
    print(f"Background IoU: {per_class_iou[0].item():.2f}%")
    print(f"Animal IoU:     {per_class_iou[1].item():.2f}%")
    print(f"Mean IoU:       {per_class_iou.mean().item():.2f}%")

    # mean_iou = iou_metric.compute()
    # print(f"{mean_iou.item() * 100:.2f}")



