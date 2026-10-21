import json
import re
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

GT_FOLDER = Path("/media/rdr2/RDR2_dataset_processed_test/InstanceSegmentationMasks")
PRED_FOLDER = Path("./gsam_results_instance_animals/masks")
PRED_JSON = Path("./gsam_results_instance_animals/instance_predictions.json")
OUTPUT_JSON = Path("./gsam_results_instance_animals/instance_ap_results.json")

# creating a dictionary to map the gt files by the id
gt_by_capture = defaultdict(list)
for filepath in GT_FOLDER.glob("*.png"):
    match = re.search(r"mSemInstance_(\d+)_", filepath.name)
    if match:
        gt_by_capture[match.group(1)].append(filepath)
print(f"Total unique GT captures indexed: {len(gt_by_capture)}")

with open(PRED_JSON) as f:
    predictions = json.load(f)

metric = MeanAveragePrecision(iou_type="segm")
#Iterate directly over predictions
for imageid, preds in tqdm(predictions.items(), desc="Evaluating Masks", unit="img"):

    match = re.search(r"o_(\d+)", imageid)
    if not match:
        continue
    capture_id = match.group(1)

    # # Loading ground truth masks
    # gt_list = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0 for p in gt_by_capture.get(capture_id, [])]
    # target_masks = torch.from_numpy(np.stack(gt_list)) if gt_list else torch.zeros((0, 1, 1), dtype=torch.bool)
    
    # # Load predicted masks
    # pred_list = [cv2.imread(str(PRED_FOLDER / p["mask_file"]), cv2.IMREAD_GRAYSCALE) > 0 for p in preds]
    # scores = [p["confidence"] for p in preds]
    # pred_masks = torch.from_numpy(np.stack(pred_list)) if pred_list else torch.zeros((0, 1, 1), dtype=torch.bool)
    
    #load the grouth truth and the prediction masks
    gt_list = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0 for p in gt_by_capture.get(capture_id, [])]
    pred_list = [cv2.imread(str(PRED_FOLDER / p["mask_file"]), cv2.IMREAD_GRAYSCALE) > 0 for p in preds]
    scores = [p["confidence"] for p in preds]

    # Resolve true image dimensions for empty tensors
    if gt_list:
        h, w = gt_list[0].shape
    elif pred_list:
        h, w = pred_list[0].shape
    else:
        continue

    target_masks = torch.from_numpy(np.stack(gt_list)) if gt_list else torch.zeros((0, h, w), dtype=torch.bool)
    pred_masks = torch.from_numpy(np.stack(pred_list)) if pred_list else torch.zeros((0, h, w), dtype=torch.bool)
    
    # Update metric
    metric.update(
        preds=[{
            "masks": pred_masks, 
            "scores": torch.tensor(scores, dtype=torch.float32), 
            "labels": torch.zeros(len(scores), dtype=torch.int64)
            }],
        target=[{
            "masks": target_masks, 
            "labels": torch.zeros(len(gt_list), dtype=torch.int64)
            }]
    )

#Compute and save
result = metric.compute()
results = {}
for k, v in result.items():
    if "map" in k or "mar" in k:
        results[k] = float(v)

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)

print(json.dumps(results, indent=2))