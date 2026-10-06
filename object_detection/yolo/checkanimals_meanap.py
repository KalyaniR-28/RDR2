import json
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

with open("all_captures.json", "r") as f:
    gt_data = json.load(f)

with open("./pred_animals15/animals_predictions.json", "r") as f:
    pred_data = json.load(f)

metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
# Since we are just checking with one prompt "animals", it is mapped to just one ID 0
EVAL_CLASS_ID = 0

def extract_capture_id(pred_key) :
    # "o_1000003_12" -> "1000003"
    parts = pred_key.split("_")
    return parts[1] if len(parts) > 1 else pred_key

for pred_key, pred_items in pred_data.items():
    capture_id = extract_capture_id(pred_key)
    
    if capture_id not in gt_data:
        print(f"Warning: Capture ID {capture_id} not found in GT JSON. Skipping.")
        continue

    gt_entry = gt_data[capture_id]
    gt_entities = gt_entry.get("Entities", [])

    # the torchmetrics expect a ground truth tensor and a prediction tensor
    gt_boxes = []
    gt_labels = []
    for ent in gt_entities:
        box = ent.get("BoundingBoxCalculated")
        if box:
            gt_boxes.append(box)
            gt_labels.append(EVAL_CLASS_ID)

    if len(gt_boxes) > 0:
        target_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        target_labels = torch.tensor(gt_labels, dtype=torch.int64)
    else:
        target_boxes = torch.zeros((0, 4), dtype=torch.float32)
        target_labels = torch.zeros((0,), dtype=torch.int64)

    target = [{"boxes": target_boxes, "labels": target_labels}]

    # creating the prediction tensor for the particular image
    p_boxes = []
    p_scores = []
    p_labels = []

    for item in pred_items:
        box = item.get("box")
        conf = item.get("confidence", 0.0)
        if box:
            p_boxes.append(box)
            p_scores.append(conf)
            p_labels.append(EVAL_CLASS_ID)

    if len(p_boxes) > 0:
        pred_boxes = torch.tensor(p_boxes, dtype=torch.float32)
        pred_scores = torch.tensor(p_scores, dtype=torch.float32)
        pred_labels = torch.tensor(p_labels, dtype=torch.int64)
    else:
        pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
        pred_scores = torch.zeros((0,), dtype=torch.float32)
        pred_labels = torch.zeros((0,), dtype=torch.int64)

    preds = [{
        "boxes": pred_boxes,
        "scores": pred_scores,
        "labels": pred_labels
    }]

  
    metric.update(preds, target)

#Computing AP
results = metric.compute()

print("\nAP")
print(f"mAP [IoU=0.50:0.95]: {results['map'].item():.4f}")
print(f"mAP [IoU=0.50]:      {results['map_50'].item():.4f}")
print(f"mAP [IoU=0.75]:      {results['map_75'].item():.4f}")
print(f"mAP (Small objects):  {results['map_small'].item():.4f}")
print(f"mAP (Medium objects): {results['map_medium'].item():.4f}")
print(f"mAP (Large objects):  {results['map_large'].item():.4f}")
print(f"Recall 100:          {results['mar_100'].item():.4f}")
