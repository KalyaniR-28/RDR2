
import json
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

GT_PATH = "/media/rdr2/RDR2_dataset_processed_test/all_captures.json"
PRED_PATH = "/home/sparackal/GroundingDINO/animals_predictions.json"

print("Loading ground truth...")
with open(GT_PATH, "r") as f:
    gt_data = json.load(f)

print("Loading predictions...")
with open(PRED_PATH, "r") as f:
    pred_data = json.load(f)

metric = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    class_metrics=True
)

EVAL_CLASS_ID = 0


def extract_capture_id(pred_key):
    parts = pred_key.split("_")
    return parts[1] if len(parts) > 1 else pred_key


matched_images = 0
total_images = len(pred_data)

print(f"Evaluating {total_images} frames...")

for idx, (pred_key, pred_items) in enumerate(pred_data.items()):
    capture_id = extract_capture_id(pred_key)

    if capture_id not in gt_data:
        continue

    matched_images += 1
    gt_entities = gt_data[capture_id].get("Entities", [])

    gt_boxes = []
    gt_labels = []

    for ent in gt_entities:
        box = ent.get("BoundingBoxCalculated")
        if box:
            gt_boxes.append([float(coord) for coord in box])
            gt_labels.append(EVAL_CLASS_ID)

    target_boxes = (
        torch.tensor(gt_boxes, dtype=torch.float32)
        if gt_boxes
        else torch.zeros((0, 4), dtype=torch.float32)
    )

    target_labels = (
        torch.tensor(gt_labels, dtype=torch.int64)
        if gt_labels
        else torch.zeros((0,), dtype=torch.int64)
    )

    target = [{
        "boxes": target_boxes,
        "labels": target_labels
    }]

    pred_boxes_list = []
    pred_scores_list = []
    pred_labels_list = []

    for item in pred_items:
        box = item.get("bbox")
        if box is None:
            box = item.get("box")

        if box:
            pred_boxes_list.append([float(coord) for coord in box])
            pred_scores_list.append(float(item.get("confidence", 0.0)))
            pred_labels_list.append(EVAL_CLASS_ID)

    pred_boxes = (
        torch.tensor(pred_boxes_list, dtype=torch.float32)
        if pred_boxes_list
        else torch.zeros((0, 4), dtype=torch.float32)
    )

    pred_scores = (
        torch.tensor(pred_scores_list, dtype=torch.float32)
        if pred_scores_list
        else torch.zeros((0,), dtype=torch.float32)
    )

    pred_labels = (
        torch.tensor(pred_labels_list, dtype=torch.int64)
        if pred_labels_list
        else torch.zeros((0,), dtype=torch.int64)
    )

    preds = [{
        "boxes": pred_boxes,
        "scores": pred_scores,
        "labels": pred_labels
    }]

    metric.update(preds, target)

    if (idx + 1) % 5000 == 0:
        print(f"Processed {idx + 1}/{total_images} frames")

results = metric.compute()

print(f"\nMatched images: {matched_images}/{total_images}")
print(f"mAP (IoU 0.50:0.95): {results['map'].item():.4f}")
print(f"mAP@50:             {results['map_50'].item():.4f}")
print(f"mAP@75:             {results['map_75'].item():.4f}")
print(f"mAP (small):        {results['map_small'].item():.4f}")
print(f"mAP (medium):       {results['map_medium'].item():.4f}")
print(f"mAP (large):        {results['map_large'].item():.4f}")
print(f"Recall@100:         {results['mar_100'].item():.4f}")
