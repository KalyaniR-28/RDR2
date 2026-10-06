import json
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

with open("all_captures.json", "r") as f:
    gt_data = json.load(f)

with open("./predfine/fine_predictions.json", "r") as f:
    pred_data = json.load(f)

with open("gt_Fine_labelIds_mapping.json", "r") as f:
    id_to_name = {v: k for k, v in json.load(f).items()}
    
metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)

def extract_capture_id(pred_key):
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

    # torchmetrics expect the gt and predicted in the form of tensor
    gt_boxes = []
    gt_labels = []
    for ent in gt_entities:
        box = ent.get("BoundingBoxCalculated")
        class_id = ent.get("FineClassNameId")
        
        if box and class_id is not None:
            gt_boxes.append(box)
            gt_labels.append(int(class_id))

    if len(gt_boxes) > 0:
        target_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        target_labels = torch.tensor(gt_labels, dtype=torch.int64)
    else:
        target_boxes = torch.zeros((0, 4), dtype=torch.float32)
        target_labels = torch.zeros((0,), dtype=torch.int64)

    target = [{"boxes": target_boxes, "labels": target_labels}]

    #creating the prediction tensor for images
    p_boxes = []
    p_scores = []
    p_labels = []

    for item in pred_items:
        box = item.get("box")
        conf = item.get("confidence", 0.0)
        pred_id = item.get("id")

        if box and pred_id is not None:
            p_boxes.append(box)
            p_scores.append(conf)
            p_labels.append(int(pred_id))

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

results = metric.compute()

print("\n--- Mean Average Precision (Overall) ---")
print(f"mAP [IoU=0.50:0.95]: {results['map'].item():.4f}")
print(f"mAP [IoU=0.50]:      {results['map_50'].item():.4f}")
print(f"mAP [IoU=0.75]:      {results['map_75'].item():.4f}")
print(f"mAP (Small):         {results['map_small'].item():.4f}")
print(f"mAP (Medium):        {results['map_medium'].item():.4f}")
print(f"mAP (Large):         {results['map_large'].item():.4f}")
print(f"Recall 100:          {results['mar_100'].item():.4f}")

# per class AP calculated
if "map_per_class" in results and results["map_per_class"].numel() > 0:
    print("\nPer-Class AP [IoU=0.50:0.95]")
    classes = results["classes"].tolist()
    ap_per_class = results["map_per_class"].tolist()
    for cls_id, cls_ap in zip(results["classes"].tolist(), results["map_per_class"].tolist()):
        class_name = id_to_name.get(cls_id, f"ID_{cls_id}")
        print(f"{class_name:<15}: {cls_ap:.4f}")
        
