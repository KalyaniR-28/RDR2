import os
import json
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

PRED_PATH = "/home/sparackal/owl/Owlvit_coarse/owlvit_predictions_coarse.json"
GT_PATH = "/media/rdr2/RDR2_dataset_processed_test/all_captures.json"
IMAGE_DIR = "/media/rdr2/RDR2_dataset_processed_test/PNG"

with open(PRED_PATH) as f:
    pred_data = json.load(f)

with open(GT_PATH) as f:
    gt_data = json.load(f)

metric = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    class_metrics=True
)

pred_by_image = {}

for p in pred_data:
    pred_by_image.setdefault(p["file_name"], []).append(p)

image_files = sorted(
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".png")
)

print(f"Starting evaluation across {len(image_files)} frames...", flush=True)

matched = 0

for i, filename in enumerate(image_files):

    capture_id = filename.split("_")[1]

    if capture_id not in gt_data:
        continue

    matched += 1
    entities = gt_data[capture_id].get("Entities", [])
    gt_boxes = []
    gt_labels = []

    for e in entities:
        box = e.get("BoundingBoxCalculated")
        class_id = e.get("CoarseClassNameId")

        if box is not None and class_id is not None:
            gt_boxes.append([float(x) for x in box])
            gt_labels.append(int(class_id))

    # OWL-ViT predictions
    predictions = pred_by_image.get(filename, [])

    pred_boxes = []
    pred_scores = []
    pred_labels = []

    for p in predictions:
        box = p.get("bbox")
        score = p.get("score")
        class_id = p.get("class_id")

        if box is not None and class_id is not None:
            pred_boxes.append([float(x) for x in box])
            pred_scores.append(float(score))
            pred_labels.append(int(class_id))

    target = [{
        "boxes": torch.tensor(
            gt_boxes, dtype=torch.float32
        ).reshape(-1, 4),
        "labels": torch.tensor(
            gt_labels, dtype=torch.int64
        )
    }]

    preds = [{
        "boxes": torch.tensor(
            pred_boxes, dtype=torch.float32
        ).reshape(-1, 4),
        "scores": torch.tensor(
            pred_scores, dtype=torch.float32
        ),
        "labels": torch.tensor(
            pred_labels, dtype=torch.int64
        )
    }]

    metric.update(preds, target)

    if (i + 1) % 5000 == 0:
        print(
            f"Processed {i + 1}/{len(image_files)} frames...",
            flush=True
        )

print(f"\nFinished pairing {matched}/{len(image_files)} images.")

res = metric.compute()

print("OWL-ViT COARSE RESULTS")
print(
    f"mAP [0.50:0.95]: {res['map'].item():.4f} "
    f"({res['map'].item()*100:.2f}%)"
)
print(
    f"mAP@50         : {res['map_50'].item():.4f} "
    f"({res['map_50'].item()*100:.2f}%)"
)
print(
    f"mAP@75         : {res['map_75'].item():.4f} "
    f"({res['map_75'].item()*100:.2f}%)"
)
print(f"mAP (Small)    : {res['map_small'].item():.4f}")
print(f"mAP (Medium)   : {res['map_medium'].item():.4f}")
print(f"mAP (Large)    : {res['map_large'].item():.4f}")
print(f"Recall@100     : {res['mar_100'].item():.4f}")
