import json
from pathlib import Path
from collections import defaultdict

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

PRED_PATH = "/home/sparackal/owl/Owlvit_fine/owlvit_predictions_fine.json"
GT_PATH = "/media/rdr2/RDR2_dataset_processed_test/all_captures.json"
IMAGE_DIR = Path("/media/rdr2/RDR2_dataset_processed_test/PNG")


with open(PRED_PATH, "r") as f:
    pred_data = json.load(f)

with open(GT_PATH, "r") as f:
    gt_data = json.load(f)


pred_by_image = defaultdict(list)

for pred in pred_data:
    pred_by_image[pred["file_name"]].append(pred)


id_to_class = {}

for capture in gt_data.values():
    for entity in capture.get("Entities", []):
        class_id = entity.get("FineClassNameId")
        class_name = entity.get("FineClassName")

        if class_id is not None and class_name is not None:
            id_to_class[int(class_id)] = class_name


metric = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    class_metrics=True,
)

image_files = sorted(IMAGE_DIR.glob("*.png"))
zero_prediction_images = 0

for image_path in image_files:
    filename = image_path.name
    capture_id = filename.split("_")[1]

    gt_entities = gt_data.get(capture_id, {}).get("Entities", [])
    predictions = pred_by_image.get(filename, [])

    if not predictions:
        zero_prediction_images += 1

    gt_boxes, gt_labels = [], []

    for entity in gt_entities:
        bbox = entity.get("BoundingBoxCalculated")
        class_id = entity.get("FineClassNameId")

        if bbox is not None and class_id is not None:
            gt_boxes.append(bbox)
            gt_labels.append(int(class_id))

    pred_boxes, pred_labels, pred_scores = [], [], []

    for pred in predictions:
        bbox = pred.get("bbox")
        class_id = pred.get("class_id")
        score = pred.get("score")

        if bbox is not None and class_id is not None and score is not None:
            pred_boxes.append(bbox)
            pred_labels.append(int(class_id))
            pred_scores.append(float(score))

    pred_batch = [
        {
            "boxes": torch.tensor(
                pred_boxes, dtype=torch.float32
            ).reshape(-1, 4),
            "labels": torch.tensor(
                pred_labels, dtype=torch.int64
            ),
            "scores": torch.tensor(
                pred_scores, dtype=torch.float32
            ),
        }
    ]

    gt_batch = [
        {
            "boxes": torch.tensor(
                gt_boxes, dtype=torch.float32
            ).reshape(-1, 4),
            "labels": torch.tensor(
                gt_labels, dtype=torch.int64
            ),
        }
    ]

    metric.update(pred_batch, gt_batch)


results = metric.compute()

print("\nOWL-ViT FINE CLASS RESULTS")
print(f"Images evaluated: {len(image_files)}")
print(f"Images with zero predictions: {zero_prediction_images}")

print(f"mAP [IoU=0.50:0.95]: {results['map'].item() * 100:.2f}%")
print(f"mAP [IoU=0.50]     : {results['map_50'].item() * 100:.2f}%")
print(f"mAP [IoU=0.75]     : {results['map_75'].item() * 100:.2f}%")
print(f"mAP (Small objects): {results['map_small'].item() * 100:.2f}%")
print(f"mAP (Medium objects): {results['map_medium'].item() * 100:.2f}%")
print(f"mAP (Large objects): {results['map_large'].item() * 100:.2f}%")
print(f"Recall (max 100 det): {results['mar_100'].item() * 100:.2f}%")

print("\nPer-class AP [IoU=0.50:0.95]")

for class_id, ap in zip(
    results["classes"].tolist(),
    results["map_per_class"].tolist(),
):
    class_name = id_to_class.get(int(class_id), f"Class {class_id}")
    print(f"{class_name}: {ap * 100:.2f}%")