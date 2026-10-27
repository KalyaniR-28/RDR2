import os
import json
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

GT_PATH = "/media/rdr2/RDR2_dataset_processed_test/all_captures.json"
PRED_PATH = "/home/sparackal/owl/OWLViT_animals/owlvit_predictions_animal.json"
IMAGE_DIR = "/media/rdr2/RDR2_dataset_processed_test/PNG"

with open(GT_PATH) as f:
    gt_data = json.load(f)

with open(PRED_PATH) as f:
    pred_data = json.load(f)

metric = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    class_metrics=True
)

def get_capture_id(filename):
    return filename.split("_")[1]

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

    predictions = pred_by_image.get(filename, [])
    capture_id = get_capture_id(filename)

    if capture_id not in gt_data:
        continue

    matched += 1

    entities = gt_data[capture_id].get("Entities", [])

    gt_boxes = [
        e["BoundingBoxCalculated"]
        for e in entities
        if e.get("BoundingBoxCalculated") is not None
    ]

    pred_boxes = [p["bbox"] for p in predictions]
    pred_scores = [p["score"] for p in predictions]

    target = [{
        "boxes": torch.tensor(
            gt_boxes, dtype=torch.float32
        ).reshape(-1, 4),
        "labels": torch.zeros(
            len(gt_boxes), dtype=torch.int64
        )
    }]

    preds = [{
        "boxes": torch.tensor(
            pred_boxes, dtype=torch.float32
        ).reshape(-1, 4),
        "scores": torch.tensor(
            pred_scores, dtype=torch.float32
        ),
        "labels": torch.zeros(
            len(pred_boxes), dtype=torch.int64
        )
    }]

    metric.update(preds, target)

    if (i + 1) % 5000 == 0:
        print(
            f"Processed {i + 1}/{len(image_files)} frames...",
            flush=True
        )

print(f"\nFinished pairing {matched}/{len(image_files)} images.")

results = metric.compute()

print("  OWL-ViT ANIMAL RESULTS")
print("=" * 50)
print(
    f"mAP [IoU=0.50:0.95] : "
    f"{results['map'].item():.4f} "
    f"({results['map'].item()*100:.2f}%)"
)
print(
    f"mAP [IoU=0.50]      : "
    f"{results['map_50'].item():.4f} "
    f"({results['map_50'].item()*100:.2f}%)"
)
print(
    f"mAP [IoU=0.75]      : "
    f"{results['map_75'].item():.4f} "
    f"({results['map_75'].item()*100:.2f}%)"
)
print(f"mAP (Small objects) : {results['map_small'].item():.4f}")
print(f"mAP (Medium objects): {results['map_medium'].item():.4f}")
print(f"mAP (Large objects) : {results['map_large'].item():.4f}")
print(f"Recall (max 100 det): {results['mar_100'].item():.4f}")