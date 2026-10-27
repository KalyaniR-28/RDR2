import json
import re
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

PRED_PATH = (
    "/home/sparackal/GroundingDINO/coarse_predictions_remove_combined.json"
)
GT_PATH = "/media/rdr2/RDR2_dataset_processed_test/all_captures.json"
MAPPING_PATH = (
    "/media/rdr2/RDR2_dataset_processed_test/gt_Coarse_labelIds_mapping.json"
)

with open(MAPPING_PATH) as f:
  raw = json.load(f)
gt_classes = [
    c.lower().replace("_", " ").strip()
    for c in raw
    if c.lower() != "background"
]
class_to_id = {c: i for i, c in enumerate(gt_classes)}
id_to_class = {i: c for c, i in class_to_id.items()}


def resolve_phrase(phrase):
  clean = re.sub(r"[^\w\s]", "", phrase.lower().replace("##", "")).strip()
  if clean in class_to_id:
    return class_to_id[clean]
  if len(clean) >= 3:
    for c in gt_classes:
      if clean in c:
        return class_to_id[c]
    for c in sorted(gt_classes, key=len, reverse=True):
      if c in clean:
        return class_to_id[c]
  if len(clean) == 2:
    for c in gt_classes:
      if c.startswith(clean):
        return class_to_id[c]
  return None


with open(GT_PATH) as f:
  gt_data = json.load(f)
with open(PRED_PATH) as f:
  pred_data = json.load(f)

metric = MeanAveragePrecision(
    box_format="xyxy", iou_type="bbox", class_metrics=True
)

for idx, (key, preds) in enumerate(pred_data.items()):
  cid = key.split("_")[1] if len(key.split("_")) > 1 else key
  if cid not in gt_data:
    continue

  gt_boxes, gt_labels = [], []
  for ent in gt_data[cid].get("Entities", []):
    cname = ent.get("CoarseClassName", "").lower().replace("_", " ").strip()
    box = ent.get("BoundingBoxCalculated")
    if cname in class_to_id and box:
      gt_boxes.append([float(x) for x in box])
      gt_labels.append(class_to_id[cname])

  pr_boxes, pr_scores, pr_labels = [], [], []
  for p in preds:
    mid = resolve_phrase(p.get("class", ""))
    box = p.get("bbox") or p.get("box")
    if mid is not None and box:
      pr_boxes.append([float(x) for x in box])
      pr_scores.append(float(p.get("confidence", 0.0)))
      pr_labels.append(mid)

  target = [{
      "boxes": (
          torch.tensor(gt_boxes, dtype=torch.float32)
          if gt_boxes
          else torch.zeros((0, 4))
      ),
      "labels": (
          torch.tensor(gt_labels, dtype=torch.int64)
          if gt_labels
          else torch.zeros((0,), dtype=torch.int64)
      ),
  }]
  pred = [{
      "boxes": (
          torch.tensor(pr_boxes, dtype=torch.float32)
          if pr_boxes
          else torch.zeros((0, 4))
      ),
      "scores": (
          torch.tensor(pr_scores, dtype=torch.float32)
          if pr_scores
          else torch.zeros((0,))
      ),
      "labels": (
          torch.tensor(pr_labels, dtype=torch.int64)
          if pr_labels
          else torch.zeros((0,), dtype=torch.int64)
      ),
  }]

  metric.update(pred, target)
  if (idx + 1) % 5000 == 0:
    print(f"Processed {idx + 1}/{len(pred_data)} frames...", flush=True)

res = metric.compute()
print("\n" + "=" * 54)
print(f"mAP [0.50:0.95]: {res['map'].item():.4f} ({res['map'].item()*100:.2f}%)")
print(
    f"mAP@50         : {res['map_50'].item():.4f}"
    f" ({res['map_50'].item()*100:.2f}%)"
)
print(
    f"mAP@75         : {res['map_75'].item():.4f}"
    f" ({res['map_75'].item()*100:.2f}%)"
)
print(f"mAP (Small)    : {res['map_small'].item():.4f}")
print(f"mAP (Medium)   : {res['map_medium'].item():.4f}")
print(f"mAP (Large)    : {res['map_large'].item():.4f}")
print(f"Recall@100     : {res['mar_100'].item():.4f}")


if "map_per_class" in res:
  print(f"\n{'Class Name':<20} | {'AP (0.50:0.95)':<15}")
  print("-" * 38)
  for idx, name in id_to_class.items():
    if idx < len(res["map_per_class"]):
      val = res["map_per_class"][idx].item()
      if val >= 0:
        print(f"{name:<20} | {val:.4f}")