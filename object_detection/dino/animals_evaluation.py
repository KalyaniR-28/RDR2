import json
import numpy as np

PREDICTIONS_FILE = "/home/sparackal/GroundingDINO/animals_predictions.json"
GROUND_TRUTH_FILE = "/media/rdr2/RDR2_dataset_processed_test/all_captures.json"
IOU_THRESHOLD = 0.5


def get_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def frame_id(filename):
    parts = filename.split("_")
    return parts[1] if len(parts) >= 2 else filename


print("Loading predictions...")
with open(PREDICTIONS_FILE) as f:
    predictions = json.load(f)

print("Loading ground truth...")
with open(GROUND_TRUTH_FILE) as f:
    gt_raw = json.load(f)

gt_by_image = {}
missing_captures = set()
total_gt_boxes = 0

for image_name in predictions:
    fid = frame_id(image_name)
    boxes = []
    if fid in gt_raw:
        boxes = [
            e["BoundingBoxCalculated"]
            for e in gt_raw[fid].get("Entities", [])
            if e.get("BoundingBoxCalculated") is not None
        ]
    else:
        missing_captures.add(fid)
    gt_by_image[image_name] = {"boxes": boxes, "matched": [False] * len(boxes)}
    total_gt_boxes += len(boxes)

detections = sorted(
    (
        {"image": img, "conf": float(d["confidence"]), "box": d["bbox"]}
        for img, dets in predictions.items()
        for d in dets
        if d.get("class", "").lower().strip() == "animal"
    ),
    key=lambda x: x["conf"],
    reverse=True,
)

print(f"\nPrediction images : {len(predictions)}")
print(f"GT animal boxes   : {total_gt_boxes}")
print(f"Total predictions : {len(detections)}")
print(f"Missing captures  : {len(missing_captures)}")
print(f"IoU threshold     : {IOU_THRESHOLD}")

# Match predictions to GT (greedy, one GT box per prediction max)
tp = np.zeros(len(detections))
fp = np.zeros(len(detections))

for i, det in enumerate(detections):
    gt = gt_by_image[det["image"]]
    best_iou, best_idx = 0.0, -1
    for j, box in enumerate(gt["boxes"]):
        if gt["matched"][j]:
            continue
        iou = get_iou(det["box"], box)
        if iou > best_iou:
            best_iou, best_idx = iou, j
    if best_iou >= IOU_THRESHOLD and best_idx != -1:
        tp[i] = 1
        gt["matched"][best_idx] = True
    else:
        fp[i] = 1

tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
recalls = tp_cum / total_gt_boxes
precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(float).eps)

recalls = np.concatenate(([0.0], recalls, [1.0]))
precisions = np.concatenate(([0.0], precisions, [0.0]))
for i in range(len(precisions) - 1, 0, -1):
    precisions[i - 1] = max(precisions[i - 1], precisions[i])

idx = np.where(recalls[1:] != recalls[:-1])[0]
ap50 = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])

tp_count, fp_count = int(np.sum(tp)), int(np.sum(fp))
fn_count = total_gt_boxes - tp_count
precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
recall = tp_count / total_gt_boxes if total_gt_boxes > 0 else 0.0

print(f"\nTrue positives  : {tp_count}")
print(f"False positives : {fp_count}")
print(f"False negatives : {fn_count}")
print(f"Precision       : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall          : {recall:.4f} ({recall * 100:.2f}%)")
print(f"AP50            : {ap50:.6f} ({ap50 * 100:.2f}%)")