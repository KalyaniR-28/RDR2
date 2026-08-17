import json
import numpy as np

# --- FILE PATHS ---
PRED_JSON = "./predcoarse/coarse_predictions.json"
GT_JSON = "./all_captures.json"
IOU_THRESH = 0.5  # Standard IoU threshold


def compute_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two boxes [xmin, ymin, xmax, ymax]."""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


# 1. Load Data
with open(PRED_JSON, "r") as f:
    pred_data = json.load(f)

with open(GT_JSON, "r") as f:
    gt_raw = json.load(f)

# 2. Extract Capture ID from prediction key (e.g. "o_1000003_0" -> "1000003")
def parse_capture_id(pred_key):
    parts = pred_key.split("_")
    if len(parts) >= 2:
        return parts[1]  # returns "1000003"
    return pred_key


# 3. Find all unique classes across Predictions and Ground Truth
classes = set()

# From predictions
for img_dets in pred_data.values():
    for det in img_dets:
        classes.add(det["class_name"].lower())

# From ground truth (using CoarseClassName or SimpleClassName)
for capture_info in gt_raw.values():
    for entity in capture_info.get("Entities", []):
        classes.add(entity["CoarseClassName"].lower())

print(f"Classes evaluated: {list(classes)}\n")

# 4. Compute AP per Class
ap_per_class = {}

for cls in classes:
    # Build Ground Truth lookup for this class per prediction key
    gt_by_pred_key = {}
    total_gt_count = 0

    for pred_key in pred_data.keys():
        cid = parse_capture_id(pred_key)
        
        # Get ground truth entities for this capture ID
        cls_gt_boxes = []
        if cid in gt_raw:
            for entity in gt_raw[cid].get("Entities", []):
                # Matching against CoarseClassName (e.g., 'deer', 'bird')
                if entity["CoarseClassName"].lower() == cls:
                    cls_gt_boxes.append(entity["BoundingBoxCalculated"])

        gt_by_pred_key[pred_key] = {
            "boxes": cls_gt_boxes,
            "matched": [False] * len(cls_gt_boxes)
        }
        total_gt_count += len(cls_gt_boxes)

    if total_gt_count == 0:
        continue

    # Gather all predictions for this class
    preds = []
    for pred_key, pred_list in pred_data.items():
        for p in pred_list:
            if p["class_name"].lower() == cls:
                preds.append({
                    "pred_key": pred_key,
                    "confidence": p["confidence"],
                    "box": p["box"]
                })

    # Sort predictions by confidence score descending
    preds.sort(key=lambda x: x["confidence"], reverse=True)

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    # Match Predictions to Ground Truth
    for i, pred in enumerate(preds):
        pkey = pred["pred_key"]
        pred_box = pred["box"]

        gt_info = gt_by_pred_key[pkey]
        gt_boxes = gt_info["boxes"]

        max_iou = 0.0
        max_idx = -1

        for j, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_idx = j

        # Check IoU threshold & duplicate match
        if max_iou >= IOU_THRESH and max_idx != -1 and not gt_info["matched"][max_idx]:
            tp[i] = 1
            gt_info["matched"][max_idx] = True
        else:
            fp[i] = 1

    # Accumulate TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / total_gt_count
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(float).eps)

    # Compute AP (Area Under PR Curve)
    recalls_padded = np.concatenate(([0.0], recalls, [1.0]))
    precisions_padded = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions_padded) - 1, 0, -1):
        precisions_padded[i - 1] = np.maximum(precisions_padded[i - 1], precisions_padded[i])

    indices = np.where(recalls_padded[1:] != recalls_padded[:-1])[0]
    ap = np.sum((recalls_padded[indices + 1] - recalls_padded[indices]) * precisions_padded[indices + 1])

    ap_per_class[cls] = ap
    print(f"Class: {cls.upper():<12} | Total GT: {total_gt_count:<4} | Detections: {len(preds):<4} | AP@50: {ap:.4f}")

# Overall mAP@50
if ap_per_class:
    mAP = np.mean(list(ap_per_class.values()))
    print("-" * 60)
    print(f"Overall mAP@50: {mAP:.4f}")