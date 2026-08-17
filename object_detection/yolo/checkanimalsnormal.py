import json
import numpy as np

# --- CONFIGURATION ---
PRED_JSON = "./pred_animals_birds/animal_bird_predictions.json"
GT_JSON = "./all_captures.json"
IOU_THRESH = 0.5

# Map the ground truth 'SimpleClassName' to your YOLO prompts
CLASS_MAPPING = {
    "mammal": "animal",
    "bird": "bird"
}

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

def parse_capture_id(pred_key):
    parts = pred_key.split("_")
    if len(parts) >= 2:
        return parts[1]  # "o_1000003_0" -> "1000003"
    return pred_key

# We are only evaluating the classes you prompted for
target_classes = ["animal", "bird"]
print(f"Evaluating classes: {target_classes}\n")

ap_per_class = {}

for cls in target_classes:
    gt_by_pred_key = {}
    total_gt_count = 0

    # 2. Extract Ground Truth
    for pred_key in pred_data.keys():
        cid = parse_capture_id(pred_key)
        
        cls_gt_boxes = []
        if cid in gt_raw:
            for entity in gt_raw[cid].get("Entities", []):
                # Get the SimpleClassName (e.g., "mammal") and map it (e.g., to "animal")
                raw_gt_class = entity.get("SimpleClassName", "").lower()
                mapped_gt_class = CLASS_MAPPING.get(raw_gt_class, "")
                
                if mapped_gt_class == cls:
                    cls_gt_boxes.append(entity["BoundingBoxCalculated"])

        gt_by_pred_key[pred_key] = {
            "boxes": cls_gt_boxes,
            "matched": [False] * len(cls_gt_boxes)
        }
        total_gt_count += len(cls_gt_boxes)

    if total_gt_count == 0:
        print(f"Class: {cls.upper():<12} | No Ground Truth objects found.")
        continue

    # 3. Extract Predictions
    preds = []
    for pred_key, pred_list in pred_data.items():
        for p in pred_list:
            pred_class = p["class_name"].lower()
            
            # Skip the empty string background class
            if pred_class == "":
                continue
                
            if pred_class == cls:
                preds.append({
                    "pred_key": pred_key,
                    "confidence": p["confidence"],
                    "box": p["box"]
                })

    preds.sort(key=lambda x: x["confidence"], reverse=True)

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    # 4. Match Predictions to Ground Truth
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

        if max_iou >= IOU_THRESH and max_idx != -1 and not gt_info["matched"][max_idx]:
            tp[i] = 1
            gt_info["matched"][max_idx] = True
        else:
            fp[i] = 1

    # 5. Calculate AP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / total_gt_count
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(float).eps)

    recalls_padded = np.concatenate(([0.0], recalls, [1.0]))
    precisions_padded = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions_padded) - 1, 0, -1):
        precisions_padded[i - 1] = np.maximum(precisions_padded[i - 1], precisions_padded[i])

    indices = np.where(recalls_padded[1:] != recalls_padded[:-1])[0]
    ap = np.sum((recalls_padded[indices + 1] - recalls_padded[indices]) * precisions_padded[indices + 1])

    ap_per_class[cls] = ap
    print(f"Class: {cls.upper():<12} | Total GT: {total_gt_count:<6} | Detections: {len(preds):<6} | AP@50: {ap:.4f}")

# 6. Overall mAP
if ap_per_class:
    mAP = np.mean(list(ap_per_class.values()))
    print("-" * 60)
    print(f"Overall mAP@50: {mAP:.4f}")