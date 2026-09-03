import json
import numpy as np

PRED_JSON = "/home/sparackal/GroundingDINO/coarse_predictions_remove_combined.json"
GT_JSON = "/media/rdr2/RDR2_dataset_processed_test/all_captures.json"
COARSE_MAPPING_JSON = "/media/rdr2/RDR2_dataset_processed_test/gt_Coarse_labelIds_mapping.json"
IOU_THRESHOLD = 0.50


def normalize(name):
    return name.lower().strip().replace("_", " ")


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def parse_capture_id(filename):
    parts = filename.split("_")
    if len(parts) < 2:
        return None
    return parts[1]


def compute_ap50(gt_boxes_per_image, predictions, total_gt_count):
    matched = {k: [False] * len(v) for k, v in gt_boxes_per_image.items()}
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))

    for i, pred in enumerate(predictions):
        gt_boxes = gt_boxes_per_image.get(pred["pred_key"], [])
        best_iou, best_idx = 0.0, -1
        for j, gt_box in enumerate(gt_boxes):
            if matched[pred["pred_key"]][j]:
                continue
            iou = compute_iou(pred["box"], gt_box)
            if iou > best_iou:
                best_iou, best_idx = iou, j
        if best_iou >= IOU_THRESHOLD and best_idx != -1:
            tp[i] = 1
            matched[pred["pred_key"]][best_idx] = True
        else:
            fp[i] = 1

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recalls = tp_cum / total_gt_count
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(float).eps)

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap50 = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])

    return ap50, int(np.sum(tp)), int(np.sum(fp))


with open(COARSE_MAPPING_JSON) as f:
    coarse_mapping = json.load(f)

coarse_classes = [c for c in coarse_mapping if c.lower() != "background"]
normalized_to_original = {normalize(c): c for c in coarse_classes}

with open(PRED_JSON) as f:
    pred_data = json.load(f)

with open(GT_JSON) as f:
    gt_raw = json.load(f)

gt_by_image = {}

for pred_key in pred_data:
    capture_id = parse_capture_id(pred_key)
    gt_by_image[pred_key] = {normalize(c): [] for c in coarse_classes}

    if capture_id is None or capture_id not in gt_raw:
        continue

    for entity in gt_raw[capture_id].get("Entities", []):
        cls = normalize(entity.get("CoarseClassName", ""))
        bbox = entity.get("BoundingBoxCalculated")
        if cls in gt_by_image[pred_key] and bbox is not None:
            gt_by_image[pred_key][cls].append(bbox)

ap_per_class = {}

for class_name in coarse_classes:
    norm_class = normalize(class_name)

    gt_for_class = {k: v.get(norm_class, []) for k, v in gt_by_image.items()}
    total_gt_count = sum(len(v) for v in gt_for_class.values())

    predictions = sorted(
        (
            {"pred_key": key, "confidence": float(p["confidence"]), "box": p["bbox"]}
            for key, preds in pred_data.items()
            for p in preds
            if normalize(p.get("class", "")) == norm_class
        ),
        key=lambda x: x["confidence"],
        reverse=True,
    )

    if total_gt_count == 0:
        print(f"{class_name:<18} GT: {total_gt_count:<6} Pred: {len(predictions):<6} AP50: N/A")
        continue

    ap50, tp_count, fp_count = compute_ap50(gt_for_class, predictions, total_gt_count)
    ap_per_class[norm_class] = ap50
    fn_count = total_gt_count - tp_count

    print(
        f"{class_name:<18} GT: {total_gt_count:<6} Pred: {len(predictions):<6} "
        f"TP: {tp_count:<6} FP: {fp_count:<6} FN: {fn_count:<6} AP50: {ap50:.4f}"
    )

if ap_per_class:
    map50 = np.mean(list(ap_per_class.values()))
    print()
    for norm_class, ap in ap_per_class.items():
        print(f"AP50 {normalized_to_original[norm_class]:<18}: {ap:.6f} ({ap * 100:.2f}%)")
    print(f"mAP50: {map50:.6f} ({map50 * 100:.2f}%)")
else:
    print("No classes had ground truth boxes.")