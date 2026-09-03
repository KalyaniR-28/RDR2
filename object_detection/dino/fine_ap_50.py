import json
import numpy as np


PRED_JSON = "/home/sparackal/GroundingDINO/fine_predictions_remove_combined.json"
GT_JSON = "/media/rdr2/RDR2_dataset_processed_test/all_captures.json"
FINE_MAPPING_JSON = "/media/rdr2/RDR2_dataset_processed_test/gt_Fine_labelIds_mapping.json"

IOU_THRESHOLD = 0.50


def normalize(name):
    return name.lower().strip().replace("_", " ")


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area1 = (
        max(0.0, box1[2] - box1[0])
        * max(0.0, box1[3] - box1[1])
    )

    area2 = (
        max(0.0, box2[2] - box2[0])
        * max(0.0, box2[3] - box2[1])
    )

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def parse_capture_id(filename):
    parts = filename.split("_")
    return parts[1] if len(parts) >= 2 else None


def compute_ap50(gt_boxes, preds, n_gt):
    matched = {
        k: [False] * len(v)
        for k, v in gt_boxes.items()
    }

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, pred in enumerate(preds):

        boxes = gt_boxes.get(
            pred["pred_key"],
            []
        )

        best_iou = 0.0
        best_idx = -1

        for j, gt in enumerate(boxes):

            if matched[pred["pred_key"]][j]:
                continue

            iou = compute_iou(
                pred["box"],
                gt
            )

            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou >= IOU_THRESHOLD and best_idx != -1:
            tp[i] = 1
            matched[pred["pred_key"]][best_idx] = True
        else:
            fp[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / n_gt

    precision = tp / np.maximum(
        tp + fp,
        np.finfo(float).eps
    )

    recall = np.concatenate(
        ([0.0], recall, [1.0])
    )

    precision = np.concatenate(
        ([0.0], precision, [0.0])
    )

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(
            precision[i - 1],
            precision[i]
        )

    idx = np.where(
        recall[1:] != recall[:-1]
    )[0]

    ap = np.sum(
        (recall[idx + 1] - recall[idx])
        * precision[idx + 1]
    )

    return ap, int(tp[-1]) if len(tp) else 0, int(fp[-1]) if len(fp) else 0


with open(FINE_MAPPING_JSON) as f:
    fine_mapping = json.load(f)

fine_classes = [
    c for c in fine_mapping
    if c.lower() != "background"
]

normalized_to_original = {
    normalize(c): c
    for c in fine_classes
}


with open(PRED_JSON) as f:
    pred_data = json.load(f)

with open(GT_JSON) as f:
    gt_raw = json.load(f)


gt_by_image = {}

for pred_key in pred_data:

    capture_id = parse_capture_id(
        pred_key
    )

    gt_by_image[pred_key] = {
        normalize(c): []
        for c in fine_classes
    }

    if capture_id is None or capture_id not in gt_raw:
        continue

    for entity in gt_raw[capture_id].get(
        "Entities", []
    ):

        cls = normalize(
            entity.get(
                "FineClassName",
                ""
            )
        )

        bbox = entity.get(
            "BoundingBoxCalculated"
        )

        if (
            cls in gt_by_image[pred_key]
            and bbox is not None
        ):
            gt_by_image[pred_key][cls].append(
                bbox
            )


ap_per_class = {}


for class_name in fine_classes:

    cls = normalize(class_name)

    gt = {
        k: v.get(cls, [])
        for k, v in gt_by_image.items()
    }

    n_gt = sum(
        len(v)
        for v in gt.values()
    )

    preds = sorted(
        (
            {
                "pred_key": key,
                "confidence": float(
                    p["confidence"]
                ),
                "box": p["bbox"]
            }

            for key, items in pred_data.items()

            for p in items

            if normalize(
                p.get("class", "")
            ) == cls
        ),

        key=lambda x: x["confidence"],
        reverse=True
    )

    if n_gt == 0:
        print(
            f"{class_name:<20} "
            f"GT: {n_gt:<6} "
            f"Pred: {len(preds):<6} "
            f"AP50: N/A"
        )
        continue

    ap, tp, fp = compute_ap50(
        gt,
        preds,
        n_gt
    )

    ap_per_class[cls] = ap

    fn = n_gt - tp

    print(
        f"{class_name:<20} "
        f"GT: {n_gt:<6} "
        f"Pred: {len(preds):<6} "
        f"TP: {tp:<6} "
        f"FP: {fp:<6} "
        f"FN: {fn:<6} "
        f"AP50: {ap:.4f}"
    )


if ap_per_class:

    map50 = np.mean(
        list(ap_per_class.values())
    )

    print()

    for cls, ap in ap_per_class.items():

        print(
            f"AP50 {normalized_to_original[cls]:<20}: "
            f"{ap:.6f} "
            f"({ap * 100:.2f}%)"
        )

    print(
        f"mAP50: {map50:.6f} "
        f"({map50 * 100:.2f}%)"
    )

else:

    print("No classes had ground truth boxes.")