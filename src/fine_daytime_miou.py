import os
import re
import json
import csv
import numpy as np
from PIL import Image
from collections import defaultdict

# ---- CONFIG: adjust these paths ----
LABEL_MAPPING_JSON = "configs/gt_Fine_labelIds_mapping.json"
PRED_DIR = "/home/sparackal/sam/fineloop"                                     # o_<id>_<suffix>_semantic.png
GT_DIR = "/media/sparackal/My Passport/RDR2_dataset_processed_test/SemanticSegmentationMasks"  # adjust to your actual GT folder
PRED_BG_VALUE = 255  # value your predicted masks use for background

with open(LABEL_MAPPING_JSON, "r") as f:
    label_mapping = json.load(f)

bg_label = label_mapping["background"]
class_ids = sorted(label_mapping.values())           # includes background internally, for correct accounting
num_classes = len(class_ids)
id_to_index = {cid: i for i, cid in enumerate(class_ids)}
index_to_name = {id_to_index[cid]: name for name, cid in label_mapping.items()}
bg_index = id_to_index[bg_label]

# ---- Step 1: collect GT files, ONLY *_coarse.png ----
gt_pattern = re.compile(r"mSemSegmentation_(\d+)_fine\.png$")
gt_files = {}
for fname in os.listdir(GT_DIR):
    m = gt_pattern.match(fname)
    if m:
        gt_files[m.group(1)] = os.path.join(GT_DIR, fname)

print(f"Found {len(gt_files)} ground truth 'coarse' masks.")

# ---- Step 2: collect predicted files, grouped by id -> suffix ----
pred_pattern = re.compile(r"o_(\d+)_(\d+)_semantic\.png$")
pred_files = defaultdict(dict)
for fname in os.listdir(PRED_DIR):
    m = pred_pattern.match(fname)
    if m:
        img_id, suffix = m.group(1), m.group(2)
        pred_files[img_id][suffix] = os.path.join(PRED_DIR, fname)

suffixes = sorted({s for d in pred_files.values() for s in d.keys()}, key=int)
print(f"Found suffix variants: {suffixes}")


def remap_to_class_index(arr, is_prediction):
    """Map raw pixel values into contiguous class indices 0..num_classes-1.
    Unrecognized pixels -> -1 (ignored)."""
    out = np.full(arr.shape, -1, dtype=np.int32)
    for cid, idx in id_to_index.items():
        out[arr == cid] = idx
    if is_prediction:
        out[arr == PRED_BG_VALUE] = bg_index
    return out


def confusion_matrix(gt_idx, pred_idx):
    valid = (gt_idx >= 0) & (pred_idx >= 0)
    conf = np.bincount(
        num_classes * gt_idx[valid] + pred_idx[valid],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return conf


def iou_from_confusion(conf):
    intersection = np.diag(conf)
    union = conf.sum(axis=1) + conf.sum(axis=0) - intersection
    iou = np.full(num_classes, np.nan)
    present = union > 0
    iou[present] = intersection[present] / union[present]
    return iou


# ---- Step 3: evaluate each suffix SEPARATELY ----
all_results = {}

for suffix in suffixes:
    total_conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    matched = 0

    for img_id, gt_path in gt_files.items():
        if img_id not in pred_files or suffix not in pred_files[img_id]:
            continue

        pred_path = pred_files[img_id][suffix]

        gt_arr = np.array(Image.open(gt_path))
        pred_arr = np.array(Image.open(pred_path))

        if gt_arr.shape != pred_arr.shape:
            pred_arr = np.array(
                Image.fromarray(pred_arr).resize((gt_arr.shape[1], gt_arr.shape[0]), Image.NEAREST)
            )

        gt_idx = remap_to_class_index(gt_arr, is_prediction=False)
        pred_idx = remap_to_class_index(pred_arr, is_prediction=True)

        total_conf += confusion_matrix(gt_idx, pred_idx)
        matched += 1

    iou = iou_from_confusion(total_conf)
    all_results[suffix] = (iou, matched)
    print(f"Suffix {suffix}: matched {matched} images against ground truth")

# ---- Step 4: report per-ANIMAL-class IoU and mIoU (background excluded), per suffix ----
for suffix, (iou, matched) in all_results.items():
    print(f"\n=== Suffix '{suffix}'  (n={matched} images) ===")

    for idx in range(num_classes):
        if idx == bg_index:
            continue  # skip background in reporting
        name = index_to_name[idx]
        score = iou[idx]
        print(f"  {name:20s}: {'N/A (not present)' if np.isnan(score) else f'{score:.4f}'}")

    animal_scores = np.delete(iou, bg_index)
    animal_scores = animal_scores[~np.isnan(animal_scores)]
    miou = animal_scores.mean() if len(animal_scores) else float("nan")

    print(f"  --> mIoU (animal classes only): {miou:.4f}")

# ---- Step 5: compact summary across suffixes, for the day/night comparison ----
print("\n=== Summary: mIoU by time-of-day (suffix) ===")
for suffix in suffixes:
    iou, matched = all_results[suffix]
    animal_scores = np.delete(iou, bg_index)
    animal_scores = animal_scores[~np.isnan(animal_scores)]
    miou = animal_scores.mean() if len(animal_scores) else float("nan")
    print(f"  Suffix {suffix:>3s}  (n={matched:4d}):  mIoU = {miou:.4f}")

# ---- Step 6: Save results to CSV ----
with open("miou_per_class_by_daytime_fine.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Header row: class_name, suffix_0, suffix_7, suffix_12, suffix_17, suffix_20
    header = ["class_name"] + [f"suffix_{s}" for s in suffixes]
    writer.writerow(header)

    # One row per animal class, background excluded
    for idx in range(num_classes):
        if idx == bg_index:
            continue
        row = [index_to_name[idx]]
        for suffix in suffixes:
            iou, matched = all_results[suffix]
            score = iou[idx]
            row.append(f"{score:.4f}" if not np.isnan(score) else "N/A")
        writer.writerow(row)

    # Final row: overall mIoU per suffix
    miou_row = ["mIoU (animal classes only)"]
    for suffix in suffixes:
        iou, matched = all_results[suffix]
        animal_scores = np.delete(iou, bg_index)
        animal_scores = animal_scores[~np.isnan(animal_scores)]
        miou = animal_scores.mean() if len(animal_scores) else float("nan")
        miou_row.append(f"{miou:.4f}")
    writer.writerow(miou_row)

    # Also record how many images were matched per suffix, for transparency
    matched_row = ["n_images_matched"]
    for suffix in suffixes:
        _, matched = all_results[suffix]
        matched_row.append(str(matched))
    writer.writerow(matched_row)

print("\nSaved per-class, per-suffix mIoU results to miou_per_class_by_daytime_fine.csv")