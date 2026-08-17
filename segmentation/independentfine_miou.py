import os
import re
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv

# ---- CONFIG ----
LABEL_MAPPING_JSON = "configs/gt_Fine_labelIds_mapping.json"
PRED_DIR = "/home/sparackal/sam/fineloop"                                              # o_<id>_<suffix>_semantic.png
GT_DIR = "/media/sparackal/My Passport/RDR2_dataset_processed_test/SemanticSegmentationMasks"  # mSemSegmentation_<id>_coarse.png
PRED_BG_VALUE = 255  # value your predicted masks use for background

with open(LABEL_MAPPING_JSON, "r") as f:
    label_mapping = json.load(f)

bg_label = label_mapping["background"]
class_ids = sorted(label_mapping.values())  # background included internally, for correct accounting
num_classes = len(class_ids)
id_to_index = {cid: i for i, cid in enumerate(class_ids)}
index_to_name = {id_to_index[cid]: name for name, cid in label_mapping.items()}
bg_index = id_to_index[bg_label]
import os
import re
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv

# ---- CONFIG ----
LABEL_MAPPING_JSON = "configs/gt_Fine_labelIds_mapping.json"
PRED_DIR = "/home/sparackal/sam/fineloop"                          # o_<id>_<suffix>_semantic.png
GT_DIR = "/media/sparackal/My Passport/RDR2_dataset_processed_test/SemanticSegmentationMasks"  # mSemSegmentation_<id>_coarse.png
PRED_BG_VALUE = 255  # value your predicted masks use for background

with open(LABEL_MAPPING_JSON, "r") as f:
    label_mapping = json.load(f)

bg_label = label_mapping["background"]
class_ids = sorted(label_mapping.values())  # background included internally
num_classes = len(class_ids)
id_to_index = {cid: i for i, cid in enumerate(class_ids)}
index_to_name = {id_to_index[cid]: name for name, cid in label_mapping.items()}
bg_index = id_to_index[bg_label]

# ---- Step 1: collect GT files ----
gt_pattern = re.compile(r"mSemSegmentation_(\d+)_fine\.png$")
gt_files = {}
for fname in os.listdir(GT_DIR):
    m = gt_pattern.match(fname)
    if m:
        gt_files[m.group(1)] = os.path.join(GT_DIR, fname)

print(f"Found {len(gt_files)} ground truth 'fine' masks.")

# ---- Step 2: collect predicted images ----
pred_pattern = re.compile(r"o_(\d+)_(\d+)_semantic\.png$")
pred_files = []
for fname in os.listdir(PRED_DIR):
    m = pred_pattern.match(fname)
    if m:
        img_id = m.group(1)
        if img_id in gt_files:
            pred_files.append({
                "filename": fname,
                "pred_path": os.path.join(PRED_DIR, fname),
                "gt_path": gt_files[img_id],
            })

print(f"Found {len(pred_files)} predicted images matched to a ground truth.")


def remap_to_class_index(arr, is_prediction):
    out = np.full(arr.shape, -1, dtype=np.int32)
    for cid, idx in id_to_index.items():
        out[arr == cid] = idx
    if is_prediction:
        out[arr == PRED_BG_VALUE] = bg_index
    return out


def confusion_matrix(gt_idx, pred_idx, n_classes):
    valid = (gt_idx >= 0) & (pred_idx >= 0)
    conf = np.bincount(
        n_classes * gt_idx[valid] + pred_idx[valid],
        minlength=n_classes ** 2
    ).reshape(n_classes, n_classes)
    return conf


def iou_from_confusion(conf, n_classes):
    intersection = np.diag(conf)
    union = conf.sum(axis=1) + conf.sum(axis=0) - intersection
    iou = np.full(n_classes, np.nan)
    present = union > 0
    iou[present] = intersection[present] / union[present]
    return iou


# ---- Step 3: Loop and process dataset ----
total_conf_multiclass = np.zeros((num_classes, num_classes), dtype=np.int64)
total_conf_binary = np.zeros((2, 2), dtype=np.int64)
matched = 0

for item in tqdm(pred_files, desc="Evaluating all images"):
    try:
        gt_arr = np.array(Image.open(item["gt_path"]))
        pred_arr = np.array(Image.open(item["pred_path"]))

        if gt_arr.shape != pred_arr.shape:
            pred_arr = np.array(
                Image.fromarray(pred_arr).resize((gt_arr.shape[1], gt_arr.shape[0]), Image.NEAREST)
            )

        # 1. Standard Multi-class indexes
        gt_idx = remap_to_class_index(gt_arr, is_prediction=False)
        pred_idx = remap_to_class_index(pred_arr, is_prediction=True)
        total_conf_multiclass += confusion_matrix(gt_idx, pred_idx, num_classes)

        # 2. Convert to Binary (0 = Background, 1 = Foreground/Animal)
        # Anything that is NOT the background index becomes 1 (Foreground)
        gt_binary = np.where(gt_idx == bg_index, 0, 1)
        pred_binary = np.where(pred_idx == bg_index, 0, 1)
        
        # Ensure invalid elements (-1) remain invalid
        gt_binary[gt_idx == -1] = -1
        pred_binary[pred_idx == -1] = -1
        
        total_conf_binary += confusion_matrix(gt_binary, pred_binary, 2)
        matched += 1

    except Exception as e:
        print(f"\n[ERROR] {item['filename']}: {e}")

print(f"\nAggregated {matched} images.")

# ---- Step 4: Compute Metrics ----
# Multi-class Scores
iou_multi = iou_from_confusion(total_conf_multiclass, num_classes)
animal_scores = np.delete(iou_multi, bg_index)
animal_scores = animal_scores[~np.isnan(animal_scores)]
miou = animal_scores.mean() if len(animal_scores) else float("nan")

# Binary Scores (0: Background, 1: Foreground)
iou_binary = iou_from_confusion(total_conf_binary, 2)

# ---- Step 5: Print Results ----
print("\n=== Binary Foreground vs Background IoU ===")
print(f"  Background IoU : {iou_binary[0]:.4f}")
print(f"  Foreground IoU : {iou_binary[1]:.4f}")
print(f"  Binary mIoU    : {np.nanmean(iou_binary):.4f}")

print("\n=== Per-class IoU (Dataset level, Animals only) ===")
for idx in range(num_classes):
    if idx == bg_index:
        continue
    name = index_to_name[idx]
    score = iou_multi[idx]
    print(f"  {name:20s}: {'N/A (never present)' if np.isnan(score) else f'{score:.4f}'}")

print(f"\nDataset mIoU (animal classes only): {miou:.4f}")

# ---- Save all results to CSV ----
with open("miou_per_class.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    # Section 1: Binary Metrics
    writer.writerow(["=== BINARY METRICS ==="])
    writer.writerow(["Metric", "IoU"])
    writer.writerow(["Binary Background", f"{iou_binary[0]:.4f}"])
    writer.writerow(["Binary Foreground (All Animals)", f"{iou_binary[1]:.4f}"])
    writer.writerow(["Binary Overall mIoU", f"{np.nanmean(iou_binary):.4f}"])
    writer.writerow([])
    
    # Section 2: Multi-class Metrics
    writer.writerow(["=== MULTI-CLASS ANIMAL METRICS ==="])
    writer.writerow(["class_name", "IoU"])
    for idx in range(num_classes):
        if idx == bg_index:
            continue
        score = iou_multi[idx]
        writer.writerow([index_to_name[idx], f"{score:.4f}" if not np.isnan(score) else "N/A"])
    writer.writerow(["Animal-only mIoU", f"{miou:.4f}"])

print("\nSaved all results to miou_per_class.csv")
# ---- Step 1: collect GT files, ONLY *_coarse.png ----
gt_pattern = re.compile(r"mSemSegmentation_(\d+)_fine\.png$")
gt_files = {}
for fname in os.listdir(GT_DIR):
    m = gt_pattern.match(fname)
    if m:
        gt_files[m.group(1)] = os.path.join(GT_DIR, fname)

print(f"Found {len(gt_files)} ground truth 'fine' masks.")

# ---- Step 2: collect ALL predicted images (every suffix), each matched to its GT ----
pred_pattern = re.compile(r"o_(\d+)_(\d+)_semantic\.png$")
pred_files = []
for fname in os.listdir(PRED_DIR):
    m = pred_pattern.match(fname)
    if m:
        img_id = m.group(1)
        if img_id in gt_files:
            pred_files.append({
                "filename": fname,
                "pred_path": os.path.join(PRED_DIR, fname),
                "gt_path": gt_files[img_id],
            })

print(f"Found {len(pred_files)} predicted images (all time-of-day variants) matched to a ground truth.")


def remap_to_class_index(arr, is_prediction):
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


# ---- Step 3: process every image independently, pool all pixel counts into ONE confusion matrix ----
total_conf = np.zeros((num_classes, num_classes), dtype=np.int64)
matched = 0

for item in tqdm(pred_files, desc="Evaluating all images"):
    try:
        gt_arr = np.array(Image.open(item["gt_path"]))
        pred_arr = np.array(Image.open(item["pred_path"]))

        if gt_arr.shape != pred_arr.shape:
            pred_arr = np.array(
                Image.fromarray(pred_arr).resize((gt_arr.shape[1], gt_arr.shape[0]), Image.NEAREST)
            )

        gt_idx = remap_to_class_index(gt_arr, is_prediction=False)
        pred_idx = remap_to_class_index(pred_arr, is_prediction=True)

        total_conf += confusion_matrix(gt_idx, pred_idx)
        matched += 1

    except Exception as e:
        print(f"\n[ERROR] {item['filename']}: {e}")

print(f"\nAggregated {matched} images into one confusion matrix.")

# ---- Step 4: compute per-class IoU and overall mIoU (background excluded) ----
iou = iou_from_confusion(total_conf)

print("\n=== Per-class IoU (dataset-level, all images pooled) ===")
for idx in range(num_classes):
    if idx == bg_index:
        continue
    name = index_to_name[idx]
    score = iou[idx]
    print(f"  {name:20s}: {'N/A (never present)' if np.isnan(score) else f'{score:.4f}'}")

animal_scores = np.delete(iou, bg_index)
animal_scores = animal_scores[~np.isnan(animal_scores)]
miou = animal_scores.mean() if len(animal_scores) else float("nan")

print(f"\nDataset mIoU (animal classes only): {miou:.4f}")

# ---- Save results to CSV ----
with open("miou_per_class.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["class_name", "IoU"])
    for idx in range(num_classes):
        if idx == bg_index:
            continue
        score = iou[idx]
        writer.writerow([index_to_name[idx], f"{score:.4f}" if not np.isnan(score) else "N/A"])
    writer.writerow(["mIoU", f"{miou:.4f}"])

print("Saved per-class results to miou_per_class.csv")