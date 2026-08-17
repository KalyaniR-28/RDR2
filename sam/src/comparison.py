# import os
# import sys
# import json
# import numpy as np
# from PIL import Image
# from tqdm import tqdm


# def main():
#     GT_DIR = "/media/sparackal/My Passport/RDR2_dataset_processed_test/SemanticSegmentationMasks"
#     PRED_DIR = "test_animal_prompt"
#     LABEL_MAPPING_JSON = "configs/gt_Coarse_labelIds_mapping.json"

#     if not os.path.exists(GT_DIR) or not os.path.exists(PRED_DIR) or not os.path.exists(LABEL_MAPPING_JSON):
#         print("[ERROR] Check GT_DIR, PRED_DIR, and LABEL_MAPPING_JSON paths.")
#         sys.exit(1)

#     with open(LABEL_MAPPING_JSON, "r") as f:
#         label_mapping = json.load(f)

#     bg_label = label_mapping["background"]

#     gt_files = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".png"))
#     pred_files = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".png"))

#     print(f"--> Found {len(gt_files)} GT files, {len(pred_files)} prediction files.")
#     if len(gt_files) != len(pred_files):
#         print("[CRITICAL WARNING] Counts do not match — positional pairing is unsafe. Stopping.")
#         sys.exit(1)

#     print("\n--> Sample pairing:")
#     for gt_name, pred_name in list(zip(gt_files, pred_files))[:5]:
#         print(f"    GT: {gt_name:<45} <-> PRED: {pred_name}")
#     print()

#     TP = FP = FN = TN = 0

#     for gt_name, pred_name in tqdm(zip(gt_files, pred_files), total=len(gt_files), desc="Scoring"):
#         gt_path = os.path.join(GT_DIR, gt_name)
#         pred_path = os.path.join(PRED_DIR, pred_name)

#         gt = np.array(Image.open(gt_path))
#         pred = np.array(Image.open(pred_path))

#         if gt.shape != pred.shape:
#             # Nearest-neighbor only — never interpolate label/binary masks
#             pred = np.array(
#                 Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
#             )

#         # Collapse GT to binary: any real class (0-37) -> "animal", background label -> "background".
#         # Prediction is already binary (0 = animal, 255 = background) from the test_animal_prompt script.
#         gt_is_animal = (gt != bg_label)
#         pred_is_animal = (pred == 0)

#         TP += int(np.sum(gt_is_animal & pred_is_animal))
#         FP += int(np.sum(~gt_is_animal & pred_is_animal))
#         FN += int(np.sum(gt_is_animal & ~pred_is_animal))
#         TN += int(np.sum(~gt_is_animal & ~pred_is_animal))

#     print(f"Scored {len(gt_files)} image pairs.\n")

#     denom_iou = TP + FP + FN
#     denom_recall = TP + FN
#     denom_precision = TP + FP
#     total = TP + FP + FN + TN

#     iou_animal = TP / denom_iou if denom_iou > 0 else 0.0
#     recall_animal = TP / denom_recall if denom_recall > 0 else 0.0     # = Acc for the animal class
#     precision_animal = TP / denom_precision if denom_precision > 0 else 0.0
#     pixel_accuracy = (TP + TN) / total if total > 0 else 0.0

#     f1_animal = (
#         2 * precision_animal * recall_animal / (precision_animal + recall_animal)
#         if (precision_animal + recall_animal) > 0 else 0.0
#     )

#     print(f"{'Metric':<20} {'Value':>10}")
#     print("-" * 31)
#     print(f"{'IoU (animal)':<20} {iou_animal:>10.4f}")
#     print(f"{'Recall (animal)':<20} {recall_animal:>10.4f}")
#     print(f"{'Precision (animal)':<20} {precision_animal:>10.4f}")
#     print(f"{'F1 (animal)':<20} {f1_animal:>10.4f}")
#     print(f"{'Pixel Accuracy':<20} {pixel_accuracy:>10.4f}  (overall, both classes)")
#     print(f"\nRaw counts — TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")


# if __name__ == "__main__":
#     main()
    
    
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- 1. THE MATH ENGINE ---
def calculate_iou_metrics(gt_mask, pred_mask, num_classes=2, ignore_index=255):
    """Hyper-fast mIoU calculation using NumPy bincount."""
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(f"Shape mismatch! GT: {gt_mask.shape} | Pred: {pred_mask.shape}")
    
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()
    
    valid_pixels = (gt_flat != ignore_index) & (gt_flat >= 0) & (gt_flat < num_classes)
    gt_flat = gt_flat[valid_pixels]
    pred_flat = pred_flat[valid_pixels]
    
    confusion_matrix = np.bincount(
        num_classes * gt_flat + pred_flat, 
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    
    return confusion_matrix

# --- 2. THE BINARY BATCH EVALUATOR ---
def evaluate_binary_dataset(gt_dir, pred_dir, num_classes=2):
    global_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Grab both lists of files and sort them
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
    
    print(f"--> Found {len(pred_files)} predictions and {len(gt_files)} ground truths.")
    if len(pred_files) != len(gt_files):
        print("\n[CRITICAL WARNING] Folder counts do not match!")
        return None, None

    valid_comparisons = 0
    print("--> Commencing BINARY (Foreground vs Background) evaluation...\n")
    
    for pred_filename, gt_filename in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="Evaluating"):
        
        pred_path = os.path.join(pred_dir, pred_filename)
        gt_path = os.path.join(gt_dir, gt_filename)
            
        try:
            # Load images as standard arrays
            raw_gt = np.array(Image.open(gt_path))
            raw_pred = np.array(Image.open(pred_path))
            
            if raw_gt.ndim == 3: raw_gt = raw_gt[:, :, 0]
            if raw_pred.ndim == 3: raw_pred = raw_pred[:, :, 0]
            
            # ==========================================================
            # THE BINARY CONVERTER: 
            # Turn 255 into 0 (Background), and any Animal ID into 1 (Foreground)
            # ==========================================================
            gt_binary = np.zeros_like(raw_gt, dtype=np.uint8)
            gt_binary[raw_gt != 255] = 1  
            
            pred_binary = np.zeros_like(raw_pred, dtype=np.uint8)
            pred_binary[raw_pred != 255] = 1 
            
            # Calculate metrics using the new Binary masks
            cm = calculate_iou_metrics(gt_binary, pred_binary, num_classes=num_classes)
            global_confusion_matrix += cm
            valid_comparisons += 1
            
        except Exception as e:
            print(f"\n[ERROR] Failed comparing '{pred_filename}' with '{gt_filename}': {e}")

    # --- 3. FINAL DATASET CALCULATION ---
    print(f"\nSuccessfully evaluated {valid_comparisons} matching image pairs.")
    print("Calculating Final Binary Metrics...")
    
    tp = np.diag(global_confusion_matrix)
    fp = global_confusion_matrix.sum(axis=0) - tp
    fn = global_confusion_matrix.sum(axis=1) - tp
    
    union = tp + fp + fn
    valid_classes = union > 0
    
    final_iou_per_class = np.full(num_classes, np.nan)
    final_iou_per_class[valid_classes] = tp[valid_classes] / union[valid_classes]
    
    final_miou = np.nanmean(final_iou_per_class)
    final_accuracy = np.sum(tp) / np.sum(global_confusion_matrix) if np.sum(global_confusion_matrix) > 0 else 0
    
    print("-" * 40)
    print(f"Binary Background IoU (Class 0) : {final_iou_per_class[0] * 100:.2f}%")
    print(f"Binary Animal IoU (Class 1)     : {final_iou_per_class[1] * 100:.2f}%")
    print("-" * 40)
    print(f"OVERALL BINARY mIoU             : {final_miou * 100:.2f}%")
    print(f"OVERALL PIXEL ACCURACY          : {final_accuracy * 100:.2f}%")
    print("-" * 40)
    
    return final_miou, final_iou_per_class

# --- 4. EXECUTION ---
if __name__ == "__main__":
    GROUND_TRUTH_FOLDER = "/media/sparackal/My Passport/RDR2_dataset_processed_test/SemanticSegmentationMasks"
    PREDICTIONS_FOLDER = "test_animal_prompt"
    
    evaluate_binary_dataset(GROUND_TRUTH_FOLDER, PREDICTIONS_FOLDER, num_classes=2)
