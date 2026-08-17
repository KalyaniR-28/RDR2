import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_confusion_matrix(pred, gt, num_classes):
    """
    Computes the confusion matrix for a single pair of images.
    pred and gt must be 1D numpy arrays (flattened images).
    """
    # Create a mask to only consider valid pixels (e.g., ignore boundaries if needed,
    # though usually all pixels in gt are valid if it's 0 to num_classes-1).
    # Here we assume all pixels in the range [0, num_classes-1] are valid.
    mask = (gt >= 0) & (gt < num_classes)
    
    # Calculate confusion matrix using a fast bincount trick
    # bincount calculates frequencies of values. By combining gt and pred into a single integer,
    # we can count occurrences of each (gt, pred) pair.
    hist = np.bincount(
        num_classes * gt[mask] + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    
    return hist

def main():
    # --- PATHS ---
    # Update these paths to match your actual directories
    GT_DIR = "/media/sparackal/My Passport/RDR2_dataset_processed_test/SemanticSegmentationMasks" 
    PRED_DIR = os.path.join(os.getcwd(), "ccoarseout") 
    LABEL_MAPPING_JSON = "configs/gt_Coarse_labelIds_mapping.json"
    
    if not os.path.exists(GT_DIR):
        print(f"[ERROR] Ground truth directory not found: {GT_DIR}")
        sys.exit(1)
    if not os.path.exists(PRED_DIR):
        print(f"[ERROR] Prediction directory not found: {PRED_DIR}")
        sys.exit(1)

    with open(LABEL_MAPPING_JSON, "r") as f:
        label_mapping = json.load(f)

    bg_label = label_mapping["background"] 
    num_classes = len(label_mapping) # Total classes including background
    
    # Create a reverse mapping to get names from IDs
    id_to_name = {int(v): k for k, v in label_mapping.items()}
    
    # --- INITIALIZE CONFUSION MATRIX ---
    total_hist = np.zeros((num_classes, num_classes), dtype=np.float64)

    # Find and SORT matching images
    valid_extensions = ('.png', '.bmp')
    
    # Sorting is critical here to ensure "first matches first"
    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.lower().endswith(valid_extensions)])
    
    gt_files = sorted([f for f in os.listdir(GT_DIR) if f.startswith("o_") and f.lower().endswith(valid_extensions)])
    print(f"Found {len(pred_files)} prediction files and {len(gt_files)} GT files to evaluate.")
    
    if len(pred_files) != len(gt_files):
        print("[WARNING] The number of predictions and ground truths do not match!")
        print("The script will evaluate up to the shortest list.")

    # --- EVALUATION LOOP ---
    # zip() pairs them up: (pred_1, gt_1), (pred_2, gt_2), etc.
    total_pairs = min(len(pred_files), len(gt_files))
    
    for pred_name, gt_name in tqdm(zip(pred_files, gt_files), desc="Evaluating", total=total_pairs):
        
        gt_path = os.path.join(GT_DIR, gt_name)
        pred_path = os.path.join(PRED_DIR, pred_name)
        
        try:
            # Load images as numpy arrays
            pred_img = np.array(Image.open(pred_path))
            gt_img = np.array(Image.open(gt_path))
            
            # Map the 255 background back to its original bg_label for the confusion matrix
            pred_img[pred_img == 255] = bg_label
            gt_img[gt_img == 255] = bg_label
            
            # Flatten arrays
            pred_flat = pred_img.flatten()
            gt_flat = gt_img.flatten()
            
            # Accumulate confusion matrix
            total_hist += compute_confusion_matrix(pred_flat, gt_flat, num_classes)
            
        except Exception as e:
            print(f"\n[ERROR] processing {pred_name} and {gt_name}: {e}")

    # --- CALCULATE METRICS FROM CONFUSION MATRIX ---
    diag = np.diag(total_hist)
    row_sum = total_hist.sum(axis=1)
    col_sum = total_hist.sum(axis=0)
    
    tp = diag
    fp = col_sum - diag
    fn = row_sum - diag
    
    epsilon = 1e-6
    
    union = tp + fp + fn
    iou_per_class = tp / (union + epsilon)
    precision_per_class = tp / (col_sum + epsilon)
    recall_per_class = tp / (row_sum + epsilon)
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + epsilon)

    pixel_accuracy = np.nansum(diag) / (np.nansum(total_hist) + epsilon)
    
    fg_indices = [i for i in range(num_classes) if i != bg_label]
    
    miou = np.nanmean(iou_per_class[fg_indices])
    macc = np.nanmean(recall_per_class[fg_indices])
    mprecision = np.nanmean(precision_per_class[fg_indices])
    mf1 = np.nanmean(f1_per_class[fg_indices])

    # --- PRINT RESULTS ---
    print("\n" + "="*50)
    print(f"{'OVERALL METRICS (Excluding Background)':^50}")
    print("="*50)
    print(f"Overall Pixel Accuracy : {pixel_accuracy:>10.4f}")
    print(f"Mean IoU (mIoU)        : {miou:>10.4f}")
    print(f"Mean Accuracy (mAcc)   : {macc:>10.4f}")
    print(f"Mean Precision         : {mprecision:>10.4f}")
    print(f"Mean F1 Score          : {mf1:>10.4f}")
    
    print("\n" + "="*60)
    print(f"{'PER-CLASS METRICS':^60}")
    print("="*60)
    print(f"{'Class Name':<15} | {'IoU':<8} | {'Acc(Rec)':<8} | {'Prec':<8} | {'F1':<8}")
    print("-" * 60)
    
    for i in range(num_classes):
        name = id_to_name.get(i, f"Class_{i}")
        if i == bg_label:
            name += " (BG)"
            
        print(f"{name:<15} | {iou_per_class[i]:>8.4f} | {recall_per_class[i]:>8.4f} | {precision_per_class[i]:>8.4f} | {f1_per_class[i]:>8.4f}")
        
    print("-" * 60)
    
    print("\n[DEBUG] Raw Counts (Aggregated across dataset):")
    print(f"Total True Positives (Foreground): {np.sum(tp[fg_indices]):.0f}")
    print(f"Total False Positives (Foreground): {np.sum(fp[fg_indices]):.0f}")
    print(f"Total False Negatives (Foreground): {np.sum(fn[fg_indices]):.0f}")
if __name__ == "__main__":
    main()