import json
import numpy as np

PREDICTIONS_FILE = "./pred_animals/animals_predictions.json"
GROUND_TRUTH_FILE = "./all_captures.json"

# 0.5 means the predicted box must cover at least 50% of the actual box to be considered a correct match.
MIN_OVERLAP = 0.5 

def get_iou(pred_bbox, true_bbox):
   
    overlap_x1 = max(pred_bbox[0], true_bbox[0])
    overlap_y1 = max(pred_bbox[1], true_bbox[1])
    overlap_x2 = min(pred_bbox[2], true_bbox[2])
    overlap_y2 = min(pred_bbox[3], true_bbox[3])

    
    overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
    
    
    area_pred = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    area_true = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])

    
    total_area = area_pred + area_true - overlap_area
    
    if total_area <= 0:
        return 0.0
        
    return overlap_area / total_area



with open(PREDICTIONS_FILE, "r") as file:
    yolo_outputs = json.load(file)

with open(GROUND_TRUTH_FILE, "r") as file:
    raw_annotations = json.load(file)


def extract_base_frame_id(filename):
    
    #The ground truth dictionary just uses the core ID '1000003'. So to just get that.
    segments = filename.split("_")
    if len(segments) >= 2:
        return segments[1]  
    return filename


labels_to_evaluate = ["animals"]
print(f"Starting evaluation for: {labels_to_evaluate}\n")


final_scores = {}

for current_label in labels_to_evaluate:
    
    image_to_gt_map = {}
    total_actual_objects = 0

    # here we the creating the ground truth checklist
    for image_name in yolo_outputs.keys():
        frame_id = extract_base_frame_id(image_name)
        
        actual_boxes = []
        if frame_id in raw_annotations:
            for entity in raw_annotations[frame_id].get("Entities", []):
                
                # since since the ground truth does not  have specific name , we treat every bounding box prediction as animals            
                actual_boxes.append(entity["BoundingBoxCalculated"])
        image_to_gt_map[image_name] = {
            "target_boxes": actual_boxes,
            "already_matched": [False] * len(actual_boxes)
        }
        total_actual_objects += len(actual_boxes)

    if total_actual_objects == 0:
        print(f"Skipping {current_label.upper()} - no objects found in ground truth.")
        continue
    
    scored_detections = []
    
    for image_name, box_list in yolo_outputs.items():
        for detection in box_list:
            found_class = detection["class_name"].lower()
            
            if found_class == "":
                continue
                
            # Catch both singular and plural versions of the prompt just in case
            if found_class == "animals" or found_class == "animal":
                scored_detections.append({
                    "image_name": image_name,
                    "conf": detection["confidence"],
                    "coords": detection["box"]
                })

    # We must sort the predictions by confidence from highest to lowest.
    # So to  calculate ap properly, we have to judge the model's most confident guesses first.
    scored_detections.sort(key=lambda x: x["conf"], reverse=True)

  
    true_positives = np.zeros(len(scored_detections))
    false_positives = np.zeros(len(scored_detections))

    for i, guess in enumerate(scored_detections):
        img_name = guess["image_name"]
        guess_box = guess["coords"]

        # Pull up the ground truth checklist for this specific image
        gt_data = image_to_gt_map[img_name]
        valid_boxes = gt_data["target_boxes"]

        best_overlap = 0.0
        best_box_index = -1

        # Test our guess against every actual box in the image to find the closest fit
        for j, true_box in enumerate(valid_boxes):
            overlap = get_iou(guess_box, true_box)
            if overlap > best_overlap:
                best_overlap = overlap
                best_box_index = j

        if best_overlap >= MIN_OVERLAP and best_box_index != -1 and not gt_data["already_matched"][best_box_index]:
            true_positives[i] = 1
            gt_data["already_matched"][best_box_index] = True # Cross it off the checklist
        else:
            false_positives[i] = 1



    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)

    
    recalls = cumulative_tp / total_actual_objects
    
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, np.finfo(float).eps)

    # Pad the start and end of the graph so we can calculate the area properly
    recalls_padded = np.concatenate(([0.0], recalls, [1.0]))
    precisions_padded = np.concatenate(([0.0], precisions, [0.0]))

    # Standard AP smoothing: The precision curve zig-zags a lot. We smooth it out by 
    # making sure the line only goes down, never up (monotonically decreasing).
    for i in range(len(precisions_padded) - 1, 0, -1):
        precisions_padded[i - 1] = np.maximum(precisions_padded[i - 1], precisions_padded[i])

    # Find the exact points where the Recall value actually changed
    change_points = np.where(recalls_padded[1:] != recalls_padded[:-1])[0]
    
    # Calculate the Area Under the Curve 
    ap = np.sum((recalls_padded[change_points + 1] - recalls_padded[change_points]) * precisions_padded[change_points + 1])

    final_scores[current_label] = ap
    print(f"Label: {current_label.upper():<12} | Total Targets: {total_actual_objects:<6} | Model Guesses: {len(scored_detections):<6} | AP@50: {ap:.4f}")


if final_scores:
    mean_ap = np.mean(list(final_scores.values()))
    print("-" * 60)
    print(f"Overall mAP@50: {mean_ap:.4f}")