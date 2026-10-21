import os
import glob
import json
from pathlib import Path
import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from GroundingDINO.groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = ("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

INPUT_FOLDER = "/media/rdr2/RDR2_dataset_processed_test/PNG/"
OUTPUT_FOLDER = "./gsam_results_instance_animals"

DIR_ANNOTATED = os.path.join(OUTPUT_FOLDER, "annotated_sam")
DIR_MASKS = os.path.join(OUTPUT_FOLDER, "masks")

os.makedirs(DIR_ANNOTATED, exist_ok=True)
os.makedirs(DIR_MASKS, exist_ok=True)

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Predict classes and hyper-param for GroundingDINO
CLASSES = ["animals"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    sam_scores = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        result_masks.append(masks[best_idx])
        sam_scores.append(float(scores[best_idx]))
        
    if len(result_masks) == 0:
        height, width = image.shape[:2]
        return (
            np.zeros((0, height, width), dtype=bool),
            np.zeros((0,), dtype=np.float32),
        )
    return (np.stack(result_masks, axis=0),np.asarray(sam_scores, dtype=np.float32))

image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

print(f"Total {len(image_paths)} images to check.")

all_dino_detections = {}
all_instance_predictions = {}
mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()

for image_index, image_path in enumerate(image_paths, start=1):
    stem = Path(image_path).stem
    print(f"[{image_index}/{len(image_paths)}] Processing {stem}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"WARNING: Could not read image: {image_path}")
        all_dino_detections[stem] = []
        all_instance_predictions[stem] = []
        continue

    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    if len(detections.xyxy) > 0:
        nms_indices = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD,
        ).cpu().numpy()

        detections.xyxy = detections.xyxy[nms_indices]
        detections.confidence = detections.confidence[nms_indices]
        detections.class_id = detections.class_id[nms_indices]

    number_of_detections = len(detections.xyxy)

    print(f"  Grounding DINO detections: {number_of_detections}")
    all_dino_detections[stem] = [
        {
            "class_name": CLASSES[int(class_id)],
            "confidence": round(float(confidence), 4),
            "box": [int(round(c)) for c in box]}
        for box, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id)
        ]

    if number_of_detections == 0:
# Save the original image in the SAM annotation folder
        cv2.imwrite(os.path.join(DIR_ANNOTATED,f"{stem}_sam_annotated.jpg",),image,)
        all_instance_predictions[stem] = []
        print("  No detections; no instance masks saved.")
        continue
    
    instance_masks, sam_scores = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # Attach individual masks to detections for visualization
    detections.mask = instance_masks
    all_instance_predictions[stem] = []

    for instance_index, (mask,box,confidence,class_id,sam_score,) in enumerate(zip(instance_masks,detections.xyxy,detections.confidence,detections.class_id,sam_scores,)):
        
        instance_mask = mask.astype(np.uint8)
        mask_filename = (f"{stem}_instance_{instance_index}.png")
        mask_path = os.path.join(DIR_MASKS,mask_filename,)
        cv2.imwrite(mask_path, instance_mask)

        # Save metadata linking the mask to its detection
        all_instance_predictions[stem].append(
            {
                "mask_file": mask_filename,
                "instance_id": instance_index,
                "class_name": CLASSES[int(class_id)],
                "class_id": int(class_id),
                "confidence": round(float(confidence), 4),
                "sam_score": round(float(sam_score), 4),
                "box": [int(round(float(c))) for c in box],
            }
        )
    
    labels = [(f"Instance {i} | "f"{CLASSES[int(class_id)]} "f"{float(confidence):.2f}")
        for i, (class_id, confidence) in enumerate(zip(detections.class_id,detections.confidence,))
    ]

    annotated_sam = mask_annotator.annotate(scene=image.copy(),detections=detections,)
    annotated_sam = box_annotator.annotate(scene=annotated_sam,detections=detections,labels=labels,)

    cv2.imwrite(os.path.join(DIR_ANNOTATED,f"{stem}_sam_annotated.jpg",),annotated_sam,)
    print(f"  Saved {len(instance_masks)} individual instance masks.")


with open(os.path.join(OUTPUT_FOLDER, "detections.json"), "w", encoding="utf-8") as f:
    json.dump(all_dino_detections, f, indent=4)

with open(os.path.join(OUTPUT_FOLDER, "instance_predictions.json"), "w", encoding="utf-8") as f:
    json.dump(all_instance_predictions, f, indent=4)

print("\nAll images processed successfully.")

print(f"Individual instance masks: {DIR_MASKS}")
print(f"SAM annotations: {DIR_ANNOTATED}")

