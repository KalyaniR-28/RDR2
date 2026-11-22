
import os
import json
from pathlib import Path
from ultralytics import YOLOE

inputfolder = "/media/rdr2/RDR2_dataset_processed_test/PNG/"
outputfolder = "./pred_coarse_default"
os.makedirs(outputfolder, exist_ok=True)
with open(
    "/media/rdr2/RDR2_dataset_processed_test/gt_Coarse_labelIds_mapping.json",
    "r"
) as f:
    coarse_mapping = json.load(f)

coarse_mapping = {
    name: class_id
    for name, class_id in coarse_mapping.items()
    if name.lower() != "background"
}

prompts = list(coarse_mapping.keys())

model = YOLOE("pretrain/yoloe-v8l-seg.pt")
model.set_classes(prompts, model.get_text_pe(prompts))

image_paths = sorted([
    p for p in Path(inputfolder).iterdir()
    if p.suffix.lower() == ".png"
])

all_detections = {}

for img_path in image_paths:

    img_name = img_path.stem

    results = model.predict(
        str(img_path),
        conf=0.25,
        verbose=False
    )

    results[0].save(
        filename=os.path.join(
            outputfolder,
            f"{img_name}_prediction.jpg"
        )
    )

    detections = []

    if results[0].boxes is not None:
        for box in results[0].boxes:

            class_index = int(box.cls[0])
            class_name = prompts[class_index]
            confidence = float(box.conf[0])
            class_id = coarse_mapping[class_name]
            coords = [round(x) for x in box.xyxy[0].tolist()]

            detections.append({
                "class_name": class_name,
                "id": class_id,
                "confidence": round(confidence, 4),
                "box": coords
            })

    all_detections[img_name] = detections

output_json = os.path.join(
    outputfolder,
    "coarse_predictions.json"
)

with open(output_json, "w") as f:
    json.dump(all_detections, f, indent=4)

print(f"Images processed: {len(all_detections)}")
print(f"Total detections: {sum(len(v) for v in all_detections.values())}")
print(f"Predictions saved to: {output_json}")
