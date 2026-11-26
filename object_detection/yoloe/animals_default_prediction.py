#animalsefault
import os
import json
from pathlib import Path
from ultralytics import YOLOE

inputfolder = "/media/rdr2/RDR2_dataset_processed_test/PNG/"
outputfolder = "./pred_animals_default"

os.makedirs(outputfolder, exist_ok=True)

prompt = ["animals"]
model = YOLOE("pretrain/yoloe-v8l-seg.pt")
model.set_classes(prompt, model.get_text_pe(prompt))

image_extensions = (".png", ".jpg", ".jpeg")

image_paths = sorted([
    str(p)
    for p in Path(inputfolder).iterdir()
    if p.suffix.lower() in image_extensions
])

all_detections = {}

for img_path in image_paths:

    img_name = Path(img_path).stem

    results = model.predict(
        img_path,
        conf=0.25,
        verbose=False
    )

    results[0].save(
        filename=os.path.join(outputfolder, f"{img_name}_prediction.jpg")
    )

    detections = []

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:

                class_name = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                coords = [round(x) for x in box.xyxy[0].tolist()]

                detections.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "box": coords
                })

    all_detections[img_name] = detections

output_json = os.path.join(
    outputfolder,
    "animals_predictions.json"
)

with open(output_json, "w") as f:
    json.dump(all_detections, f, indent=4)

print(f"Processed images: {len(all_detections)}")
print(f"Total detections: {sum(len(v) for v in all_detections.values())}")
print(f"Predictions saved to: {output_json}")
