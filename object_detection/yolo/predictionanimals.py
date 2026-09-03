import os
import json
from pathlib import Path
from ultralytics import YOLOWorld

inputfolder = "/media/rdr2/RDR2_dataset_processed_test/PNG/"
outputfolder = "./pred_animals_birds"

os.makedirs(outputfolder, exist_ok=True)
prompt = ["animal","bird",""]
model = YOLOWorld("yolov8x-worldv2.pt")
model.set_classes(prompt)
image_extensions = (".png", ".jpg", ".jpeg")

image_paths = [
    str(p)
    for p in Path(inputfolder).iterdir()
    if p.suffix.lower() in image_extensions
]

print(f"Found {len(image_paths)} images")
all_detections = {}
for img_path in image_paths:
    img_name = Path(img_path).stem
    results = model.predict(img_path,agnostic_nms=True,conf=0.15)
    annotated_path = os.path.join(outputfolder,f"{img_name}_prediction.jpg")
    results[0].save(filename=annotated_path)

    detections = []

    for result in results:
        for box in result.boxes:

            class_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            coords = [round(x) for x in box.xyxy[0].tolist()]

            detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "box": coords
            })

            print(
                f"[{img_name}] "
                f"[{class_name.upper()}] "
                f"Conf: {confidence:.2f} "
                f"Box: {coords}"
            )

    all_detections[img_name] = detections
output_json = os.path.join(
    outputfolder,
    "animal_bird_predictions.json"
)

with open(output_json, "w") as f:
    json.dump(all_detections, f, indent=4)

print("\nDone!")
print(f"Predictions saved to {output_json}")