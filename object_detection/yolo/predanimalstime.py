import os
import json
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLOWorld

inputfolder = "/media/rdr2/RDR2_dataset_processed_test/PNG/"
outputfolder = "./pred_animals_birds_time"

os.makedirs(outputfolder, exist_ok=True)
prompt = ["animal", "bird", ""]
model = YOLOWorld("yolov8x-worldv2.pt")
model.set_classes(prompt)
image_extensions = (".png", ".jpg", ".jpeg")

image_paths = [
    str(p)
    for p in Path(inputfolder).iterdir()
    if p.suffix.lower() in image_extensions
]

print(f"Found {len(image_paths)} images")

# Dictionary to hold detections grouped by daytime suffix
# Example: {"12": {"o_1000003_12": [...]}, "7": {"o_1000003_7": [...]}}
detections_by_daytime = defaultdict(dict)

for img_path in image_paths:
    img_name = Path(img_path).stem
    
    # Extract the daytime suffix (e.g., "12" from "o_1000003_12")
    daytime_suffix = img_name.split('_')[-1]
    
    # Create a subfolder for this specific time of day
    time_folder = os.path.join(outputfolder, f"time_{daytime_suffix}")
    os.makedirs(time_folder, exist_ok=True)

    results = model.predict(img_path, agnostic_nms=True, conf=0.15)

    # Save annotated image into its respective daytime folder
    annotated_path = os.path.join(time_folder, f"{img_name}_prediction.jpg")
    results[0].save(filename=annotated_path)

    detections = []

    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls[0])]
            
            # Skip empty class names if your prompt includes ""
            if class_name == "":
                continue
                
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

    # Add the image's detections to the correct daytime group
    detections_by_daytime[daytime_suffix][img_name] = detections

# Save a separate JSON file for each daytime group in the main output folder
for daytime, dets in detections_by_daytime.items():
    output_json = os.path.join(
        outputfolder,
        f"animal_bird_predictions_time_{daytime}.json"
    )
    with open(output_json, "w") as f:
        json.dump(dets, f, indent=4)
    print(f"Saved {len(dets)} images for time_{daytime} to {output_json}")

print("\nDone!")