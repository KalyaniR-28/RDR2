import json
import os
from pathlib import Path
from ultralytics import YOLOWorld

inputfolder = "/media/rdr2/RDR2_dataset_processed_test/PNG/"
outputfolder = "./predfine"
os.makedirs(outputfolder, exist_ok=True)

with open("gt_Fine_labelIds_mapping.json", "r") as file:
    coarsefile = json.load(file)
prompt = []
for label in coarsefile.keys():
    if label != "background":
        prompt.append(label)
prompt.append("")

model = YOLOWorld("yolov8x-worldv2.pt")
model.set_classes(prompt)
image_extensions = (".png", ".jpg", ".jpeg")

image_paths = []
for p in Path(inputfolder).iterdir():
    if p.suffix.lower() in image_extensions:
        image_paths.append(str(p))
        
print(f"Found {len(image_paths)} images in {inputfolder}")
all_detections={}
for img_path in image_paths:
    img_name = Path(img_path).stem 
    results = model.predict(img_path, agnostic_nms=True)
    # Save annotated image
    annotated_path = os.path.join(outputfolder, f"{img_name}_predfine.jpg")
    results[0].save(filename=annotated_path)

    detections = []
    for result in results:
        for box in result.boxes:
            classificationname = model.names[int(box.cls[0])]

            if classificationname == "":
                continue

            final_id = coarsefile[classificationname]
            confidence = float(box.conf[0])
            coords = [round(x) for x in box.xyxy[0].tolist()]

            detections.append({
                "class_name": classificationname,
                "id": final_id,
                "confidence": round(confidence, 4),
                "box": coords
            })
            
            print(f"[{img_name}] [{classificationname.upper()}] ID: {final_id} | Conf: {confidence:.2f} | Box: {coords}")
    all_detections[img_name] = detections
output_json = os.path.join(outputfolder,"fine_predictions.json")
with open(output_json, "w") as f:
    json.dump(all_detections, f, indent=4)
print(f"\nDone/")