import os
import json
import torch
from PIL import Image, ImageDraw
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from tqdm import tqdm

IMAGE_DIR = "/media/rdr2/RDR2_dataset_processed_test/PNG"

OUTPUT_DIR = "/home/sparackal/owl/OWLViT_animals"

PREDICTIONS_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "owlvit_predictions_animal.json"
)

MODEL_NAME = "google/owlvit-base-patch32"

TEXTS = ["animal"]

THRESHOLD = 0.10
NUM_IMAGES = None
BATCH_SIZE = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

processor = OwlViTProcessor.from_pretrained(MODEL_NAME)

model = OwlViTForObjectDetection.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

image_files = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".png")
])

if NUM_IMAGES is not None:
    image_files = image_files[:NUM_IMAGES]

print("Images selected:", len(image_files))

all_predictions = []

num_batches = (
    len(image_files) + BATCH_SIZE - 1
) // BATCH_SIZE

for start in tqdm(
    range(0, len(image_files), BATCH_SIZE),
    total=num_batches,
    desc="Processing"
):

    batch_files = image_files[
        start:start + BATCH_SIZE
    ]

    images = []
    valid_files = []

    for path in batch_files:
        try:
            images.append(Image.open(path).convert("RGB"))
            valid_files.append(path)
        except Exception as e:
            print("ERROR:", path, e)

    if not images:
        continue

    batch_texts = [TEXTS for _ in images]

    inputs = processor(
        text=batch_texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor(
        [image.size[::-1] for image in images],
        device=device
    )

    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=THRESHOLD,
        text_labels=batch_texts
    )

    for image_path, image, result in zip(
        valid_files,
        images,
        results
    ):

        filename = os.path.basename(image_path)

        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        for score, box, label in zip(
            result["scores"],
            result["boxes"],
            result["text_labels"]
        ):

            score = float(score.cpu())

            box = [
                float(x)
                for x in box.cpu().tolist()
            ]

            all_predictions.append({
                "file_name": filename,
                "class": label,
                "score": round(score, 4),
                "bbox": [
                    round(x, 2)
                    for x in box
                ]
            })

            draw.rectangle(
                box,
                outline="red",
                width=3
            )

            draw.text(
                (box[0], max(0, box[1] - 15)),
                f"{label} {score:.3f}",
                fill="white"
            )

        draw_image.save(
            os.path.join(OUTPUT_DIR, filename)
        )

with open(PREDICTIONS_JSON_PATH, "w") as f:
    json.dump(
        all_predictions,
        f,
        indent=2
    )

print("\nTEST COMPLETE")
print("Images processed:", len(image_files))
print("Total detections:", len(all_predictions))
print("AP50 input JSON:", PREDICTIONS_JSON_PATH)