import os
import json
import torch
from PIL import Image, ImageDraw
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from tqdm import tqdm


IMAGE_DIR = "/media/rdr2/RDR2_dataset_processed_test/PNG"
CATEGORY_JSON_PATH = "/media/rdr2/RDR2_dataset_processed_test/gt_Fine_labelIds_mapping.json"

OUTPUT_DIR = "/home/sparackal/owl/Owlvit_fine"
PREDICTIONS_JSON_PATH = os.path.join(OUTPUT_DIR, "owlvit_predictions_fine.json")

MODEL_NAME = "google/owlvit-base-patch32"

THRESHOLD = 0.10
NUM_IMAGES = None
BATCH_SIZE = 4


with open(CATEGORY_JSON_PATH, "r") as f:
    category_data = json.load(f)

RAW_NAMES = [name for name in category_data.keys() if name.lower() != "background"]


def clean_class_name(name):
    return name.replace("_", " ")


TEXTS = [clean_class_name(name) for name in RAW_NAMES]
CLEANED_TO_RAW = dict(zip(TEXTS, RAW_NAMES))


print(f"\nLoaded {len(TEXTS)} fine categories (excluding background):")
print(TEXTS)


os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\nOWL-ViT FINE LABEL FULL DATASET RUN")
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME).to(device)
model.eval()

print("OWL-ViT loaded successfully.")


image_files = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".png")
])

if NUM_IMAGES is not None:
    image_files = image_files[:NUM_IMAGES]

print("\nImages selected:", len(image_files))

if not image_files:
    raise RuntimeError("No PNG images found.")


all_predictions = []

num_batches = (len(image_files) + BATCH_SIZE - 1) // BATCH_SIZE


for start in tqdm(
    range(0, len(image_files), BATCH_SIZE),
    total=num_batches,
    desc="Processing batches"
):

    batch_files = image_files[start:start + BATCH_SIZE]
    images, valid_files = [], []

    for image_path in batch_files:
        try:
            images.append(Image.open(image_path).convert("RGB"))
            valid_files.append(image_path)
        except Exception as e:
            print("ERROR:", image_path, e)

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

    for image_path, image, result in zip(valid_files, images, results):

        filename = os.path.basename(image_path)

        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        for score, box, label in zip(
            result["scores"],
            result["boxes"],
            result["text_labels"]
        ):

            score_value = float(score.cpu())

            x1, y1, x2, y2 = [
                float(x)
                for x in box.cpu().tolist()
            ]

            raw_name = CLEANED_TO_RAW[label]
            class_id = category_data[raw_name]

            all_predictions.append({
                "file_name": filename,
                "FineClassName": raw_name,
                "class_id": class_id,
                "score": round(score_value, 4),
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ],
            })

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

            text = f"{raw_name} {score_value:.3f}"

            try:
                bbox = draw.textbbox((x1, y1), text)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = len(text) * 8, 15

            ty = max(0, y1 - th - 4)

            draw.rectangle(
                [x1, ty, x1 + tw + 6, ty + th + 4],
                fill="red"
            )

            draw.text(
                (x1 + 3, ty + 2),
                text,
                fill="white"
            )

        draw_image.save(
            os.path.join(OUTPUT_DIR, filename),
            format="PNG"
        )


with open(PREDICTIONS_JSON_PATH, "w") as f:
    json.dump(all_predictions, f, indent=2)


print(f"\nSaved {len(all_predictions)} predictions to {PREDICTIONS_JSON_PATH}")
print("TEST COMPLETE")
print("Output images:", OUTPUT_DIR)
print("Predictions JSON:", PREDICTIONS_JSON_PATH)