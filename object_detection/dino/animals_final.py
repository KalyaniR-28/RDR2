import os
import glob
import json
import torch
from PIL import Image
from torchvision.ops import box_convert

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate
)

INPUT_FOLDER = (
    "/media/rdr2/RDR2_dataset_processed_test/PNG/"
)

OUTPUT_FOLDER = (
    "/home/sparackal/GroundingDINO/Output_animals/"
)

PREDICTION_JSON = (
    "/home/sparackal/GroundingDINO/"
    "animals_predictions.json"
)

CONFIG_PATH = (
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)

WEIGHTS_PATH = (
    "weights/groundingdino_swint_ogc.pth"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

TEXT_PROMPT = "animal"

print("\n==============================================")
print("Grounding DINO - Animal Detection")
print("==============================================")

print(
    f"Prompt          : {TEXT_PROMPT}"
)

print(
    f"Box threshold   : {BOX_THRESHOLD}"
)

print(
    f"Text threshold  : {TEXT_THRESHOLD}"
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"\nRunning on device: {device}"
)

print(
    "Loading Grounding DINO model..."
)

model = load_model(
    CONFIG_PATH,
    WEIGHTS_PATH,
    device=device
)

image_paths = sorted(
    glob.glob(
        os.path.join(
            INPUT_FOLDER,
            "*.png"
        )
    )
)

print(
    f"\nFound {len(image_paths)} images to process."
)

all_predictions = {}

for idx, img_path in enumerate(image_paths):

    filename = os.path.basename(img_path)

    print(
        f"\n[{idx + 1}/{len(image_paths)}] "
        f"{filename}"
    )

    save_path = os.path.join(
        OUTPUT_FOLDER,
        f"det_{filename}"
    )

    image_source, image = load_image(
        img_path
    )

    h, w, _ = image_source.shape

    boxes, logits, phrases = predict(

        model=model,

        image=image,

        caption=TEXT_PROMPT,

        box_threshold=BOX_THRESHOLD,

        text_threshold=TEXT_THRESHOLD,

        device=device

    )

    image_predictions = []


    print(
        f"Detected objects: "
        f"{len(boxes)}"
    )

    boxes_pixel = (
        boxes
        * torch.Tensor(
            [w, h, w, h]
        )
    )


    xyxy = box_convert(
        boxes=boxes_pixel,
        in_fmt="cxcywh",
        out_fmt="xyxy"
    ).numpy()

    for box, logit, phrase in zip(
        xyxy,
        logits,
        phrases
    ):

        x1, y1, x2, y2 = box

        prediction = {

            "class": phrase,

            "confidence": float(
                logit
            ),

            "bbox": [

                float(x1),
                float(y1),
                float(x2),
                float(y2)

            ]

        }


        image_predictions.append(
            prediction
        )


        print(

            f"    {phrase:<20} "

            f"confidence: "
            f"{float(logit):.3f} "

            f"bbox: "

            f"["
            f"{float(x1):.1f}, "
            f"{float(y1):.1f}, "
            f"{float(x2):.1f}, "
            f"{float(y2):.1f}"
            f"]"

        )

    all_predictions[filename] = (
        image_predictions
    )

    annotated_frame = annotate(

        image_source=image_source,

        boxes=boxes,

        logits=logits,

        phrases=phrases

    )

    Image.fromarray(
        annotated_frame
    ).save(save_path)


    print(
        f"    Saved → {save_path}"
    )

    if (idx + 1) % 100 == 0:

        with open(
            PREDICTION_JSON,
            "w"
        ) as f:

            json.dump(
                all_predictions,
                f,
                indent=2
            )


        print(
            f"\n    Checkpoint saved "
            f"after {idx + 1} images."
        )

with open(
    PREDICTION_JSON,
    "w"
) as f:

    json.dump(
        all_predictions,
        f,
        indent=2
    )
print("Finished processing all images!")

print(
    f"Total images processed: "
    f"{len(image_paths)}"
)

print(
    "\nAnnotated images:"
)

print(
    OUTPUT_FOLDER
)

print(
    "\nPrediction JSON:"
)

print(
    PREDICTION_JSON
)
