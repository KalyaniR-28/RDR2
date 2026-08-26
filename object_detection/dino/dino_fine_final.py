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
    "/home/sparackal/GroundingDINO/"
    "Output_Fine_all/"
)

PREDICTION_JSON = (
    "/home/sparackal/GroundingDINO/"
    "fine_predictions_remove_combined.json"
)

JSON_PATH = (
    "/media/rdr2/RDR2_dataset_processed_test/"
    "gt_Fine_labelIds_mapping.json"
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

REMOVE_COMBINED = True

if not os.path.exists(JSON_PATH):

    raise FileNotFoundError(
        f"Could not find JSON file: {JSON_PATH}"
    )


with open(
    JSON_PATH,
    "r"
) as f:

    label_dict = json.load(f)

categories = [

    label.replace("_", " ")

    for label in label_dict.keys()

    if label.lower() != "background"

]


TEXT_PROMPT = (
    " . ".join(categories)
    + " ."
)


print(
    "Grounding DINO - Fine Categories",
    flush=True
)

print(
    f"\nNumber of categories: "
    f"{len(categories)}",
    flush=True
)


print(
    "\nCategories:",
    flush=True
)

for i, category in enumerate(categories):

    print(
        f"{i:2d}: {category}",
        flush=True
    )


print(
    "\nGenerated prompt:",
    flush=True
)

print(
    TEXT_PROMPT,
    flush=True
)


print(
    "\nExperiment settings:",
    flush=True
)

print(
    f"Box threshold      : "
    f"{BOX_THRESHOLD}",
    flush=True
)

print(
    f"Text threshold     : "
    f"{TEXT_THRESHOLD}",
    flush=True
)

print(
    f"Remove combined    : "
    f"{REMOVE_COMBINED}",
    flush=True
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"\nRunning on device: "
    f"{device}",
    flush=True
)


print(
    "Loading Grounding DINO model...",
    flush=True
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
    f"\nFound {len(image_paths)} images to process.",
    flush=True
)


if len(image_paths) == 0:

    raise RuntimeError(
        f"No PNG images found in: "
        f"{INPUT_FOLDER}"
    )

all_predictions = {}


for idx, img_path in enumerate(image_paths):

    filename = os.path.basename(
        img_path
    )


    print(
        f"\n[{idx + 1}/{len(image_paths)}] "
        f"{filename}",
        flush=True
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

        device=device,

        remove_combined=REMOVE_COMBINED

    )

    image_predictions = []


    print(
        f"Detected objects: "
        f"{len(boxes)}",
        flush=True
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

            f"    {phrase:<30} "

            f"confidence: "
            f"{float(logit):.3f} "

            f"bbox: "

            f"["
            f"{float(x1):.1f}, "
            f"{float(y1):.1f}, "
            f"{float(x2):.1f}, "
            f"{float(y2):.1f}"
            f"]",

            flush=True

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
    ).save(
        save_path
    )


    print(
        f"    Saved → "
        f"{save_path}",
        flush=True
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
            "\n==============================================",
            flush=True
        )

        print(
            f"Checkpoint saved after "
            f"{idx + 1} images.",
            flush=True
        )

        print(
            f"Prediction JSON: "
            f"{PREDICTION_JSON}",
            flush=True
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

print(
    f"Total images processed: "
    f"{len(image_paths)}",
    flush=True
)

print(
    "\nAnnotated images:",
    flush=True
)

print(
    OUTPUT_FOLDER,
    flush=True
)

print(
    "\nPrediction JSON:",
    flush=True
)

print(
    PREDICTION_JSON,
    flush=True
)