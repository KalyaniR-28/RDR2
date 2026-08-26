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


INPUT_FOLDER = "/media/rdr2/RDR2_dataset_processed_test/PNG/"
OUTPUT_FOLDER = "/home/sparackal/GroundingDINO/Output_Fine_all/"
PREDICTION_JSON = "/home/sparackal/GroundingDINO/fine_predictions_remove_combined.json"

JSON_PATH = "/media/rdr2/RDR2_dataset_processed_test/gt_Fine_labelIds_mapping.json"

CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
REMOVE_COMBINED = True


os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


with open(JSON_PATH, "r") as f:
    label_dict = json.load(f)


categories = [
    label.replace("_", " ")
    for label in label_dict
    if label.lower() != "background"
]


TEXT_PROMPT = " . ".join(categories) + " ."


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(f"Device: {device}")
print(f"Categories: {len(categories)}")


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


if not image_paths:
    raise RuntimeError(
        f"No PNG images found in {INPUT_FOLDER}"
    )


print(f"Images: {len(image_paths)}")


all_predictions = {}


for idx, img_path in enumerate(image_paths):

    filename = os.path.basename(
        img_path
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


    image_predictions = []


    for box, logit, phrase in zip(
        xyxy,
        logits,
        phrases
    ):

        x1, y1, x2, y2 = box

        image_predictions.append({

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

        })


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
            f"Processed "
            f"{idx + 1}/{len(image_paths)}"
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
    f"Finished: "
    f"{len(image_paths)} images"
)

print(
    f"Predictions: "
    f"{PREDICTION_JSON}"
)