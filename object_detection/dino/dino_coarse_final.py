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


# ============================================================
# 1. PATHS
# ============================================================

INPUT_FOLDER = (
    "/media/rdr2/RDR2_dataset_processed_test/PNG/"
)

OUTPUT_FOLDER = (
    "/home/sparackal/GroundingDINO/"
    "Output_coarse_final/"
)

PREDICTION_JSON = (
    "/home/sparackal/GroundingDINO/"
    "coarse_predictions_remove_combined.json"
)

JSON_PATH = (
    "/media/rdr2/RDR2_dataset_processed_test/"
    "gt_Coarse_labelIds_mapping.json"
)

CONFIG_PATH = (
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)

WEIGHTS_PATH = (
    "weights/groundingdino_swint_ogc.pth"
)


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 3. THRESHOLDS
# ============================================================

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25


# ============================================================
# 4. REMOVE COMBINED
# ============================================================

REMOVE_COMBINED = True


# ============================================================
# 5. LOAD COARSE CATEGORIES
# ============================================================

if not os.path.exists(JSON_PATH):

    raise FileNotFoundError(
        f"Could not find JSON file: {JSON_PATH}"
    )


with open(
    JSON_PATH,
    "r"
) as f:

    label_dict = json.load(f)


# Remove background
# Replace underscores with spaces

categories = [

    label.replace("_", " ")

    for label in label_dict.keys()

    if label.lower() != "background"

]


# ============================================================
# 6. CREATE TEXT PROMPT
# ============================================================

TEXT_PROMPT = (
    " . ".join(categories)
    + " ."
)


# ============================================================
# 7. PRINT SETTINGS
# ============================================================

print(
    "\n==============================================",
    flush=True
)

print(
    "Grounding DINO - Coarse Categories",
    flush=True
)

print(
    "==============================================",
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


# ============================================================
# 8. LOAD MODEL
# ============================================================

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


# ============================================================
# 9. FIND ALL PNG IMAGES
# ============================================================

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


# ============================================================
# 10. STORAGE FOR PREDICTIONS
# ============================================================

all_predictions = {}


# ============================================================
# 11. PROCESS ALL IMAGES
# ============================================================

for idx, img_path in enumerate(image_paths):

    filename = os.path.basename(
        img_path
    )


    print(
        f"\n[{idx + 1}/{len(image_paths)}] "
        f"{filename}",
        flush=True
    )


    # --------------------------------------------------------
    # Output image path
    # --------------------------------------------------------

    save_path = os.path.join(
        OUTPUT_FOLDER,
        f"det_{filename}"
    )


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image_source, image = load_image(
        img_path
    )


    # --------------------------------------------------------
    # Get original image dimensions
    # --------------------------------------------------------

    h, w, _ = image_source.shape


    # --------------------------------------------------------
    # Grounding DINO prediction
    # --------------------------------------------------------

    boxes, logits, phrases = predict(

        model=model,

        image=image,

        caption=TEXT_PROMPT,

        box_threshold=BOX_THRESHOLD,

        text_threshold=TEXT_THRESHOLD,

        device=device,

        remove_combined=REMOVE_COMBINED

    )


    # --------------------------------------------------------
    # Storage for this image
    # --------------------------------------------------------

    image_predictions = []


    print(
        f"Detected objects: "
        f"{len(boxes)}",
        flush=True
    )


    # ========================================================
    # CONVERT GROUNDING DINO BOXES
    #
    # Grounding DINO returns normalized:
    #
    #     cx, cy, width, height
    #
    # Convert to pixel:
    #
    #     x1, y1, x2, y2
    #
    # Same approach as the animals experiment.
    # ========================================================

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


    # ========================================================
    # PROCESS EACH DETECTION
    # ========================================================

    for box, logit, phrase in zip(

        xyxy,

        logits,

        phrases

    ):

        x1, y1, x2, y2 = box


        # ----------------------------------------------------
        # Create prediction entry
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Print prediction
        # ----------------------------------------------------

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


    # ========================================================
    # STORE PREDICTIONS FOR THIS IMAGE
    # ========================================================

    all_predictions[filename] = (
        image_predictions
    )


    # ========================================================
    # CREATE ANNOTATED IMAGE
    #
    # IMPORTANT:
    # annotate() receives the original normalized
    # Grounding DINO boxes.
    # ========================================================

    annotated_frame = annotate(

        image_source=image_source,

        boxes=boxes,

        logits=logits,

        phrases=phrases

    )


    # ========================================================
    # SAVE ANNOTATED IMAGE
    # ========================================================

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


    # ========================================================
    # SAVE CHECKPOINT EVERY 100 IMAGES
    # ========================================================

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

        print(
            "==============================================",
            flush=True
        )


# ============================================================
# 12. SAVE FINAL PREDICTION JSON
# ============================================================

with open(
    PREDICTION_JSON,
    "w"
) as f:

    json.dump(
        all_predictions,
        f,
        indent=2
    )


# ============================================================
# 13. FINISHED
# ============================================================

print(
    "\n==============================================",
    flush=True
)

print(
    "Finished processing all images!",
    flush=True
)

print(
    "==============================================",
    flush=True
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

print(
    "==============================================",
    flush=True
)