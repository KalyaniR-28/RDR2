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
# 4. PROMPT
# ============================================================

TEXT_PROMPT = "animal"


# ============================================================
# 5. PRINT SETTINGS
# ============================================================

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


# ============================================================
# 6. LOAD MODEL
# ============================================================

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


# ============================================================
# 7. FIND ALL PNG IMAGES
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
    f"\nFound {len(image_paths)} images to process."
)


# ============================================================
# 8. STORAGE FOR PREDICTIONS
# ============================================================

all_predictions = {}


# ============================================================
# 9. PROCESS ALL IMAGES
# ============================================================

for idx, img_path in enumerate(image_paths):

    filename = os.path.basename(img_path)

    print(
        f"\n[{idx + 1}/{len(image_paths)}] "
        f"{filename}"
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

        device=device

    )


    # --------------------------------------------------------
    # Storage for this image
    # --------------------------------------------------------

    image_predictions = []


    print(
        f"Detected objects: "
        f"{len(boxes)}"
    )


    # ========================================================
    # CONVERT GROUNDING DINO BOXES
    #
    # Grounding DINO returns normalized boxes in:
    #
    #     cx, cy, width, height
    #
    # Convert them to pixel coordinates:
    #
    #     x1, y1, x2, y2
    #
    # This follows the conversion approach used in the
    # Grounding DINO inference implementation.
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
    # annotate() expects the original normalized
    # Grounding DINO boxes, not xyxy pixel boxes.
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
    ).save(save_path)


    print(
        f"    Saved → {save_path}"
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
            f"\n    Checkpoint saved "
            f"after {idx + 1} images."
        )


# ============================================================
# 10. SAVE FINAL PREDICTION JSON
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
# 11. FINISHED
# ============================================================

print("\n==============================================")
print("Finished processing all images!")
print("==============================================")

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

print("==============================================")