import os
import sys
import json
import gc
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from transformers import Sam3Model, Sam3Processor
except ImportError as e:
    print(f"\n[ERROR] Import failed! Actual error: {e}")
    sys.exit(1)


def main():
    # Paths
    IMAGE_DIR = "/media/sparackal/My Passport/RDR2_dataset_processed_test/PNG"
    LABEL_MAPPING_JSON = "configs/gt_Coarse_labelIds_mapping.json"
    OUTPUT_DIR = os.path.join(os.getcwd(), "coarseloop")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(IMAGE_DIR) or not os.path.exists(LABEL_MAPPING_JSON):
        print("[ERROR] Please check that your IMAGE_DIR and JSON paths are correct.")
        sys.exit(1)

    with open(LABEL_MAPPING_JSON, "r") as f:
        label_mapping = json.load(f)

    bg_label = label_mapping["background"]
    prompts_to_check = [name for name in label_mapping.keys() if name.lower() != "background"]
    print(f"{len(prompts_to_check)} animal categories.")

    # Hardware
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- OFFICIAL TRANSFORMERS LOADING API ---
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    model.eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    # Check images
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    images_list = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    images_list.sort()

    # Calibrated thresholds
    GLOBAL_PRESENCE_THRESHOLD = 0.50   # gate: is this class in the image at all
    PIXEL_CONFIDENCE_THRESHOLD = 0.65  # gate: is this specific pixel confidently this class

    print(f"--> Target Dataset Size: {len(images_list)} images.")
    print(f"--> Commencing semantic label-mask pipeline (Presence*Class Gate "
          f"{GLOBAL_PRESENCE_THRESHOLD}, Pixel Threshold {PIXEL_CONFIDENCE_THRESHOLD})...")
    skipped_count=0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")), torch.inference_mode():
        for index, img_name in enumerate(tqdm(images_list, desc="Generating Semantic Masks")):
            try:
                mask_output_filename = f"{os.path.splitext(img_name)[0]}_semantic.png"
                mask_save_path = os.path.join(OUTPUT_DIR, mask_output_filename)

                # If the file is already there from your previous run, skip it
                if os.path.exists(mask_save_path):
                    skipped_count += 1
                    continue
                img_path = os.path.join(IMAGE_DIR, img_name)
                raw_image = Image.open(img_path).convert("RGB")
                w, h = raw_image.size

                # Canvas init: background label everywhere, remapped to 255 at save time
                semantic_canvas = np.full((h, w), bg_label, dtype=np.uint8)
                logit_tracker = np.full((h, w), -float("inf"), dtype=np.float32)

                # --- Confirmed "Efficient Multi-Prompt Inference" pattern (HF docs) ---
                # Compute vision embeddings ONCE per image, reuse across all 38 prompts.
                # model.forward() explicitly accepts vision_embeds, documented as
                # mutually exclusive with pixel_values, precisely for this reuse case.
                img_inputs = processor(images=raw_image, return_tensors="pt").to(device)
                vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

                for prompt in prompts_to_check:
                    target_id = int(label_mapping[prompt])

                    text_inputs = processor(text=prompt, return_tensors="pt").to(device)
                    outputs = model(vision_embeds=vision_embeds, **text_inputs)

                    # --- 1. RECOGNITION GATE ---
                    # Confirmed formula (transformers source, Sam3ImageSegmentationOutput docstring):
                    #   final_scores = pred_logits.sigmoid() * presence_logits.sigmoid()
                    # pred_logits: (batch, num_queries) per-query class confidence
                    # presence_logits: (batch, 1) whole-image "is this concept present" signal
                    # We take the max combined score across queries as the whole-image
                    # gate for this class/prompt.
                    pred_logits = outputs.pred_logits              # (1, num_queries)
                    presence_logits = outputs.presence_logits       # (1, 1)

                    query_scores = torch.sigmoid(pred_logits) * torch.sigmoid(presence_logits)
                    presence_score = query_scores.max().item()

                    if presence_score < GLOBAL_PRESENCE_THRESHOLD:
                        continue

                    # --- 2. LOCALIZATION: per-pixel confidence from semantic_seg ---
                    logits = outputs.semantic_seg  # [batch, 1, H, W]
                    resized_logits = F.interpolate(
                        logits, size=(h, w), mode="bilinear", align_corners=False
                    ).squeeze(0).squeeze(0)

                    prob_map = torch.sigmoid(resized_logits).to(torch.float32).cpu().numpy()
                    raw_logits_np = resized_logits.to(torch.float32).cpu().numpy()

                    # --- 3. Overlap resolution: threshold AND beat competing classes ---
                    winning_pixels = (prob_map > PIXEL_CONFIDENCE_THRESHOLD) & (raw_logits_np > logit_tracker)

                    if np.any(winning_pixels):
                        semantic_canvas[winning_pixels] = target_id
                        logit_tracker[winning_pixels] = raw_logits_np[winning_pixels]

                # Save result, remapping background label to white (255)
                mask_output_filename = f"{os.path.splitext(img_name)[0]}_semantic.png"
                mask_save_path = os.path.join(OUTPUT_DIR, mask_output_filename)

                output_mask = semantic_canvas.copy()
                output_mask[output_mask == bg_label] = 255

                Image.fromarray(output_mask, mode="L").save(mask_save_path)

                if index % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                with open("pipeline_errors.log", "a") as log_file:
                    log_file.write(f"Error on frame {img_name}: {str(e)}\n")

    print(f"\nCalibrated, overlap-free semantic masks saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()