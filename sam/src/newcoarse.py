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
    OUTPUT_DIR = os.path.join(os.getcwd(), "ccoarseout")
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

    # --- OFFICIAL LOADING API ---
    model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
    model.eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    # Check images
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    images_list = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    images_list.sort()

    # Calibrated thresholds (matches the presence-head-gated pipeline)
    GLOBAL_PRESENCE_THRESHOLD = 0.50
    PIXEL_CONFIDENCE_THRESHOLD = 0.65

    print(f"--> Target Dataset Size: {len(images_list)} images.")
    print(f"--> Commencing semantic label-mask pipeline (Presence Head {GLOBAL_PRESENCE_THRESHOLD}, "
          f"Pixel Threshold {PIXEL_CONFIDENCE_THRESHOLD})...")

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        for index, img_name in enumerate(tqdm(images_list, desc="Generating Semantic Masks")):
            try:
                img_path = os.path.join(IMAGE_DIR, img_name)
                raw_image = Image.open(img_path).convert("RGB")
                w, h = raw_image.size

                # Canvas init: background label everywhere, remapped to 255 at save time
                semantic_canvas = np.full((h, w), bg_label, dtype=np.uint8)
                logit_tracker = np.full((h, w), -float("inf"), dtype=np.float32)

                # --- OFFICIAL "Efficient Multi-Prompt Inference" pattern ---
                # Precompute vision embeddings once per image
                img_inputs = processor(images=raw_image, return_tensors="pt").to(model.device)
                vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

                for prompt in prompts_to_check:
                    target_id = int(label_mapping[prompt])

                    text_inputs = processor(text=prompt, return_tensors="pt").to(model.device)
                    outputs = model(vision_embeds=vision_embeds, **text_inputs)

                    # 1. RECOGNITION: presence head gate (decoupled recognition/localization)
                    presence_logit = outputs.presence_logits.squeeze()
                    presence_score = torch.sigmoid(presence_logit).item()
                    if presence_score < GLOBAL_PRESENCE_THRESHOLD:
                        continue

                    # 2. LOCALIZATION: semantic_seg -> resize -> sigmoid
                    logits = outputs.semantic_seg  # [batch, 1, H, W]
                    resized_logits = F.interpolate(
                        logits, size=(h, w), mode="bilinear", align_corners=False
                    ).squeeze(0).squeeze(0)

                    prob_map = torch.sigmoid(resized_logits).to(torch.float32).cpu().numpy()
                    raw_logits_np = resized_logits.to(torch.float32).cpu().numpy()

                    # 3. Overlap resolution: threshold AND beat competing classes
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

    print(f"\nCalibrated, overlap-free masks saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()