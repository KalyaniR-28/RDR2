import os
import sys
import json
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
    IMAGE_DIR = "/media/sparackal/My Passport/RDR2_dataset_processed_test/PNG"
    LABEL_MAPPING_JSON = "configs/gt_Coarse_labelIds_mapping.json"
    OUTPUT_DIR = os.path.join(os.getcwd(), "concat_animal_prompt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(IMAGE_DIR):
        print("[ERROR] Please check that your IMAGE_DIR path is correct.")
        sys.exit(1)

    if not os.path.exists(LABEL_MAPPING_JSON):
        print("[ERROR] Please check that your LABEL_MAPPING_JSON path is correct.")
        sys.exit(1)

    with open(LABEL_MAPPING_JSON, "r") as f:
        label_mapping = json.load(f)

    category_names = [name for name in label_mapping.keys() if name.lower() != "background"]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
    model.eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    images_list = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    images_list.sort()

    PROMPT = ", ".join(category_names)
    PIXEL_CONFIDENCE_THRESHOLD = 0.5

    print(f"--> Testing single concatenated prompt ({len(category_names)} categories) "
          f"on {len(images_list)} sample images...")
    print(f"--> Prompt text: {PROMPT}")

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        for img_name in tqdm(images_list, desc="Testing"):
            try:
                img_path = os.path.join(IMAGE_DIR, img_name)
                raw_image = Image.open(img_path).convert("RGB")
                w, h = raw_image.size

                inputs = processor(images=raw_image, text=PROMPT, return_tensors="pt").to(model.device)
                outputs = model(**inputs)

                # --- DIAGNOSTIC PRINTS ---
                print(f"\n--- {img_name} ---")
                print("semantic_seg shape:", outputs.semantic_seg.shape)
                print("pred_masks shape:", outputs.pred_masks.shape)

                logits = outputs.semantic_seg  # [batch, 1, H, W]
                resized_logits = F.interpolate(
                    logits, size=(h, w), mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)

                prob_map = torch.sigmoid(resized_logits).to(torch.float32).cpu().numpy()

                unique_vals = np.unique(prob_map.round(2))
                print("Number of distinct probability values (rounded):", len(unique_vals))
                print("Sample values:", unique_vals[:10])

                # Build binary mask: white background, black = "one of the 38 categories" detected
                binary_mask = np.full((h, w), 255, dtype=np.uint8)
                binary_mask[prob_map > PIXEL_CONFIDENCE_THRESHOLD] = 0

                out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_concat_test.png")
                Image.fromarray(binary_mask, mode="L").save(out_path)

            except Exception as e:
                print(f"[FRAME ERROR] {img_name}: {str(e)}")

    print(f"\nTest masks saved to: {OUTPUT_DIR}")
    print("Look at these images. If they show clean, class-distinguishable silhouettes,")
    print("that would suggest the concatenation works. If it's one undifferentiated blob")
    print("(or nothing meaningful at all), that confirms concatenation can't replace")
    print("the 38-prompt loop — remember: this single mask has no way to tell you")
    print("WHICH of the 38 categories any given pixel belongs to, even if it 'works'.")


if __name__ == "__main__":
    main()