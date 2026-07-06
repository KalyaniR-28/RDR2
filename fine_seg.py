import os
import sys
import json
import gc
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# 1. Environment and Path Verification
try:
    import sam3
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

# To THIS:
except ImportError as e:
    print(f"\n[ERROR] Import failed! Actual error: {e}")
    import traceback
sys.exit(1)

def main():
    # 2. Paths Configuration
    IMAGE_DIR = "/media/sparackal/My Passport/RDR2_dataset_processed_test/PNG"
    LABEL_MAPPING_JSON = "gt_Fine_labelIds_mapping.json"
    OUTPUT_DIR = os.path.join(os.getcwd(), "fineout")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(IMAGE_DIR) or not os.path.exists(LABEL_MAPPING_JSON):
        print("[ERROR] Please check that your IMAGE_DIR and JSON paths are correct.")
        sys.exit(1)

    # 3. Load Integer Label Mappings
    with open(LABEL_MAPPING_JSON, "r") as f:
        label_mapping = json.load(f)
    
    bg_label = label_mapping["background"]
    prompts_to_check = [name for name in label_mapping.keys() if name.lower() != "background"]
    print(f"--> Loaded {len(prompts_to_check)} animal categories.")

    # 4. Initialize Hardware
    print("--> Initializing SAM 3 on GPU...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_sam3_image_model()
    model.to("cuda")
    model.eval()
    processor = Sam3Processor(model)

    # Gather images
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    images_list = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    images_list.sort()
    
    print(f"--> Target Dataset Size: {len(images_list)} images.")

    # 5. Semantic Inference Loop with Confidence Battle
    print("--> Commencing semantic label-mask pipeline (Resolving overlaps)...")
    
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        for index, img_name in enumerate(tqdm(images_list, desc="Generating ID Masks")):
            try:
                img_path = os.path.join(IMAGE_DIR, img_name)
                raw_image = Image.open(img_path).convert("RGB")
                w, h = raw_image.size
                
                # Step A: Initialize winning canvas and confidence canvas
                semantic_canvas = np.full((h, w), bg_label, dtype=np.uint8)
                confidence_canvas = np.zeros((h, w), dtype=np.float32)
                
                # Step B: Compute image embeddings
                inference_state = processor.set_image(raw_image)
                
                # Step C: Check every animal
                for prompt in prompts_to_check:
                    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                    masks = output["masks"]
                    scores = output.get("scores", output.get("iou_predictions", None))
                    
                    if scores is None:
                        scores = torch.ones(masks.shape[0], device=masks.device)
                    
                    if masks.shape[0] > 0:
                        if masks.ndim == 4:
                            masks = masks.squeeze(1)

                        target_id = int(label_mapping[prompt])

                        # Step D: Battle for the pixels based on confidence
                        for i in range(masks.shape[0]):
                            mask_np = masks[i].cpu().numpy()
                            mask_score = scores[i].item()
                            
                            # Only overwrite if the AI is more confident about this animal
                            winning_pixels = mask_np & (mask_score > confidence_canvas)
                            
                            semantic_canvas[winning_pixels] = target_id
                            confidence_canvas[winning_pixels] = mask_score

                # Step E: Save result
                mask_output_filename = f"{os.path.splitext(img_name)[0]}_semantic.png"
                mask_save_path = os.path.join(OUTPUT_DIR, mask_output_filename)
                Image.fromarray(semantic_canvas, mode="L").save(mask_save_path)

                # Memory Safeguard
                if index % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                                        
            except Exception as e:
                with open("pipeline_errors.log", "a") as log_file:
                    log_file.write(f"Error on frame {img_name}: {str(e)}\n")

    print(f"\n--> Success! Overlap-free masks saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()