import os
import numpy as np
import torch
from PIL import Image
from glob import glob
from tqdm import tqdm  # Nice progress bar for batch processing
from transformers import Sam3Processor, Sam3Model

# 1. Setup paths and directories
dataset_dir = "/media/sparackal/My Passport/RDR2_dataset_processed_test/PNG"
output_dir = "animalsinstant"
os.makedirs(output_dir, exist_ok=True)

# 2. Setup device and load authenticated model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# 3. Get all images from the dataset folder
valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(valid_extensions)]

print(f"Found {len(image_files)} images to process.")

# 4. Loop through each image
for filename in tqdm(image_files, desc="Segmenting Dataset"):
    try:
        # Load image
        image_path = os.path.join(dataset_dir, filename)
        image_pil = Image.open(image_path).convert("RGB")
        W, H = image_pil.size

        # Generate input tensors with text prompt (captures people, horses, wildlife)
        inputs = processor(images=image_pil, text="animals", return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process outputs to native pixel resolution
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,       
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        # Create an inverted background canvas (Start with all WHITE)
        predicted_mask = np.full((H, W), 255, dtype=np.uint8)

        # Overwrite detected masks to BLACK (0)
        if len(results["masks"]) > 0:
            for mask in results["masks"]:
                bool_mask = mask.cpu().numpy().astype(bool)
                predicted_mask[bool_mask] = 0

        # Save predicted mask with the same filename but as a PNG mask
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}_mask.png")
        
        output_image = Image.fromarray(predicted_mask)
        output_image.save(output_path)

    except Exception as e:
        print(f"\nError processing {filename}: {e}")

print(f"\nProcessing complete! All masks saved to: {output_dir}")