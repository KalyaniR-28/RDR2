# OWL-ViT: Environment Setup and Object Detection

This guide explains how to set up the Python environment and run OWL-ViT for animal detection on the RDR2 dataset.

## 1. Create and Activate the Conda Environment

```bash
conda create -n owlvit python=3.11 -y
conda activate owlvit
```

## 2. Install PyTorch and TorchVision

Install the CUDA 12.1-compatible versions:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## 3. Install Required Dependencies

```bash
pip install transformers==5.17.0 accelerate==1.15.0 pillow==12.3.0 matplotlib==3.11.2 numpy==2.4.6 pycocotools==2.0.11
```

## 4. Download and Load the OWL-ViT Model

The scripts use the pretrained `google/owlvit-base-patch32` model from Hugging Face. The model weights and processor are downloaded automatically the first time `from_pretrained()` is called.

```python
from transformers import OwlViTProcessor, OwlViTForObjectDetection

model_name = "google/owlvit-base-patch32"

processor = OwlViTProcessor.from_pretrained(model_name)
model = OwlViTForObjectDetection.from_pretrained(model_name)
```

No separate checkpoint download is required. Internet access is needed for the initial download unless the model is already cached locally.

## 5. Dataset and Annotation Files

The scripts use the following RDR2 dataset files on Pepper:

- **Images:** `/media/rdr2/RDR2_dataset_processed_test/PNG`
- **Ground-truth annotations:** `/media/rdr2/RDR2_dataset_processed_test/all_captures.json`
- **Coarse label mapping:** `/media/rdr2/RDR2_dataset_processed_test/gt_Coarse_labelIds_mapping.json`
- **Fine label mapping:** `/media/rdr2/RDR2_dataset_processed_test/gt_Fine_labelIds_mapping.json`

## 6. Run the Detection Scripts

Navigate to the OWL-ViT directory from the repository root:

```bash
cd object_detection/owl
```

Run animal detection:

```bash
python owlvit_animals.py
```

Run coarse-level detection:

```bash
python owl_coarse_full.py
```

Run fine-level detection:

```bash
python owl_vit_fine.py
```
