# Grounded-SAM: Environment Setup

How to set up the Python environment for running Grounded-SAM animal instance segmentation using Grounding DINO and Segment Anything Model (SAM).

# Create and Activate the Conda Environment

```bash
conda create -n <env_name> python=3.10 -y
conda activate <env_name>
```

# Install PyTorch and TorchVision



```bash
pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```



# Clone and Install Grounding DINO

Clone the official Grounding DINO repository:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
```

Set the CUDA toolkit path:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
```

Install Grounding DINO:

```bash
pip install --no-build-isolation -e .
```

Return to the project root:

```bash
cd ..
```

#Install SAM and Supporting Dependencies

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Install the required supporting packages:

```bash
pip install --no-deps supervision==0.16.0

pip install opencv-python==4.11.0.86  numpy==1.26.4 transformers==4.48.3 tqdm addict yapf timm
```

# Download the Model Checkpoints

```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Keep both checkpoint files in the project root directory, alongside the Python script.

# Run the Segmentation Script

Run the script:

```bash
python -u checkinstanceanimals.py
```

Replace `checkinstanceanimals.py` with the actual Python script filename if it differs.

# Expected Outputs

The pipeline generates the following outputs, depending on the script configuration:

- **Annotated images:** Images containing predicted segmentation masks and bounding boxes.
- **Instance masks:** Individual binary masks for detected animals.
- **`detections.json`:** Grounding DINO detection results.
- **`instance_predictions.json`:** Instance-level mask metadata, confidence scores, and bounding boxes.
