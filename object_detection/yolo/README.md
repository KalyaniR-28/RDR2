## YOLO-World Animal Detection

This section explains how to set up and run YOLO-World for animal detection. These instructions apply only to YOLO-World and do not require installing the dependencies for SAM or GroundingDINO.

## Requirements

- Conda or Miniconda
- Python 3.11
- A compatible NVIDIA GPU and driver for GPU acceleration (optional)

## Create the Conda Environment and activate

```bash
conda create -n yoloenv python=3.11 -y
conda activate yoloenv
```

## Install Dependencies

Install PyTorch and TorchVision:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Install Ultralytics:

```bash
pip install ultralytics==8.4.112
```

The PyTorch command installs CUDA 12.1 builds. For CPU-only execution, install the appropriate CPU builds instead.

### 4. Configure Input Images

Update the input image directory or image paths in `predictionanimals.py` to match your local setup.

### 5. Run the Script

Navigate to the directory containing `predictionanimals.py` and execute:

```bash
python predictionanimals.py
```

The first run may download model weights or other required resources.

### 6. Outputs

The script generates animal detection predictions. Check `predictionanimals.py` for the configured output directory and output file formats.
