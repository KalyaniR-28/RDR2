# YOLOE Object Detection

This folder contains YOLOE scripts for animal and coarse-class detection on the RDR2 dataset.

## Requirements

- Conda or Miniconda
- Python 3.10 (tested with 3.10.22)
- Compatible NVIDIA GPU (tested on RTX 3090)

## Create the environment

```bash
conda create -n yoloe python=3.10 -y
conda activate yoloe
```

## Install dependencies

Install PyTorch and TorchVision with CUDA 13.0:

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
```

Install Ultralytics:

```bash
pip install ultralytics==8.3.39
```

## Configure input files

Update the dataset and output paths in the scripts if needed. Place the YOLOE checkpoint at:

```text
pretrain/yoloe-v8l-seg.pt
```

## Run the scripts

Run the prediction scripts first, followed by their AP evaluation scripts:

```bash
python animals_default_prediction.py
python animals_default_ap.py

python coarse_default.py
python coarse_default_ap.py
```

Run these commands from the YOLOE project directory.

## Outputs

The prediction scripts save per-image detections as JSON and annotated images in their configured output folders. The AP scripts evaluate the saved predictions against the RDR2 ground-truth annotations.

## Tested environment

- Python: 3.10.22
- PyTorch: 2.14.1+cu130
- TorchVision: 0.29.1
- Ultralytics: 8.3.39
- CUDA runtime: 13.0
- GPU: NVIDIA GeForce RTX 3090
