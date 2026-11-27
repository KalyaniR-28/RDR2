# SPAR: RDR2 Animal Segmentation

This repository contains the implementation and evaluation setup for applying **SPAR (Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation)** to animal segmentation on the Red Dead Redemption 2 (RDR2) dataset.

The experiments use **SPAR MaskCLIP (ViT-B)**, pretrained on 25K images from SA-1B, for binary, coarse-grained, and fine-grained animal segmentation.

## 1. Clone the Repository

Clone the SPAR repository and navigate to its root directory.

```bash
git clone <SPAR_REPOSITORY_URL>
cd SPAR
```

## 2. Environment Setup

Create and activate a Conda environment using Python 3.10.

```bash
conda create -n spar python=3.10 -y
conda activate spar
```

### Install PyTorch

The environment uses PyTorch 2.1.0 with CUDA 11.8.

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

### Install MMCV

```bash
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

### Install the remaining dependencies

```bash
pip install \
    mmengine==0.10.6 \
    mmsegmentation==1.2.2 \
    open_clip_torch==2.29.0 \
    torchmetrics==1.6.0 \
    scikit-learn==1.6.1 \
    openpyxl==3.1.5 \
    pycocotools==2.0.8 \
    numpy==1.26.4 \
    pytorch_lightning==2.5.3
```

## 3. DINOv3 Preparation

Clone the DINOv3 repository:

```bash
git clone https://github.com/facebookresearch/dinov3.git
```

Copy the `dinov3` directory from the cloned repository into the root of the SPAR repository, as required by the implementation.

## 4. Pretrained Model

The experiments use **SPAR MaskCLIP (ViT-B)** pretrained on 25K images from SA-1B.

Download the pretrained SPAR MaskCLIP checkpoint from the official SPAR repository and place it in the `checkpoints/` directory.

Checkpoint used:

```text
checkpoints/spar_checkpoint.pth
```

## 5. RDR2 Dataset Setup

The RDR2 dataset is placed inside the `data/` directory using a symbolic link to avoid duplicating the dataset.

Create the data directory:

```bash
mkdir -p data
```

Create a symbolic link to the dataset:

```bash
ln -n -s </path/to/RDR2_dataset_processed_test data>
```

Replace `/path/to/RDR2_dataset_processed_test` with the actual location of the dataset on your system.

## 6. Evaluation

Animal segmentation is evaluated using custom dataset configurations and evaluation scripts for binary, coarse-grained, and fine-grained segmentation.

Run the evaluation from the root of the SPAR repository.

### Binary segmentation

```bash
python eval.py -model maskclip -dataset cfg_binary_rdr2 -checkpoint checkpoints/spar_checkpoint.pth
```

### Coarse-grained segmentation

```bash
python eval_coarse.py -model maskclip -dataset cfg_coarse_rdr2 -checkpoint checkpoints/spar_checkpoint.pth
```

### Fine-grained segmentation

```bash
python eval_fine.py -model maskclip -dataset cfg_fine_rdr2 -checkpoint checkpoints/spar_checkpoint.pth
```
