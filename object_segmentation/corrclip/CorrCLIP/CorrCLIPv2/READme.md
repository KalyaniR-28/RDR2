# CorrCLIPv2: Open-Vocabulary Segmentation on Custom RDR2 Dataset

This repository provides a reproducible pipeline for performing **zero-shot / open-vocabulary semantic segmentation** on a custom Red Dead Redemption 2 (RDR2) animal dataset using CorrCLIPv2.

The pipeline combines:

- **MetaCLIP (ViT-B/16):** Text-image representation and open-vocabulary recognition.
- **EntitySeg:** Class-agnostic mask proposals.
- **C-RADIOv3:** Visual feature extraction for region-level correspondence.
- **COCO Memory Bank:** Precomputed CLIP and visual feature embeddings used for feature retrieval and correspondence.

The objective is to evaluate open-vocabulary segmentation on custom RDR2 images without training the segmentation model specifically on the target dataset.

---

## 1. Repository Setup

Clone the CorrCLIP repository and navigate to the CorrCLIPv2 directory:

```bash
git clone https://github.com/zdk258/CorrCLIP.git

cd CorrCLIP/CorrCLIPv2
```

## 2. Environment Setup

### 2.1 Create Conda Environment

```bash
conda create -n corrclipv2 python=3.10 -y

conda activate corrclipv2
```

### 2.2 Install PyTorch with CUDA 12.1

```bash
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 Install OpenMMLab Dependencies

```bash
pip install -U openmim

mim install mmengine==0.10.7

mim install "mmcv>=2.0.0"

mim install "mmsegmentation>=1.2.2"
```

### 2.4 Install Additional Dependencies

```bash
pip install faiss-gpu timm opencv-python ftfy regex huggingface_hub
```

**Note:** Ensure that the installed MMCV, FAISS, PyTorch, and CUDA versions are compatible with the available GPU environment.

## 3. Install EntitySeg Dependencies

CorrCLIPv2 uses EntitySeg as its default mask proposal generator. EntitySeg depends on Detectron2, CropFormer, and custom CUDA/C++ extensions.

### 3.1 Install Detectron2

From the CorrCLIPv2 directory, install Detectron2:

```bash
python -m pip install --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git'
```

### 3.2 Build CropFormer EntityAPI

Navigate to the EntityAPI Python API directory:

```bash
cd mask_generators/CropFormer/entity_api/PythonAPI
```

Compile the C++ extensions:

```bash
make
```

Return to the CorrCLIPv2 root directory:

```bash
cd -
```

### 3.3 Build Mask2Former CUDA Operations

Navigate to the Mask2Former pixel decoder operations directory:

```bash
cd mask_generators/CropFormer/mask2former/modeling/pixel_decoder/ops
```

Build and install the custom CUDA extensions:

```bash
python setup.py build install
```

Return to the CorrCLIPv2 root directory:

```bash
cd -
```

**Note:** Run these commands inside the activated Conda environment. Ensure that the CUDA toolkit and compiler are compatible with the installed PyTorch version.

## 4. Pretrained Models and Memory Bank

CorrCLIPv2 requires pretrained model weights and precomputed COCO memory bank embeddings.

Download or prepare the following resources in the CorrCLIPv2 working directory.

### 4.1 MetaCLIP Backbone (ViT-B/16)

Download the pretrained MetaCLIP checkpoint:

```bash
wget -c -O open_clip_pytorch_model.bin \
"https://huggingface.co/timm/vit_base_patch16_clip_224.metaclip_2pt5b/resolve/main/open_clip_pytorch_model.bin"
```

### 4.2 COCO Memory Bank

CorrCLIPv2 uses precomputed COCO memory bank embeddings for CLIP and visual feature correspondence.

Create the required directories:

```bash
mkdir -p memory_bank/clip_embeddings
mkdir -p memory_bank/vfm_embeddings
```

Download the pretrained memory bank files from Hugging Face:

```bash
huggingface-cli download dk258/CorrCLIPv2 \
    memory_bank/clip_embeddings/coco_metaclip_b.pt \
    --repo-type dataset \
    --local-dir .

huggingface-cli download dk258/CorrCLIPv2 \
    memory_bank/clip_embeddings/coco_dfnclip_b.pt \
    --repo-type dataset \
    --local-dir .

huggingface-cli download dk258/CorrCLIPv2 \
    memory_bank/vfm_embeddings/coco_radiov3.faiss \
    --repo-type dataset \
    --local-dir .
```

The downloaded files will be available in the following structure:

```text
memory_bank/
├── clip_embeddings/
│   ├── coco_metaclip_b.pt
│   └── coco_dfnclip_b.pt
│
└── vfm_embeddings/
    └── coco_radiov3.faiss
```

These files provide the precomputed COCO CLIP and C-RADIOv3 visual feature embeddings required by CorrCLIPv2.

### 4.3 EntitySeg Proposal Model

The EntitySeg checkpoint is hosted on Hugging Face and requires accepting the dataset access conditions.

1. Visit the [Adobe EntitySeg Hugging Face repository](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg).
2. Accept the access conditions.
3. Generate a Hugging Face access token with the required permissions.

Download the checkpoint:

```bash
wget --header="Authorization: Bearer YOUR_HF_TOKEN" \
-c -O Mask2Former_hornet_3x_576d0b.pth \
"https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/resolve/main/CropFormer_model/Entity_Segmentation/Mask2Former_hornet_3x/Mask2Former_hornet_3x_576d0b.pth"
```

Replace `YOUR_HF_TOKEN` with your Hugging Face access token.

Keep the token private and do not commit it to GitHub.

## 5. Custom RDR2 Dataset Setup

### 5.1 Dataset Structure

The experiments use the processed RDR2 animal segmentation dataset.

### 5.2 Link the Dataset

Create a data directory inside the CorrCLIPv2 repository:

```bash
mkdir -p data
```

Create a symbolic link to the dataset:

```bash
ln -s <path_to_dataset>
```

Replace `<path_to_dataset>` with the absolute path to your RDR2 dataset.


### 5.3 Define the Custom Vocabulary

Create a text file named `classes_rdr2.txt` inside the `data/` directory.

```bash
cat << 'EOF' > data/classes_rdr2.txt
background
animal
EOF
```

The vocabulary defines two semantic categories:

| Class Index | Class Name |
|---|---|
| 0 | background |
| 1 | animal |

The class order must be consistent with the segmentation labels used by the inference pipeline.

## 6. Custom Dataset Registration

Create a Python file named `custom_datasets.py` in the CorrCLIPv2 working directory.

Register the RDR2 dataset using the MMSegmentation dataset registry.


## 7. Inference Configuration

Create the configuration file:

```text
configs/cfg_rdr2_animals.py
```

Configure the following components:

- **Base configuration:** `base_config.py`
- **Custom dataset:** `RDR2AnimalDataset`
- **Vocabulary:** `data/classes_rdr2.txt`
- **CLIP backbone:** MetaCLIP ViT-B/16
- **Visual feature model:** C-RADIOv3
- **Mask generator:** EntitySeg
- **Memory bank:** COCO
- **Image directory:** `PNG`
- **Ground-truth directory:** `SemanticSegmentationMasks`

The configuration should also define the test dataloader, image preprocessing pipeline, visualization hook, and prediction output path.

## 8. Running Inference

Once the environment, pretrained checkpoints, memory bank, dataset, and configuration are ready, execute:

```bash
python eval.py \
    --config configs/cfg_rdr2_animals.py
```

This runs CorrCLIPv2 inference on the custom RDR2 dataset.

---

## References

- [CorrCLIP Repository](https://github.com/zdk258/CorrCLIP)
- [MetaCLIP ViT-B/16](https://huggingface.co/timm/vit_base_patch16_clip_224.metaclip_2pt5b)
- [Adobe EntitySeg](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [MMEngine](https://github.com/open-mmlab/mmengine)