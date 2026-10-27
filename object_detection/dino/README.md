# Grounding DINO Object Detection

This section explains how to set up and run Grounding DINO for zero-shot object detection.

## Create and Activate the Conda Environment

Create a Conda environment with Python 3.10 and activate it.

```bash
conda create -n gdino_env python=3.10 -y
conda activate gdino_env
```

## Install PyTorch and TorchVision

Install PyTorch with CUDA 12.1 support.

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Verify the installation and check CUDA availability.

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Install Required Dependencies

Clone the official Grounding DINO repository and install its dependencies.

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO

pip install -r requirements.txt
pip install -e .
```

If the repository is already cloned, navigate to its directory before installing the dependencies.

## Download and Load the Grounding DINO Model

Download the pretrained Grounding DINO Swin-T checkpoint from the official [Grounding DINO repository](https://github.com/IDEA-Research/GroundingDINO) and place it in the `weights` directory.

The configuration and checkpoint paths should be:

```text
groundingdino/config/GroundingDINO_SwinT_OGC.py
weights/groundingdino_swint_ogc.pth
```

Load the model using the following Python code:

```python
from groundingdino.util.inference import load_model

model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)
```

## Dataset and Annotation Paths

Update the dataset and annotation paths in the detection and evaluation scripts according to the location of your dataset.

The required files may include:

```text
dataset/
├── PNG/
├── all_captures.json
├── gt_Coarse_labelIds_mapping.json
└── gt_Fine_labelIds_mapping.json
```

Ensure that the image directory, annotation files, prediction JSON files, and output directories match the paths specified in the scripts.

## Run the Detection Scripts

Run the required detection scripts from the project directory.

```bash
python dino_coarse_final.py
python dino_fine_final.py
python animals_final.py
```

The scripts perform object detection using Grounding DINO. Make sure the model configuration, checkpoint, dataset paths, and output directories are correctly configured before running them.

### Detection Parameters

The following parameters can be adjusted in the detection scripts:

```python
TEXT_PROMPT = "animal . bird"
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
```

- `TEXT_PROMPT`: Specifies the text categories used to detect objects.
- `BOX_THRESHOLD`: Sets the minimum confidence threshold for predicted bounding boxes.
- `TEXT_THRESHOLD`: Sets the minimum text-matching threshold.

Adjust the prompt and thresholds according to the detection task.

## Evaluation Using TorchMetrics

TorchMetrics is used to evaluate the detection predictions against the ground-truth annotations. The evaluation scripts use `MeanAveragePrecision` from `torchmetrics.detection.mean_ap`.

Install the required evaluation dependencies:

```bash
pip install torchmetrics
pip install pycocotools
```

The evaluation scripts compare predicted bounding boxes with ground-truth bounding boxes and calculate standard object detection metrics.

### Evaluation Scripts

Run the following scripts from the project directory:

**1. Animal Detection Evaluation**

```bash
python ap_animals_new.py
```

Evaluates animal detection predictions against the ground-truth annotations.

**2. Coarse-Grained Classification Evaluation**

```bash
python ap_new_coarse.py
```

Evaluates predictions using coarse-grained animal categories.

**3. Fine-Grained Classification Evaluation**

```bash
python ap_new_fine.py
```

Evaluates predictions using fine-grained animal categories.

### Evaluation Metrics

The evaluation scripts report the following metrics, where supported by the evaluation configuration:

- **mAP (IoU 0.50:0.95):** Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95.
- **mAP@50:** Average Precision at an IoU threshold of 0.50.
- **mAP@75:** Average Precision at an IoU threshold of 0.75.
- **mAP for small, medium, and large objects:** Average Precision grouped by object size.
- **Recall@100:** Average recall with a maximum of 100 detections per image.
- **Per-class AP:** Average Precision for individual classes, when class-wise metrics are enabled and reported.

Before running the evaluation scripts, ensure that the prediction JSON files exist, the ground-truth annotations are accessible, and the paths in each script are correctly configured.