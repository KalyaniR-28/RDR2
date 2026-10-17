## YOLO-World: Environment Setup
The following instructions describe how to set up a separate Conda environment to run YOLO-World.

# Create the Conda environment and activate it

conda create -n yoloenv python=3.11 -y
conda activate yoloenv


# Install the dependencies
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install Ultralytics:
pip install ultralytics==8.4.112


# Run the prediction script
Navigate to the directory containing predictionanimals.py and run:
python predictionanimals.py


# Note: The PyTorch command above installs CUDA 12.1 builds.
