# Installation

This codebase is tested using Python 3.8.13 with PyTorch 1.13.0. Follow the below steps to create an environment and install dependencies.

### Setup Virtual Environment

#### Option 1: Use NVIDIA Docker (Recommended)

```bash
# Run base image
docker pull nvcr.io/nvidia/pytorch:22.07-py3
docker run -it --gpus all --ipc=host --rm --name=tc_clip nvcr.io/nvidia/pytorch:22.07-py3

# Setup environment
pip install -r requirements.txt

# Save docker image, ...
```

#### Option 2: Use Conda Environment

```bash
# Create a conda environment
conda create -y --name tc_clip python=3.8

# Activate the environment
conda activate vclip

# Install PyTorch
conda install pytorch==1.13.0 torchvision==0.14.0 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt
```

### Install Apex for Enabling Mixed-Precision Training

**Note:** Ensure that you have the system CUDA of the same version as the PyTorch CUDA version to properly install Apex.

1. Clone the Apex library:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
```

2. Replace the [cached_cast](https://github.com/NVIDIA/apex/blob/810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c/apex/amp/utils.py#L90-L122) function in `apex/amp/utils.py` with the modified version in [scripts/apex_custom_cached_cast.py](../scripts/apex_custom_cached_cast.py). This is to enable multiple forwards during training. See [PR #1282](https://github.com/NVIDIA/apex/pull/1282) for details.

3. Finally, install Apex:

```bash
pip install --upgrade pip
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```
