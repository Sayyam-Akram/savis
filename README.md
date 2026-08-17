# SAVIS: Audio-Visual Instance Segmentation

[![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2603.01431) <!-- Update this link with your paper URL -->
[![Dataset](https://img.shields.io/badge/Dataset-AVISeg-blue.svg)](https://onedrive.live.com/?id=%2Fpersonal%2F3c9af704fb61931d%2FDocuments&viewid=75c49926%2D2803%2D4280%2D994e%2Da8047deca96b&listurl=%2Fpersonal%2F3c9af704fb61931d%2FDocuments&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvYy8zYzlhZjcwNGZiNjE5MzFkL0VURERsaVE4elpGR21ZeGxMVlB5aTNzQmlzX2ZkalgwdzhtSmh5UW5ZVlNkWEE%5FZT1XdDdwVWI&ga=1)
[![Model Weights](https://img.shields.io/badge/Model_Weights-Download-green.svg)](https://drive.google.com/file/d/13DR2U54zjZwswYSYp4xg1TgsMrxrRbE4/view?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SAVIS (Audio-Visual Instance Segmentation) is a state-of-the-art framework designed for segmenting and tracking sounding objects in video sequences. By leveraging joint audio-visual representations, attention-based fusion, and temporal modeling, SAVIS delivers precise instance-level segmentation masks and tracks them across video timelines.

This repository hosts the core implementation of **SAVIS (SAVISM Research Framework)**.

---

## 📊 Performance & Ablation Study

A key architectural enhancement in the development of SAVIS is the transition of the audio modality feature extractor:

* **VGGish (Baseline)**: Utilizes the legacy VGGish audio encoder. While VGGish is lightweight, it struggles with fine-grained temporal audio boundary alignment, resulting in a lower overall FSLA metric.
* **BEATs (Proposed)**: Employs the BEATs (Bidirectional Encoder representations from Audio Transformers) encoder (768-dimension output) with frame-level Audio-Guided Contrastive Learning (AGCL) loss.

### Ablation Study Table: VGGish vs. BEATs

| Audio Encoder | FSLA | FSLAn | FSLAs | FSLAm | HOTA | mAP | DetA | AssA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VGGish** | 41.78 | **40.28** | 29.63 | 49.70 | 61.92 | **41.80** | 55.28 | 70.89 |
| **BEATs** | **43.49** | 18.90 | **34.72** | **52.29** | **62.38** | 40.73 | **55.86** | **71.41** |

---

## 🛠️ Installation

This guide is optimized for Ubuntu 24.04 and RTX 4080 (CUDA 12.1 + PyTorch 2.1.0).

### 1. Clone the Repository
```bash
git clone https://github.com/Sayyam-Akram/savis.git
cd savis
```

### 2. Setup the Environment
```bash
conda create --name savis python=3.8 -y
conda activate savis

# Install PyTorch
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install OpenCV, Ninja, and other core dependencies
pip install -U opencv-python ninja
pip install -r requirements.txt
```

### 3. Install Detectron2 from Source
```bash
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_HOME=/usr
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```

### 4. Compile Deformable Attention Operators
```bash
export CUDA_HOME=/usr
export TORCH_CUDA_ARCH_LIST="8.9"
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../
```

---

## 💾 Dataset & Model Weights

### Dataset Setup
1. Download the [AVISeg Dataset](https://onedrive.live.com/?id=%2Fpersonal%2F3c9af704fb61931d%2FDocuments&viewid=75c49926%2D2803%2D4280%2D994e%2Da8047deca96b&listurl=%2Fpersonal%2F3c9af704fb61931d%2FDocuments&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvYy8zYzlhZjcwNGZiNjE5MzFkL0VURERsaVE4elpGR21ZeGxMVlB5aTNzQmlzX2ZkalgwdzhtSmh5UW5ZVlNkWEE%5FZT1XdDdwVWI&ga=1).
2. Extract the dataset files to the `datasets/` folder:
```bash
mkdir -p datasets
# Unzip AVISeg.zip inside the datasets/ directory
```

### Pretrained Weights
We use the **BEATs** audio backbone and pretrained visual backbones.
1. Download the [Model Weights](https://drive.google.com/file/d/13DR2U54zjZwswYSYp4xg1TgsMrxrRbE4/view?usp=sharing) and the pretrained backbones/BEATs weight files from [OneDrive](https://onedrive.live.com/?id=%2Fpersonal%2F3c9af704fb61931d%2FDocuments&viewid=75c49926%2D2803%2D4280%2D994e%2Da8047deca96b&listurl=%2Fpersonal%2F3c9af704fb61931d%2FDocuments&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvYy8zYzlhZjcwNGZiNjE5MzFkL0VURERsaVE4elpGR21ZeGxMVlB5aTNzQmlzX2ZkalgwdzhtSmh5UW5ZVlNkWEE%5FZT1XdDdwVWI&ga=1). Place them in the correct directories:
```bash
mkdir -p pre_models checkpoints
```
* Place the backbone weights in `pre_models/`.
* Place the audio encoder checkpoint (`BEATs_iter3_plus_AS2M.pt`) in the root directory.
* Place the trained model checkpoint (`AVISM_R50_IN.pth`) in `checkpoints/`.

---

## 🚀 Running SAVIS

### Training
To train the model on a single GPU (with gradient accumulation):
```bash
python train_net.py --config-file configs/avism/R50/avism_R50_IN.yaml
```

### Evaluation
To evaluate a trained checkpoint:
```bash
python train_net.py --config-file configs/avism/R50/avism_R50_IN.yaml --eval-only MODEL.WEIGHTS checkpoints/AVISM_R50_IN.pth
```

---

## 🎨 Qualitative Results

Below are qualitative results demonstrating SAVIS tracking and segmenting sounding objects over time:

### Example 1: Segmenting Sounding Instruments
![Qualitative Result 1](docs/images/qualitative_result.jpg)

### Example 2: Multi-Object Audio-Visual Tracking and Segmentation
![Qualitative Result 2](docs/images/qualitative_result_2.jpg)

---

## 🤝 Acknowledgement

We thank the great work from Detectron2, Mask2Former and VITA. We also highly appreciate the great work from the authors of AVISM (AVIS baseline) on which our framework is built.
