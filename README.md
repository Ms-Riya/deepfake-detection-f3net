# Multi-Scale Frequency Injection for Robust Deepfake Detection

> Extended F3Net with Adaptive DCT Scale Fusion and Compression-Aware Embeddings  
> **Riya Sarkar & Prof. Preet Kanwal** · PES University, Bengaluru · 2025

---

## Overview

This repository contains the full implementation of our deepfake detection system, which extends [F3Net (Li et al., CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Frequency-Aware_Discriminative_Feature_Learning_Supervised_by_Single-Center_Loss_for_Face_CVPR_2021_paper.pdf) with two novel modules:

- **Multi-Scale DCT (MSDCT)** — parallel frequency feature extraction at 8×8, 16×16, and 32×32 patch sizes, fused by per-image learned softmax attention
- **Compression-Level Embedding** — a learned conditioning vector that tells the classifier how much to trust high-frequency evidence based on the JPEG quality tier of the input

The key architectural insight is **mid-backbone injection**: rather than projecting frequency features back to 3-channel pixel space (which destroys 97.7% of the representation), we inject them directly into Xception at block-2, where channel capacity is large enough to absorb the full 128-channel frequency representation.

---

## Results

### FaceForensics++ c23 Test Set (22,349 samples)

| Model | AUC | Balanced Acc | Real Acc | Fake Acc | Params | FPS |
|---|---|---|---|---|---|---|
| F3Net baseline | 0.9004 | 0.8151 | 0.7773 | 0.8530 | 43.6M | 591 |
| + MSDCT only (A) | 0.9794 | — | — | — | 43.8M | — |
| + Compression only (B) | 0.8670 | — | — | — | 43.7M | — |
| **Ours (A + B)** | **0.9761** | **0.9090** | **0.8590** | **0.9591** | **43.9M** | **273** |

### Cross-Compression Robustness (simulated c40)

| Model | c23 AUC | c40 AUC | AUC Drop |
|---|---|---|---|
| F3Net baseline | 0.9004 | 0.7567 | −0.1437 |
| **Ours** | **0.9675** | **0.8488** | **−0.1187** |

**17.4% reduction in compression-induced degradation.**  
All inference benchmarks on NVIDIA RTX 4090, batch size 16. 273 FPS is well above the 25 FPS threshold for real-time video screening.

---

## Architecture

```
Input Face (3×299×299)
        │
        ├──────────────────────────────────────┐
        │                                      │
   Xception Early                         MSDCT Module
  (blocks 1–2)                    ┌──────────┴──────────┐
        │                    8×8 DCT   16×16 DCT   32×32 DCT
        │                         └──────────┬──────────┘
        │                              Scale Attention
        │                           (softmax, per-image)
        │                                    │
        │                          Conv 3×3 → 128ch
        │                                    │
        └──────────── + ─────────────────────┘
             (1×1 proj 128→256, element-wise add)
                          │
                   Xception Late
                  (blocks 3–12)
                          │
                       MixBlock  ←── LFS Branch
                          │
                    GAP → 512-dim
                          │
            concat [512-dim ; 64-dim comp. embedding]
                          │
                  MLP 576→256→128→1
                          │
                        Logit
```

---

## Repository Structure

```
deepfake-detection-f3net/
│
├── extract_faces.py        # MTCNN face extraction from FF++ videos
├── dataset.py              # FFppDataset, WeightedRandomSampler, get_dataloaders()
├── xception.py             # Full Xception from scratch + pretrained weight loading
├── f3net.py                # F3Net baseline (FAD + LFS + MixBlock)
├── multiscale_dct.py       # MSDCT module — multi-scale DCT feature extractor
├── compression_aug.py      # JPEG augmentation + CompressionEmbedding
├── f3net_novel.py          # Combined novel model (MSDCT + compression embedding)
├── train.py                # Baseline training script (20 epochs)
├── train_v2.py             # Improved baseline attempt (not recommended)
├── train_novel.py          # Novel model training (25 epochs, dual LR)
└── README.md
```

---

## Setup

### Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install facenet-pytorch opencv-python pillow numpy tqdm scikit-learn matplotlib
```

Tested with:
- Python 3.13
- PyTorch 2.x + CUDA 12.2
- NVIDIA RTX 4090 (24 GB VRAM)

### Pretrained Xception Weights

Download the ImageNet-pretrained Xception weights and place them at the project root:

```bash
wget https://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth
```

### Dataset

This project uses [FaceForensics++](https://github.com/ondyari/FaceForensics). Request access from the authors and download the c23 videos. Expected structure:

```
face forensics/FaceForensics++_C23/
├── original/          # 1000 real videos
├── Deepfakes/         # 1000 manipulated videos
├── Face2Face/
├── FaceSwap/
└── NeuralTextures/
```

---

## Usage

### 1. Extract Faces

```bash
python extract_faces.py
```

Runs MTCNN (confidence 0.90, 30% margin) on all videos, sampling 32 frames each. Outputs to `dataset/train`, `dataset/val`, `dataset/test`.

### 2. Train Baseline

```bash
python train.py
```

20 epochs, Adam, LR 2e-4, cosine annealing. Checkpoint saved to `checkpoints/best_model.pth`.

### 3. Train Novel Model

```bash
python train_novel.py
```

25 epochs, dual learning rate (backbone: 2e-5, novel layers: 2e-4), JPEG augmentation active. Checkpoint saved to `checkpoints_novel/best_model.pth`.

### 4. Inference

```python
import torch
from f3net_novel import F3NetNovel
from multiscale_dct import MultiScaleDCT, SingleScaleDCT

# Patch SingleScaleDCT to grayscale (avoids OOM at full resolution)
def forward_grayscale(self, x):
    import torch.nn.functional as F_nn
    gray = (0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3])
    freq = F_nn.conv2d(gray, self.dct_filters, stride=1, padding=self.size//2)
    freq = freq * self.freq_weight
    return self.encoder(freq[:, :3, :, :])

SingleScaleDCT.forward = forward_grayscale  # apply BEFORE model instantiation

# Load model
model = F3NetNovel(mode="Mix").cuda()
ckpt = torch.load("checkpoints_novel/best_model.pth")
model.load_state_dict(ckpt['model'], strict=False)
model.eval()

# Run inference
# x: tensor of shape (B, 3, 299, 299), normalised to [-1, 1]
# compression_level: 0 = c0, 1 = c23, 2 = c40
with torch.no_grad():
    logit, scale_weights = model(x.cuda(), compression_level=1)
    prob = torch.sigmoid(logit).item()
    print(f"Fake probability: {prob:.3f}")
    print(f"Scale weights (8×8, 16×16, 32×32): {scale_weights.cpu().numpy()}")
```

> **Important:** Always apply the `forward_grayscale` patch before instantiating the model. Running both baseline and novel model simultaneously will OOM on 24 GB VRAM — load one at a time.

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Input normalisation | [−1, 1] | DCT requires consistent scaling; ImageNet stats break frequency analysis |
| Injection point | Xception block-2 | Avoids 128→3 channel bottleneck; pilot at pixel-input gave AUC 0.841 |
| JPEG augmentation range | Quality 65–95, p=0.5 | Brackets c23–c40 gap without destroying all discriminative signal |
| Backbone LR | 10× lower than novel layers | Preserves pretrained Xception features during fine-tuning |
| Loss | BCEWithLogitsLoss (no pos_weight) | pos_weight=0.25 degraded performance in experiments |
| Frame sampling | Official FF++ JSON splits | Prevents frame-level leakage across train/val/test |

---

## Ablation: Why Mid-Backbone Injection Matters

| Injection strategy | AUC |
|---|---|
| Pixel-level input (128→3 bottleneck) | 0.841 |
| Xception block-2 (ours) | **0.9761** |

The 9.5 pp gap is the single most impactful design choice in the whole system.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{kanwal2025msdct,
  title   = {Multi-Scale Frequency Injection for Robust Deepfake Detection:
             Extended F3Net with Adaptive DCT Scale Fusion and
             Compression-Aware Embeddings},
  author  = {Kanwal, Preet and Sarkar, Riya},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## Acknowledgements

- [F3Net](https://github.com/yyk-wew/F3Net) — Li et al., CVPR 2021, the baseline this work extends
- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Rossler et al., ICCV 2019, dataset and official splits
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN implementation used for face extraction
- PES University for computing resources

---

## License

This code is released for academic and non-commercial research use only.  
© 2025 Riya Sarkar, Preet Kanwal — PES University
