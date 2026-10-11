# 🌿 GARDEN: Generative Augmentation for Robust Device-Effect Normalization via Source-Conditioned Cycle Consistency

Official implementation of the paper:
**"GARDEN: Generative Augmentation for Robust Device-Effect Normalization via Source-Conditioned Cycle Consistency"**, submitted to **ICASSP 2027**.

---

## 🔎 Overview

Acoustic Scene Classification (ASC) models degrade severely when test audio
comes from **recording devices never seen during training**. Prior
translation-based solutions (e.g., Microphone Conversion) require a separate
generator per device pair and knowledge of the test-device identity — which is
unavailable in practice.

**GARDEN** reformulates the problem as **many-to-one device normalization**:

- **Reference domain A** — recordings from the reference device A.
- **Domain B** — recordings from *all* other devices (real, simulated, and
  IR-augmented).
- A single **unconditional forward generator `G : B → A`** strips the device
  signature and maps any input to the reference domain.
- A **source-conditioned reverse generator `F : (A, e_s) → B`** closes the
  cycle during training. It receives an embedding `e_s` of the source device
  or IR cluster via FiLM, resolving the ambiguity of mapping back into the
  heterogeneous domain B.

Device variability is broadened by two complementary augmentations:

1. **Conditional latent diffusion** — a VAE + U-Net model generates 100,000
   device- and scene-conditioned log-mel spectrograms using dual
   classifier-free guidance (device weight `w_d = 6`, scene weight `w_c = 2`).
2. **Impulse-response (IR) augmentation** — 320 microphone IRs, grouped into
   `M = 32` spectral classes via cosine-based k-means, are convolved with
   domain-B clips to simulate unseen microphones.

> 🔑 **Key property:** at inference time, only `G` is needed. No device
> identity, no device-specific generator, and no samples from the test device
> are required — so the same `G` handles completely unseen hardware.

---

## 📁 Repository Structure

```text
garden/
├── configs/
│   ├── diffusion_yaml.yaml      # Hyperparameters for latent diffusion training
│   ├── gan_yaml.yaml            # Hyperparameters for cycle-consistent GAN training
│   └── vae_yaml.yaml            # Hyperparameters for VAE fine-tuning
├── garden/
│   ├── data/
│   │   ├── dataset.py           # Dataset loaders for training and evaluation splits
│   │   ├── features.py          # Mel-spectrogram feature extraction utilities
│   │   └── ir_augment.py        # Impulse-response (IR) convolution and augmentation
│   ├── models/
│   │   ├── __init__.py          # Model module exports
│   │   ├── classifiers.py       # Downstream ASC backbones (ResNet-50, EfficientNet-B0, CNN14, AST, ViT-B/16)
│   │   ├── cyclegan.py          # Unconditional G, source-conditioned F (FiLM), PatchGAN discriminators
│   │   ├── diffusion.py         # Latent diffusion U-Net with dual-condition guidance
│   │   └── vae.py               # VAE encoder/decoder adapted from AudioLDM
│   ├── __init__.py              # Package initialization
│   ├── metrics.py               # Evaluation metrics (device-wise and macro accuracy)
│   └── utils.py                 # General helper utilities
├── scripts/
│   ├── cluster.py               # Cosine-based k-means clustering of IR profiles (M = 32)
│   ├── eval.py                  # Evaluation of saved checkpoints on test splits
│   ├── extract.py               # Mel-spectrogram extraction from raw audio
│   ├── fig.py                   # Visualizations (e.g., t-SNE of CLS embeddings)
│   ├── gen_pool.py              # Generation of the synthetic diffusion augmentation pool
│   ├── pred.py                  # Prediction on individual audio clips
│   ├── tra_clf.py               # Training of downstream ASC classifiers
│   ├── tra_diff.py              # Training of the conditional latent diffusion model
│   ├── tra_gan.py               # Training of the cycle-consistent GAN (G and F)
│   ├── tra_kd.py                # Knowledge distillation / CP-Mobile training
│   └── tra_va.py                # VAE fine-tuning
├── .gitignore                   # Git ignore rules
├── pyproject.toml               # Project metadata and build configuration
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Python dependencies
└── utils_py.py                  # Supplementary utility scripts
```

### Component Overview

- **`configs/`** — YAML files specifying hyperparameters, data paths, and
  training settings for each of the three training stages (VAE, diffusion,
  GAN). Edit the paths in these files before running the scripts.
- **`garden/data/`** — Audio loading and splitting (`dataset.py`), log-mel
  feature computation (`features.py`), and IR convolution with class-based
  sampling (`ir_augment.py`).
- **`garden/models/`** — All network architectures: the AudioLDM-based VAE,
  the conditional diffusion U-Net (FiLM-modulated by timestep, device, and
  scene embeddings), the CycleGAN generators/discriminators, and the five
  downstream classifier backbones.
- **`scripts/`** — Executable entry points covering the full pipeline:
  feature extraction → generative training → augmentation-pool generation →
  GAN training → classifier training → evaluation and prediction.

---

## ⚙️ Installation

**Requirements:** Python 3.9+, PyTorch with CUDA support. Full training was
performed on 2× NVIDIA RTX 3090 (24 GB); inference and evaluation run on a
single GPU.

1. Clone the repository and navigate to the project directory:

```bash
git clone <REPOSITORY_URL>
cd garden
```

2. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Pretrained Checkpoints

Due to the large size of the trained weights, checkpoints could not be
uploaded directly to the repository and are hosted on Google Drive:

| Checkpoint | Contents |
|---|---|
| [Link 1 — VAE & Diffusion](https://drive.google.com/file/d/1g-6KJ7_yqhp6H2CPHPqjgKkv-tJ9OM9L/view?usp=sharing) | Weights of the first generative stage: the fine-tuned mel-spectrogram VAE (encoder/decoder) and the conditional latent diffusion U-Net |
| [Link 2 — Classifiers](https://drive.google.com/file/d/1ar2sMYx21VqlRA59YtGjDkw0KpWNTKJ2/view?usp=sharing) | **All saved checkpoints of the trained classifiers** — the five ASC backbones (ResNet-50, EfficientNet-B0, CNN14, AST, ViT-B/16), each trained on G-normalized (reference-domain) spectrograms, and other parts of trained models mentioned in paper comes here. |
| [Link 3 — Generators, Discriminators & CP-Mobile](https://drive.google.com/file/d/1yRYhNqSM7Ge-VFVa7Xf43zhVaTo03yPj/view?usp=sharing) | The cycle-consistent LSGAN weights: forward generator `G` (the only module needed at inference), source-conditioned reverse generator `F`, PatchGAN discriminators `D_A` / `D_B`, and the CP-Mobile (KD) classifier used in the DCASE 2025 comparison |

**Usage:** Download the saved model checkpoints from the links above and place
them in the `checkpoints/` folder. Then load the models using the loading code
provided in this repository (see `scripts/eval.py` and `scripts/pred.py`)
before running inference or evaluation.
Note that **only `G` is required for test-time device normalization**.

---

## 🚀 Step-by-Step Guide

### Step 1 — Extract Mel-Spectrograms

Audio is converted to mono and resampled to **32 kHz**. Power spectra are
computed with a **4096-point FFT** and a **hop of 502 samples**, projected
onto **256 mel bands**, and converted to decibels with the dynamic range
clipped to 80 dB below the clip maximum. Each one-second clip yields a
**256 × 64 log-mel matrix**, which is the common front-end representation for
all stages. For the generative stages, values are linearly rescaled to
[−1, 1] and mapped back to decibels before classification.

```bash
python scripts/extract.py --config configs/vae_yaml.yaml \
    --data_dir /path/to/audio/dataset --output_dir /path/to/extracted_mels
```

### Step 2 — Train the Pipeline

The training pipeline consists of six sequential stages:

**2.1 — Fine-tune the VAE.** The pre-trained AudioLDM mel-spectrogram
autoencoder is adapted to the 256-mel representation (30 epochs,
reconstruction + KL objective, KL weight 1e-8). It compresses each
256 × 64 spectrogram into an 8 × 64 × 16 latent and is **frozen afterwards**:

```bash
python scripts/tra_va.py --config configs/vae_yaml.yaml
```

**2.2 — Train the conditional latent diffusion model.** The U-Net denoiser
(128 base channels, multipliers 1/2/4, self-attention at 16 × 4) is trained
for 200 epochs with AdamW (lr 1e-4, batch 64, EMA 0.9999) using
dual-condition classifier-free guidance on device `d` and scene `c`:

```bash
python scripts/tra_diff.py --config configs/diffusion_yaml.yaml
```

**2.3 — Generate the synthetic diffusion pool.** 100,000 balanced
mel-spectrograms are sampled with DDIM (50 steps, `w_d = 6`, `w_c = 2`),
with device and scene labels drawn uniformly:

```bash
python scripts/gen_pool.py --output_dir /path/to/synthetic_pool
```

**2.4 — Cluster impulse responses.** K = 320 microphone IRs are represented
by mean-centered, L2-normalized mel-band log-magnitude profiles and grouped
into M = 32 classes via cosine-based k-means. The class index serves as the
source label `s` during augmentation:

```bash
python scripts/cluster.py --ir_bank_path /path/to/irs --num_clusters 32
```

**2.5 — Train the cycle-consistent GAN.** `G` and `F` use CycleGAN ResNet
generators (9 residual blocks); `D_A`/`D_B` are 70 × 70 PatchGANs. Training
uses the least-squares loss, λ_cyc = 10, Adam (β = 0.5/0.999), and 400
epochs (lr 5e-5, linearly decayed over the last 200). Each step draws 32
domain-A and 32 domain-B samples, with the domain-B half balanced across
four device sources and four IR classes:

```bash
python scripts/tra_gan.py --config configs/gan_yaml.yaml
```

**2.6 — Train downstream classifiers / knowledge distillation.** Classifiers
are trained on domain-A representations: every domain-B clip is first mapped
by `G` to the reference domain:

```bash
python scripts/tra_clf.py --backbone resnet50
# Or, for CP-Mobile with knowledge distillation:
python scripts/tra_kd.py
```

> ⏱️ **Approximate training time** on 2× RTX 3090: ~18 h (diffusion),
> ~68 h (GAN), ~7 h (one classifier).

### Step 3 — Evaluate and Run Checkpoints

At inference, **only the forward generator `G`** is used: each test clip —
from a known or completely unseen device — is mapped to the reference
domain A and then classified. No device identity is required.

Evaluate saved checkpoints on the test splits:

```bash
python scripts/eval.py --checkpoint_dir checkpoints/ --split test
```

Run prediction on an individual audio clip:

```bash
python scripts/pred.py --checkpoint checkpoints/best_model.pth --input clip.wav
```

Reproduce the t-SNE embedding visualization (Fig. 3 of the paper):

```bash
python scripts/fig.py --checkpoint_dir checkpoints/
```

---

## 📊 Results

**Device generalization (TAU, DCASE 2025 Task 1 protocol, mean over five
backbones):**

| Model | Known | Unseen | Gap |
|---|---|---|---|
| Baseline | 59.9 | 45.7 | 14.2 |
| RFN | 60.2 | 50.9 | 9.3 |
| Freq-MixStyle | **60.4** | 49.6 | 10.8 |
| EG-CL | 59.7 | 55.6 | 4.1 |
| MC-200-Adapt | 59.7 | 58.3 | 1.4 |
| **GARDEN (ours)** | 58.8 | **58.6** | **0.2** |

GARDEN achieves the **highest unseen-device accuracy (58.6%)** and the
**smallest known–unseen gap (0.2%)** among the compared methods, at the cost
of slightly lower known-device accuracy.

**DCASE 2025 evaluation split (organizer-scored):** applied to a lightweight
CP-Mobile–KD classifier, GARDEN raises unseen-device accuracy from 55.7% to
**61.1%** and reduces the known–unseen gap from **6.4% to 0.5%**.

---

## 🧩 Extending GARDEN

The framework is task-agnostic: whenever device and task labels are
available, the same two-stage recipe (conditional generation + IR
augmentation + source-conditioned cycle consistency) can be applied to other
audio tasks such as sound event detection or keyword spotting.
