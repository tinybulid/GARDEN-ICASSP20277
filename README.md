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
GARDEN-ICASSP20277/
├── garden/                          # Implementation package
│   ├── checkpoints/
│   │   └── readme.md                # Checkpoint download links and usage (see below)
│   ├── configs/
│   │   ├── diffusion_yaml.yaml      # Settings for latent diffusion training
│   │   ├── gan_yaml.yaml            # Settings for cycle-consistent GAN training
│   │   └── vae_yaml.yaml            # Settings for VAE fine-tuning
│   ├── garden/                      # Core Python package
│   │   ├── data/
│   │   │   ├── dataset.py           # Clip lists, Mel datasets, CycleSampler, augmentation builders
│   │   │   ├── features.py          # Log-mel feature extraction (LogMel, audio loading)
│   │   │   └── ir_augment.py        # IR clustering (cluster_irs) and IR convolution augmentation
│   │   ├── models/
│   │   │   ├── __init__.py          # Model module exports
│   │   │   ├── classifiers.py       # SceneClassifier / GardenClassifier and the five ASC backbones
│   │   │   ├── cyclegan.py          # CycleGAN: unconditional G, source-conditioned F (FiLM), PatchGAN D_A/D_B
│   │   │   ├── diffusion.py         # LatentDiffusion U-Net with dual-condition guidance and DDIM sampling
│   │   │   └── vae.py               # ConditionalVAE encoder/decoder adapted from AudioLDM
│   │   ├── __init__.py              # Package initialization
│   │   ├── metrics.py               # predict() and scores() — device-wise / macro accuracy
│   │   └── utils.py                 # Paths (CKPT_DIR, POOL_DIR, ...), seeding, EMA, checkpoint I/O
│   ├── scripts/
│   │   ├── cluster.py               # Cosine-based k-means clustering of IR profiles (default M = 32)
│   │   ├── eval.py                  # Evaluate saved classifier checkpoints; writes results/*.json
│   │   ├── extract.py               # Pre-compute and cache log-mel spectrograms for all clips
│   │   ├── fig.py                   # Full ablation pipeline for Fig. 2 (prep → train → collect → plot)
│   │   ├── gen_pool.py              # Sample the synthetic diffusion augmentation pool (default 100k)
│   │   ├── pred.py                  # Predictions on the TAU evaluation split (DCASE-style CSV output)
│   │   ├── tra_clf.py               # Train downstream ASC classifiers (with or without GARDEN's G)
│   │   ├── tra_diff.py              # Train the conditional latent diffusion model
│   │   ├── tra_gan.py               # Train the cycle-consistent GAN (G and F)
│   │   ├── tra_kd.py                # CP-Mobile training with multi-teacher knowledge distillation
│   │   └── tra_va.py                # Fine-tune the conditional VAE
│   ├── gitignore_file.gitignore     # Git ignore rules
│   ├── pyproject.toml               # Project metadata and build configuration
│   ├── readme.md                    # Package-level documentation
│   ├── requirements.txt             # Python dependencies
│   └── utils_py.py                  # Supplementary utility scripts
└── README.md                        # Main project documentation (this file)
```

### Component Overview

- **`garden/garden/data/`** — Clip listing and dataset classes (`dataset.py`),
  log-mel feature computation (`features.py`), and IR clustering/convolution
  (`ir_augment.py`).
- **`garden/garden/models/`** — All network architectures: the AudioLDM-based
  conditional VAE, the latent diffusion U-Net (FiLM-modulated by timestep,
  device, and scene embeddings), the CycleGAN generators/discriminators, and
  the downstream classifier backbones.
- **`garden/garden/utils.py`** — Central path configuration (data,
  checkpoints, diffusion pool, IR clusters), seeding, EMA, and checkpoint
  save/load helpers used by all scripts.
- **`garden/scripts/`** — Executable entry points covering the full pipeline:
  feature extraction → generative training → augmentation-pool generation →
  IR clustering → GAN training → classifier training → evaluation and
  prediction. All hyperparameters are exposed as command-line arguments with
  the paper's values as defaults.
- **`garden/checkpoints/`** — Target folder for the pretrained weights
  (download links below).

---

## ⚙️ Installation

**Requirements:** Python 3.9+, PyTorch with CUDA support. Full training was
performed on 2× NVIDIA RTX 3090 (24 GB); inference and evaluation run on a
single GPU.

1. Clone the repository and navigate to the project directory:

```bash
git clone <REPOSITORY_URL>
cd GARDEN-ICASSP20277/garden
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

All scripts read data and checkpoint locations from the paths defined in
`garden/utils.py`, and every hyperparameter defaults to the value used in the
paper — so each stage can be run with no arguments.

### Step 1 — Extract Mel-Spectrograms

Audio is converted to mono and resampled to **32 kHz**. Power spectra are
computed with a **4096-point FFT** and a **hop of 502 samples**, projected
onto **256 mel bands**, and converted to decibels with the dynamic range
clipped to 80 dB below the clip maximum. Each one-second clip yields a
**256 × 64 log-mel matrix**. This script pre-computes and caches the log-mel
features for all training and test clips:

```bash
python scripts/extract.py --workers 8
```

### Step 2 — Train the Pipeline

The training pipeline consists of six sequential stages:

**2.1 — Fine-tune the VAE.** The conditional VAE (adapted from AudioLDM) is
fine-tuned on the training pool for 30 epochs, conditioned on device and
scene. The resulting `vae.pt` checkpoint is frozen and reused by all later
stages:

```bash
python scripts/tra_va.py --epochs 30 --batch-size 32 --lr 5e-5
```

**2.2 — Train the conditional latent diffusion model.** The U-Net denoiser is
trained for 200 epochs with AdamW (lr 1e-4, batch 64) and an EMA of the
weights (decay 0.9999), on latents produced by the frozen VAE. The EMA
weights are saved in `diffusion.pt` and used for sampling:

```bash
python scripts/tra_diff.py --epochs 200 --batch-size 64 --lr 1e-4 --ema 0.9999
```

**2.3 — Generate the synthetic diffusion pool.** 100,000 mel-spectrograms are
sampled with DDIM (50 steps, `w_d = 6`, `w_c = 2`), with device and scene
labels drawn uniformly over all training devices and scenes. The pool
(`mels.npy`, `devices.npy`, `scenes.npy`) is written to the pool directory:

```bash
python scripts/gen_pool.py --n 100000 --steps 50 --wd 6.0 --wc 2.0
```

**2.4 — Cluster impulse responses.** The microphone IRs are represented by
mean-centered, L2-normalized mel-band log-magnitude profiles and grouped into
M = 32 classes via cosine-based k-means. The cluster assignment file is used
as the source label `s` during augmentation:

```bash
python scripts/cluster.py --clusters 32
```

**2.5 — Train the cycle-consistent GAN.** `G` and `F` are trained with the
least-squares loss, λ_cyc = 10, Adam (β = 0.5/0.999), for 400 epochs (lr
5e-5 held for the first 200 epochs — `--hold` — then linearly decayed). The
`--mode` flag selects the F variant used in the ablations: `cond`
(source-conditioned F, full model), `uncond` (F without source conditioning),
or `none` (no F). The forward generator is additionally exported as
`cyclegan_G.pt`:

```bash
python scripts/tra_gan.py --mode cond --aug both --epochs 400 --hold 200 --lambda-cyc 10.0
```

**2.6 — Train downstream classifiers / knowledge distillation.** Classifiers
are trained on domain-A representations: every domain-B clip is first mapped
by the frozen `G` to the reference domain. Backbone-specific learning rates
are applied automatically, and the best checkpoint (by macro accuracy) is
saved with its scores:

```bash
python scripts/tra_clf.py --backbone resnet50 --gan cyclegan_G.pt
# Available backbones: resnet50, efficientnet_b0, cnn14, ast, vit_b16
# Baseline without GARDEN normalization:
python scripts/tra_clf.py --backbone resnet50 --no-garden
# CP-Mobile with multi-teacher knowledge distillation (teachers in checkpoints/teachers/):
python scripts/tra_kd.py
```

> ⏱️ **Approximate training time** on 2× RTX 3090: ~18 h (diffusion),
> ~68 h (GAN), ~7 h (one classifier).

### Step 3 — Evaluate and Run Checkpoints

At inference, **only the forward generator `G`** is used: each test clip —
from a known or completely unseen device — is mapped to the reference
domain A and then classified. No device identity is required.

Evaluate saved classifier checkpoints on the development-test split
(by default all five `clf_<backbone>.pt` checkpoints; per-device,
known/unseen, and macro accuracies are written to `results/results.json`
with mean ± std across backbones):

```bash
python scripts/eval.py --ckpts clf_resnet50.pt clf_vit_b16.pt
```

Generate predictions on the TAU evaluation split (DCASE-style tab-separated
CSV with per-scene probabilities, for submission to the organizers):

```bash
python scripts/pred.py --ckpt cpmobile_kd.pt
```

Reproduce the full ablation study of Fig. 2 (prepares IR clusterings and
diffusion pools, trains GANs in all three modes and classifiers for every
panel setting, collects scores, and renders the plots to
`results/fig2/fig2.png`/`.pdf`):

```bash
python scripts/fig.py --stages prep train collect plot --panels a b c d e
```

---

## 🧩 Extending GARDEN

The framework is task-agnostic: whenever device and task labels are
available, the same two-stage recipe (conditional generation + IR
augmentation + source-conditioned cycle consistency) can be applied to other
audio tasks such as sound event detection or keyword spotting.
