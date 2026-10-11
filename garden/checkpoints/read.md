# Pretrained Checkpoints

Due to the large size of the trained model weights, they could not be
uploaded directly to this repository. All checkpoints are hosted on
Google Drive and can be downloaded from the links below.

## Contents

- **[Link 1 — VAE & Diffusion](https://drive.google.com/file/d/1g-6KJ7_yqhp6H2CPHPqjgKkv-tJ9OM9L/view?usp=sharing)**
  Weights of the first generative stage: the fine-tuned mel-spectrogram
  VAE (encoder/decoder) and the conditional latent diffusion U-Net used
  to generate device- and scene-conditioned log-mel spectrograms.

- **[Link 2 — Classifiers](https://drive.google.com/file/d/1ar2sMYx21VqlRA59YtGjDkw0KpWNTKJ2/view?usp=sharing)**
  All saved checkpoints of the trained classifiers are stored in this link.
  This includes the trained weights of the five downstream ASC backbones
  (ResNet-50, EfficientNet-B0, CNN14, AST, ViT-B/16), trained on
  G-normalized (reference-domain) spectrograms.

- **[Link 3 — Generators, Discriminators & CP-Mobile](https://drive.google.com/file/d/1yRYhNqSM7Ge-VFVa7Xf43zhVaTo03yPj/view?usp=sharing)**
  Weights of the cycle-consistent LSGAN: the forward generator `G`, the source-conditioned reverse
  generator `F`, the PatchGAN discriminators `D_A` / `D_B`, plus the
  CP-Mobile (KD) classifier used in the DCASE 2025 comparison.

## Usage

Download the saved model checkpoints from the links above and place
them in this folder (`checkpoints/`). Then load the models using the
loading scripts/code provided in this repository before running
inference or evaluation.
Note that only `G` is required for test-time device normalization.
