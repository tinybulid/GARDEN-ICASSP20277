# GARDEN

**Generative Augmentation for Robust Device-Effect Normalization via Source-Conditioned Cycle Consistency**

[Project page](https://acoustilight.github.io/GARDEN/)

GARDEN is a many-to-one device normalization framework for acoustic scene classification. A single generator `G` maps recordings from any non-reference device to the reference device A, including devices never seen in training, without needing to know which device a test clip came from.

Training has three parts:

- A conditional latent diffusion model with dual classifier-free guidance generates device- and scene-conditioned log-mel spectrograms.
- Microphone impulse responses (320 IRs grouped into 32 spectral clusters) add more device variability.
- A cycle-consistent LSGAN is trained on real and synthetic data. The forward generator `G` removes device effects. The reverse generator `F` is conditioned on the source device or IR cluster.

Only `G` is used at inference.

## Installation

```bash
git clone https://github.com/<user>/GARDEN.git
cd GARDEN
pip install -e .
```

## Data

The code expects this layout under `data/`:

```
data/
├── TAU-urban-acoustic-scenes-2022-mobile-development/
│   └── audio/
├── TAU-urban-acoustic-scenes-2025-mobile-evaluation/
│   ├── audio/
│   └── meta.csv
├── split_setup/
│   ├── split25.csv
│   └── test.csv
├── replays/
│   ├── r1/audio/
│   ├── ...
│   └── r8/audio/
└── mic_irs/
```

- **TAU Urban Acoustic Scenes 2022 Mobile:** the development set, with the official DCASE 2025 Task 1 25% split.
- **Evaluation split:** devices D and S7–S10. `meta.csv` lists `filename` and `source_label`. Its labels are private, so it is used only to write prediction files.
- **Physical replays (R1–R8):** these use the same filenames as the TAU clips.
  - R1–R4 are used for training and Known evaluation.
  - R5–R8 are used only for Unseen evaluation.
- **Microphone impulse responses:** 320 IRs from the multi-angle, multi-distance microphone IR dataset.

The pipeline generates `mels/`, `diffusion_pool/` and `ir_clusters.npz` itself.

## Pipeline

Run from the repo root, in order:

```bash
python scripts/ext_mel.py
python scripts/clu_ir.py
python scripts/tra_vae.py
python scripts/tra_diff.py
python scripts/gen_pool.py
python scripts/tra_gan.py
python scripts/tra_clf.py --backbone resnet50
python scripts/eval.py
```

- Run `tra_clf.py` once for each backbone: `resnet50`, `efficientnet_b0`, `cnn14`, `ast`, `vit_b16`.
- Each classifier keeps the checkpoint with the best device-macro accuracy on the development-test split.
- Add `--no-garden` to `tra_clf.py` to train the baseline without device normalization.
- `eval.py` writes per-device, Known, Unseen and Gap accuracies, with mean ± std over the backbones, to `results/results.json`.

## DCASE 2025 comparison (Table 2)

```bash
python scripts/tra_kd.py
python scripts/tra_kd.py --no-garden
python scripts/eval.py --ckpts cpmobile_kd.pt cpmobile_kd_base.pt --out cpmobile
python scripts/pred.py --ckpt cpmobile_kd.pt
python scripts/pred.py --ckpt cpmobile_kd_base.pt
```

- `tra_kd.py` trains a CP-Mobile student with knowledge distillation from an ensemble of three teachers, using temperature 2 and label weight 0.02.
- With GARDEN, non-A clips are mapped by `G` before both the student and the teachers see them.
- `pred.py` writes evaluation-split predictions in DCASE format to `results/<checkpoint>_eval.csv`.

## Ablations (Fig. 2)

```bash
python scripts/fig2.py
```

`fig2.py` reproduces all five panels of Fig. 2:

- It changes only the plotted setting, keeping every other setting at the full configuration.
- The number of training updates is the same in every condition.
- Each condition is trained with source-conditioned `F`, unconditioned `F`, and without `F`, and the results are averaged over the five backbones.
- Completed runs are skipped, so the script can be restarted.
- Output goes to `results/fig2/` as a JSON file, one CSV per panel, and the figure.
- Use `--panels a b` or `--stages collect plot` to run part of it.

## Checkpoints

Put all weights in `checkpoints/`:

| File | Contents |
|---|---|
| `vae.pt` | Fine-tuned conditional VAE |
| `diffusion.pt` | Latent diffusion model (with EMA weights) |
| `cyclegan.pt` | Full CycleGAN (`G`, `F`, `D_A`, `D_B`) |
| `cyclegan_G.pt` | Device-normalization generator used at inference |
| `clf_<backbone>.pt`, `clf_<backbone>_base.pt` | Scene classifiers with and without GARDEN |
| `teachers/T1.pt` … `teachers/T3.pt` | Knowledge-distillation teachers (TorchScript, log-mel input) |
| `cpmobile_kd.pt`, `cpmobile_kd_base.pt` | CP-Mobile students with and without GARDEN |
| `fig2_*.pt` | Ablation runs |
| `Cnn14_mAP=0.431.pth` | PANNs weights for CNN14 initialization |

## Results

These are device-macro accuracies (%) from the paper.

**Table 1 (averaged over the five backbones):**

| Model | Known | Unseen | Gap |
|---|---|---|---|
| Baseline | 59.9 ± 1.6 | 45.7 ± 1.7 | 14.2 |
| GARDEN | 58.8 ± 1.3 | 58.6 ± 1.3 | 0.2 |

**Table 2 (DCASE 2025 evaluation split, scored by the organizers):**

| Model | Known | Unseen | Gap |
|---|---|---|---|
| CP-Mobile–KD | 62.1 | 55.7 | 6.4 |
| CP-Mobile–KD + GARDEN | 61.6 | 61.1 | 0.5 |

## Citation

```bibtex
@misc{sharifi2026garden,
  title  = {GARDEN: Generative Augmentation for Robust Device-Effect Normalization via Source-Conditioned Cycle Consistency},
  author = {Sharifi, Hossein and Raoufi, Behnam and Tavakoli, Reza and Ahmadi, Siavash and Khalaj, Babak and Alirezaie, Marjan and Bagheri Shouraki, Saeed and Rohban, Mohammadhosein and Diba, Ali and Sabokrou, Mohammad},
  year   = {2026}
}
```

## License

MIT
