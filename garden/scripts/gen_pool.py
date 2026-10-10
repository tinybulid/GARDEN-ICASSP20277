import argparse
import math
from pathlib import Path

import numpy as np
import torch

from garden.data.dataset import SCENES, TRAIN_DEVICES
from garden.data.features import N_MELS
from garden.models.diffusion import LatentDiffusion
from garden.models.vae import ConditionalVAE
from garden.utils import POOL_DIR, get_device, load_checkpoint, seed_everything


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--wd", type=float, default=6.0)
    p.add_argument("--wc", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=POOL_DIR)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()

    vae = ConditionalVAE(len(TRAIN_DEVICES), len(SCENES))
    vae.load_state_dict(load_checkpoint("vae.pt")["model"])
    vae = vae.to(device).eval()

    diffusion = LatentDiffusion(len(TRAIN_DEVICES), len(SCENES))
    diffusion.load_state_dict(load_checkpoint("diffusion.pt")["ema"])
    diffusion = diffusion.to(device).eval()

    pairs = torch.cartesian_prod(torch.arange(len(TRAIN_DEVICES)), torch.arange(len(SCENES)))
    labels = pairs.repeat(math.ceil(args.n / len(pairs)), 1)[: args.n]
    labels = labels[torch.randperm(args.n)]

    args.out.mkdir(parents=True, exist_ok=True)
    mels = np.lib.format.open_memmap(args.out / "mels.npy", mode="w+", dtype=np.float16, shape=(args.n, N_MELS, 64))
    for i in range(0, args.n, args.batch_size):
        d, c = labels[i : i + args.batch_size].to(device).unbind(1)
        z = diffusion.sample(d, c, steps=args.steps, wd=args.wd, wc=args.wc)
        mels[i : i + len(d)] = vae.from_latent(z, d, c)[:, 0].cpu().numpy().astype(np.float16)
    mels.flush()

    np.save(args.out / "devices.npy", labels[:, 0].numpy())
    np.save(args.out / "scenes.npy", labels[:, 1].numpy())


if __name__ == "__main__":
    main()
