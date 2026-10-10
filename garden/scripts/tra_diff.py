
import argparse

import torch
from torch.utils.data import DataLoader

from garden.data.dataset import SCENES, TRAIN_DEVICES, MelDataset, train_clips
from garden.models.diffusion import LatentDiffusion
from garden.models.vae import ConditionalVAE
from garden.utils import EMA, get_device, load_checkpoint, save_checkpoint, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()

    vae = ConditionalVAE(len(TRAIN_DEVICES), len(SCENES))
    vae.load_state_dict(load_checkpoint("vae.pt")["model"])
    vae = vae.to(device).eval().requires_grad_(False)

    model = LatentDiffusion(len(TRAIN_DEVICES), len(SCENES)).to(device)
    ema = EMA(model, args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loader = DataLoader(
        MelDataset(train_clips(), unit=True), args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )

    model.train()
    for epoch in range(args.epochs):
        for x, c, d in loader:
            x, c, d = x.to(device), c.to(device), d.to(device)
            z0 = vae.to_latent(x, d, c)
            loss = model.loss(z0, d, c)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model)
        save_checkpoint("diffusion.pt", model=model.state_dict(), ema=ema.model.state_dict(), epoch=epoch)


if __name__ == "__main__":
    main()
