
import argparse

import torch
from torch.utils.data import DataLoader

from garden.data.dataset import SCENES, TRAIN_DEVICES, MelDataset, train_clips
from garden.models.vae import ConditionalVAE
from garden.utils import get_device, save_checkpoint, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()

    vae = ConditionalVAE(len(TRAIN_DEVICES), len(SCENES)).to(device)
    loader = DataLoader(
        MelDataset(train_clips(), unit=True), args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)

    vae.train()
    for epoch in range(args.epochs):
        for x, c, d in loader:
            x, c, d = x.to(device), c.to(device), d.to(device)
            loss = vae.loss(x, d, c)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        save_checkpoint("vae.pt", model=vae.state_dict(), epoch=epoch)


if __name__ == "__main__":
    main()
