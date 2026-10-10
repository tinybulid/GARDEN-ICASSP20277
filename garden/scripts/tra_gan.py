import argparse
import random
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from garden.data.dataset import AUGS, DOMAIN_B, CycleSampler, build_augmentations, train_clips
from garden.models.cyclegan import MODES, CycleGAN, lr_lambda
from garden.utils import IR_CLUSTER_FILE, POOL_DIR, get_device, save_checkpoint, seed_everything


class Steps(Dataset):
    def __init__(self, sampler):
        self.sampler = sampler

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, i):
        return self.sampler.draw()


def seed_worker(_):
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--hold", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lambda-cyc", type=float, default=10.0)
    p.add_argument("--mode", choices=MODES, default="cond")
    p.add_argument("--aug", choices=AUGS, default="both")
    p.add_argument("--n-irs", type=int, default=None)
    p.add_argument("--pool-size", type=int, default=None)
    p.add_argument("--pool", type=Path, default=POOL_DIR)
    p.add_argument("--ir-clusters", type=Path, default=IR_CLUSTER_FILE)
    p.add_argument("--name", default="cyclegan")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()

    pool, irs = build_augmentations(args.aug, args.pool, args.pool_size, args.n_irs, args.seed, args.ir_clusters)
    sampler = CycleSampler(train_clips(), pool, irs)
    loader = DataLoader(
        Steps(sampler), batch_size=None, num_workers=args.workers,
        worker_init_fn=seed_worker, pin_memory=True, persistent_workers=args.workers > 0,
    )

    n_sources = len(DOMAIN_B) + (irs.n_classes if irs is not None else 0)
    gan = CycleGAN(n_sources, mode=args.mode, lambda_cyc=args.lambda_cyc).to(device)
    gens = [m for m in (gan.G, gan.F) if m is not None]
    discs = [m for m in (gan.D_A, gan.D_B) if m is not None]
    opt_g = torch.optim.Adam(chain(*(m.parameters() for m in gens)), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(chain(*(m.parameters() for m in discs)), lr=args.lr, betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda(args.hold, args.epochs))
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda(args.hold, args.epochs))

    gan.train()
    for epoch in range(args.epochs):
        for xa, xb, sb in loader:
            xa, xb, sb = xa.to(device), xb.to(device), sb.to(device)

            g_loss, fake_a, fake_b = gan.generator_loss(xa, xb, sb)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            d_loss = gan.discriminator_loss(xa, xb, fake_a, fake_b)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

        sched_g.step()
        sched_d.step()
        save_checkpoint(
            f"{args.name}.pt", model=gan.state_dict(), opt_g=opt_g.state_dict(),
            opt_d=opt_d.state_dict(), epoch=epoch, mode=args.mode, n_sources=n_sources,
        )

    save_checkpoint(f"{args.name}_G.pt", model=gan.G.state_dict())


if __name__ == "__main__":
    main()
