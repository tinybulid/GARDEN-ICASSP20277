import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from garden.data.dataset import (
    AUGS, REFERENCE, SCENES, TRAIN_DEVICES, ClassifierData, MelDataset,
    budget_sampler, build_augmentations, test_clips, train_clips,
)
from garden.metrics import predict, scores
from garden.models.classifiers import BACKBONES, GardenClassifier, SceneClassifier
from garden.models.cyclegan import Generator
from garden.utils import IR_CLUSTER_FILE, POOL_DIR, get_device, load_checkpoint, save_checkpoint, seed_everything

LR = {"resnet50": 1e-4, "efficientnet_b0": 1e-4, "cnn14": 1e-4, "ast": 1e-5, "vit_b16": 1e-5}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=BACKBONES, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gan", default="cyclegan_G.pt")
    p.add_argument("--no-garden", action="store_true")
    p.add_argument("--aug", choices=AUGS, default="both")
    p.add_argument("--n-irs", type=int, default=None)
    p.add_argument("--pool-size", type=int, default=None)
    p.add_argument("--pool", type=Path, default=POOL_DIR)
    p.add_argument("--ir-clusters", type=Path, default=IR_CLUSTER_FILE)
    p.add_argument("--name", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()
    ref = TRAIN_DEVICES.index(REFERENCE)
    garden = not args.no_garden
    name = args.name or f"clf_{args.backbone}" + ("" if garden else "_base")

    if garden:
        G = Generator()
        G.load_state_dict(load_checkpoint(args.gan)["model"])
        model = GardenClassifier(args.backbone, G, len(SCENES))
    else:
        model = SceneClassifier(args.backbone, len(SCENES))
    model = model.to(device)

    pool, irs = build_augmentations(args.aug, args.pool, args.pool_size, args.n_irs, args.seed, args.ir_clusters)
    data = ClassifierData(train_clips(), pool, irs)
    train_loader = DataLoader(
        data, args.batch_size, sampler=budget_sampler(data),
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )

    test = test_clips()
    labels = np.array([c.scene for c in test])
    devices = np.array([c.device for c in test])
    test_loader = DataLoader(
        MelDataset(test), args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr or LR[args.backbone])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(train_loader))

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        for x, c, d in train_loader:
            x, c, d = x.to(device), c.to(device), d.to(device)
            logits = model(x, d == ref) if garden else model(x)
            loss = F.cross_entropy(logits, c)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        res = scores(predict(model, test_loader, garden, device).argmax(1).numpy(), labels, devices)
        if res["macro"] > best:
            best = res["macro"]
            save_checkpoint(
                f"{name}.pt", model=model.state_dict(), backbone=args.backbone,
                garden=garden, replays=True, epoch=epoch, scores=res,
            )


if __name__ == "__main__":
    main()
