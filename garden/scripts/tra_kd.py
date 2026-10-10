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
from garden.models.classifiers import GardenClassifier, SceneClassifier, to_reference
from garden.models.cyclegan import Generator
from garden.utils import CKPT_DIR, IR_CLUSTER_FILE, POOL_DIR, get_device, load_checkpoint, save_checkpoint, seed_everything

TEACHER_DIR = CKPT_DIR / "teachers"
N_TEACHERS = 3


def load_teachers(root, n, device):
    return [torch.jit.load(str(root / f"T{i}.pt"), map_location=device).eval() for i in range(1, n + 1)]


def kd_loss(s, t, y, temperature, label_weight):
    kd = F.kl_div(F.log_softmax(s / temperature, 1), F.softmax(t / temperature, 1), reduction="batchmean")
    return label_weight * F.cross_entropy(s, y) + (1 - label_weight) * kd * temperature**2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--label-weight", type=float, default=0.02)
    p.add_argument("--teachers", type=Path, default=TEACHER_DIR)
    p.add_argument("--n-teachers", type=int, default=N_TEACHERS)
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
    name = args.name or "cpmobile_kd" + ("" if garden else "_base")

    teachers = load_teachers(args.teachers, args.n_teachers, device)

    if garden:
        G = Generator()
        G.load_state_dict(load_checkpoint(args.gan)["model"])
        student = GardenClassifier("cpmobile", G, len(SCENES)).to(device)
        head = student.clf
    else:
        student = SceneClassifier("cpmobile", len(SCENES)).to(device)
        head = student

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

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    for epoch in range(args.epochs):
        student.train()
        for x, c, d in train_loader:
            x, c, d = x.to(device), c.to(device), d.to(device)
            if garden:
                x = to_reference(student.G, x, d == ref)
            with torch.no_grad():
                t = torch.stack([m(x) for m in teachers]).mean(0)
            loss = kd_loss(head(x), t, c, args.temperature, args.label_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        res = scores(predict(student, test_loader, garden, device).argmax(1).numpy(), labels, devices)
        if res["macro"] > best:
            best = res["macro"]
            save_checkpoint(
                f"{name}.pt", model=student.state_dict(), backbone="cpmobile",
                garden=garden, replays=True, epoch=epoch, scores=res,
            )


if __name__ == "__main__":
    main()
