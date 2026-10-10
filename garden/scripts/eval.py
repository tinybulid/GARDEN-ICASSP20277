import argparse
import json

import numpy as np
from torch.utils.data import DataLoader

from garden.data.dataset import MelDataset, test_clips
from garden.metrics import predict, scores
from garden.models.classifiers import BACKBONES, load_classifier
from garden.utils import CKPT_DIR, ROOT, get_device, load_checkpoint

RESULTS_DIR = ROOT / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[f"clf_{b}.pt" for b in BACKBONES])
    p.add_argument("--out", default="results")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    device = get_device()
    cache = {}

    def data(replays):
        if replays not in cache:
            clips = test_clips(replays)
            loader = DataLoader(MelDataset(clips), args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            cache[replays] = (loader, np.array([c.scene for c in clips]), np.array([c.device for c in clips]))
        return cache[replays]

    results = {}
    for name in args.ckpts:
        if not (CKPT_DIR / name).exists():
            continue
        ckpt = load_checkpoint(name)
        model = load_classifier(ckpt).to(device)
        loader, labels, devices = data(ckpt.get("replays", True))
        results[name] = scores(predict(model, loader, ckpt["garden"], device).argmax(1).numpy(), labels, devices)

    keys = sorted(set.intersection(*(set(r) for r in results.values()))) if results else []
    summary = {
        k: {"mean": float(np.mean([r[k] for r in results.values()])),
            "std": float(np.std([r[k] for r in results.values()]))}
        for k in keys
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{args.out}.json", "w") as f:
        json.dump({"models": results, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
