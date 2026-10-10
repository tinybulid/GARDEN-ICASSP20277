import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from garden.data.dataset import REFERENCE, SCENES, TRAIN_DEVICES, eval_clips
from garden.data.features import LogMel, load_audio
from garden.metrics import predict
from garden.models.classifiers import load_classifier
from garden.utils import ROOT, TAU_EVAL_DIR, get_device, load_checkpoint

RESULTS_DIR = ROOT / "results"


class EvalAudio(Dataset):
    def __init__(self, items):
        self.items = items
        self.logmel = LogMel()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, dev = self.items[i]
        ref = TRAIN_DEVICES.index(REFERENCE) if dev == REFERENCE else -1
        return self.logmel(load_audio(path))[None], 0, ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="cpmobile_kd.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    device = get_device()
    ckpt = load_checkpoint(args.ckpt)
    model = load_classifier(ckpt).to(device)

    items = eval_clips()
    loader = DataLoader(EvalAudio(items), args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    probs = torch.softmax(predict(model, loader, ckpt["garden"], device), 1).numpy()

    df = pd.DataFrame({
        "filename": [str(path.relative_to(TAU_EVAL_DIR)) for path, _ in items],
        "scene_label": [SCENES[i] for i in probs.argmax(1)],
    })
    for j, s in enumerate(SCENES):
        df[s] = probs[:, j]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"{args.ckpt.removesuffix('.pt')}_eval.csv", sep="\t", index=False)


if __name__ == "__main__":
    main()
