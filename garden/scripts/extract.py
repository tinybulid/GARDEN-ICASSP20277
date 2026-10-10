
import argparse

import numpy as np
from torch.utils.data import DataLoader, Dataset

from garden.data.dataset import mel_path, test_clips, train_clips
from garden.data.features import LogMel, load_audio


class Extract(Dataset):
    def __init__(self, clips):
        self.clips = clips
        self.logmel = LogMel()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        c = self.clips[i]
        out = mel_path(c)
        if not out.exists():
            np.save(out, self.logmel(load_audio(c.path)).numpy().astype(np.float32))
        return i


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    clips = train_clips() + test_clips()
    for folder in {mel_path(c).parent for c in clips}:
        folder.mkdir(parents=True, exist_ok=True)

    for _ in DataLoader(Extract(clips), batch_size=256, num_workers=args.workers):
        pass


if __name__ == "__main__":
    main()
