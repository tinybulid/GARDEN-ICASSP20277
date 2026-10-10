import random
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, RandomSampler

from ..utils import IR_CLUSTER_FILE, MEL_DIR, POOL_DIR, REPLAY_DIR, SPLIT_DIR, TAU_DIR, TAU_EVAL_DIR
from .features import LogMel, load_audio, to_db, to_unit
from .ir_augment import load_bank

SCENES = [
    "airport", "bus", "metro", "metro_station", "park",
    "public_square", "shopping_mall", "street_pedestrian", "street_traffic", "tram",
]
REFERENCE = "a"
TRAIN_DEVICES = ["a", "b", "c", "s1", "s2", "s3", "r1", "r2", "r3", "r4"]
DOMAIN_B = TRAIN_DEVICES[1:]
UNSEEN_DEVICES = ["s4", "s5", "s6", "r5", "r6", "r7", "r8"]
REPLAYS = [f"r{i}" for i in range(1, 9)]
AUGS = ("none", "diff", "ir", "both")
FULL_POOL = 100_000

Clip = namedtuple("Clip", "path scene device")


def _clip(root, name, device=None):
    stem = Path(name).stem
    return Clip(root / name, SCENES.index(stem.split("-")[0]), device or stem.split("-")[-1])


def _names(csv):
    return pd.read_csv(csv, sep="\t")["filename"].tolist()


def train_clips(replays=True):
    names = _names(SPLIT_DIR / "split25.csv")
    clips = [_clip(TAU_DIR, n) for n in names]
    for r in (REPLAYS[:4] if replays else []):
        clips += [_clip(REPLAY_DIR / r, n, r) for n in names]
    return clips


def test_clips(replays=True):
    names = _names(SPLIT_DIR / "test.csv")
    clips = [_clip(TAU_DIR, n) for n in names]
    for r in (REPLAYS if replays else []):
        clips += [_clip(REPLAY_DIR / r, n, r) for n in names]
    return clips


def eval_clips():
    meta = pd.read_csv(TAU_EVAL_DIR / "meta.csv", sep="\t")
    return [(TAU_EVAL_DIR / f, str(d).lower()) for f, d in zip(meta["filename"], meta["source_label"])]


def device_index(device):
    return TRAIN_DEVICES.index(device) if device in TRAIN_DEVICES else -1


def mel_path(clip):
    return MEL_DIR / clip.device / f"{clip.path.stem}.npy"


def load_mel(clip):
    return torch.from_numpy(np.load(mel_path(clip))).float()[None]


class MelDataset(Dataset):
    def __init__(self, clips, unit=False):
        self.clips = clips
        self.unit = unit

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        c = self.clips[i]
        x = load_mel(c)
        return (to_unit(x) if self.unit else x), c.scene, device_index(c.device)


class DiffusionPool(Dataset):
    def __init__(self, root=POOL_DIR, size=None):
        self.mels = np.load(root / "mels.npy", mmap_mode="r")
        devices = np.load(root / "devices.npy")
        scenes = np.load(root / "scenes.npy")
        idx = np.arange(len(devices))
        if size is not None:
            per = size // (len(TRAIN_DEVICES) * len(SCENES))
            idx = np.concatenate([
                np.flatnonzero((devices == d) & (scenes == c))[:per]
                for d in range(len(TRAIN_DEVICES)) for c in range(len(SCENES))
            ])
        self.idx = idx
        self.devices = devices[idx]
        self.scenes = scenes[idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.mels[self.idx[i]], dtype=np.float32))[None]
        return x, int(self.scenes[i]), int(self.devices[i])

    def indices(self, device):
        return np.flatnonzero(self.devices == TRAIN_DEVICES.index(device))


def build_augmentations(aug="both", pool_dir=POOL_DIR, pool_size=None, n_irs=None, seed=0, clusters=IR_CLUSTER_FILE):
    pool = DiffusionPool(pool_dir, pool_size) if aug in ("diff", "both") and pool_size != 0 else None
    irs = load_bank(n_irs, seed, clusters) if aug in ("ir", "both") else None
    return pool, irs


class ClassifierData(Dataset):
    def __init__(self, clips, pool=None, irs=None, p_ir=0.5):
        self.clips = clips
        self.pool = pool
        self.irs = irs
        self.p_ir = p_ir
        self.logmel = LogMel()

    def __len__(self):
        return len(self.clips) + (len(self.pool) if self.pool is not None else 0)

    def __getitem__(self, i):
        if i >= len(self.clips):
            x, c, d = self.pool[i - len(self.clips)]
            return to_db(x), c, d
        clip = self.clips[i]
        if self.irs is not None and clip.device != REFERENCE and random.random() < self.p_ir:
            h = self.irs.sample(random.choice(self.irs.classes))
            x = self.logmel(self.irs.apply(load_audio(clip.path), h))[None]
        else:
            x = load_mel(clip)
        return x, clip.scene, device_index(clip.device)


def budget_sampler(data):
    return RandomSampler(data, num_samples=len(data.clips) + FULL_POOL)


class CycleSampler:
    def __init__(self, clips, pool=None, irs=None, batch=32, n_devices=4, n_irs=4, per_source=4):
        self.batch = batch
        self.pool, self.irs = pool, irs
        self.n_devices = n_devices
        self.n_irs = n_irs if irs is not None else 0
        self.ir_clips = per_source
        self.device_clips = (batch - self.n_irs * per_source) // n_devices
        self.real = {d: [c for c in clips if c.device == d] for d in TRAIN_DEVICES}
        self.fake = {d: pool.indices(d) for d in TRAIN_DEVICES} if pool is not None else None
        self.b_clips = [c for c in clips if c.device != REFERENCE]
        self.steps = (len(self.real[REFERENCE]) + FULL_POOL // len(TRAIN_DEVICES)) // batch
        self.logmel = LogMel()

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(len(self)):
            yield self.draw()

    def _real(self, d, n):
        return [to_unit(load_mel(c)) for c in random.choices(self.real[d], k=n)]

    def _fake(self, d, n):
        return [self.pool[i][0] for i in np.random.choice(self.fake[d], n)]

    def _mix(self, d, n):
        if self.pool is None:
            return self._real(d, n)
        k = n // 2
        return self._real(d, k) + self._fake(d, n - k)

    def _ir(self, m, n):
        out = []
        for c in random.choices(self.b_clips, k=n):
            wave = self.irs.apply(load_audio(c.path), self.irs.sample(m))
            out.append(to_unit(self.logmel(wave))[None])
        return out

    def draw(self):
        xa = self._mix(REFERENCE, self.batch)
        xb, sb = [], []
        for d in random.sample(DOMAIN_B, self.n_devices):
            xb += self._mix(d, self.device_clips)
            sb += [DOMAIN_B.index(d)] * self.device_clips
        if self.n_irs:
            classes = self.irs.classes
            picks = random.sample(classes, self.n_irs) if len(classes) >= self.n_irs else random.choices(classes, k=self.n_irs)
            for m in picks:
                xb += self._ir(m, self.ir_clips)
                sb += [len(DOMAIN_B) + m] * self.ir_clips
        return torch.stack(xa), torch.stack(xb), torch.tensor(sb)
