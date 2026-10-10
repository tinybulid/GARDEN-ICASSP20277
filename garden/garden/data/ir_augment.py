import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from ..utils import IR_CLUSTER_FILE, IR_DIR
from .features import N_FFT, N_MELS, SR, load_audio


def ir_files(root=IR_DIR):
    return sorted(root.rglob("*.wav"))


def band_profiles(irs):
    fb = AF.melscale_fbanks(N_FFT // 2 + 1, 0.0, SR / 2, N_MELS, SR)
    mag = torch.stack([torch.fft.rfft(h, n=N_FFT).abs() for h in irs])
    r = 20 * torch.log10(mag @ fb + 1e-8)
    r = r - r.mean(1, keepdim=True)
    return F.normalize(r, dim=1)


def _kpp(q, M, g):
    idx = [torch.randint(len(q), (1,), generator=g).item()]
    for _ in range(M - 1):
        dist = (1 - q @ q[idx].T).min(1).values.clamp_min(1e-12)
        idx.append(torch.multinomial(dist, 1, generator=g).item())
    return q[idx]


def spherical_kmeans(q, M=32, n_init=10, iters=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    best, best_score = None, -float("inf")
    for _ in range(n_init):
        mu = _kpp(q, M, g)
        for _ in range(iters):
            sim = q @ mu.T
            labels = sim.argmax(1)
            new = torch.zeros_like(mu).index_add_(0, labels, q)
            empty = new.norm(dim=1) == 0
            if empty.any():
                new[empty] = q[sim.max(1).values.argsort()[: int(empty.sum())]]
            new = F.normalize(new, dim=1)
            if torch.allclose(new, mu):
                break
            mu = new
        labels = (q @ mu.T).argmax(1)
        score = (q * mu[labels]).sum().item()
        if score > best_score:
            best, best_score = (labels, mu), score
    return best


def cluster_irs(M=32, seed=0, out=IR_CLUSTER_FILE):
    files = ir_files()
    q = band_profiles([load_audio(f, length=None) for f in files])
    labels, centers = spherical_kmeans(q, M, seed=seed)
    np.savez(out, labels=labels.numpy(), centers=centers.numpy(), files=np.array([str(f) for f in files]))
    return labels, centers


def ir_subset(n, seed=0, clusters=IR_CLUSTER_FILE):
    K = len(np.load(clusters)["labels"])
    if n is None or n >= K:
        return None
    return sorted(random.Random(seed).sample(range(K), n))


def load_bank(n_irs=None, seed=0, clusters=IR_CLUSTER_FILE):
    if n_irs == 0:
        return None
    return IRBank(clusters, keep=ir_subset(n_irs, seed, clusters))


class IRBank:
    def __init__(self, clusters=IR_CLUSTER_FILE, keep=None):
        meta = np.load(clusters)
        self.labels = meta["labels"]
        self.n_classes = len(meta["centers"])
        keep = range(len(self.labels)) if keep is None else keep
        self.irs = {k: load_audio(meta["files"][k], length=None) for k in keep}
        self.by_class = defaultdict(list)
        for k in keep:
            self.by_class[int(self.labels[k])].append(k)
        self.classes = sorted(self.by_class)

    def sample(self, m):
        return self.irs[random.choice(self.by_class[m])]

    @staticmethod
    def apply(wave, h):
        return AF.fftconvolve(wave, h)[: wave.numel()]
