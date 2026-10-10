import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
CONFIG_DIR = ROOT / "configs"

TAU_DIR = DATA_DIR / "TAU-urban-acoustic-scenes-2022-mobile-development"
TAU_EVAL_DIR = DATA_DIR / "TAU-urban-acoustic-scenes-2025-mobile-evaluation"
SPLIT_DIR = DATA_DIR / "split_setup"
REPLAY_DIR = DATA_DIR / "replays"
IR_DIR = DATA_DIR / "mic_irs"
MEL_DIR = DATA_DIR / "mels"
POOL_DIR = DATA_DIR / "diffusion_pool"
IR_CLUSTER_FILE = DATA_DIR / "ir_clusters.npz"


def load_config(name):
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LabelEmbedding(nn.Module):
    def __init__(self, n, dim=512):
        super().__init__()
        self.table = nn.Embedding(n, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, y):
        return self.mlp(self.table(y))


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p, 1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def save_checkpoint(name, **state):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(state, CKPT_DIR / name)


def load_checkpoint(name, map_location="cpu"):
    return torch.load(CKPT_DIR / name, map_location=map_location)
