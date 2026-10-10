import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from garden.data.ir_augment import cluster_irs
from garden.models.classifiers import BACKBONES
from garden.utils import CKPT_DIR, DATA_DIR, IR_CLUSTER_FILE, POOL_DIR, ROOT

SCRIPTS = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "fig2"
MODES = ("cond", "uncond", "none")
STAGES = ("prep", "train", "collect", "plot")
DEFAULT = {"aug": "both", "n_irs": 320, "pool_size": 100_000, "clusters": 32, "wd": 6.0, "wc": 2.0}
W_SUM = DEFAULT["wd"] + DEFAULT["wc"]
PANELS = {
    "a": ("n_irs", [0, 8, 16, 33, 66, 100, 150, 200, 240, 268, 320]),
    "b": ("pool_size", [0, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 75_000, 90_000, 100_000]),
    "c": ("clusters", [4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 240]),
    "d": ("dw", [-6, -4, -2, 0, 2, 4, 6, 8]),
    "e": ("aug", ["none", "diff", "ir", "both"]),
}
TITLES = {
    "a": "a) Number of IR sources used",
    "b": "b) Number of generated diffusion samples",
    "c": "c) IR clusters for conditioning F",
    "d": "d) Guidance balance Δw = wd − wc",
    "e": "e) Augmentation setting",
}
SERIES = [
    ("known_cond", "Known (full)", "0.3", "o-"),
    ("unseen_cond", "Unseen (full)", "tab:blue", "s-"),
    ("unseen_uncond", "Unseen: w/o cond. F", "orange", "^--"),
    ("unseen_none", "Unseen: w/o F", "hotpink", "d:"),
]


def config(key, value):
    cfg = dict(DEFAULT)
    if key == "dw":
        cfg["wd"], cfg["wc"] = (W_SUM + value) / 2, (W_SUM - value) / 2
    else:
        cfg[key] = value
    return cfg


def tag(cfg):
    return f"{cfg['aug']}_k{cfg['n_irs']}_p{cfg['pool_size']}_m{cfg['clusters']}_wd{cfg['wd']:g}_wc{cfg['wc']:g}"


def clusters_file(cfg):
    if cfg["clusters"] == DEFAULT["clusters"]:
        return IR_CLUSTER_FILE
    return DATA_DIR / f"ir_clusters_m{cfg['clusters']}.npz"


def pool_dir(cfg):
    if (cfg["wd"], cfg["wc"]) == (DEFAULT["wd"], DEFAULT["wc"]):
        return POOL_DIR
    return DATA_DIR / f"diffusion_pool_wd{cfg['wd']:g}_wc{cfg['wc']:g}"


def data_args(cfg, seed):
    return [
        "--aug", cfg["aug"], "--n-irs", cfg["n_irs"], "--pool-size", cfg["pool_size"],
        "--pool", pool_dir(cfg), "--ir-clusters", clusters_file(cfg), "--seed", seed,
    ]


def gan_name(cfg, mode):
    return f"fig2_{mode}_{tag(cfg)}"


def clf_name(cfg, mode, backbone):
    return f"{gan_name(cfg, mode)}_{backbone}"


def base_name(backbone):
    return f"fig2_base_{backbone}"


def conditions(panels):
    found = {}
    for p in panels:
        key, values = PANELS[p]
        for v in values:
            cfg = config(key, v)
            found[tag(cfg)] = cfg
    return list(found.values())


def exists(name):
    return (CKPT_DIR / f"{name}.pt").exists()


def run(script, *args):
    subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], check=True)


def prepare(panels, args):
    for cfg in conditions(panels):
        f = clusters_file(cfg)
        if not f.exists():
            cluster_irs(M=cfg["clusters"], seed=args.seed, out=f)
        d = pool_dir(cfg)
        if not (d / "scenes.npy").exists():
            run("gen_pool.py", "--wd", cfg["wd"], "--wc", cfg["wc"], "--out", d, "--seed", args.seed)


def train(panels, args):
    for b in BACKBONES:
        if not exists(base_name(b)):
            run("tra_clf.py", "--backbone", b, "--no-garden", "--epochs", args.clf_epochs,
                "--name", base_name(b), *data_args(DEFAULT, args.seed))
    for cfg in conditions(panels):
        for mode in MODES:
            g = gan_name(cfg, mode)
            if not exists(f"{g}_G"):
                run("tra_gan.py", "--mode", mode, "--epochs", args.gan_epochs, "--hold", args.gan_epochs // 2,
                    "--name", g, *data_args(cfg, args.seed))
            for b in BACKBONES:
                n = clf_name(cfg, mode, b)
                if not exists(n):
                    run("tra_clf.py", "--backbone", b, "--gan", f"{g}_G.pt", "--epochs", args.clf_epochs,
                        "--name", n, *data_args(cfg, args.seed))


def collect(panels):
    cache = {}

    def score(n):
        if n not in cache:
            cache[n] = torch.load(CKPT_DIR / f"{n}.pt", map_location="cpu")["scores"] if exists(n) else None
        return cache[n]

    def avg(names, key):
        vals = [s[key] for s in map(score, names) if s is not None]
        return float(np.mean(vals)) if vals else math.nan

    results = {"baseline_known": avg([base_name(b) for b in BACKBONES], "known"), "panels": {}}
    for p in panels:
        key, values = PANELS[p]
        rows = []
        for v in values:
            cfg = config(key, v)
            row = {"x": v}
            for mode in MODES:
                names = [clf_name(cfg, mode, b) for b in BACKBONES]
                row[f"known_{mode}"] = avg(names, "known")
                row[f"unseen_{mode}"] = avg(names, "unseen")
            rows.append(row)
        results["panels"][p] = rows

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "fig2.json", "w") as f:
        json.dump(results, f, indent=2)
    for p, rows in results["panels"].items():
        with open(OUT_DIR / f"fig2_{p}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return results


def label(p, v):
    if p == "b":
        return f"{v // 1000}k" if v else "0"
    if p == "e":
        return {"none": "None", "diff": "Diffusion\nOnly", "ir": "IR\nOnly", "both": "Both"}[v]
    return str(v)


def plot(results):
    import matplotlib.pyplot as plt

    panels = results["panels"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6), sharey=True, squeeze=False)
    for ax, (p, rows) in zip(axes[0], panels.items()):
        xs = np.arange(len(rows))
        if p == "e":
            w = 0.2
            for i, (k, name, color, _) in enumerate(SERIES):
                ax.bar(xs + (i - 1.5) * w, [r[k] for r in rows], w, color=color, label=name)
        else:
            for k, name, color, style in SERIES:
                ax.plot(xs, [r[k] for r in rows], style, color=color, label=name, markersize=3)
        ax.axhline(results["baseline_known"], color="red", ls="--", label="Known baseline")
        ax.set_xticks(xs)
        ax.set_xticklabels([label(p, r["x"]) for r in rows], rotation=0 if p == "e" else 90, fontsize=7)
        ax.set_xlabel(TITLES[p])
    axes[0][0].set_ylabel("Accuracy (%)")
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=5, frameon=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig2.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2.pdf", bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--panels", nargs="+", choices=tuple(PANELS), default=list(PANELS))
    p.add_argument("--gan-epochs", type=int, default=400)
    p.add_argument("--clf-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if "prep" in args.stages:
        prepare(args.panels, args)
    if "train" in args.stages:
        train(args.panels, args)
    results = None
    if "collect" in args.stages:
        results = collect(args.panels)
    if "plot" in args.stages:
        if results is None:
            with open(OUT_DIR / "fig2.json") as f:
                results = json.load(f)
        plot(results)


if __name__ == "__main__":
    main()
