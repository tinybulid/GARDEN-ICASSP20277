
import argparse
from pathlib import Path

from garden.data.ir_augment import cluster_irs
from garden.utils import IR_CLUSTER_FILE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=IR_CLUSTER_FILE)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cluster_irs(M=args.clusters, seed=args.seed, out=args.out)


if __name__ == "__main__":
    main()
