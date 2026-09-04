#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import FIGSIZE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=160)
    args = parser.parse_args()

    meta = json.loads(args.meta.read_text())
    binary = Path(meta["binary"])
    xyzw = np.fromfile(binary, dtype=np.float32).reshape(-1, 4)
    x = xyzw[:, 0]
    z = xyzw[:, 2]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist2d(x, z, bins=args.bins, range=[[x.min(), x.max()], [z.min(), z.max()]])
    ax.set_aspect("equal")
    ax.set_xlabel("x (a0)")
    ax.set_ylabel("z (a0)")
    nlm = meta.get("nlm", [0, 0, 0])
    ax.set_title(f"n={nlm[0]} l={nlm[1]} m={nlm[2]}  N={meta['n']}")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
