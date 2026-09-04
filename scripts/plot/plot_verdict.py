#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from style import FIGSIZE, KERNEL_COLORS, THROUGHPUT_LABEL


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsons", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bw-copy", type=Path, help="bw_copy JSON for 16 B roof")
    parser.add_argument("--curand", type=Path, help="curand_philox JSON")
    args = parser.parse_args()

    labels = []
    values = []
    colors = []
    for path in args.jsons:
        rec = load(path)
        name = rec["kernel"]
        labels.append(name)
        values.append(rec["gsamples_per_s"])
        colors.append(KERNEL_COLORS.get(name, "#4c78a8"))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel(THROUGHPUT_LABEL)
    if args.bw_copy is not None:
        bw = load(args.bw_copy)
        roof = bw["achieved_gbps"] / 16.0
        ax.axhline(roof, color=KERNEL_COLORS["bw_roof"], linestyle="--",
                   label=f"16 B roof {roof:.2f}")
    if args.curand is not None:
        curand = load(args.curand)
        ax.axhline(curand["gsamples_per_s"], color=KERNEL_COLORS["curand_raw"],
                   linestyle=":", label="cuRAND Philox raw")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
