#!/usr/bin/env python3
"""Plot Figure 5 from precomputed time-scan tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT / "src"))

from rqc_randompermutations import fidtheorfull


OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "x"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Figure 5 from momentsFid files.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RESULTS_DIR / "fig5")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "Fig5.pdf")
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--ensemble-U", default="CUE")
    parser.add_argument("--ensemble-K", default="CUE")
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.1, 0.015])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.1, 0.015])
    parser.add_argument("--ps", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.015])
    return parser


def _fmt(x: float) -> str:
    return str(x)


def find_first(root: Path, filename: str):
    candidates = [filename, f"{filename}.txt"]
    for candidate in candidates:
        matches = list(root.rglob(candidate))
        if matches:
            if len(matches) > 1:
                print(f"Warning: multiple matches for {candidate}. Using: {matches[0]}")
            return matches[0]
    return None


def main() -> None:
    args = _build_parser().parse_args()

    data = []
    descriptors = []
    for kapa in args.kappas:
        for alpha in args.alphas:
            for p in args.ps:
                name = (
                    f"momentsFid_U_{args.ensemble_U}__K_{args.ensemble_K}"
                    f"_L_{args.L}_rank_{args.rank}"
                    f"_kappa_{_fmt(kapa)}_alpha_{_fmt(alpha)}_p_{_fmt(p)}"
                )
                path = find_first(args.data_root, name)
                if path is None:
                    print(f"{kapa} {alpha} {p} does not exist")
                    continue
                try:
                    avgfid = np.loadtxt(path)
                    descriptors.append([str(kapa), str(alpha), str(p)])
                    data.append(avgfid)
                except Exception as exc:
                    print(f"Failed to load {path}: {exc}")

    if not data:
        raise FileNotFoundError(f"No figure5 tables found under {args.data_root}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(descriptors)):
        color = OKABE_ITO[i % len(OKABE_ITO)]
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(data[i][:, 0], data[i][:, 1], marker, color=color, label=descriptors[i][0] + " , " + descriptors[i][1] + " , " + descriptors[i][2])
        ax.plot(
            data[i][:, 0],
            fidtheorfull(float(descriptors[i][0]), float(descriptors[i][1]), float(descriptors[i][2]), args.L, data[i][-1, 0]),
            "-",
            color=color,
        )

    ax.set_ylabel(r"$\overline{\mathcal{F}}$", fontsize=20)
    ax.set_xlabel(r"$T$", fontsize=20)
    ax.legend(title=r"$\kappa\ ,\ \alpha\ ,\ p$", fontsize=14, title_fontsize=16, loc=(0.5, 0.1))
    ax.set_yscale("log")

    xbins = "auto"
    ybins = "auto"
    sci = False

    ax.tick_params(labelsize=18)
    ax.tick_params(which="both", width=1.0, length=4, direction="in", top=True, right=True)
    ax.tick_params(which="minor", length=2.5)

    if ax.get_xscale() == "log":
        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=tuple(d / 10 for d in range(2, 10))))
        ax.xaxis.set_major_formatter(mpl.ticker.LogFormatter(base=10.0))
    else:
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=xbins, prune=None, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        if sci:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useMathText=True)

    if ax.get_yscale() == "log":
        ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=tuple(d / 10 for d in range(2, 10))))
        ax.yaxis.set_major_formatter(mpl.ticker.LogFormatter(base=10.0))
    else:
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=ybins, prune=None, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        if sci:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
