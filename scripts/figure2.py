#!/usr/bin/env python3
"""Plot Fig. 2 panels from the generated fidelity tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT / "src"))

from rqc_randompermutations import fidtheor, find_corresponding_index


OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "x"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Fig. 2a/b/c.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RESULTS_DIR / "fig2")
    parser.add_argument("--panel", choices=["a", "b", "c", "all"], default="all")
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "Fig2.pdf")
    return parser


def find_first(root: Path, filename: str):
    matches = list(root.rglob(filename))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"Warning: multiple matches for {filename}. Using: {matches[0]}")
    return matches[0]


def load_fidelity_tables(root: Path, L: int = 6):
    ranks = [1, 2, 4, 8, 15]
    Ts = [1, 6]
    ensembles = ["CUE", "FF"]
    data = []
    descriptors = []

    for ensemble_u in ensembles:
        for ensemble_k in ensembles:
            for rank in ranks:
                for T in Ts:
                    base = f"momentsFid_U_{ensemble_u}__K_{ensemble_k}_L_{L}_T_{T}_rank_{rank}_smallkappas"
                    path = find_first(root, base)
                    if path is None:
                        path = find_first(root, base + ".txt")
                    if path is None:
                        continue
                    avgfid = np.loadtxt(path)
                    descriptors.append([ensemble_u, ensemble_k, str(rank), str(T)])
                    data.append(avgfid)

    return data, descriptors, ranks


def _style_axes(ax, label, title=None):
    ax.set_xlabel(r"$\kappa$", fontsize=20)
    ax.set_ylabel(r"$\overline{\mathcal{F}}$", fontsize=20)
    ax.annotate(label, [-0.01, 0.02], fontsize=24)
    if title:
        ax.set_title(title)
    ax.tick_params(labelsize=18)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins="auto", prune=None, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins="auto", prune=None, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(which="both", width=1.0, length=4)
    ax.tick_params(which="minor", length=2.5)
    ax.tick_params(axis="both", direction="in", top=True, right=True)
    ax.ticklabel_format(style="sci", scilimits=(-3, 3), useMathText=True)


def plot_panel_a(ax, data, descriptors, L=6):
    moment = 1
    indices6 = [
        find_corresponding_index(descriptors, ["CUE", "CUE", "4", "6"]),
        find_corresponding_index(descriptors, ["FF", "CUE", "4", "6"]),
        find_corresponding_index(descriptors, ["CUE", "FF", "4", "6"]),
    ]
    indices1 = [
        find_corresponding_index(descriptors, ["CUE", "CUE", "4", "1"]),
        find_corresponding_index(descriptors, ["FF", "CUE", "4", "1"]),
        find_corresponding_index(descriptors, ["CUE", "FF", "4", "1"]),
    ]

    legend_handles = []
    for i in range(len(indices6)):
        ax.plot(data[indices6[i]][:, 0], data[indices6[i]][:, moment], MARKERS[i], color=OKABE_ITO[i], markevery=slice(i, None, 3))
        ax.plot(data[indices1[i]][:, 0], data[indices1[i]][:, moment], MARKERS[i], color=OKABE_ITO[i], markevery=slice(i, None, 3))
        legend_handles.append(
            Line2D([], [], marker=MARKERS[i], color=OKABE_ITO[i], linestyle="None", label=r"$u\in\ \mathrm{" + descriptors[indices6[i]][0] + r"}$ , $k_{\mu}\in\ \mathrm{" + descriptors[indices6[i]][1] + r"}$")
        )

    x = np.linspace(data[indices1[0]][0, 0], data[indices1[0]][-1, 0], 100)
    ax.plot(x, fidtheor(x, L, 6), "-", linewidth=1.0, color="black")
    ax.plot(x, fidtheor(x, L, 1), "-", linewidth=1.0, color="black")
    legend_handles.append(Line2D([], [], color="black", linestyle="-", label=r"$\overline{\mathcal{F}}_{\mathrm{theor}}(\kappa)$"))

    ax.legend(handles=legend_handles, fontsize=16, ncol=2)
    ax.annotate("$T=6$", [0.28, 0.1], fontsize=18)
    ax.annotate("$T=1$", [0.28, 0.55], fontsize=18)
    _style_axes(ax, "$(a)$")


def plot_panel_b(ax, data, descriptors, L=6):
    moment = 1
    ranks = [1, 2, 4, 8, 15]
    indices6 = [find_corresponding_index(descriptors, ["FF", "CUE", str(rank), "6"]) for rank in ranks]
    indices1 = [find_corresponding_index(descriptors, ["FF", "CUE", str(rank), "1"]) for rank in ranks]

    legend_handles = []
    for i in range(len(indices6)):
        ax.plot(data[indices6[i]][:, 0], data[indices6[i]][:, moment], MARKERS[i], color=OKABE_ITO[i], markevery=slice(i, None, 5))
        ax.plot(data[indices1[i]][:, 0], data[indices1[i]][:, moment], MARKERS[i], color=OKABE_ITO[i], markevery=slice(i, None, 5))
        legend_handles.append(Line2D([], [], marker=MARKERS[i], color=OKABE_ITO[i], linestyle="None", label=r"$r =\ $" + str(ranks[i])))

    x = np.linspace(data[indices1[0]][0, 0], data[indices1[0]][-1, 0], 100)
    ax.plot(x, fidtheor(x, L, 6), "-", linewidth=1.0, color="black")
    ax.plot(x, fidtheor(x, L, 1), "-", linewidth=1.0, color="black")
    legend_handles.append(Line2D([], [], color="black", linestyle="-", label=r"$\overline{\mathcal{F}}_{\mathrm{theor}}(\kappa)$"))

    ax.legend(handles=legend_handles, fontsize=16, ncol=2)
    ax.annotate("$T=6$", [0.28, 0.1], fontsize=18)
    ax.annotate("$T=1$", [0.28, 0.55], fontsize=18)
    _style_axes(ax, "$(b)$")


def plot_panel_c(ax, data, descriptors, L=6):
    moment = 1
    ranks = [1, 2, 4, 8]
    indices6 = [find_corresponding_index(descriptors, ["FF", "FF", str(rank), "6"]) for rank in ranks]
    indices1 = [find_corresponding_index(descriptors, ["FF", "FF", str(rank), "1"]) for rank in ranks]
    ref_cue = [
        find_corresponding_index(descriptors, ["FF", "CUE", "2", "6"]),
        find_corresponding_index(descriptors, ["FF", "CUE", "2", "1"]),
    ]

    legend_handles = []
    for i in range(len(indices6)):
        ax.plot(data[indices6[i]][:, 0], data[indices6[i]][:, moment], MARKERS[i], color=OKABE_ITO[i], linestyle=":", markevery=slice(i, None, 2))
        ax.plot(data[indices1[i]][:, 0], data[indices1[i]][:, moment], MARKERS[i], color=OKABE_ITO[i], linestyle=":", markevery=slice(i, None, 2))
        legend_handles.append(Line2D([], [], marker=MARKERS[i], color=OKABE_ITO[i], linestyle=":", label=r"$r =\ $" + str(ranks[i])))

    x = np.linspace(data[indices1[0]][0, 0], data[indices1[0]][-1, 0], 100)
    ax.plot(x, fidtheor(x, L, 6), "-", linewidth=1.0, color="black")
    ax.plot(x, fidtheor(x, L, 1), "-", linewidth=1.0, color="black")
    ax.plot(data[ref_cue[0]][:, 0], data[ref_cue[0]][:, moment], color="gray", linestyle="--")
    ax.plot(data[ref_cue[1]][:, 0], data[ref_cue[1]][:, moment], color="gray", linestyle="--")
    legend_handles.append(Line2D([], [], color="gray", linestyle="--", label=r"$u\in\ \mathrm{FF}$" "\n" r"$k_\mu\in\ \mathrm{CUE}$"))
    legend_handles.append(Line2D([], [], color="black", linestyle="-", label=r"$\overline{\mathcal{F}}_{\mathrm{theor}}(\kappa)$"))

    ax.legend(handles=legend_handles, fontsize=16, ncol=2)
    ax.annotate("$T=6$", [0.28, 0.1], fontsize=18)
    ax.annotate("$T=1$", [0.28, 0.55], fontsize=18)
    _style_axes(ax, "$(c)$")


def main() -> None:
    args = _build_parser().parse_args()
    data, descriptors, _ = load_fidelity_tables(args.data_root, L=args.L)

    if not data:
        raise FileNotFoundError(f"No fidelity tables found under {args.data_root}")

    if args.panel == "all":
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        plot_panel_a(axes[0], data, descriptors)
        plot_panel_b(axes[1], data, descriptors)
        plot_panel_c(axes[2], data, descriptors)
        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved figure: {args.output}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    if args.panel == "a":
        plot_panel_a(ax, data, descriptors)
    elif args.panel == "b":
        plot_panel_b(ax, data, descriptors)
    else:
        plot_panel_c(ax, data, descriptors)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()