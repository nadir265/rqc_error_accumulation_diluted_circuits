#!/usr/bin/env python3
"""Plot Fig. 3 from the Fig. 2 fidelity tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT / "src"))

from rqc_randompermutations import find_corresponding_index, kapatilde


OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "8"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Fig. 3 using the Fig. 2 data tables.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RESULTS_DIR / "fig2")
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "Fig3v1.pdf")
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
                    candidates = [
                        f"momentsFid_U_{ensemble_u}__K_{ensemble_k}_L_{L}_T_{T}_rank_{rank}_smallkappas.txt",
                        f"momentsFid_U_{ensemble_u}__K_{ensemble_k}_L_{L}_T_{T}_rank_{rank}.txt",
                    ]
                    path = None
                    for name in candidates:
                        path = find_first(root, name)
                        if path is not None:
                            break
                    if path is None:
                        continue
                    avgfid = np.loadtxt(path)
                    descriptors.append([ensemble_u, ensemble_k, str(rank), str(T)])
                    data.append(avgfid)

    return data, descriptors


def main() -> None:
    args = _build_parser().parse_args()
    data, descriptors = load_fidelity_tables(args.data_root, L=args.L)
    if not data:
        raise FileNotFoundError(f"No fidelity tables found under {args.data_root}")

    ranks = [1, 2, 4, 8, 15]
    ranksff = [1, 2, 4, 8]

    indices_cuecue_1 = []
    indices_cuecue_6 = []
    indices_ffff_1 = []
    indices_ffff_6 = []
    for rank in ranks:
        indices_cuecue_1.append(find_corresponding_index(descriptors, ["CUE", "CUE", str(rank), "1"]))
        indices_cuecue_6.append(find_corresponding_index(descriptors, ["CUE", "CUE", str(rank), "6"]))
    for rank in ranksff:
        indices_ffff_1.append(find_corresponding_index(descriptors, ["FF", "FF", str(rank), "1"]))
        indices_ffff_6.append(find_corresponding_index(descriptors, ["FF", "FF", str(rank), "6"]))

    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    rank_index = 4
    index = indices_cuecue_6[rank_index]
    kapas = data[index][:, 0]
    for moment in range(2, 5):
        index = indices_cuecue_6[rank_index]
        ax[0].plot(
            kapas,
            data[index][:, moment],
            MARKERS[moment - 2],
            fillstyle="full",
            color=OKABE_ITO[rank_index],
            markevery=slice(0, None, 2),
        )
        ax[0].plot(
            kapas,
            np.power(data[index][:, 1], moment),
            MARKERS[moment - 2],
            fillstyle="none",
            markersize=10,
            color=OKABE_ITO[rank_index],
            markevery=slice(0, None, 2),
        )

    ax[0].legend(
        handles=[
            Line2D([], [], marker=MARKERS[0], linestyle="None", color=OKABE_ITO[rank_index], label=r"$\overline{\mathcal{F}^k}$", fillstyle="full"),
            Line2D([], [], marker=MARKERS[0], linestyle="None", color=OKABE_ITO[rank_index], label=r"$\overline{\mathcal{F}}^k$", fillstyle="none"),
        ],
        fontsize=14,
        loc="upper right",
        frameon=True,
        borderaxespad=0.5,
    )

    index = indices_cuecue_1[0]
    kapas = data[index][:, 0]
    kapastilde = kapatilde(kapas, args.L)
    moment_handles = []
    rank_handles = []

    for moment in range(2, 5):
        for i in range(len(ranks)):
            index = indices_cuecue_1[i]
            delta = data[index][:, moment] - np.power(data[index][:, 1], moment)
            numerator = 16 * delta[1:] * ranks[i]
            denominator = sp.special.binom(moment, 2) * (1 - kapastilde[1:]) ** (moment - 2) * kapastilde[1:] ** 2
            ax[1].plot(
                kapas[1:],
                np.divide(numerator, denominator),
                MARKERS[moment - 2],
                fillstyle="full",
                color=OKABE_ITO[i],
            )
        moment_handles.append(Line2D([], [], marker=MARKERS[moment - 2], linestyle="None", color="grey", label="$k=$" + str(moment)))

    for i in range(len(ranks)):
        rank_handles.append(Line2D([], [], color=OKABE_ITO[i], linestyle="-", label="$r=$" + str(ranks[i])))

    fig.subplots_adjust(wspace=0, hspace=0.1)
    ax[0].set_ylabel(r"$\overline{\mathcal{F}^k}\ ,\ \overline{\mathcal{F}}^k$", fontsize=20)
    ax[1].set_ylabel(r"$\Delta_k/\Delta_k(\hat{\kappa})$", fontsize=20)
    fig.supxlabel(r"$\kappa$", fontsize=20)
    ax[1].legend(handles=moment_handles + rank_handles, title="Rank $r$", ncol=3, loc="lower left", fontsize=14, title_fontsize=14)
    ax[1].set_ylim(-0.1, 0.4)
    ax[1].annotate(r"$T=1$", xy=(0.45, 0.85), xycoords="axes fraction", fontsize=18)
    ax[0].annotate(r"$T=6$", xy=(0.45, 0.85), xycoords="axes fraction", fontsize=18)
    ax[0].set_yscale("log")

    xbins = "auto"
    ybins = "auto"
    sci = True
    for i in range(2):
        current_ax = ax[i]
        current_ax.tick_params(labelsize=18)
        current_ax.tick_params(which="both", width=1.0, length=4, direction="in", top=True, right=True)
        current_ax.tick_params(which="minor", length=2.5)

        if current_ax.get_xscale() == "log":
            current_ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
            current_ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=tuple(d / 10 for d in range(2, 10))))
            current_ax.xaxis.set_major_formatter(mpl.ticker.LogFormatter(base=10.0))
        else:
            current_ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=xbins, prune=None, steps=[1, 2, 2.5, 5, 10]))
            current_ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            if sci:
                current_ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useMathText=True)

        if current_ax.get_yscale() == "log":
            current_ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
            current_ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=tuple(d / 10 for d in range(2, 10))))
            current_ax.yaxis.set_major_formatter(mpl.ticker.LogFormatter(base=10.0))
        else:
            current_ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=ybins, prune=None, steps=[1, 2, 2.5, 5, 10]))
            current_ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            if sci:
                current_ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()