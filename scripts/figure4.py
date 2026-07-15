#!/usr/bin/env python3
"""Plot Figure 4 from precomputed scaling tables.

This script expects files named like:
scaling_momentsFid_U_<U>__K_<K>_T_<T>_rank_<r>kapa_<kappa>
"""

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

from rqc_randompermutations import find_corresponding_index


OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "8"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Fig. 4 from scaling tables.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RESULTS_DIR / "fig4")
    parser.add_argument("--variant", choices=["v1", "v2"], default="v2", help="v1: CUE-only, v2: CUE and FF")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "Fig4v2.pdf")
    parser.add_argument("--moment", type=int, default=2)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 15])
    parser.add_argument("--ranks-ff", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--kapas", type=float, nargs="+", default=[0.0005, 0.025, 0.3])
    parser.add_argument("--Ts", type=int, nargs="+", default=[1, 8, 10, 12])
    return parser


def find_first(root: Path, filename: str):
    candidates = [filename, f"{filename}.txt"]
    for candidate in candidates:
        matches = list(root.rglob(candidate))
        if matches:
            if len(matches) > 1:
                print(f"Warning: multiple matches for {candidate}. Using: {matches[0]}")
            return matches[0]
    return None


def load_scaling_data(root: Path, ranks: list[int], kapas: list[float], Ts: list[int]):
    ensembles = ["CUE", "FF"]
    data_l = []
    descriptors_l = []
    for ensemble_u in ensembles:
        for ensemble_k in ensembles:
            for rank in ranks:
                for kapa in kapas:
                    for T in Ts:
                        name = f"scaling_momentsFid_U_{ensemble_u}__K_{ensemble_k}_T_{T}_rank_{rank}kapa_{kapa}"
                        path = find_first(root, name)
                        if path is None:
                            print(f"{ensemble_u} {ensemble_k} {rank} {kapa} {T} does not exist")
                            continue
                        try:
                            avgfid = np.loadtxt(path)
                            descriptors_l.append([ensemble_u, ensemble_k, str(rank), str(kapa), str(T)])
                            data_l.append(avgfid)
                        except Exception as exc:
                            print(f"Failed to load {path}: {exc}")
    return data_l, descriptors_l


def _build_indices(descriptors_l: list[list[str]], ranksff: list[int]):
    out = {
        "cue_00005_tl": [],
        "cue_0025_tl": [],
        "cue_03_tl": [],
        "ff_00005_tl": [],
        "ff_0025_tl": [],
        "ff_03_tl": [],
    }
    for rank in ranksff:
        out["cue_00005_tl"].append(find_corresponding_index(descriptors_l, ["CUE", "CUE", str(rank), "0.0005", "10"]))
        out["cue_0025_tl"].append(find_corresponding_index(descriptors_l, ["CUE", "CUE", str(rank), "0.025", "10"]))
        out["cue_03_tl"].append(find_corresponding_index(descriptors_l, ["CUE", "CUE", str(rank), "0.3", "10"]))
        out["ff_00005_tl"].append(find_corresponding_index(descriptors_l, ["FF", "FF", str(rank), "0.0005", "10"]))
        out["ff_0025_tl"].append(find_corresponding_index(descriptors_l, ["FF", "FF", str(rank), "0.025", "10"]))
        out["ff_03_tl"].append(find_corresponding_index(descriptors_l, ["FF", "FF", str(rank), "0.3", "10"]))
    return out


def _plot_pair(ax_top, ax_bottom, data_l, idx, rank_color, moment, include_ff, ff_idx=None):
    ls = data_l[idx][:, 0]
    delta = data_l[idx][:, moment] - np.power(data_l[idx][:, 1], moment)
    phi = np.divide(np.power(np.abs(delta), 1 / moment), np.abs(data_l[idx][:, 1]))
    ax_top.plot(ls, np.abs(delta), MARKERS[0], fillstyle="full", color=rank_color, linestyle=":")
    ax_bottom.plot(ls, phi, MARKERS[0], fillstyle="full", color=rank_color, linestyle=":")

    if include_ff and ff_idx is not None:
        ls_f = data_l[ff_idx][:, 0]
        delta_f = data_l[ff_idx][:, moment] - np.power(data_l[ff_idx][:, 1], moment)
        phi_f = np.divide(np.power(np.abs(delta_f), 1 / moment), np.abs(data_l[ff_idx][:, 1]))
        ax_top.plot(ls_f, np.abs(delta_f), MARKERS[0], fillstyle="none", color=rank_color, linestyle=":")
        ax_bottom.plot(ls_f, phi_f, MARKERS[0], fillstyle="none", color=rank_color, linestyle=":")


def plot_figure4(data_l, descriptors_l, ranksff, moment, variant, output: Path):
    indices = _build_indices(descriptors_l, ranksff)
    include_ff = variant == "v2"

    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    ax = ax.ravel()

    dummy_handles = []
    for i in range(len(ranksff)):
        _plot_pair(
            ax[0],
            ax[3],
            data_l,
            indices["cue_00005_tl"][i],
            OKABE_ITO[i],
            moment,
            include_ff,
            indices["ff_00005_tl"][i] if include_ff else None,
        )
        _plot_pair(
            ax[1],
            ax[4],
            data_l,
            indices["cue_0025_tl"][i],
            OKABE_ITO[i],
            moment,
            include_ff,
            indices["ff_0025_tl"][i] if include_ff else None,
        )
        _plot_pair(
            ax[2],
            ax[5],
            data_l,
            indices["cue_03_tl"][i],
            OKABE_ITO[i],
            moment,
            include_ff,
            indices["ff_03_tl"][i] if include_ff else None,
        )
        line, = ax[5].plot([], [], color=OKABE_ITO[i], label=str(ranksff[i]))
        dummy_handles.append(line)

    line, = ax[5].plot([], [], MARKERS[0], fillstyle="full", color="gray", linestyle="None", label=r"$u, k_\mu\in\mathrm{CUE}$")
    dummy_handles.append(line)
    if include_ff:
        line, = ax[5].plot([], [], MARKERS[0], fillstyle="none", color="gray", linestyle="None", label=r"$u, k_\mu\in\mathrm{FF}$")
        dummy_handles.append(line)

    ax[1].axvline(6.6213, linestyle=":", c="gray")
    ax[0].set_ylabel(r"$\Delta_" + str(moment) + r"$", fontsize=18)
    ax[3].set_ylabel(r"$\Phi_" + str(moment) + r"$", fontsize=18)

    ax[3].set_xlabel("$L=T$", fontsize=18)
    ax[4].set_xlabel("$L=T$", fontsize=18)
    ax[5].set_xlabel("$L=T$", fontsize=18)

    panel_labels = ["$(a)$", "$(b)$", "$(c)$", "$(d)$", "$(e)$", "$(f)$"]
    for i, label in enumerate(panel_labels):
        ax[i].annotate(label, [0.05, 0.05], fontsize=18, xycoords="axes fraction")

    ax[2].set_yscale("log")
    ax[5].set_yscale("log")
    ax[0].set_title("Deviation", fontsize=18)
    ax[1].set_title("Weak concentration", fontsize=18)
    ax[2].set_title("Strong concentration", fontsize=18)

    xbins = "auto"
    ybins = "auto"
    sci = False
    for i in range(6):
        a = ax[i]
        a.tick_params(labelsize=18)
        a.tick_params(which="both", width=1.0, length=4, direction="in", top=True, right=True)
        a.tick_params(which="minor", length=2.5)

        if a.get_xscale() == "log":
            a.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
            a.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=tuple(d / 10 for d in range(2, 10))))
            a.xaxis.set_major_formatter(mpl.ticker.LogFormatter(base=10.0))
        else:
            a.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=xbins, prune=None, steps=[1, 2, 2.5, 5, 10]))
            a.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            if sci:
                a.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useMathText=True)

        if a.get_yscale() == "log":
            a.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
            a.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=tuple(d / 10 for d in range(2, 10))))
            a.yaxis.set_major_formatter(mpl.ticker.LogFormatter(base=10.0))
        else:
            a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=ybins, prune=None, steps=[1, 2, 2.5, 5, 10]))
            a.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            if sci:
                a.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.legend(
        handles=dummy_handles,
        labels=[h.get_label() for h in dummy_handles],
        title=r"Rank $r$",
        alignment="left",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=6,
        fontsize=14,
        title_fontsize=14,
        frameon=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure: {output}")


def main() -> None:
    args = _build_parser().parse_args()
    data_l, descriptors_l = load_scaling_data(args.data_root, args.ranks, args.kapas, args.Ts)
    if not data_l:
        raise FileNotFoundError(f"No scaling files found under {args.data_root}")
    plot_figure4(data_l, descriptors_l, args.ranks_ff, args.moment, args.variant, args.output)


if __name__ == "__main__":
    main()