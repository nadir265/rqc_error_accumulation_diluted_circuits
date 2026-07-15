#!/usr/bin/env python3
"""Generate the data tables used for Fig. 2 of the paper.

This reproduces the notebook logic for the average fidelity moments as plain
text files, one file per ensemble/rank/time combination.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT / "src"))

from rqc_randompermutations import (
    compute_overlap_pure_dilutedRQCFF,
    compute_overlap_pure_dilutedRQCv1,
    ff_rank,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Fig. 2 fidelity tables.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS_DIR / "fig2")
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--Ts", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 15])
    parser.add_argument("--ensemble-U", nargs="+", default=["CUE", "FF"])
    parser.add_argument("--ensemble-K", nargs="+", default=["CUE", "FF"])
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--divs", type=int, default=15)
    parser.add_argument("--kappa-min", type=float, default=0.0)
    parser.add_argument("--kappa-max", type=float, default=0.3)
    parser.add_argument("--small-kappas", action="store_true", help="Use the historical _smallkappas suffix")
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def _effective_rank(rank: int, ensemble_k: str) -> int:
    return ff_rank(rank) if ensemble_k == "FF" else rank


def main() -> None:
    args = _build_parser().parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    suffix = "_smallkappas" if args.small_kappas else ""
    kappas = np.linspace(args.kappa_min, args.kappa_max, args.divs)

    for ensemble_u in args.ensemble_U:
        for ensemble_k in args.ensemble_K:
            for rank in args.ranks:
                effective_rank = _effective_rank(rank, ensemble_k)
                if ensemble_k == "FF" and rank not in {1, 2, 4, 8}:
                    print(f"Skipping unsupported FF rank {rank}; effective rank is {effective_rank}.")
                    continue

                for T in args.Ts:
                    avg_fs = np.zeros((args.divs, 4), dtype=float)
                    for k, kapa in enumerate(kappas):
                        print(f"Generating U={ensemble_u} K={ensemble_k} rank={rank} T={T} kappa={kapa:.4f}")
                        for _ in range(args.reps):
                            if ensemble_u == "FF":
                                avg = float(np.real(compute_overlap_pure_dilutedRQCFF(
                                    args.L,
                                    T,
                                    float(kapa),
                                    effective_rank,
                                    ensemble_u,
                                    ensemble_k,
                                )))
                            else:
                                avg = float(np.real(compute_overlap_pure_dilutedRQCv1(
                                    args.L,
                                    T,
                                    float(kapa),
                                    effective_rank,
                                    ensemble_u,
                                    ensemble_k,
                                    gamma=0.0,
                                    p=0.0,
                                )))
                            for moment in range(4):
                                avg_fs[k, moment] += avg ** (moment + 1)

                    avg_fs /= args.reps
                    table = np.zeros((args.divs, 5), dtype=float)
                    table[:, 0] = kappas
                    table[:, 1:] = avg_fs

                    filename = (
                        f"momentsFid_U_{ensemble_u}__K_{ensemble_k}"
                        f"_L_{args.L}_T_{T}_rank_{effective_rank}{suffix}.txt"
                    )
                    path = outdir / filename
                    np.savetxt(path, table, header="columns are moment, column 0 is kappa values")
                    print(f"Saved {path}")


if __name__ == "__main__":
    main()