#!/usr/bin/env python3
"""Generate data tables for Figure 5.

This reproduces the notebook loop that scans time T for fixed
(kappa, alpha, p) triplets and stores moments of fidelity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT / "src"))

from rqc_randompermutations import compute_overlap_pure_dilutedRQCv1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Figure 5 data tables.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS_DIR / "fig5")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--T-min", type=int, default=1)
    parser.add_argument("--T-max", type=int, default=30)
    parser.add_argument("--ensemble-U", default="CUE")
    parser.add_argument("--ensemble-K", default="CUE")
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.001, 0.01, 0.01, 0.1, 0.015])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.001, 0.01, 0.01, 0.1, 0.015])
    parser.add_argument("--ps", type=float, nargs="+", default=[0.001, 0.0, 0.01, 0.1, 0.015])
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def _fmt(x: float) -> str:
    return str(x)


def main() -> None:
    args = _build_parser().parse_args()
    np.random.seed(args.seed)

    Ts = np.arange(args.T_min, args.T_max + 1)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not (len(args.kappas) == len(args.alphas) == len(args.ps)):
        raise ValueError("--kappas, --alphas, and --ps must have the same length (tuple semantics).")

    for kapa, alpha, p in zip(args.kappas, args.alphas, args.ps):
        avg_fs = np.zeros((len(Ts), 4), dtype=float)
        for k, T in enumerate(Ts):
            print(f"Generating kappa={kapa} alpha={alpha} p={p} T={T}")
            for _ in range(args.reps):
                avg = float(
                    np.real(
                        compute_overlap_pure_dilutedRQCv1(
                            args.L,
                            int(T),
                            float(kapa),
                            args.rank,
                            args.ensemble_U,
                            args.ensemble_K,
                            float(alpha),
                            float(p),
                        )
                    )
                )
                for moment in range(4):
                    avg_fs[k, moment] += avg ** (moment + 1)

        avg_fs /= args.reps
        table = np.zeros((len(Ts), 5), dtype=float)
        table[:, 0] = Ts
        table[:, 1:] = avg_fs

        name = (
            f"momentsFid_U_{args.ensemble_U}__K_{args.ensemble_K}"
            f"_L_{args.L}_rank_{args.rank}"
            f"_kappa_{_fmt(kapa)}_alpha_{_fmt(alpha)}_p_{_fmt(p)}.txt"
        )
        path = args.outdir / name
        np.savetxt(path, table, header="columns are moment, column 0 is T values")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
