#!/usr/bin/env python3
"""Generate scaling tables used by Figure 4.

Each output file corresponds to one (ensemble_U, ensemble_K, kappa, rank) pair and
contains rows over system sizes L. By default we use the paper preset L in {4,6,8,10}
with T=L and save under T-tag 10 for compatibility with figure4.py.
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

from rqc_randompermutations import compute_overlap_pure_dilutedRQCv1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Figure 4 scaling tables.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS_DIR / "fig4")
    parser.add_argument("--ensemble-U", nargs="+", default=["CUE"])
    parser.add_argument("--ensemble-K", nargs="+", default=["CUE"])
    parser.add_argument("--Ls", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.0005, 0.025, 0.3])
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--p", type=float, default=0.0)
    parser.add_argument("--T-tag", type=int, default=10, help="T value embedded in output filename")
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def _fmt(x: float) -> str:
    return str(x)


def main() -> None:
    args = _build_parser().parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)

    for ensemble_u in args.ensemble_U:
        for ensemble_k in args.ensemble_K:
            for rank in args.ranks:
                for kappa in args.kappas:
                    table = np.zeros((len(args.Ls), 5), dtype=float)
                    for idx, L in enumerate(args.Ls):
                        T = int(L)  # paper preset for Fig. 4 uses T=L
                        sums = np.zeros(4, dtype=float)
                        print(
                            f"Generating U={ensemble_u} K={ensemble_k} rank={rank} "
                            f"kappa={kappa} L={L} T={T}"
                        )
                        for _ in range(args.reps):
                            avg = float(
                                np.real(
                                    compute_overlap_pure_dilutedRQCv1(
                                        L=int(L),
                                        T=T,
                                        kappa=float(kappa),
                                        rank=int(rank),
                                        ensemble_unitaries=ensemble_u,
                                        ensemble_kraus=ensemble_k,
                                        gamma=float(args.gamma),
                                        p=float(args.p),
                                    )
                                )
                            )
                            for moment in range(4):
                                sums[moment] += avg ** (moment + 1)

                        table[idx, 0] = float(L)
                        table[idx, 1:] = sums / float(args.reps)

                    name = (
                        f"scaling_momentsFid_U_{ensemble_u}__K_{ensemble_k}"
                        f"_T_{args.T_tag}_rank_{rank}kapa_{_fmt(kappa)}.txt"
                    )
                    out_path = args.outdir / name
                    np.savetxt(out_path, table, header="columns are moment, column 0 is L values")
                    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
