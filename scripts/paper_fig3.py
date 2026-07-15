#!/usr/bin/env python3
"""Paper wrapper for Figure 3: ensure Fig. 2 data exists, then plot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper Figure 3 workflow.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RESULTS_DIR / "fig2")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "paper" / "Fig3.pdf")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true", help="Regenerate data even if files exist")
    return parser


def _expected_files(data_root: Path) -> list[Path]:
    ranks = [1, 2, 4, 8, 15]
    Ts = [1, 6]
    ensembles = ["CUE", "FF"]
    files: list[Path] = []
    for ensemble_u in ensembles:
        for ensemble_k in ensembles:
            for rank in ranks:
                if ensemble_k == "FF" and rank not in {1, 2, 4, 8}:
                    continue
                for T in Ts:
                    name = f"momentsFid_U_{ensemble_u}__K_{ensemble_k}_L_6_T_{T}_rank_{rank}_smallkappas.txt"
                    files.append(data_root / name)
    return files


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _build_parser().parse_args()
    scripts = SCRIPT_DIR
    expected = _expected_files(args.data_root)
    missing = [p for p in expected if not p.exists()]

    if args.force or missing:
        args.data_root.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                str(scripts / "generate_fig2_data.py"),
                "--outdir",
                str(args.data_root),
                "--L",
                "6",
                "--Ts",
                "1",
                "6",
                "--ranks",
                "1",
                "2",
                "4",
                "8",
                "15",
                "--ensemble-U",
                "CUE",
                "FF",
                "--ensemble-K",
                "CUE",
                "FF",
                "--reps",
                "1000",
                "--divs",
                "15",
                "--kappa-min",
                "0.0",
                "--kappa-max",
                "0.3",
                "--small-kappas",
                "--seed",
                str(args.seed),
            ]
        )
    else:
        print(f"Using existing Figure 2/3 data under {args.data_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(scripts / "figure3.py"),
            "--data-root",
            str(args.data_root),
            "--L",
            "6",
            "--output",
            str(args.output),
        ]
    )


if __name__ == "__main__":
    main()
