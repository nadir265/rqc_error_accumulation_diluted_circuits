#!/usr/bin/env python3
"""Paper wrapper for Figure 4: ensure scaling data exists, then plot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper Figure 4 workflow.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RESULTS_DIR / "fig4")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "paper" / "Fig4.pdf")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true", help="Regenerate data even if files exist")
    return parser


def _expected_files(data_root: Path) -> list[Path]:
    ranks = [1, 2, 4, 8]
    kappas = [0.0005, 0.025, 0.3]
    files: list[Path] = []
    for rank in ranks:
        for kappa in kappas:
            name = f"scaling_momentsFid_U_CUE__K_CUE_T_10_rank_{rank}kapa_{kappa}.txt"
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
                str(scripts / "generate_fig4_data.py"),
                "--outdir",
                str(args.data_root),
                "--ensemble-U",
                "CUE",
                "--ensemble-K",
                "CUE",
                "--Ls",
                "4",
                "6",
                "8",
                "10",
                "--ranks",
                "1",
                "2",
                "4",
                "8",
                "--kappas",
                "0.0005",
                "0.025",
                "0.3",
                "--reps",
                "1000",
                "--gamma",
                "0.0",
                "--p",
                "0.0",
                "--T-tag",
                "10",
                "--seed",
                str(args.seed),
            ]
        )
    else:
        print(f"Using existing Figure 4 scaling data under {args.data_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(scripts / "figure4.py"),
            "--data-root",
            str(args.data_root),
            "--variant",
            "v1",
            "--ranks",
            "1",
            "2",
            "4",
            "8",
            "--kapas",
            "0.0005",
            "0.025",
            "0.3",
            "--Ts",
            "10",
            "--output",
            str(args.output),
        ]
    )


if __name__ == "__main__":
    main()
