#!/usr/bin/env python3
"""Paper wrapper for Figure 5: ensure time-scan data exists, then plot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper Figure 5 workflow.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RESULTS_DIR / "fig5")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "paper" / "Fig5.pdf")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true", help="Regenerate data even if files exist")
    return parser


def _expected_files(data_root: Path) -> list[Path]:
    tuples = [(0.001, 0.001, 0.001), (0.01, 0.01, 0.0), (0.01, 0.01, 0.01), (0.1, 0.1, 0.1), (0.015, 0.015, 0.015)]
    files: list[Path] = []
    for k, a, p in tuples:
        name = f"momentsFid_U_CUE__K_CUE_L_6_rank_1_kappa_{k}_alpha_{a}_p_{p}.txt"
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
                str(scripts / "generate_fig5_data.py"),
                "--outdir",
                str(args.data_root),
                "--L",
                "6",
                "--rank",
                "1",
                "--reps",
                "1000",
                "--T-min",
                "1",
                "--T-max",
                "30",
                "--ensemble-U",
                "CUE",
                "--ensemble-K",
                "CUE",
                "--kappas",
                "0.001",
                "0.01",
                "0.01",
                "0.1",
                "0.015",
                "--alphas",
                "0.001",
                "0.01",
                "0.01",
                "0.1",
                "0.015",
                "--ps",
                "0.001",
                "0.0",
                "0.01",
                "0.1",
                "0.015",
                "--seed",
                str(args.seed),
            ]
        )
    else:
        print(f"Using existing Figure 5 data under {args.data_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(scripts / "figure5.py"),
            "--data-root",
            str(args.data_root),
            "--L",
            "6",
            "--rank",
            "1",
            "--ensemble-U",
            "CUE",
            "--ensemble-K",
            "CUE",
            "--kappas",
            "0.001",
            "0.01",
            "0.01",
            "0.1",
            "0.015",
            "--alphas",
            "0.001",
            "0.01",
            "0.01",
            "0.1",
            "0.015",
            "--ps",
            "0.001",
            "0.0",
            "0.01",
            "0.1",
            "0.015",
            "--output",
            str(args.output),
        ]
    )


if __name__ == "__main__":
    main()
