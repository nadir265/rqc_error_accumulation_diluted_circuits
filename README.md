# Reproducibility Package: Random-Permutation RQC Simulations

This package contains the code necessary to reproduce all figures in the paper "Error Accumulation in Dissipative Quantum Circuits" by Nadir Samos Sáenz de Buruaga, Rodrigo M.C. Pereira, Sergey Denisov, Karol Zyczkowski and Pedro Ribeiro.

Everything is written in Python and requires usual scientific packages.
## What Is Included

- **src/rqc_randompermutations/core.py**: core overlap and state-update functions.
- **scripts/**: figure-generation and data-generation entry points.
  - **paper_fig*.py**: single command to ensure data exists or generate, then plot.
  - **generate_fig*_data.py**: pure data generation (standalone use optional).
  - **figure*.py**: pure plotting from pre-generated data (standalone use optional).
- **requirements.txt** and **pyproject.toml**: optional environment files if you want a clean install on another machine.
- **scripts/results/**: output folder for generated figures and intermediate data (auto-created).

## Running It

```bash
python3 scripts/paper_fig2.py
```

## Generating Paper Figures

Each paper figure can be generated with a single command. The wrapper checks whether the required data exists locally; if not, it generates the data and then produces the figure.

### Figure 2 (Ensemble and Rank Dependence)

```bash
python3 scripts/paper_fig2.py
```

This generates or reuses fidelity tables for:
- Ensembles: CUE, FF (all four combinations: CUE-CUE, CUE-FF, FF-CUE, FF-FF)
- Ranks: 1, 2, 4, 8, 15 (FF limited to 1, 2, 4, 8)
- Times: T ∈ {1, 6}
- Dissipation: κ ∈ [0, 0.3] (15 points)
- Trials: 1000 per (κ, rank, T, ensemble) tuple

### Figure 3 (Concentration Analysis)

```bash
python3 scripts/paper_fig3.py
```

Reuses the Figure 2 data to derive concentration bounds and deviation scaling.

### Figure 4 (Scaling with System Size)

```bash
python3 scripts/paper_fig4.py
```

Generates or reuses scaling tables for:
- Ensemble: CUE-CUE only
- Ranks: 1, 2, 4, 8
- Dissipation: κ ∈ {0.0005, 0.025, 0.3}
- System sizes: L ∈ {4, 6, 8, 10} (with T=L)
- Trials: 1000 per (κ, rank, L) tuple

### Figure 5 (Time Evolution)

```bash
python3 scripts/paper_fig5.py
```

Generates or reuses time-scan data for:
- Ensemble: CUE-CUE only
- Rank: 1
- Five fixed (κ, α, p) tuples:
  - (0.001, 0.001, 0.001)
  - (0.01, 0.01, 0.0)
  - (0.01, 0.01, 0.01)
  - (0.1, 0.1, 0.1)
  - (0.015, 0.015, 0.015)
- Times: T ∈ {1, 2, …, 30}
- Trials: 1000 per time point

### Figure 6 is generated in a Mathematica file

## Advanced Usage

### Force Data Regeneration

To force regeneration of all data even if files already exist:

```bash
python3 scripts/paper_fig2.py --force
```

### Standalone Data Generation

Generate data without plotting (e.g., for analysis or external visualization):

```bash
python3 scripts/generate_fig2_data.py --reps 1000
python3 scripts/generate_fig5_data.py --reps 1000
```

### Standalone Plotting

Plot from existing data without regeneration:

```bash
python3 scripts/figure2.py --panel all
python3 scripts/figure3.py
python3 scripts/figure5.py
```

### Customizing Parameters

Edit generator scripts to adjust parameters such as `--reps`, `--kappas`, etc., or pass command-line overrides. For example:

```bash
python scripts/generate_fig2_data.py \
  --outdir my_data \
  --reps 500 \
  --divs 20 \
  --kappa-max 0.5 \
  --small-kappas
```

## Core Physics

- **compute_overlap_pure_dilutedRQCv1**: fidelity calculation for CUE ensembles with optional dissipation (α, p).
- **compute_overlap_pure_dilutedRQCFF**: fidelity calculation using fermionic routing (exact fermionic permutation swaps). Used when ensemble_U = "FF".
- **ff_rank**: maps abstract rank to effective fermionic rank (e.g., rank 1 → FF rank 1).

All functions use NumPy/SciPy for numerical simulation, with both NumPy and Python random seeds set for reproducibility.

## File Structure

```
rqc-randompermutations-repro/
├── README.md                       # This file
├── requirements.txt                # Optional dependency list
├── src/rqc_randompermutations/
│   ├── __init__.py                 # Package exports
│   └── core.py                     # Core physics functions
└── scripts/
    ├── paper_fig2.py               # Figure 2 one-command wrapper
    ├── paper_fig3.py               # Figure 3 one-command wrapper
    ├── paper_fig4.py               # Figure 4 one-command wrapper
    ├── paper_fig5.py               # Figure 5 one-command wrapper
    ├── generate_fig2_data.py        # Figure 2 data generator
    ├── generate_fig4_data.py        # Figure 4 data generator
    ├── generate_fig5_data.py        # Figure 5 data generator
    ├── figure2.py                  # Figure 2 plotter
    ├── figure3.py                  # Figure 3 plotter
    ├── figure4.py                  # Figure 4 plotter
    └── figure5.py                  # Figure 5 plotter

scripts/results/
├── paper/                          # Final figures output by paper_fig*.py scripts
├── fig2/                           # Figure 2 data tables (intermediate)
├── fig4/                           # Figure 4 scaling tables (intermediate)
└── fig5/                           # Figure 5 time-scan tables (intermediate)
```

## Notes for Referees

- **Transparency**: Each figure comes with a separate wrapper script that shows exactly what data is needed and generated.
- **Determinism**: All scripts set both NumPy and Python random seeds for exact reproducibility across runs.
- **Validation**: Use `--force` flag to verify that fresh data regeneration produces the same figures.
- **Computation time**: Full Fig. 2 with 1000 trials can take 1–2 hours on a modern CPU; Figs. 4 and 5 scale similarly. Consider starting with `--reps 100` for quick testing.

## Troubleshooting

- **Missing data**: Ensure scripts have write access to `scripts/results/`.
- **Memory issues**: Reduce `--reps` or `--divs` if running on memory-constrained machines.
- **Slow execution**: The simulation is inherently Monte Carlo; reducing `--reps` speeds up testing.

## References

All core physics and numerical methods follow implementations extracted and verified against the paper's analytical results.
