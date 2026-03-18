# PPS-QJ

Python implementation of waiting-time Monte Carlo and partial post-selection (PPS) quantum-jump algorithms, with both exact and Gaussian backends.

## What is in this repo

- Waiting-time Monte Carlo (`waiting_time_mc`)
- PPS Monte Carlo (`pps_mc`, Procedure C)
- Procedure A and Procedure B reference implementations
- Exact small-system backend (statevector)
- Gaussian large-system backend (covariance/orbital representation)
- Unit tests covering simple limits up to cross-backend checks

Detailed code/theory mapping is in:
- `docs/architecture_theory_map.md`

## Project layout

- `pps_qj/algorithms/`: trajectory algorithms
- `pps_qj/backends/`: exact and Gaussian state backends
- `pps_qj/models/`: model constructors
- `pps_qj/core/`: numerical helpers and RNG utilities
- `pps_qj/observables/`: analysis helpers
- `tests/`: test suite
- `examples/validation.py`: quick validation script
- `continuousmeasurementslatex/`: theory notes (LaTeX)
- `notebooks/quickstart_debug.ipynb`: interactive testing notebook

## Environment setup

## 1) Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install package dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

This project currently requires:
- `numpy>=1.24`

## 3) Optional tools for interactive work

```bash
python3 -m pip install jupyter matplotlib
```

## Running tests

Run full suite:

```bash
python3 -m unittest discover -s tests -v
```

Run targeted subsets:

```bash
python3 -m unittest tests/test_core.py -v
python3 -m unittest tests/test_algorithms.py -v
python3 -m unittest tests/test_eq130_and_procedures.py -v
python3 -m unittest tests/test_gaussian_backend.py -v
```

## Run a validation script

```bash
python3 examples/validation.py
```

## How to run and debug specific parts

## Quick model/algorithm switch

Main control is `SimulationConfig`:
- `backend`: `"exact"` or `"gaussian"`
- `method`: `"waiting_time_mc"`, `"pps_mc"`, `"procedure_a"`, `"procedure_b"`
- `seed`: fixed seed for reproducibility

## Reproducible debugging

Use fixed seeds and single trajectories first:

```python
cfg = SimulationConfig(T=1.0, zeta=0.6, n_traj=1, seed=123, backend="exact", method="pps_mc")
```

Then inspect a trajectory record:
- `times`
- `candidate_jump_times`
- `accepted_jump_times`
- `channels`
- `n_clicks`
- `observables["purity_trace"]`

## Step-through debugging with pdb

```bash
python3 -m pdb -m unittest tests/test_gaussian_backend.py
```

You can also add `breakpoint()` in `pps_qj/algorithms/waiting_time.py` or backend methods and rerun a targeted test.

## Interactive notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open:
- `notebooks/quickstart_debug.ipynb`

The notebook includes:
- single-projector sanity checks (`zeta=1` and `zeta=0` limits)
- Procedure A/B/C comparison
- acceptance-law check against the analytic expression
- exact vs Gaussian cross-check on click statistics

## Theory notes

Primary theory source is in:
- `continuousmeasurementslatex/sections/sec5new.tex`
- `continuousmeasurementslatex/sections/sec5_partial_post_selection.tex`

For code-to-equation references, use:
- `docs/architecture_theory_map.md`
