# PPS-QJ

Focused implementation of partial post-selection and finite-horizon Doob waiting-time Monte Carlo for the monitored free-fermion chain.

## Active code path

The live package surface is the direct Doob/PPS stack:

- `pps_qj/exact_backend.py`
  Exact spin-1/2 reference backend: Jordan-Wigner construction, ordinary WTMC, Procedures A/B/C, Lindblad integration.
- `pps_qj/gaussian_backend.py`
  Gaussian covariance/orbital backend for larger systems.
- `pps_qj/backward_pass.py`
  Exact and Gaussian backward evolutions for the tilted/Doob construction.
- `pps_qj/overlaps.py`
  Gaussian overlap formulas used in the forward Doob rates.
- `pps_qj/doob_wtmc.py`
  Exact and Gaussian finite-horizon Doob trajectory samplers.
- `pps_qj/part6_validation.py`
  Validation harness for the Part 6 benchmark checklist.
- `pps_qj/extended_validation.py`
  Extended validation harness for the compensator, martingale, observable, entropy, scaling, and report-plot checks.
- `tests/test_doob_wtmc.py`
  Fast regression checks for the current Doob implementation.

Archived legacy code, notebooks, and scripts were moved to `trash/`.

## Repository layout

- `pps_qj/`
  Active package code.
- `tests/`
  Active tests for the current code path.
- `docs/architecture_theory_map.md`
  Code-to-theory map for the active modules.
- `notebooks/doob_wtmc_validation.ipynb`
  Working notebook for the Doob/PPS validation workflow.
- `trash/`
  Archived legacy framework, scripts, tests, notebooks, and generated outputs.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .[dev]
```

Core runtime dependencies:

- `numpy>=1.24`
- `scipy>=1.11`

Optional for plots/notebooks:

```bash
python3 -m pip install matplotlib jupyter
```

## Running tests

Current fast regression suite:

```bash
./.venv/bin/pytest tests/test_doob_wtmc.py -q
```

If you want the slower validation harness instead of the fast pytest checks:

```bash
./.venv/bin/python run_part6_validation.py
```

That writes artifacts into `outputs/part6_validation_full/`.

For the heavier post-Part-6 benchmark and report plots:

```bash
./.venv/bin/python run_extended_validation.py
```

That writes artifacts into `outputs/extended_validation/`.

## Theory notes

Primary implementation references are in:

- `continuousmeasurementslatex/`
- `docs/architecture_theory_map.md`

Use equation labels and subsection names from the LaTeX notes rather than hard-coded equation numbers, since the manuscript is still evolving.
