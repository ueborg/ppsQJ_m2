# PPS-QJ Code Architecture and Theory Map

This document maps the Python implementation to the LaTeX theory notes in `continuousmeasurementslatex/`.

## 1) High-level architecture

The code follows the layered architecture described in `sections/sec5_partial_post_selection.tex` ("Code architecture", around label `sec:code-architecture`).

- `pps_qj/backends/`: state representations and low-level state updates.
- `pps_qj/algorithms/`: waiting-time and PPS trajectory algorithms.
- `pps_qj/models/`: model constructors (single projector, exact spin chain, Gaussian free-fermion model).
- `pps_qj/observables/`: post-processing helpers.
- `pps_qj/simulator.py`: orchestration (build backend, run one trajectory, run ensembles).
- `tests/`: sanity/consistency tests from simple to more physical cases.

## 2) Theory -> code map

## 2.1 Core equations and trajectory algorithms

- `pps_qj/core/numerics.py::heff_from`
  - Implements the effective Hamiltonian construction
  - Theory: `sec5new.tex`, label `eq:Heff_general`

- `pps_qj/algorithms/waiting_time.py::sample_waiting_time`
  - Solves `S(tau) = r` via bracketing+bisection when no closed form is available.
  - Theory: `sec5new.tex`, labels `eq:survival`, `eq:waiting-time-cond`

- `pps_qj/algorithms/waiting_time.py::sample_channel`
  - Samples channel with probabilities proportional to `\langle L_j^\dagger L_j \rangle`.
  - Theory: `sec5new.tex`, label `eq:channel-select`

- `pps_qj/algorithms/waiting_time.py::run_waiting_time_trajectory`
  - Standard waiting-time MC (W1-W6).
  - Theory: `sec5new.tex`, subsection "The standard waiting-time MC algorithm (no PPS)"

- `pps_qj/algorithms/waiting_time.py::run_pps_mc_trajectory`
  - PPS algorithm (P1-P6): candidate jump, coin-flip accept/reject with `zeta`, jump only if accepted.
  - Theory: `sec5new.tex`, subsection "PPS modification: accept/reject at the jump point"

## 2.2 Procedure A / B / C correspondence

- `pps_qj/algorithms/procedures.py::run_procedure_a`
  - Run-level rejection (generate full Born trajectory, accept with probability `zeta^n`).
  - Theory: `sec5new.tex`, coin-flip section (Procedure A), labels `eq:coinflip_protocol`, `eq:coinflip_equals_tilt`

- `pps_qj/algorithms/procedures.py::run_procedure_b`
  - Sequential conditioning (abort on first failed coin).
  - Theory: `sec5new.tex`, Procedure B / local conditioning, label `eq:local_coinflip_probs`

- `pps_qj/algorithms/waiting_time.py::run_pps_mc_trajectory`
  - Procedure C (never abort run; rejected candidate continues on no-click branch).
  - Theory: `sec5new.tex`, subsection "Monte Carlo Clicks (Procedure C)"

## 2.3 Model builders

- `pps_qj/models/single_projector.py`
  - Single-projector model `L = sqrt(gamma) P`, optional Hamiltonian.
  - Theory: `sec5new.tex`, labels `eq:tau-analytic`, `eq:full_postselection`

- `pps_qj/models/spin_chain.py`
  - Exact (2^L) model with Jordan-Wigner operators, hopping Hamiltonian, jump projectors from `d_j` modes.
  - Theory: `sec5_partial_post_selection.tex`, labels `eq:JW`, `eq:H_hopping`, `eq:d_def`, `eq:Md_relation`

- `pps_qj/models/free_fermion.py`
  - Gaussian free-fermion model for larger L; builds Majorana effective matrix and jump-channel Majorana pairs.
  - Theory: `sec5_partial_post_selection.tex`, labels `eq:majorana_def`, `eq:majorana_pair`, `eq:q_from_Gamma`

## 2.4 Backends

- `pps_qj/backends/exact.py`
  - Full statevector backend for exact small-L simulations.
  - Implements no-click propagation with `H_eff`, jump application, and single-projector analytic waiting-time shortcut.

- `pps_qj/backends/gaussian.py`
  - Gaussian backend using orbital matrix `V` / covariance matrix `Gamma`.
  - No-click propagation: Majorana-space exponential (linked to `eq:nocl_cov`).
  - Channel rates: `q_j = (1 - Gamma[a_j,b_j]) / 2` (labels `eq:q_from_Gamma`, `eq:channel_prob_summary`).
  - Jump covariance update: rank-2 antisymmetric update + pair locking (label `eq:full_update`).
  - Survival computation: no-click norm formula used by waiting-time root-finding.

## 2.5 Observables

- `pps_qj/observables/basic.py::acceptance_fraction`
  - Empirical acceptance fraction (used for Eq. 130-style checks in tests).

- `pps_qj/observables/basic.py::entanglement_entropy_gamma`
  - Entanglement entropy from covariance-spectrum formula.
  - Theory: `sec5_partial_post_selection.tex`, label `eq:EE_majorana`

## 3) Ground-up validation ladder

Use this order when debugging from simplest logic upward:

1. Core numerics and probability utilities
   - `python3 -m unittest tests/test_core.py -v`

2. Single-projector limits and PPS invariants
   - `python3 -m unittest tests/test_algorithms.py -v`
   - Checks purity preservation and `zeta=1` / `zeta=0` limits.

3. Procedure equivalence and acceptance formula
   - `python3 -m unittest tests/test_eq130_and_procedures.py -v`
   - Verifies Procedure A/B/C consistency and acceptance law against the analytical expression.

4. Gaussian backend consistency
   - `python3 -m unittest tests/test_gaussian_backend.py -v`
   - Checks `Gamma` invariants, survival monotonicity, and cross-backend click statistics.

5. Full regression
   - `python3 -m unittest discover -s tests -v`

## 4) Notes on theory references

The LaTeX is still evolving, so equation numbers may shift. In code/docs, prefer equation labels (for example `eq:waiting-time-cond`) and subsection names, which are stable anchors across edits.
