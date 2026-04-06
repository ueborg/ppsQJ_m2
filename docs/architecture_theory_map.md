# PPS-QJ Architecture and Theory Map

This document describes the active implementation after the legacy simulator stack was archived to `trash/`.

## Active module map

### `pps_qj/exact_backend.py`

Reference spin-1/2 backend for small systems.

- Jordan-Wigner fermion operators and Neel initial state
- Hopping Hamiltonian and monitored bond projectors
- Ordinary waiting-time Monte Carlo
- Procedure A, Procedure B, Procedure C
- Lindblad superoperator and density-matrix integration

Theory anchors:

- Jordan-Wigner construction
- Free-fermion hopping chain
- Bond projectors `P_j = d_j^\dagger d_j`
- Born-rule waiting-time Monte Carlo
- PPS Procedures A/B/C

### `pps_qj/gaussian_backend.py`

Gaussian backend for the same monitored chain in the Majorana/covariance language.

- Neel covariance initialization
- Majorana hopping and non-Hermitian effective generators
- No-click propagation in orbital form
- Rank-2 projective jump update
- Entanglement entropy from covariance subblocks

Theory anchors:

- Site-Majorana convention
- Effective quadratic generator in Majorana form
- `q_j = (1 - \Gamma_{a_j b_j}) / 2`
- Projective covariance update for monitored jumps

### `pps_qj/backward_pass.py`

Backward evolutions for the finite-horizon Doob transform.

- Exact adjoint tilted Lindbladian propagation
- Gaussian closure ODE for `(C_t, z_t)`
- Reconstruction helpers for the Gaussian generator

Theory anchors:

- Finite-horizon backward observable `G_t`
- Gaussian closure ansatz for the tilted backward evolution
- Scalar partition-function flow and covariance ODE

### `pps_qj/overlaps.py`

Gaussian overlap formulas used in the Doob rates.

- `Tr(G rho)` for Gaussian operator plus pure Gaussian state
- Post-jump overlap helper

Theory anchors:

- Gaussian overlap / Pfaffian-determinant identities
- `Tr(G_t P_j rho P_j)` evaluation through the normalized post-jump covariance

### `pps_qj/doob_wtmc.py`

Finite-horizon Doob samplers.

- Exact Doob trajectory sampler
- Gaussian Doob trajectory sampler
- Conditioned survival functions for root-finding
- Per-segment diagnostics for survival monotonicity checks

Theory anchors:

- Doob waiting-time survival
- Finite-horizon Doob jump rates
- Root search for the next jump time

### `pps_qj/part6_validation.py`

Slow validation harness for the benchmark checklist in Part 6.

- `zeta = 1` recovery
- Small-`zeta` concentration
- Single-mode analytic checks
- Commuting-case checks
- Partition-function consistency
- Click-count PMF comparison
- Entanglement-scaling benchmark
- Conditioned-survival monotonicity
- `Q_s` vs `R_\zeta` comparison

### `pps_qj/extended_validation.py`

Slow post-Part-6 benchmark and report generator.

- overlap micro-test for `Tr(G_t rho_t)`
- compensator / Radon-Nikodym comparison between `Q_s` and `R_\zeta`
- martingale check for the backward weight process
- finite-horizon waiting-time transform check
- backward-pass stability and convergence
- look-ahead rate comparison at fixed instantaneous `q_j`
- occupation-profile and density-correlator benchmarks
- entropy benchmark with confidence bands
- system-size scaling checks
- click-time statistics and report plots A-D

Theory anchors:

- compensator identity for `dR_\zeta / dP`
- martingale structure of `Tr(G_t \rho_t) \zeta^{N_t}`
- finite-horizon truncated waiting-time law
- Gaussian closure convergence and commuting-case analytic solution
- look-ahead structure of Doob rates
- observable equivalence under weighted Born averages vs Doob sampling

### `tests/test_doob_wtmc.py`

Fast regression tests for the current Doob/PPS code path. These are not the full benchmark suite; they are the lightweight CI-facing checks.

## Supporting modules

### `pps_qj/core/numerics.py`

Small numerical utilities used by the active implementation:

- safe state normalization
- safe channel probabilities
- monotone bracket-and-bisect root finder

### `pps_qj/observables/basic.py`

Entropy helpers used by the validation code:

- entanglement entropy from exact statevectors
- entanglement entropy from covariance matrices

### `pps_qj/types.py`

Minimal shared types:

- `Tolerances`
- `JumpTrajectory`

## Archived material

The previous layered stack built around `simulator.py`, `models/`, `algorithms/`, and `backends/` was moved to `trash/legacy_package/pps_qj/`. Old tests, scripts, notebooks, and generated figures were moved alongside it under `trash/`.

That material is no longer the source of truth for the current Doob/PPS implementation.
