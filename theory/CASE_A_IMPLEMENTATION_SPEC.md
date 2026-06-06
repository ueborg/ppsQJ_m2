# Case A Implementation Specification

**Audience.** This document is written for an autonomous coding agent (Claude Code or
equivalent) who has access to this repository but no prior chat context.
It specifies, end-to-end, how to implement the "Case A" two-measurement
QJ-PPS model alongside the existing single-measurement code, run the
relevant cluster jobs, and analyze the output.

**Repository.** `ueborg/ppsQJ_m2` (this directory).
**Compute target.** Habrok HPC at RUG. Local dev on a Mac.
**Language.** Python 3.11+, numpy/scipy, plus `pfapack` for Pfaffian.

---

## 1. Project context (read this once, then you can ignore it)

The repository simulates a 1D free-fermion chain under continuous
measurement using a quantum-jump (QJ) unraveling, with a *partial post-
selection* (PPS) reweighting parameter ζ ∈ (0, 1]. The state is Gaussian
throughout and stored as the Majorana covariance matrix Γ (a 2L × 2L
real antisymmetric matrix). All dynamics preserve Gaussianity.

The existing code implements one model variant, henceforth **Case B**:
- Hamiltonian: free hopping `H = w Σ_j (c†_{j+1} c_j + h.c.)`.
- One measurement channel: `L̃_j = d†_j d_j` (Kitaev/Bogoliubov mode
  density) at rate α.
- Parametrization: λ = α/(α+w), with α+w fixed to 1.
- PPS: trajectory weight ζ^M where M = total click count.

The task here is to add a parallel model variant, **Case A**:
- Hamiltonian: **none** (H = 0).
- Two competing measurement channels:
  - `L_j  = n_j  = c†_j c_j`        (site density)            at rate γ
  - `L̃_j = d†_j d_j` (Kitaev density, same as Case B)        at rate α
- Parametrization: λ_A = α/(α+γ), with α+γ = 1.
- PPS unchanged: ζ^M, M = total click count over both channels.

**Theory predictions to confirm numerically:**

1. **Critical line pinned at λ_A = 1/2 for all ζ** by the Kells-Meidan-
   Romito self-duality (`c_j ↔ d_j`, `γ ↔ α`), which PPS preserves
   because ζ^M depends only on the total click count, not the channel
   split. This is the single sharpest claim.
2. **Two area-law phases on either side of the transition**, distinguished
   by a Z₂ topological invariant (the Pfaffian sign of the spectrally
   flattened Majorana covariance).
3. **Universality class at the critical line: Ising** (c = 1/2, ν = 1)
   from a D-class WZW field theory at θ = π. Diffusive numerics (KMR
   SciPost 2023) report ν ≈ 5/3; we expect the QJ result to drift
   toward ν = 1 as L grows, OR confirm ν ≈ 5/3 and indicate a
   different universality class for QJ vs QSD.

Background references (PDFs in `analysis/refs/` if present, or fetch
from arXiv):
- KMR 2023, SciPostPhys 14, 031 (the parent paper, QSD unraveling)
- Cao, Tilloy, De Luca 2019, SciPost Phys 7, 024 (continuous monitoring
  free fermions)
- Le Gal & Schirò 2025, arXiv:2511.22506 (replica field theory for QJ
  monitoring)

You don't need to internalize the theory; you just need to implement
the model and the Pfaffian invariant correctly. The theoretical paper
trail is documented at `theory/HANDOFF.md`.

---

## 2. Existing code you will extend

The Case B implementation lives in `pps_qj/`. The files that matter:

| File | Role |
|---|---|
| `pps_qj/gaussian_backend.py` | Gaussian state machinery: covariance, no-click evolution, projective jumps, jump probabilities, entanglement entropy. The hot path. |
| `pps_qj/cloning.py` | Population-dynamics cloning for PPS reweighting. |
| `pps_qj/doob_wtmc.py` | Alternative Doob-transformed sampler (not needed for Case A; ignore). |
| `pps_qj/observables/` | Entanglement, Rényi, correlators. |
| `scripts/` | CLI entry points and analysis. |
| `slurm/` | Job submission scripts and arrays. |

Key conventions in `gaussian_backend.py`:
- A site `j ∈ {0, ..., L-1}` has two Majorana modes at indices `2j` and `2j+1`.
- `bond_jump_pair(bond) = (2*bond, 2*bond+3)` returns the Majorana pair
  whose product `i γ_a γ_b` is measured for the Case B bond channel.
- `GaussianChainModel` caches the eigendecomposition of `h_effective`,
  the non-Hermitian generator, for fast no-click propagation. **You will
  follow this pattern**: an analogous Case A model with cached
  eigendecomposition.
- `gaussian_born_rule_trajectory(model, T, rng, ...)` is the main
  trajectory function. It loops while `t < T`, samples a Brent-found
  waiting time from the survival probability, performs a no-click
  propagation, samples a jump channel proportional to its
  click density, and applies a projective jump. Returns jump times
  and channels.
- `apply_projective_jump(covariance, jump_pair)` is channel-agnostic.
  It projects to the +1 eigenspace of `i γ_a γ_b`. **No modification
  needed.**

Read `gaussian_backend.py` once before you start, particularly:
- `bond_jump_pair`, `neel_covariance`, `effective_generator`,
  `build_gaussian_chain_model`, `gaussian_born_rule_trajectory`,
  `apply_projective_jump`, `jump_probability`.

---

## 3. What you will build

A new file `pps_qj/gaussian_backend_caseA.py` and a small adapter in
`pps_qj/cloning.py` (or a new `pps_qj/cloning_caseA.py`). A new
observable module `pps_qj/observables/topological_z2.py`. A CLI script
`scripts/run_caseA.py`. SLURM submit scripts `slurm/submit_caseA_*.sh`.

**Acceptance test for the implementation:** running

```bash
python scripts/run_caseA.py --L 32 --lambda_A 0.5 --zeta 1.0 \
    --n_clones 50 --T_factor 4 --seed 0
```

must produce a single JSON or pickle output with:
- `pf_invariant_mean` ≈ 0 ± small (transition point: equal weight on ±1)
- `entanglement_entropy_half_chain_mean` finite (area-law on both sides
  if you sweep λ_A around 0.5)
- click counts split roughly equally between site and bond channels
  (because λ_A = 1/2)

---

## 4. Implementation steps

### Step 4.1 — Add `site_jump_pair`

In `pps_qj/gaussian_backend.py`, add a sibling of `bond_jump_pair`:

```python
def site_jump_pair(site: int) -> tuple[int, int]:
    """Majorana pair for the on-site density operator.

    n_j = c†_j c_j = (1 + i γ_{2j} γ_{2j+1}) / 2.
    Measuring n_j corresponds to measuring i γ_{2j} γ_{2j+1}.

    Returns (2*site, 2*site + 1) — the two Majorana modes that live on
    the same physical site. Compare bond_jump_pair which acts on
    Majoranas on neighbouring sites.
    """
    return 2 * site, 2 * site + 1
```

Add the unit test in `tests/test_gaussian_backend.py`: at the Néel
initial state (built by `neel_covariance(L)`), the site jump probability
on site 0 is 1 (occupied) and on site 1 is 0 (empty).

### Step 4.2 — Create `pps_qj/gaussian_backend_caseA.py`

New file. Imports:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from .gaussian_backend import (
    bond_jump_pair,
    site_jump_pair,
    neel_covariance,
    orbitals_from_covariance,
    apply_projective_jump,
    jump_probability,
    project_to_physical_covariance,
    entanglement_entropy,
)
```

Define:

```python
def effective_generator_caseA(L: int, gamma_rate: float, alpha_rate: float
                              ) -> np.ndarray:
    """Non-Hermitian Majorana generator for Case A.

    No Hamiltonian. Two damping contributions:
      - on-site density measurements at rate gamma_rate, acting on the
        Majorana pairs (2j, 2j+1) for j = 0..L-1
      - bond measurements at rate alpha_rate, acting on pairs
        (2*bond, 2*bond+3) for bond = 0..L-2

    The contribution of measuring i γ_a γ_b at rate r to the generator
    is the same matrix structure as in `effective_generator` in
    gaussian_backend.py — copy that pattern and verify sign conventions
    by direct comparison.

    Returns
    -------
    np.ndarray (2L × 2L), complex
    """
    h_eff = np.zeros((2 * L, 2 * L), dtype=np.complex128)

    # Site channel
    for j in range(L):
        a, b = site_jump_pair(j)
        h_eff[a, b] -= 1j * gamma_rate
        h_eff[b, a] += 1j * gamma_rate

    # Bond channel
    for bond in range(L - 1):
        a, b = bond_jump_pair(bond)
        h_eff[a, b] -= 1j * alpha_rate
        h_eff[b, a] += 1j * alpha_rate

    return h_eff
```

**Verification step.** Before relying on this, write a small validation:
- `effective_generator_caseA(L, gamma_rate=0, alpha_rate=alpha)` minus
  the existing `effective_generator(L, w=0, alpha)` should equal the
  pure-bond term in both. Diff should be machine-zero.
- `effective_generator_caseA(L, gamma_rate=gamma, alpha_rate=0)` should
  produce a block-diagonal (site-by-site) matrix because the site
  channel doesn't couple different sites.

The model dataclass:

```python
@dataclass(frozen=True)
class GaussianCaseAModel:
    L: int
    gamma_rate: float
    alpha_rate: float
    h_effective: np.ndarray

    # Flat list of (a, b, rate) for every channel × every location
    # First L entries: site channel (rate = gamma_rate)
    # Next L-1 entries: bond channel (rate = alpha_rate)
    jump_pairs_with_rates: tuple[tuple[int, int, float], ...]

    # Numpy arrays of (a, b, rate) extracted for vectorized access
    ja: np.ndarray   # shape (2L-1,), int
    jb: np.ndarray   # shape (2L-1,), int
    rates: np.ndarray  # shape (2L-1,), float

    # Initial state (Néel half-filled), matching Case B
    gamma0: np.ndarray
    orbitals0: np.ndarray

    # Cached eigendecomposition of h_effective for fast propagation
    h_eff_evals: np.ndarray
    h_eff_V: np.ndarray
    h_eff_V_inv: np.ndarray
    h_eff_VhV: np.ndarray


def build_caseA_model(L: int, gamma_rate: float, alpha_rate: float
                      ) -> GaussianCaseAModel:
    h_eff = effective_generator_caseA(L, gamma_rate, alpha_rate)
    evals, V = np.linalg.eig(h_eff)
    V_inv = np.linalg.inv(V)

    pairs = []
    for j in range(L):
        a, b = site_jump_pair(j)
        pairs.append((a, b, gamma_rate))
    for bond in range(L - 1):
        a, b = bond_jump_pair(bond)
        pairs.append((a, b, alpha_rate))
    pairs_tuple = tuple(pairs)

    ja = np.array([p[0] for p in pairs], dtype=np.intp)
    jb = np.array([p[1] for p in pairs], dtype=np.intp)
    rates = np.array([p[2] for p in pairs], dtype=np.float64)

    gamma0 = neel_covariance(L)
    orbitals0 = orbitals_from_covariance(gamma0)

    return GaussianCaseAModel(
        L=L,
        gamma_rate=gamma_rate,
        alpha_rate=alpha_rate,
        h_effective=h_eff,
        jump_pairs_with_rates=pairs_tuple,
        ja=ja, jb=jb, rates=rates,
        gamma0=gamma0, orbitals0=orbitals0,
        h_eff_evals=evals,
        h_eff_V=V,
        h_eff_V_inv=V_inv,
        h_eff_VhV=V.conj().T @ V,
    )
```

### Step 4.3 — Case A trajectory function

The Case B trajectory `gaussian_born_rule_trajectory` in
`gaussian_backend.py` is the template. The only differences:
1. The list of (a, b) pairs has length `2L - 1` instead of `L - 1`.
2. Each pair carries its own rate: the per-pair click density is
   `rate * 0.5 * (1 - covariance[a, b])` instead of `alpha * 0.5 * (1 - ...)`.
3. After sampling a jump, the channel is one of `{site, bond}` — track it.
4. `apply_projective_jump` is unchanged.

Implement `gaussian_born_rule_trajectory_caseA` in
`gaussian_backend_caseA.py` by **copying** `gaussian_born_rule_trajectory`
verbatim and making exactly these substitutions:
- `_ja, _jb` come from the model's `ja, jb` (the flat 2L-1 lists).
- The per-pair "instantaneous click density" used in the survival
  branch-norm calculation must be weighted by `model.rates`. In Case B
  this is implicit (single rate α multiplied outside); in Case A you
  must multiply pair-wise.
- Return `jump_channels: list[int]` where 0 = site channel, 1 = bond
  channel. This is a diagnostic (count clicks per channel).

**Where the rate enters mathematically.** Cross-check by reading the
existing code carefully: the per-pair click rate density is
`rate * 0.5 * (1 - cov[a, b])`, and the total click rate is
`sum_pairs rate_pair * 0.5 * (1 - cov[a, b])`. In Case B, the rate is
constant α, so it factors out; in Case A it does not. The survival
function `branch_norm(dt) = trace(exp(-i h_eff dt) Γ exp(+i h_eff* dt))`
already includes both channels via `h_effective`, so the branch norm
calculation is unchanged structurally; only the channel-selection step
when a jump fires differs.

**Diagnostic.** When λ_A = 1/2 (and so γ = α = 1/2), the two
channels have the same total rate, and you should see roughly equal
numbers of site and bond clicks over a long trajectory.

### Step 4.4 — Cloning adapter for Case A

The cloning code in `pps_qj/cloning.py` operates on a Case B model.
For Case A, either:
- **Option 1 (recommended):** generalize `cloning.py` to accept a
  generic "model" object exposing `.h_effective`, `.gamma0`, `.orbitals0`,
  `.h_eff_evals/V/V_inv`, `.ja`, `.jb`, and `.rates`. The traj function
  is passed as a callback. Case B and Case A then share the cloning loop.
- **Option 2:** copy `cloning.py` to `cloning_caseA.py` and swap the
  trajectory call to `gaussian_born_rule_trajectory_caseA`. Less
  elegant but doesn't risk breaking Case B.

Either way, **PPS is unchanged**: every click multiplies the trajectory
weight by ζ, regardless of channel. The cloning resampling on weight
divergence is identical.

### Step 4.5 — Topological Z₂ invariant

Create `pps_qj/observables/topological_z2.py`:

```python
"""Z₂ topological invariant for class-D Gaussian Majorana states.

Implements the spectrally-flattened Pfaffian sign described in:
- Kells, Meidan, Romito, SciPostPhys 14, 031 (2023)
- Cao, Tilloy, De Luca, SciPost Phys 7, 024 (2019)

Requires the `pfapack` package: `pip install pfapack`.
"""
from __future__ import annotations
import numpy as np

try:
    from pfapack.pfaffian import pfaffian as _pf
except ImportError as e:
    raise ImportError(
        "pfapack required for topological_z2 invariant. "
        "Install with `pip install pfapack`."
    ) from e


def pfaffian_sign_invariant(covariance: np.ndarray) -> int:
    """Z2 invariant for the class-D Gaussian state with covariance Γ.

    Returns +1 (trivial phase) or -1 (non-trivial topological phase).

    Method: spectrally flatten i Γ (replace each eigenvalue by its sign),
    rotate back, take the antisymmetric real part, and return sign(Pf).
    """
    Gamma = np.asarray(covariance, dtype=np.float64)
    eigvals, U = np.linalg.eigh(1j * Gamma)
    sign_vals = np.sign(eigvals.real)
    flat = U @ np.diag(sign_vals) @ U.conj().T
    M = (-1j * flat)
    M = 0.5 * (M - M.T)
    M_real = np.real_if_close(M, tol=1e6).real
    pf_val = _pf(M_real, method='P')
    return int(np.sign(pf_val))
```

Verification: on the Néel initial state (`neel_covariance(L)`), this
should return a definite ±1 (you'll discover which by running once;
that's fine — what matters is that it's consistent across realizations
and changes sign across the transition).

### Step 4.6 — CLI entry point

Create `scripts/run_caseA.py`. It should:
- Accept arguments `--L`, `--lambda_A`, `--zeta`, `--n_clones`,
  `--T_factor` (T = T_factor * L), `--seed`, `--output`.
- Compute `gamma_rate = 1 - lambda_A`, `alpha_rate = lambda_A`.
- Build the model.
- Run `n_clones` independent trajectories (or the cloning population
  dynamics with the chosen `n_clones`).
- For each trajectory, at the **final time T**, compute:
  - half-chain entanglement entropy `S(L/2)`
  - Pfaffian-sign invariant `ν ∈ {±1}`
  - total click count and split by channel
- Save aggregate observables to a single pickle:
  ```
  {
    'L': L, 'lambda_A': lambda_A, 'zeta': zeta,
    'n_clones': n_clones, 'T': T, 'seed': seed,
    'S_half_chain_mean': ..., 'S_half_chain_err': ...,
    'pf_invariant_mean': ..., 'pf_invariant_err': ...,
    'pf_invariant_var': ...,  # Binder of nu, used for FSS
    'click_count_mean': ..., 'site_click_frac': ...,
    'B_L_mean': ..., 'B_L_err': ...,  # Binder of S(L/2) or of nu
    ...
  }
  ```

The choice of Binder observable for Case A is **the cumulant of ν**:
`Q_L = 1 - <ν^4> / (3<ν^2>^2)` — but since ν ∈ {±1}, ν² = 1 always and
this is trivial. Use instead `B_L = <ν^2>_realisations - <ν>²_realisations`
(variance of the order parameter across the realization ensemble), which
goes to 0 in either ordered phase and is finite at the transition.
Equivalently: `<|ν|>` saturates to 1 in each ordered phase and dips at
the critical point. Standard practice in class-D MIPTs is to use the
order-parameter variance; document whichever you pick in the output
metadata.

### Step 4.7 — SLURM submission

Two scripts to add in `slurm/`:

#### `slurm/submit_caseA_smoke.sh`
A single-job smoke test:
- L = 32, λ_A = 0.5, ζ = 1.0, n_clones = 50, T_factor = 4
- 1 hour walltime, 4 CPUs, 8 GB RAM
- Confirms the code runs end-to-end on Habrok.

#### `slurm/submit_caseA_scan.sh`
A job array over the production grid:
- L ∈ {32, 48, 64, 96, 128}, 5 values
- λ_A ∈ linspace(0.35, 0.65, 13), 13 values
- ζ ∈ {0.10, 0.30, 0.50, 1.00}, 4 values
- n_clones = 300, T_factor = 4
- Total: 260 jobs in a single array.

Walltimes per L (conservative):
- L = 32: 30 min
- L = 64: 1 hour
- L = 128: 4 hours

Use the existing `slurm/submit_clone_v2_fst.sh` (Case B's submit script)
as a template for the SLURM directives and module loads.

### Step 4.8 — Aggregation and analysis

Add `analysis/case_A/aggregate.py`: walk over the SLURM output
directory, load all pickles, build an aggregate dataframe keyed by
(L, λ_A, ζ).

Add `analysis/case_A/binder_analysis.py`: for each ζ, extract the
crossing λ_c(L_pair, ζ) from Binder/order-parameter-variance crossings.
The structural test is:

**For every ζ tested, the extrapolated λ_c^∞(ζ) should equal 1/2
within statistical error.**

Plot λ_c^∞(ζ) vs ζ across the four ζ values; the curve should be flat
at 0.5. **If you see a residual ζ dependence at the >2σ level, that is
either a code bug or a falsification of the theory** — report it
explicitly and do not paper over it.

Also: at λ_A = 1/2 exactly, plot S(L/2) vs log(L) for each ζ. Theory
predicts a slope c/6 = 1/12 (Ising universality, c = 1/2). If you find
slope ≈ 1/6 (c = 1, Dirac universality) instead, that's a discrepancy
worth flagging.

---

## 5. Validation gates (in order)

Do not proceed past each gate until the test passes:

**Gate 1: Generator correctness.**
- `effective_generator_caseA(8, 0.5, 0.5)` is Hermitian-conjugate-skew
  in the Majorana sense (its anti-Hermitian part is well-defined and
  has the correct sign structure). Cross-check with hand calculation
  on L = 4.

**Gate 2: Single-trajectory sanity.**
- At λ_A = 0.5, ζ = 1, L = 16, T = 64: a single trajectory produces
  click counts roughly balanced between channels (site:bond ≈ 1:1).
- The covariance remains a valid Gaussian state at all times
  (`project_to_physical_covariance` introduces zero correction).

**Gate 3: Self-duality of statistics.**
- Run 100 trajectories at (λ_A = 0.3, ζ = 1) and 100 at (λ_A = 0.7, ζ = 1)
  with paired seeds. Under the KMR self-duality, the *distributions* of
  observables in the two ensembles should match (up to a known sign
  on the Pfaffian invariant). In particular `S(L/2)` distributions
  should be statistically indistinguishable. If they differ, **the
  self-duality is broken at the level of the algorithm**, which is a
  bug.

**Gate 4: Smoke run on Habrok.**
- Submit `submit_caseA_smoke.sh`. It should complete and produce
  a valid pickle.

**Gate 5: PPS reweighting cross-check.**
- At (L = 32, λ_A = 0.5), run with ζ = 1 and ζ = 0.5 (50 clones each).
  Pfaffian-invariant variance should be qualitatively similar (NOT
  shifted), confirming PPS does not move the transition.

**Gate 6: Full scan.**
- Submit `submit_caseA_scan.sh`. Monitor for completion.

**Gate 7: Analysis.**
- Run `analysis/case_A/binder_analysis.py`. Confirm flat λ_c(ζ) ≈ 0.5
  curve.

---

## 6. Expected runtime and resource budget

| Stage | Wallclock estimate |
|---|---|
| Code implementation (steps 4.1–4.6) | 1 day of focused work |
| Local smoke test (L=16, 100 traj) | < 1 minute |
| Habrok smoke test (gate 4) | < 1 hour walltime |
| Full scan submission (gate 6) | 4–8 hours total walltime |
| Aggregation & analysis (gate 7) | 2 hours |

Habrok CPU budget for the full scan: ~260 jobs × (30 min avg) = 130
CPU-hours. Fits easily in a normal allocation.

---

## 7. Deliverables

After completing all gates, the following should exist in the repo:

```
pps_qj/
  gaussian_backend_caseA.py       (NEW, ~250 lines)
  cloning_caseA.py                (NEW or extend cloning.py)
  observables/topological_z2.py   (NEW, ~50 lines)
scripts/
  run_caseA.py                    (NEW, ~150 lines)
analysis/case_A/
  aggregate.py                    (NEW)
  binder_analysis.py              (NEW)
slurm/
  submit_caseA_smoke.sh           (NEW)
  submit_caseA_scan.sh            (NEW)
tests/
  test_caseA_backend.py           (NEW, unit tests for gates 1–2)
theory/
  CASE_A_NUMERICAL_RESULTS.md     (NEW, summary of findings)
```

The `CASE_A_NUMERICAL_RESULTS.md` writeup should contain:
1. A table of extrapolated λ_c^∞(ζ) for the four ζ values, with
   statistical error bars.
2. A statement: "Self-duality confirmed (or NOT confirmed) within
   X σ."
3. A plot of S(L/2) vs log(L) at λ_A = 0.5 for each ζ, with the
   extracted central charge.
4. A plot of `<|ν|>` vs λ_A for each L and ζ, showing the transition
   sharpen with L.
5. Honest reporting of any discrepancies — including if KMR's
   ν ≈ 5/3 is reproduced or replaced by ν = 1.

---

## 8. Things to ASK ABOUT before silently making decisions

If any of the following are ambiguous, stop and ask:
- Whether to use the cloning population dynamics (with branching) vs
  pure independent trajectories with PPS reweighting. Both are
  implemented in the Case B code; default is **cloning** with
  `n_clones = 300`.
- Which Binder-type observable to use for the topological transition.
  Default: variance of ν (the Pfaffian sign) over the realization
  ensemble. Document the choice.
- Whether to compute the Pfaffian invariant only at final time or at
  multiple times during each trajectory. Default: final time only,
  to match Case B's protocol.
- Whether T = 4L is sufficient for steady state in Case A. The Case B
  default works because measurement strength is comparable to band
  velocity; in Case A there is no velocity scale, so steady-state
  approach is set entirely by the total rate γ + α = 1. T = 4L should
  be ample, but check with a single trajectory at L = 64 whether
  observables have plateaued by T = 2L vs T = 4L.

---

## 9. Backout plan

If the implementation reveals a bug in the existing Case B code (e.g.
in `gaussian_backend.py`), **do not modify Case B silently**. Open an
issue describing the bug, and only after confirming the fix does not
change Case B's numerical output should it be applied.

If the numerical results contradict the self-duality prediction (i.e.
λ_c^∞(ζ) depends on ζ in Case A), this is a major finding that
falsifies a theoretical claim in `theory/CASE_A_IMPLEMENTATION_SPEC.md`
itself. Document the finding precisely and flag for theoretical review
before drawing any conclusions about the code being broken vs the
theory being wrong.

---

**End of specification.**
