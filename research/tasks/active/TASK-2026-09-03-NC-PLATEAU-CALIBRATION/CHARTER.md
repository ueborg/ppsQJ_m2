# TASK-2026-09-03-NC-PLATEAU-CALIBRATION — task charter

Labels: `[E]` evidence · `[I]` inference · `[C]` conjecture · `[J]` judgment.

**Mode: coordinated numerical campaign preparation.** The deliverable is a set
of runnable, validated HPC packages and a frozen analysis, not a scientific
conclusion. **Terminal state: `READY_FOR_HUMAN_SUBMISSION` at Human Gate A.**

---

## The question

> `[J]` **What is the smallest defensible `N_c` required to locate the
> transition at each relevant `L`, and does a usable asymptotic finite-`N_c`
> regime actually emerge?**

Operationally, in one sentence with named quantities:

> For `zeta = 0.35`, `T = L`, the certified guided-cloning sampler and the
> observable `OBS-CMI-001`, at which `N_c` does `Delta_N = I_{2N} - I_N` first
> satisfy the frozen plateau criteria P1–P5 at `L = 64`, `96` and `128`, and
> does the inferred cross-`L` crossing location stabilise to within
> `tau_lambda` at a **smaller** `N_c` than the absolute level does?

## Hypotheses under test

| id | statement | where |
|---|---|---|
| **P** | a high-`N_c` plateau is OBSERVABLE at `L = 64` within reach of `N_c = 8192` | campaign A |
| **H1** | the finite-`N_c` correction is an additive constant in `lambda` | campaign B |
| **H2** | it is a multiplicative rescaling | campaign B |
| **H3** | it is a resolved `lambda`-dependent shape distortion | campaign B |
| **LOC** | the crossing location converges at smaller `N_c` than the absolute level | campaigns B + B2 |
| **E1** | finite-`N_c` drift changes materially with the window count `K` | campaign E |
| **E2** | results approach a discretisation-stable continuous-time particle limit | campaign E |

## Kill criterion

`[E]` Frozen in `SUCCESS_CRITERIA.yaml` before any new datum exists.

- **P is killed** if, at `L = 64`, `Delta_N` remains resolved away from zero at
  the `4096 -> 8192` step. The report is then *even `L = 64` remains
  pre-asymptotic*, and no `I_inf` is extrapolated from a rejected model.
- **H1/H2 are killed** individually by their own `chi2` over the seven measured
  `lambda`. **Neither being rejected is an UNRESOLVED outcome, not a win for
  either.**
- **LOC is killed** if the crossing displacement per `N_c` doubling does not
  shrink, or does not fall inside `tau_lambda`.
- **E1 and E2 each kill the other**, and an intermediate result kills neither
  and must be reported `INCONCLUSIVE`.

## What this task may never conclude

`[E]` Not from any result, however it lands:

1. any `lambda_c(zeta)`, any phase-boundary law, or any exponent;
2. that the `0.2182–0.2482` window is a critical window — it is an **observed
   locator region** in `L <= 64` curves at `N_c = 1024`;
3. any general `N_c_req(L, zeta, lambda)` rule;
4. that `1/N_c` convergence is impossible. The frozen theory result is
   narrower and is preserved exactly as stated: *the standard useful
   uniform-mixing Feynman–Kac bounds do not directly transfer to the production
   mutation kernel, because the no-click branch is deterministic.* That is the
   failure of a **proof route**, not of convergence.

## Standing constraints this task operates under

`[E]` `research/RESOURCE_POLICY.md` §4: **no agent submits an HPC job**, at any
stage, gate, or approval level. `[E]` `research/state/**` is read-only and this
task does not write it. `[E]` No predecessor task directory is modified.
`[E]` The six live disputes in `research/state/disputes/` are not closed here;
nothing in this campaign bears directly on any of them, which is itself
recorded in `ASSESSMENT_AH.md`.

## Vocabulary, enforced

`[E]` `task-verified` != `canonical`. `proposed promotion` != `promoted`. The
local, read-only work done here is **T0 analysis compute**, not "no compute".
Finite-`N_c` movement is **drift**, never **bias**, because the
`N_c -> infinity` target is unknown.
