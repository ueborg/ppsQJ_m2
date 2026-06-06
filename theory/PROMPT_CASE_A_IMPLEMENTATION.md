# New-chat prompt: Case A code implementation and numerical investigation

## What this project is

You are working on `ueborg/ppsQJ_m2`, a physics research project studying
Measurement-Induced Phase Transitions (MIPTs) in a 1D Kitaev chain under
quantum-jump (QJ) dynamics with partial post-selection (PPS), parametrized
by ζ ∈ (0,1]. The codebase is at `/Users/catlover1337/Documents/ppsQJ_m2/`.

Read `theory/HANDOFF.md` first for the full project state.

## The specific task

Implement and run the **Case A** model variant, which is currently **unimplemented**.
The complete implementation specification is in:

```
theory/CASE_A_IMPLEMENTATION_SPEC.md   (612 lines — read this in full)
```

This document specifies everything you need: model definition, which code files to
extend, how to implement the two-channel jump mechanism and the Z₂ topological
invariant, the SLURM job structure, and the acceptance tests.

## The central physics question

**Is the critical line λ_c^A = 1/2 strictly independent of ζ for all ζ ∈ (0,1]?**

The theory prediction (from D-class NLSM + KMR self-duality, documented in
`theory/NLSM_FRAMEWORK.md` and `sections/sec8_replica_nlsm_pps.tex`) says yes:
the self-duality c_j ↔ d_j, α ↔ γ is preserved by PPS because ζ^M depends only
on the total click count, not on which channel fired. Therefore the self-dual line
λ_A = 1/2 is pinned for all ζ.

Compare: Case B has λ_c^B ∝ √ζ (the phase boundary moves with ζ). Case A has
no such movement by symmetry.

The secondary predictions are:
- Ising universality class (c = 1/2, ν = 1) at the critical line
- Two area-law phases distinguished by Z₂ Pfaffian topological invariant
- ν ≈ 5/3 at ζ = 1 from KMR diffusive numerics; QJ may drift toward ν = 1

## Key model differences from existing Case B code

| Quantity | Case B (exists) | Case A (to implement) |
|---|---|---|
| Hamiltonian | H = -w Σ c†_{j+1}c_j | H = 0 |
| Jump operators | L̃_j = d†_j d_j only | n_j = c†_j c_j AND L̃_j = d†_j d_j |
| Rate param | λ = α/(α+w) | λ_A = α/(α+γ), with α+γ=1 |
| Key observable | B_L (Binder cumulant of S) | Pfaffian invariant + B_L |
| Self-duality | No | Yes (α ↔ γ) |

The Majorana index pairs are:
- c_j density jump pair: `(2j, 2j+1)` — the INTRA-cell Majorana pair
- d_j density jump pair: `bond_jump_pair(j) = (2j, 2j+3)` — existing function

The hop between `apply_projective_jump(covariance, jump_pair)` is channel-agnostic
and requires no modification.

The Z₂ topological invariant: Pfaffian of the spectrally flattened Majorana covariance
matrix Q = sign(iΓ) (replace each eigenvalue λ_k of iΓ with +1 if λ_k > 0, -1 if
λ_k < 0). The topological phase is Pf(Q) = ±1. At the critical line, the cloning
ensemble should give Pf(Q) ≈ 0 (equal probability of both signs).

## Data I already have for Case B

Aggregate pickle: `/Users/catlover1337/Downloads/clone_aggregate(1).pkl` (1920 entries)
plus newer runs in `aggregate_runAC.pkl` and `aggregate_B.pkl`. All have `B_L_mean`
stored. These can serve as templates for the analysis pipeline.

## Expected deliverables

1. `pps_qj/gaussian_backend_caseA.py` — two-channel model, cached h_eff (H=0, so
   h_eff = -iα/2 Σ_j n̂_j - iγ/2 Σ_j d̂_j†d_j as a real antisymmetric Majorana matrix)
2. `pps_qj/observables/topological_z2.py` — Pfaffian invariant
3. `scripts/run_caseA.py` — CLI wrapper
4. SLURM submit scripts for a scan of (λ_A, ζ, L) at L ∈ {32,48,64,96,128}
5. Analysis script producing the λ_c^A(ζ) plot showing it is flat at 1/2

## Repository

GitHub: `ueborg/ppsQJ_m2`. The SSH push issue from Habrok means changes must be
committed locally (Mac), pushed to GitHub, then pulled on Habrok. Habrok user is
s4629701, interactive node is interactive2.
