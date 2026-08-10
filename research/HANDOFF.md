---
lifecycle: active
authoritative_for: navigation and in-flight work only
contains_scientific_values: false
last_reviewed: 2026-08-10
---

# ppsQJ_m2 — Handoff

**This file contains no scientific values.** Exponents, amplitudes, phase-boundary
numbers and other mutable conclusions live in `research/state/claims/` and are
referenced here by ID only. If you want a number, open the claim file.

## 1. Project identity

Measurement-induced phase transitions in a monitored 1D Kitaev/Majorana chain
under quantum-jump unraveling with partial post-selection parameter ζ.

- **Cut A**: α+γ=1, w=0. Two competing on-site measurements, no Hamiltonian.
  Self-dual, class D.
- **Cut B**: γ=0, α+w=1. Single Bogoliubov-density measurement plus Kitaev
  hopping. Class DIII. The main campaign.

Extends KMR (SciPost Phys 14, 031) and LMR (PRX 15, 021020), which use quantum
state diffusion rather than quantum jumps.

## 2. Current research decision

**The project is in post-audit migration, not active research.** A full
reconstruction audit (`audit/2026-08-10/`) found the previous information
architecture unsafe for delegated work. The knowledge plane under
`research/state/` was bootstrapped on 2026-08-10 and is incomplete.

No new physics investigation should begin until the migration blockers in
`research/README.md` §4 are cleared.

## 3. Active tasks

None. `research/tasks/active/` is empty.

## 4. Open disputes

| id | question |
|---|---|
| `DISP-PHI-001` | Cut B boundary exponent: φ ≈ 1/2 versus φ = 1 |
| `DISP-WINDOW-001` | Is any fitted exponent asymptotic, given window drift |
| `DISP-XI-001` | No-click length: λ⁻¹ versus λ^−1.5 versus λ⁻² |
| `DISP-CASEA-UNIV-001` | Cut A universality and the higher-ζ crossing drift |
| `DISP-SNAPSHOT-001` | Production value of snapshot averaging |
| `DISP-YZETA-001` | Consequences of Δ ≈ 2 for y_ζ and the θ₁ chain |

Full list: `research/state/disputes/`. These are **not to be resolved by
argument**. They are the queue for the first real research tasks.

## 5. Where canonical state lives

```
research/state/claims/       scientific claims, status, evidence links
research/state/evidence/     datasets, analyses, derivations, code, sources
research/state/observables/  observable and estimator definitions
research/state/disputes/     preserved unresolved disagreements
research/state/decisions/    methodology decisions and kill records
research/state/sources/      literature, with inspection level
```

Validate with `research/tools/validate_state.py`.

## 6. Operational warnings

- **Two filesystems.** Desktop Commander reaches the Mac. `bash` reaches the
  container. Never confuse them.
- **Git from the Mac only**, never from HPC. Stage specific files. Never
  `git add -A`. As of 2026-08-10 the working tree has ~100 uncommitted paths;
  see `research/GIT_MIGRATION_PLAN.md`.
- **`analysis/anchor_scan.py` is wrong** and produces plausible-looking output.
  Its kernel drops the hopping w from the measured bond. Do not use it. See
  `EV-CODE-ANCHORSCAN-001`.
- **`theory/**` is historical or unreconciled.** Check the `lifecycle:`
  front-matter before reading. Nothing under `theory/` is authoritative.
- **Data lives in two places** with no unified index:
  `~/Downloads/01_M1_Internship/Data/` and `results/` inside the repo. The
  second is newer and larger and was undocumented until the audit.
- **No HPC job** without an approved `research/experiments/<EXP-ID>.yaml`.

## 7. What to read next

1. `research/RESEARCH_CHARTER.md` — mandatory, governs all substantive work.
2. `research/README.md` — how the knowledge plane is organised, and what is
   still missing.
3. Then only the specific claim, evidence and observable files your task
   touches. **Do not preload the state.**

For historical context, and only as provenance, never as evidence:
`research/history/legacy/HANDOFF_pre_reconstruction_2026-08-10.md` and the audit
under `audit/2026-08-10/`.
