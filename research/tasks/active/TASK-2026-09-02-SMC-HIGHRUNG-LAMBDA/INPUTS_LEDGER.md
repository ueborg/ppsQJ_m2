# Inputs ledger — every file this design was derived from

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA, brief §0.

The prose summary in the brief was **not** used as the input. Every number was
re-derived from the result files below and cross-checked against the published
analyses.

## Files read

| file | sha256 (first 32) | bytes | tracked in git |
|---|---|---:|---|
| `…/TASK-2026-09-01-SMCRUCHE-READY/arm1/ARM1_FINAL_ANALYSIS.txt` | `58de6ff424d13772e7333f6fb5e114c4…` | 2130 | **no** |
| `…/TASK-2026-09-01-SMCRUCHE-READY/arm2/ARM2_FINAL_ANALYSIS.txt` | `8e32c3641703275ed3f0a9d397f0f076…` | 2242 | **no** |
| `…/TASK-2026-09-01-SMCRUCHE-READY/arm1/results/` (112 JSONs) | rollup `d53c84327faa3cb395a360c7143204e8…` | — | **no** |
| `…/TASK-2026-09-01-SMCRUCHE-READY/arm2/results/` (192 JSONs) | rollup `bb1ff49ca3e2bdd9e2bfc6a08e6db3f0…` | — | **no** |
| `…/TASK-2026-08-30-SMCSTAT/scratch/A-P96.jsonl` | `a506fc2ef193dfa2f6c357a30976bec2…` | 1626821 | **no** |
| `…/TASK-2026-08-30-SMCSTAT/scratch/A-BUD.jsonl` | `d20e65761b3f421227675bbf4c208868…` | 3034106 | **no** |
| `…/TASK-2026-08-30-SMCSTAT/scratch/numB_cells.csv` | `ce297b7f35f6808564179cef4862825c…` | 399261 | **no** |
| `/Users/catlover1337/Downloads/pps_all_realizations.csv` | `7066bac78198f5a93fa5688a5490540c…` | 8023862 | outside the repo |
| `…/TASK-2026-09-01-SMCRUCHE-READY/support/instrumented.py` | `0a33c4034cda70ea635cf715ee0b160d…` | 11396 | **yes** |

(The "rollup" is the SHA-256 of the concatenated per-file SHA-256s, in sorted
filename order.)

## The tracking problem, and what was done about it

Almost everything above is **untracked**. A package that read those paths at
runtime would fail on a clean clone — which is exactly the failure that killed
the first ARM 1 attempt and prompted `TASK-2026-09-01-SMCRUCHE-PACKFIX`.

Two rules follow, and both are enforced rather than intended:

1. **Nothing in the runtime path touches an untracked file.** `run_cell.py`
   imports only the bundled `support/instrumented.py` (tracked, SHA-gated) and
   the tracked `pps_qj` package. `preflight.py` verifies this by actually
   importing them in a subprocess and fails otherwise. The clean-clone test in
   `VALIDATION.md` runs the package out of a `git archive` in which none of the
   untracked directories above exist.

2. **Everything the analysis needs from a predecessor is snapshotted into a
   tracked file.** `frozen_inputs/predecessor_populations.csv`
   (`971d272a1aa3b0f4861975475490f4dc…`, 145,594 bytes, 528 rows) carries one
   row per independent population — its cell, `N_c`, seed, weighted mean CMI,
   within-clone variance, non-finite count, genealogy diagnostics, wall time —
   plus, per row, the path and SHA-256 of the file it came from.

The predecessor task archives were **not modified**. Nothing was added to their
git tracking, nothing was rewritten, nothing was deleted.

## Verification that the snapshot is faithful

Recomputing the per-cell statistics from `frozen_inputs/` reproduces the
published ARM1 and ARM2 final analyses digit for digit:

```
  L= 96 N_c=  32 R= 128 mean=0.62354 SEM=0.02145 Var=5.8900e-02 VIF= 48.70
  L= 96 N_c=  64 R=  64 mean=0.49871 SEM=0.02792 Var=4.9875e-02 VIF= 79.33
  L= 96 N_c= 128 R=  48 mean=0.36467 SEM=0.01934 Var=1.7955e-02 VIF= 75.10
  L= 96 N_c= 256 R=  48 mean=0.33921 SEM=0.01628 Var=1.2716e-02 VIF= 95.94
  L= 96 N_c= 512 R=  48 mean=0.26631 SEM=0.01027 Var=5.0669e-03 VIF= 93.26
  L=128 N_c=  64 R=  64 mean=0.51957 SEM=0.02494 Var=3.9811e-02 VIF= 71.79
  L=128 N_c= 128 R=  64 mean=0.42059 SEM=0.02354 Var=3.5474e-02 VIF=146.53
  L=128 N_c= 256 R=  64 mean=0.29932 SEM=0.01679 Var=1.8049e-02 VIF=177.48
```

Every value matches `ARM1_FINAL_ANALYSIS.txt` and `ARM2_FINAL_ANALYSIS.txt`, and
matches the brief's §0 summary.

## The verified input, restated

**ARM1** — L = 96, T = 96, ζ = 0.35, λ = 0.3032. Variance-scaling verdict
**SUPPORTED**, γ = +0.905, CI [+0.744, +1.082]. Read as: at this L = 96 cell,
raising `N_c` continues to buy variance reduction roughly efficiently over the
tested range. It says nothing about a universal `1/N_c` bias law.

**ARM2** — L = 128, T = 128, ζ = 0.35, λ = 0.3032, `N_c ∈ {64, 128, 256}`,
`R = 64`. Variance-scaling verdict **INCONCLUSIVE**, γ = +0.571,
CI [+0.127, +1.007]. Large, unmistakable finite-`N_c` drift in the mean through
`N_c = 256`: **−0.0990** from 64→128 and **−0.1213** from 128→256, i.e. the
second doubling moves the mean *more* than the first. The descriptive
`I = I_inf + B/N_c` fit gives positive `B`, and per
`TASK-2026-08-31-SMCCERT` that model is **not** established as a controlled
universal asymptotic bias law and is **not** used here as ground truth or as a
definition of convergence.

Also carried forward, at ARM2's L = 128 rungs: VIF 71.79 / 146.53 / 177.48 —
high and rising, which is a variance diagnostic and is explicitly **not** treated
as a predictor of bias.

## Non-authoritative by construction

Nothing in this ledger is canonical evidence. Per `CLAUDE.md`, scientific state
lives only in `research/state/**`, which this task does not write and did not
modify. These files are task-verified inputs to a preparation decision.
