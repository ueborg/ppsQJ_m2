# Stage 1 corrections

Audit 2026-08-10, Stage 2. Corrections to my own Stage 1 output, forced by
evidence found in Stage 2. Recorded rather than silently patched.

## Root cause of the Stage 1 errors

Stage 1 trusted HANDOFF's framing that production data lives under
`~/Downloads`. It does not, or not only. The repository's own `results/`
directory holds a **newer and larger** campaign (16,344 files under
`results/ruche_pull/`, every one post-2026-07-01) which HANDOFF never mentions.
Data is split across two locations with no index, and the location HANDOFF
points at is the older one.

## C1 — "No guided data at ζ = 1" is WRONG

Stage 1 `02_DATA_INVENTORY.md` gap G1 and ledger `CB-BORN-001` claimed no
guided dataset reaches ζ = 1 and that the Born endpoint is therefore an
extrapolation. **False.**

`results/boundary_aggregate.csv` (470 rows, 5634 realizations) covers
ζ ∈ {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, **1.0**}.

Additionally, and independent of that: **at ζ = 1 the PPS weights are ≡ 1**, so
cloning is a null operation and the guided-versus-non-guided distinction does
not exist there. The legacy ζ=1 data was therefore never invalidated by the
guided-estimator transition. HANDOFF says as much itself when it describes the
ζ=1 ladder as "cloning-free: weights ≡1, no population needed".

**Corrected `CB-BORN-001`.** Legacy ζ=1 Binder crossings from
`clone_aggregate(2).pkl` (L = 8…128, 22 wide pairs) give a median λ_c(1) = 0.4927.
But the pair sequence **drifts monotonically downward with L**: 0.5266 at (8,16)
to 0.4759 at (64,128), with (48,128) giving 0.4768. The agreement with 1/2 is
therefore partly an artifact of averaging small-L pairs, and the L→∞
extrapolation that `NUMERICS_STATUS_AND_PLAN.md` §5 explicitly requires would
land **below** 0.476. Status `[P]` with a downward-drift caveat, not `[V]`.

## C2 — "n_real = 5 everywhere" is WRONG

Gap G3. True of the `~/Downloads` aggregates, false of the repository campaign:
`boundary_aggregate.csv` has `nreal` ∈ {6, 12}. The ν-programme statistics
situation is better than Stage 1 reported, though still short of the ≈25 that
the 2026-06-17 LMR calibration asks for.

## C3 — L = 80 and L = 112 exist

Stage 1 listed sizes {32,48,64,96,128,160}. The July campaign adds **L = 80 and
L = 112**. This matters: the 2026-06-17 quotients analysis singled out L=80 as
"high value at 1/16 the cost of an L=160 run" because it supplies the missing
(80,160) pair. That size was run and the fact is recorded nowhere.

## C4 — The Cut A end-to-end MI observable is implemented in code

Stage 1 `M4` listed the Cut A observable claim as having "no located on-disk
source at all". It has one. `pps_qj/parallel/worker_caseA.py` is modified
(uncommitted, +16 lines) adding `_batched_compute_MI_ends` and emitting
`MI_ends_q4_mean/err` and `MI_ends_q8_mean/err`. See `CHAT_ARCHAEOLOGY.md` §2.

## C5 — The θ₁ work is committed, not chat-only

Stage 1 `M4` treated the θ₁/parity material as memory-only. It was committed in
a "Parity-resolved correction" commit together with
`analysis/compute_theta1_exact.py`, `analysis/parity_resolved_theta.py`,
`analysis/parity_resolved.png`, `analysis/parity_resolved_data.pkl` and
`theory/theta1_first_principles.md`. Evidence class `EXEC` + `CODE`.
What is chat-only is its *absence* from HANDOFF, not the work.

## C6 — The amplitude conflict is already resolved in the live manuscript

Stage 1 `03_AMPLITUDE_TRACE.md` treated the manuscript exposure as live. The
SciPost article uses **0.50√ζ** with appropriate hedging. The 0.96 exposure is
confined to `m1thesislatex`, which is the completed M1 report (deadline
2026-06-19), i.e. a historical submitted document. See `MANUSCRIPT_LINEAGE.md`.
Priority downgraded from urgent to "correct if the M1 report is ever reissued".

## What stands unchanged

Gap G2 (no data anywhere at T ≥ 2L) — the July campaign does not change this and
I did not verify T for it. Still open, still the binding constraint on ν.
M1, M2, M3 (uncommitted tree) stand. The amplitude reproduction stands.
