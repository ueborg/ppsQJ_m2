# Missing results: executed work absent from HANDOFF and memory

Audit 2026-08-10. Evidence class for all items below: `EXEC` (executed script
with preserved output) but stored **only in `/tmp`**, therefore ephemeral.

## M1 — The "open gap" in snapshot averaging has been closed. HANDOFF still calls it open.

`/tmp/crossing_prod.log`, `/tmp/crossing_prod_result.json`, timestamp
**2026-08-10 12:13**, driver `analysis/var_reduction/crossing_prod.py` (11:06).
Both postdate the last HANDOFF write (10:28).

HANDOFF's top banner states, emphatically:

> **THE OPEN GAP:** every attractive number above used **T=64**, but production
> uses **T≈L** … the production-relevant quantity `g_prod(L)` **has not been
> measured at any L.** Until it is, do not budget a specific snapshot gain.

It has now been measured, at production settings (T=L, mult=4, production
burn-in 0.25, N_c=32, R=18, ζ=0.9, L1=32/L2=64, λ ∈ {0.440, 0.474, 0.508}),
and scored on the master metric σ_λc with a full bootstrap (4000 resamples):

| estimator | λ_c | sd | G_snap on λ_c |
|---|---|---|---|
| terminal | 0.5053 | 0.0127 | 1.00 (reference) |
| dense (K=8) | 0.4942 | **0.0032** | **15.79×** |
| sparse (K=4) | 0.4936 | 0.0034 | 14.23× |

This simultaneously executes **NEXT QUEUE item (2)** (production-matched T=L
snapshot certification) and **item (4)** (bootstrapped three-point crossing
estimator). HANDOFF lists both as pending.

All three estimators agree on λ_c within ~0.9σ, so there is no bias signal.

**Audit caution, not yet resolved.** G_snap = 15.8× at K=8 is roughly double the
naive ceiling K. HANDOFF correctly notes `g_snap > K` is structurally possible
when the weighted autocorrelation sum is negative, so this is not impossible.
But the ratio is driven by the *terminal* arm's sd (0.0127) at R=18, where a
variance estimate carries roughly ±35%. The honest reading is "large, single-cell,
one L-pair, not yet a budgetable number", not "15.8×". Status `[P]`.

## M2 — The L=48 snapshot point was read and it breaks the monotone story.

`/tmp/coupsnap_L48.json`, `/tmp/coupsnap_scan.log`, timestamp **2026-08-10 10:44**.
This is **NEXT QUEUE item (1)** ("Read L=48 from `/tmp/coupsnap_scan.log`").

Paired-difference snapshot gain at fixed T=64, spacing 8, K=5:

| L | 32 | 40 | **48** | 64 |
|---|---|---|---|---|
| gain | 5.85× | 7.73× | **1.22×** | 2.66× |

The L=48 cell shows **essentially no snapshot gain**. The sequence
5.85 / 7.73 / 1.22 / 2.66 is non-monotone by a factor of six and has no
plausible physical reading as a function of L.

This is load-bearing in the negative direction. HANDOFF currently labels
snapshot averaging "**the largest confirmed lever**" `[P]`. With L=48 included,
the fixed-spacing-8 measurements are more consistent with being noise-dominated
at R=14–16 than with any L-trend, which is exactly what HANDOFF's own warning
anticipated ("DO NOT fit τ ∝ L or a sharp 48/64 threshold to these four points").

Supporting detail: at L=48 the GLS weights are again wildly non-uniform
(0.039, 0.057, 0.328, 0.062, 0.514), reproducing the overfitting signature that
led to GLS being rejected.

Per-realisation σ_λc from this run: 0.0090 (L=48, F' ≈ 14.65), 0.0076 (L=40,
F' ≈ 13.04).

## M3 — The entire 2026-07-27 → 2026-08-10 programme is uncommitted

Last commit on `main`: **9b617fa, 2026-07-26**. Working tree has 106 modified or
untracked paths. Specifically untracked (never in git, never pushed):

- `analysis/var_reduction/` — 20 scripts, the whole controlled-Doob / cloning
  / snapshot / bottleneck programme
- `theory/VARIANCE_REDUCTION.md` — a document HANDOFF cites as authority
  (§1, §5) for closed research directions
- `tests/conftest.py`, `tests/test_backward_pass_sector.py`,
  `tests/test_exact_benchmark.py` — the test suite may not run from a clean clone
- `scripts/aggregate.py` — named in HANDOFF's Operational section as *the*
  aggregate script

`theory/HANDOFF.md` itself carries +132 uncommitted lines.

Everything dated 2026-07-27, 2026-08-09 and 2026-08-10 in HANDOFF — the master
metric, the sampler-programme closure, the chunk certification, the snapshot
work, the N_c ladders — exists on exactly one machine, in one working tree,
with no backup. This is the single largest concrete risk found in Stage 1 and it
is not a documentation problem.

## M4 — Results present in memory but nowhere on disk

The following appear in project memory with no corresponding repo document:

- θ₁^SCGF = 0 by fermion parity; "BdG quantity is no-click activity, not SCGF
  derivative"; 2×2 parity-doublet effective generator reproducing L_ζ to 1e-8;
  `K_eff ~ L⁰` as the correct expectation for Δ_ζ = 1.
- Cut A order-parameter claim: end-to-end MI `I(A_left:C_right)` is correct and
  goes 0→1, CMI is peaked and is not an order parameter; region assignment must
  be ABDC.

Partial on-disk support exists: `analysis/compute_theta1_exact.py`,
`analysis/parity_resolved_theta.py`, `analysis/parity_resolved_data.pkl`,
`theory/theta1_first_principles.md` (2026-05-18), `analysis/theta1_scaling.png`.
None of these was inspected in Stage 1 and none is referenced anywhere in
HANDOFF. The Cut A observable claim has **no located on-disk source at all**
and is a Stage 2 chat-archaeology target.

Note the θ₁ material also carries a dependency on `Δ_ζ = 1`, which HANDOFF
declares corrected to Δ ≈ 2. Whether the θ₁ conclusions survive that correction
is unexamined.

## M5 — Case A production data exists; three sources say it does not

See `02_DATA_INVENTORY.md` G4. 574-record guided Case A aggregate with
per-realisation arrays. Memory pending-action "Implement Cut A code" and
HANDOFF's TL;DR "production Binder scan + FSS not yet run" are both false.
