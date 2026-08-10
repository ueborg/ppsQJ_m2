# Data inventory

Audit 2026-08-10. Read-only inspection via
`audit/2026-08-10/scripts/inspect_aggregates.py`.

## Headline: every data path in HANDOFF.md is dead

All of `~/Downloads/pps_aggregates/`, `~/Downloads/pps_clone_guided_ladder`,
`~/Downloads/pps_quicklook/`, `~/Downloads/m1thesislatex/`,
`~/Downloads/continuousmeasurements(2)/`, `~/Downloads/clone_aggregate(1).pkl`,
`~/Downloads/aggregate_runAC.pkl`, `~/Downloads/aggregate_B.pkl`, and
`~/Downloads/clone_aggregate_dense_partial.pkl` are **MISSING** at the paths
HANDOFF and project memory record. The material was reorganized under
`~/Downloads/01_M1_Internship/{Code,Data,Figures,Papers,Thesis}`.

Consequence: an agent following HANDOFF's "Data on disk" table finds nothing and
may conclude the data does not exist.

## Current aggregates (`~/Downloads/01_M1_Internship/Data/pps_aggregates/`)

Guided-cloning generation. Rebuilt 2026-06-17.

| file | n | cut | L | ζ | λ | N_c | T | per-real arrays |
|---|---|---|---|---|---|---|---|---|
| `agg_caseB_combined.pkl` | 1046 | B | 32,48,64,96,128,160 | 0.05–0.85 (15) | 0.034–0.550 (191) | 250–500 | 64–147 | no |
| `agg_caseB_perreal.pkl` | 1046 | B | same | same | same | same | same | **yes** (S_AB/S_BC/S_B/S_ABC, Rényi 2,3) |
| `agg_B_prod.pkl` | 925 | B | 32–128 | 0.05–0.85 | 191 pts | 300–500 | 64–147 | no |
| `agg_B_highL.pkl` | 121 | B | **160 only** | 0.05–0.35 | 121 pts | 250 | 128, 147 | no |
| `agg_pps_clone_guided_prod.pkl` | 877 | B | 32–128 | 0.05–0.85 | 191 pts | 300–500 | 64–147 | **yes**, full 70-field set |
| `agg_pps_clone_guided_highL.pkl` | 103 | B | 160 | 0.05–0.35 | 103 pts | 250 | 128, 147 | **yes**, full set |
| `agg_ladder.pkl` | 161 | B | 96,128,160 | 0.05–0.40 | 72 pts | **500, 600** | 128 | partial |
| `agg_A.pkl` | 574 | **A** | 32,48,64,96,128 | 0.05–0.85 | 0.42–0.58 (13, symmetric about 0.5) | 300–500 | 64,96,128 | no |
| `agg_caseA_perreal.pkl` | 574 | **A** | same | same | same | same | same | **yes** |
| `agg_pps_caseA_guided.pkl` | 574 | **A** | same | same | same | same | same | **yes**, 47-field set |

`n_real = 5` in **every** current aggregate without exception.

## Legacy aggregates (`Data/old_cloning_data/`)

Pre-guided estimator generation. Retain for the ζ=1 anchor only.

| file | n | ζ_max | has ζ=1 |
|---|---|---|---|
| `clone_aggregate_dense_full.pkl` | 3349 | 1.0 | **yes** |
| `clone_aggregate(2).pkl` | 1920 | 1.0 | **yes** |
| `ladder_fss_ready.pkl` | 891 | 1.0 | **yes** |
| `clone_aggregate_rescue.pkl` | 117 | 0.8 | no |
| `ladder_nc{250,500,800}.pkl` | 21/91/21 | 0.3 | no |
| `caseA_agg.pkl` | 102 | 0.5 | no |

## Four structural gaps the data reveals

**G1 — No guided data at ζ = 1.** Maximum ζ in every guided aggregate is 0.85.
ζ=1 exists only in the superseded non-guided sets. HANDOFF (2026-06-17) calls
"the Born-endpoint reproduction … the robust headline". On guided data that
headline is an **extrapolation from ζ ≤ 0.85**, not a measurement. The
2026-06-17 data plan made "ζ=1 Cut B ladder FIRST" its top numerical item.
It was never run.

**G2 — No data anywhere satisfies T ≥ 2L.** Observed T values are
{64, 67.6, 68.5, 82.4, 83.8, 96, 105.5, 128, 147.1}. T/L = 1.0 at L=128 and
0.80–0.92 at L=160. HANDOFF's own [V, CRITICAL] 2026-06-17 finding says the ν
tier requires T ≥ 2L and that existing L≥96 data is suspect for ν and should be
re-run rather than supplemented. **That re-run does not exist.** Every ν
statement in the project rests on data HANDOFF itself declared unfit for ν.

**G3 — n_real = 5 everywhere.** The 2026-06-17 calibration found the LMR
interpolation estimator *saturates* at n_real=5 (ν_true 2.0→1.31, 2.5→1.36,
3.0→1.31, i.e. cannot distinguish 2 from 3) and needs n_real≈25. No n_real=25
dataset exists. The ν programme is blocked on statistics that were never
collected.

**G4 — Case A production data EXISTS.** 574 records, five sizes, fifteen ζ,
thirteen λ symmetric about 0.5, with per-realisation arrays. This directly
contradicts project memory ("current code only does Cut B"; pending action
"Implement Cut A code") and contradicts HANDOFF's own TL;DR ("production Binder
scan + FSS not yet run"). HANDOFF's 2026-06-17 banner *does* list the aggregate,
so HANDOFF contradicts itself internally between its top and bottom halves.

Open sub-question: memory asserts the Cut A order parameter must be end-to-end
mutual information with ABDC region assignment, not peaked CMI. The Case A
aggregates carry the standard CMI tripartition fields (`S_AB`, `S_BC`, `S_B`,
`S_ABC`) plus `B_L`. Whether the stored region assignment matches the claimed
requirement is **unresolved by this audit** and needs a code read of
`pps_qj/parallel/worker_caseA.py`.

## Ephemeral evidence (`/tmp`, survives as of 2026-08-10)

The entire 2026-08-09/10 variance-reduction evidence base is in `/tmp` and is
**one reboot from deletion**. Present: `ascan_L{32,64}_z0.9.json`,
`bottleneck_L32.json`, `chunkcert_L32.json`, `coupsnap_L{32,40,48,64}.json`,
`d2scaling.json`, `doob_screen*_*.json`, `crossing_prod.json`, plus logs and the
canonical prototypes `csampler2.py`, `doob_galerkin.py`, `controlled_sampler.py`.

`/tmp/coupsnap_L48.json` (2026-08-10 10:44) and `/tmp/crossing_prod*.json`
(2026-08-10 12:13) **postdate the last HANDOFF edit** (10:28). They are results
that have never been written down anywhere. See `04_MISSING_RESULTS.md`.
