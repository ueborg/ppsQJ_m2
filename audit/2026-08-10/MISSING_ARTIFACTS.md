# Missing artifacts: chat-only claims and evidentiary gaps

Audit 2026-08-10, Stage 2. Every item below was searched for on disk.

## A. Chat-only, no executed artifact located

| claim | chat | what is missing | can it be re-derived? |
|---|---|---|---|
| **MIPT confirmed independently of crossings** via entropy scaling: S_{L/2} vs ln L shows log-law → area-law | `ca2b054c`, 2026-08-05 | no script, no figure, no output on disk | **Yes**, from `results/boundary_aggregate.csv` and the ruche_pull npz set. High value: this is fit-free and does not depend on any crossing estimator. |
| φ ≈ 0.50 from the fixed-ratio (64,128) bootstrapped CMI pair; linear excluded χ²/dof ≈ 1 vs 5–13; spread 0.36–0.57 across pairs | `ca2b054c` | no analysis script or output | Yes, data present |
| Cut A self-duality validated at ζ ≤ 0.25 (crossings 0.49–0.51); drift to ~0.42 at higher ζ | `ca2b054c` | no output | Yes, `results/ruche_pull/caseA_guided/` |
| Small-ζ windows mis-centred (multipliers 0.85–1.45 above λ_c); rerun needed at 0.45–1.05 | `ca2b054c` | no record | Diagnosis re-derivable; the **rerun does not exist** |
| End-to-end MI table (0.000 → 1.000 across λ) | `ca2b054c` | numbers exist only in chat text; the run was inline, output not saved | Yes, code now in `worker_caseA.py` |
| MI(ends) step **sharpens toward λ=1/2 with L** | `ca2b054c` | **the check timed out and never completed** | Not yet established at all |
| χ₂ scan p = 0.46–1.40, x_J = 1 not confirmed | `ca2b054c` | `analysis/chi2_*.py` exist (committed 2026-07-09); output not located | Yes |
| x_J ≈ 1.04 from the §9 click-update test on the critical L=800 state, decay r^{−2.07} | `601b6758`, 2026-06-17 | container-side, **container expired** | Re-runnable but the script is gone |
| ∂_g²CMI ~ L^{1.975} at L = 32–256, FSS collapse at CMI = 0.139 ± 0.003 at fixed gL=1 | `601b6758` | container-side, expired | Re-runnable, script gone |
| Doc-6 argument that first-order CMI response vanishes identically (⟨ε⟩_C = 0 on the uniformized replica cover) | `601b6758` | derivation in a chat-uploaded memo; **the memo documents themselves are not on disk** | No. The three memos (docs 5/6/7) are unrecovered. |
| Ashkin–Teller / Thirring mechanism, ν_B(ζ) = 1/(2−K) | HANDOFF 2026-06-17 summarises it | the source memo is not on disk | No |

## B. Evidence exists but the analysis output does not

- `theory/VARIANCE_REDUCTION.md` §5 retraction lives only in HANDOFF prose.
- `/tmp/pps_lmr*.py`, `/tmp/pps_global_*.py`, `/tmp/pps_calib*.py` (the entire
  2026-06-17 FSS-calibration suite, including `pps_lmr_robust.py` which produced
  the [1.5, 3] confidence set) — **gone**. `/tmp` retains only the 2026-08-09/10
  generation. The single most important negative result about ν rests on a
  script that no longer exists.
- `outputs/rc_scaling_analysis.png`, cited by `NUMERICS_STATUS_AND_PLAN.md` §2 —
  not located.

## C. Preserved this session

`audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/` — 93 files, 428 KB,
SHA-256 manifest in `recovered_ephemeral/MANIFEST_TABLE.md`. Grouping by
experiment:

| group | files | supports |
|---|---|---|
| `ascan_*`, `galerkin_*`, `doob_*`, `probe_dk`, `gate_*` | ~20 | `VR-DOOB-001`: Galerkin-predicted a*, 9.4×/16.9× path-weight reduction, correctness gates |
| `bottleneck_*` | 2 | `VR-CLOSE-001`, the sampler-programme closure |
| `chunkcert_*` | 2 | `VR-CHUNK-001`, mult=4 certification |
| `snapgain_*`, `snapscan`, `coupsnap_*` | ~10 | `VR-SNAPSHOT-001` including the **unrecorded L=48 point** |
| `crossing_prod*` | 3 | **unrecorded** T=L certification + bootstrapped crossing |
| `d2scaling`, `xo_lagged` | 4 | D₂ scaling, lagged cross-correlation nulls |
| `ncladder*`, `ladder_L64*` | 5 | N_c ladders at L=32 and L=64 |
| `mcmc_*` | 8 | trajectory MCMC closure |
| `twisted_*`, `threearm_*`, `l64ref*`, `e2e`, `tscan*`, `blocker*`, `ccl*` | ~25 | twisted cloning, three-arm benchmark, path-IS reference |

Reproducible driver scripts exist in `analysis/var_reduction/` for the
bottleneck, chunk, snapshot, coupled-snapshot, d2, xo_lagged, nc_ladder,
crossing_prod, l64_reference, meanone_gate, controlled_sampler, galerkin_control,
suffix_mcmc and threearm items. **All untracked in git.**

Copying is preservation, not verification. No claim above changes status.

## D. Unindexed data discovered in Stage 2

`results/ruche_pull/` — 16,344 files, all post-2026-07-01, absent from HANDOFF:

| subdir | files | dates |
|---|---|---|
| `pps/` (boundary + smoke) | 6,725 | 2026-07-08 → 07-23 |
| `refine_smallz/` | 3,675 | 2026-07-27 → 07-29 |
| `caseA_guided/` | 1,394 | 2026-07-26 → 07-27 |
| `logs/` | 500 | 2026-07-07 → 07-23 |
| archives | `pps_all_20260725.tgz`, `pps_logs_20260725.tgz` | |

Plus `results/boundary_aggregate.csv`: 470 rows, **5634 realizations**,
L ∈ {64, 80, 96, 112, 128}, 14 ζ **including 1.0**, λ ∈ [0.067, 0.800],
nreal 6–12, fields CMI/CMI_se/B_L/B_L_se, 69 (L,ζ) cells with 1–7 λ points each.

**Metadata gaps in this dataset**: no T recorded in the CSV, no N_c, no burn-in,
no seed provenance, no generating-script reference, no date column. The npz/json
tree presumably carries them but there is no index. Determining whether this
campaign satisfies T ≥ 2L requires reading the per-realisation JSON.

The 1–7 λ points per cell means many cells are too thin for a crossing.
