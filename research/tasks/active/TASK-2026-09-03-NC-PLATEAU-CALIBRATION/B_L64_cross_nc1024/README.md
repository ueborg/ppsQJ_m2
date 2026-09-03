# B_L64_cross_nc1024 — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign B)

CAMPAIGN B. L = 64, T = 64, the frozen 7-point transition-region grid 0.2182-0.2482 at N_c = 1024, matched R = 48. Tests whether finite-N_c distorts the SHAPE of CMI(lambda) where the low-L locator sits, not just its level.

| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |
|---:|---:|---:|---:|---:|---:|---|
| 1024 | 0.2182 | 6 | 294 | 48 | 0 | — |
| 1024 | 0.2232 | 6 | 300 | 24 | 24 | TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION/lowlamL64 |
| 1024 | 0.2282 | 6 | 307 | 48 | 0 | — |
| 1024 | 0.2332 | 6 | 314 | 24 | 24 | TASK-2026-09-02-MOCK-PRODUCTION/mockL64 |
| 1024 | 0.2382 | 6 | 321 | 48 | 0 | — |
| 1024 | 0.2432 | 6 | 327 | 24 | 24 | TASK-2026-09-02-MOCK-PRODUCTION/mockL64 |
| 1024 | 0.2482 | 6 | 334 | 48 | 0 | — |

| | |
|---|---|
| zeta | 0.35 |
| T | 64 (T = L) |
| resampling | systematic |
| target R per cell | 48 |
| array tasks | 264 |
| seeds | 33080000–33086047, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | [5.769] ms per clone-window, measured on Ruche |
| slowest task | 0.55 h predicted, 0.77 h pessimistic |
| core-hours | 136.0 predicted, 190.4 pessimistic |
| elapsed at cap %64 | 2.74 h predicted, 3.84 h pessimistic, **queue wait excluded** |
| peak memory | 1363 MB modelled from direct measurement (requesting 2G) |
| partition | **cpu_med** `--time=02:00:00` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
