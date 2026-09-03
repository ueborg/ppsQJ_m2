# B_L64_cross_nc2048 — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign B)

CAMPAIGN B. L = 64, T = 64, the frozen 7-point transition-region grid 0.2182-0.2482 at N_c = 2048, matched R = 48. Tests whether finite-N_c distorts the SHAPE of CMI(lambda) where the low-L locator sits, not just its level.

| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |
|---:|---:|---:|---:|---:|---:|---|
| 2048 | 0.2182 | 6 | 294 | 48 | 0 | — |
| 2048 | 0.2232 | 6 | 300 | 48 | 0 | — |
| 2048 | 0.2282 | 6 | 307 | 48 | 0 | — |
| 2048 | 0.2332 | 6 | 314 | 48 | 0 | — |
| 2048 | 0.2382 | 6 | 321 | 48 | 0 | — |
| 2048 | 0.2432 | 6 | 327 | 48 | 0 | — |
| 2048 | 0.2482 | 6 | 334 | 48 | 0 | — |

| | |
|---|---|
| zeta | 0.35 |
| T | 64 (T = L) |
| resampling | systematic |
| target R per cell | 48 |
| array tasks | 336 |
| seeds | 33100000–33106047, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | [5.769] ms per clone-window, measured on Ruche |
| slowest task | 1.10 h predicted, 1.53 h pessimistic |
| core-hours | 346.1 predicted, 484.5 pessimistic |
| elapsed at cap %64 | 6.58 h predicted, 9.21 h pessimistic, **queue wait excluded** |
| peak memory | 1705 MB modelled from direct measurement (requesting 3G) |
| partition | **cpu_med** `--time=03:00:00` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
