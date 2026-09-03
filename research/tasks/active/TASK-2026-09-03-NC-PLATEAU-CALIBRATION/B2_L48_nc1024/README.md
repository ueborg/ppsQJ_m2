# B2_L48_nc1024 — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign B2)

CAMPAIGN B2. L = 48, T = 48, the same frozen 7-point grid as campaign B, at N_c = 1024, matched R = 48. Puts the low-L reference curve on the SAME grid at the SAME N_c as L = 64, so the locator test of section 4B is a FULLY MATCHED cross-L comparison and not a one-sided diagnostic. On a 3-lambda grid the frozen crossing protocol flags every interior crossing ENDPOINT_INDUCED by construction; on 7 points both have a guard point on each side. The three lambdas that already exist at N_c = 1024 are TOPPED UP, never recomputed.

| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |
|---:|---:|---:|---:|---:|---:|---|
| 1024 | 0.2182 | 6 | 165 | 48 | 0 | — |
| 1024 | 0.2232 | 6 | 168 | 24 | 24 | TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION/lowlamL48 |
| 1024 | 0.2282 | 6 | 172 | 48 | 0 | — |
| 1024 | 0.2332 | 6 | 176 | 24 | 24 | TASK-2026-09-02-MOCK-PRODUCTION/mockL48 |
| 1024 | 0.2382 | 6 | 180 | 48 | 0 | — |
| 1024 | 0.2432 | 6 | 183 | 24 | 24 | TASK-2026-09-02-MOCK-PRODUCTION/mockL48 |
| 1024 | 0.2482 | 6 | 187 | 48 | 0 | — |

| | |
|---|---|
| zeta | 0.35 |
| T | 48 (T = L) |
| resampling | systematic |
| target R per cell | 48 |
| array tasks | 264 |
| seeds | 33200000–33206047, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | [3.457] ms per clone-window, measured on Ruche |
| slowest task | 0.18 h predicted, 0.26 h pessimistic |
| core-hours | 45.7 predicted, 63.9 pessimistic |
| elapsed at cap %64 | 0.92 h predicted, 1.29 h pessimistic, **queue wait excluded** |
| peak memory | 833 MB modelled from direct measurement (requesting 2G) |
| partition | **cpu_med** `--time=01:00:00` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
