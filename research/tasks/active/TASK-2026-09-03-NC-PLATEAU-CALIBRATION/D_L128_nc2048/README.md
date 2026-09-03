# D_L128_nc2048 — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign D)

CAMPAIGN D. L = 128, T = 128, lambda = 0.3032, N_c = 2048, R = 16. A SCREENING rung: R = 16 resolves a shift of the size the 512 -> 1024 step showed (-0.0602 +- 0.0234) and CANNOT certify convergence. That asymmetry is pre-registered in ../SUCCESS_CRITERIA.yaml.

| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |
|---:|---:|---:|---:|---:|---:|---|
| 2048 | 0.3032 | 6 | 1643 | 16 | 0 | — |

| | |
|---|---|
| zeta | 0.35 |
| T | 128 (T = L) |
| resampling | systematic |
| target R per cell | 16 |
| array tasks | 16 |
| seeds | 33280000–33280015, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | [33.592] ms per clone-window, measured on Ruche |
| slowest task | 31.40 h predicted, 43.96 h pessimistic |
| core-hours | 502.4 predicted, 703.3 pessimistic |
| elapsed at cap %64 | 31.40 h predicted, 43.96 h pessimistic, **queue wait excluded** |
| peak memory | 6330 MB modelled from direct measurement (requesting 9G) |
| partition | **cpu_long** `--time=72:00:00` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
