# A_L64_nc2048_topup — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign A)

CAMPAIGN A top-up. L = 64, lambda = 0.3032, N_c = 2048: 24 exact-compatible populations already exist, and 24 more are added here to bring the rung to the R = 48 the frozen tau_I demands. The existing 24 are NOT recomputed.

| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |
|---:|---:|---:|---:|---:|---:|---|
| 2048 | 0.3032 | 6 | 408 | 24 | 24 | TASK-2026-09-02-MOCK-PRODUCTION/mockL64nc2048 |

| | |
|---|---|
| zeta | 0.35 |
| T | 64 (T = L) |
| resampling | systematic |
| target R per cell | 48 |
| array tasks | 24 |
| seeds | 33000024–33000047, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | [5.769] ms per clone-window, measured on Ruche |
| slowest task | 1.34 h predicted, 1.87 h pessimistic |
| core-hours | 32.1 predicted, 45.0 pessimistic |
| elapsed at cap %64 | 1.34 h predicted, 1.87 h pessimistic, **queue wait excluded** |
| peak memory | 1708 MB modelled from direct measurement (requesting 3G) |
| partition | **cpu_med** `--time=03:00:00` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
