# A_L64_nc4096 — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign A)

CAMPAIGN A. L = 64, T = 64, lambda = 0.3032, N_c = 4096, R = 48. New top rung of the deep central ladder: the rung that decides whether a high-N_c plateau is OBSERVED rather than inferred by eye.

| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |
|---:|---:|---:|---:|---:|---:|---|
| 4096 | 0.3032 | 6 | 408 | 48 | 0 | — |

| | |
|---|---|
| zeta | 0.35 |
| T | 64 (T = L) |
| resampling | systematic |
| target R per cell | 48 |
| array tasks | 48 |
| seeds | 33020000–33020047, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | [6.568] ms per clone-window, measured on Ruche |
| slowest task | 3.05 h predicted, 4.27 h pessimistic |
| core-hours | 146.3 predicted, 204.9 pessimistic |
| elapsed at cap %64 | 3.05 h predicted, 4.27 h pessimistic, **queue wait excluded** |
| peak memory | 2774 MB modelled from direct measurement (requesting 4G) |
| partition | **cpu_long** `--time=08:00:00` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
