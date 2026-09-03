# A_L64_nc8192 — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign A)

CAMPAIGN A. L = 64, T = 64, lambda = 0.3032, N_c = 8192, R = 48. New top rung of the deep central ladder: the rung that decides whether a high-N_c plateau is OBSERVED rather than inferred by eye.

| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |
|---:|---:|---:|---:|---:|---:|---|
| 8192 | 0.3032 | 6 | 408 | 48 | 0 | — |

| | |
|---|---|
| zeta | 0.35 |
| T | 64 (T = L) |
| resampling | systematic |
| target R per cell | 48 |
| array tasks | 48 |
| seeds | 33040000–33040047, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | [7.477] ms per clone-window, measured on Ruche |
| slowest task | 6.94 h predicted, 9.72 h pessimistic |
| core-hours | 333.2 predicted, 466.5 pessimistic |
| elapsed at cap %64 | 6.94 h predicted, 9.72 h pessimistic, **queue wait excluded** |
| peak memory | 4647 MB modelled from direct measurement (requesting 7G) |
| partition | **cpu_long** `--time=18:00:00` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
