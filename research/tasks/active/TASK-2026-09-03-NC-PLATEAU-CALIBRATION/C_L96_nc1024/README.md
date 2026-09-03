# C_L96_nc1024 — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign C)

CAMPAIGN C. L = 96, T = 96, lambda = 0.3032, N_c = 1024, R = 24. Fills the L = 64 / L = 128 gap. The existing L = 96 ladder (N_c = 128, 256, 512) REJECTS a clean I = I_inf + B/N over its measured range; these rungs test whether it enters a simpler high-N regime. They do not assume it does.

| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |
|---:|---:|---:|---:|---:|---:|---|
| 1024 | 0.3032 | 6 | 922 | 24 | 0 | — |

| | |
|---|---|
| zeta | 0.35 |
| T | 96 (T = L) |
| resampling | systematic |
| target R per cell | 24 |
| array tasks | 24 |
| seeds | 33240000–33240023, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | [14.147] ms per clone-window, measured on Ruche |
| slowest task | 3.71 h predicted, 5.19 h pessimistic |
| core-hours | 89.0 predicted, 124.7 pessimistic |
| elapsed at cap %64 | 3.71 h predicted, 5.19 h pessimistic, **queue wait excluded** |
| peak memory | 2155 MB modelled from direct measurement (requesting 3G) |
| partition | **cpu_long** `--time=12:00:00` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
