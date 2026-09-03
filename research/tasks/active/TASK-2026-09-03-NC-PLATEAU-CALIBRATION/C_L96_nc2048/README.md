# C_L96_nc2048 — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign C)

CAMPAIGN C. L = 96, T = 96, lambda = 0.3032, N_c = 2048, R = 24. Fills the L = 64 / L = 128 gap. The existing L = 96 ladder (N_c = 128, 256, 512) REJECTS a clean I = I_inf + B/N over its measured range; these rungs test whether it enters a simpler high-N regime. They do not assume it does.

| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |
|---:|---:|---:|---:|---:|---:|---|
| 2048 | 0.3032 | 6 | 922 | 24 | 0 | — |

| | |
|---|---|
| zeta | 0.35 |
| T | 96 (T = L) |
| resampling | systematic |
| target R per cell | 24 |
| array tasks | 24 |
| seeds | 33260000–33260023, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | [16.106] ms per clone-window, measured on Ruche |
| slowest task | 8.45 h predicted, 11.83 h pessimistic |
| core-hours | 202.7 predicted, 283.8 pessimistic |
| elapsed at cap %64 | 8.45 h predicted, 11.83 h pessimistic, **queue wait excluded** |
| peak memory | 2574 MB modelled from direct measurement (requesting 4G) |
| partition | **cpu_long** `--time=24:00:00` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
