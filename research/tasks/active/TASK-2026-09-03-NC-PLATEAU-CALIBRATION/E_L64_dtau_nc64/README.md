# E_L64_dtau_nc64 — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign E)

CAMPAIGN E. L = 64, T = 64, lambda = 0.3032, N_c = 64, dtau_mult in {3, 6, 12} giving K = 816 / 408 / 204, matched R = 48. The Feynman-Kac weight is exact at any window size, so the TARGET MEASURE IS EXACTLY UNCHANGED across the three sub-cells; only where selection is applied moves. dtau_mult is a discretisation control and never a physical parameter.

| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 0.3032 | 3 | 816 | 48 | 0 | — |
| 64 | 0.3032 | 6 | 408 | 48 | 0 | — |
| 64 | 0.3032 | 12 | 204 | 48 | 0 | — |

| | |
|---|---|
| zeta | 0.35 |
| T | 64 (T = L) |
| resampling | systematic |
| target R per cell | 48 |
| array tasks | 144 |
| seeds | 33300000–33302047, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | [7.5] ms per clone-window, measured on Ruche |
| slowest task | 0.11 h predicted, 0.15 h pessimistic |
| core-hours | 9.1 predicted, 12.8 pessimistic |
| elapsed at cap %64 | 0.33 h predicted, 0.46 h pessimistic, **queue wait excluded** |
| peak memory | 226 MB modelled from direct measurement (requesting 1G) |
| partition | **cpu_med** `--time=01:00:00` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
