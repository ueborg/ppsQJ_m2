# B2_L48_nc512 — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign B2)

CAMPAIGN B2. L = 48, T = 48, the same frozen 7-point grid as campaign B, at N_c = 512, matched R = 48. Puts the low-L reference curve on the SAME grid at the SAME N_c as L = 64, so the locator test of section 4B is a FULLY MATCHED cross-L comparison and not a one-sided diagnostic. On a 3-lambda grid the frozen crossing protocol flags every interior crossing ENDPOINT_INDUCED by construction; on 7 points both have a guard point on each side. The three lambdas that already exist at N_c = 1024 are TOPPED UP, never recomputed.

| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 0.2182 | 6 | 165 | 48 | 0 | — |
| 512 | 0.2232 | 6 | 168 | 48 | 0 | — |
| 512 | 0.2282 | 6 | 172 | 48 | 0 | — |
| 512 | 0.2332 | 6 | 176 | 48 | 0 | — |
| 512 | 0.2382 | 6 | 180 | 48 | 0 | — |
| 512 | 0.2432 | 6 | 183 | 48 | 0 | — |
| 512 | 0.2482 | 6 | 187 | 48 | 0 | — |

| | |
|---|---|
| zeta | 0.35 |
| T | 48 (T = L) |
| resampling | systematic |
| target R per cell | 48 |
| array tasks | 336 |
| seeds | 33180000–33186047, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | [3.63] ms per clone-window, measured on Ruche |
| slowest task | 0.10 h predicted, 0.14 h pessimistic |
| core-hours | 30.5 predicted, 42.7 pessimistic |
| elapsed at cap %64 | 0.58 h predicted, 0.81 h pessimistic, **queue wait excluded** |
| peak memory | 491 MB modelled from direct measurement (requesting 1G) |
| partition | **cpu_med** `--time=01:00:00` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
