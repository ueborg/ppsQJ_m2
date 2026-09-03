# B2_L32_nc512 — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign B2)

CAMPAIGN B2. L = 32, T = 32, the same frozen 7-point grid as campaign B, at N_c = 512, matched R = 48. Puts the low-L reference curve on the SAME grid at the SAME N_c as L = 64, so the locator test of section 4B is a FULLY MATCHED cross-L comparison and not a one-sided diagnostic. On a 3-lambda grid the frozen crossing protocol flags every interior crossing ENDPOINT_INDUCED by construction; on 7 points both have a guard point on each side. The three lambdas that already exist at N_c = 1024 are TOPPED UP, never recomputed.

| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 0.2182 | 6 | 73 | 48 | 0 | — |
| 512 | 0.2232 | 6 | 74 | 48 | 0 | — |
| 512 | 0.2282 | 6 | 76 | 48 | 0 | — |
| 512 | 0.2332 | 6 | 78 | 48 | 0 | — |
| 512 | 0.2382 | 6 | 79 | 48 | 0 | — |
| 512 | 0.2432 | 6 | 81 | 48 | 0 | — |
| 512 | 0.2482 | 6 | 83 | 48 | 0 | — |

| | |
|---|---|
| zeta | 0.35 |
| T | 32 (T = L) |
| resampling | systematic |
| target R per cell | 48 |
| array tasks | 336 |
| seeds | 33120000–33126047, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | [2.018] ms per clone-window, measured on Ruche |
| slowest task | 0.02 h predicted, 0.03 h pessimistic |
| core-hours | 7.5 predicted, 10.5 pessimistic |
| elapsed at cap %64 | 0.14 h predicted, 0.20 h pessimistic, **queue wait excluded** |
| peak memory | 302 MB modelled from direct measurement (requesting 1G) |
| partition | **cpu_med** `--time=01:00:00` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
