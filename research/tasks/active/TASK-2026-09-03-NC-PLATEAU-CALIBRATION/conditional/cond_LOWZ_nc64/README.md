# cond_LOWZ_nc64 — **CONDITIONAL, BLOCKED**

> **OPTIONAL -- NOT PART OF THE zeta = 0.35 CALIBRATION**
>
> Design 2 of TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING. It is deliberately NOT in the immediate group: the programme wants the zeta = 0.35 calibration understood before it spends anything on a second zeta. Release only as an explicit decision to buy that one test now.

OPTIONAL. L = 64, T = 64, zeta = 0.10, lambda = 0.3032, N_c = 64, R = 48. 'Matched lambda' is read as THE SAME lambda, not the same offset from a putative lambda_c: matching on a critical-law offset would import the law under test.

| | |
|---|---|
| L, T | 64, 64 |
| zeta | 0.1 |
| lambda | 0.3032 |
| N_c | [64] |
| R | 48 |
| tasks | 48 |
| seeds | 33600000–33600047 |
| core-hours | 3 (4 pessimistic) |
| slowest task | 0.05 h (0.08 h pessimistic) |
| partition / time / mem | cpu_med / 01:00:00 / 1G |

`run_preflight.sh` exits 3 while blocked, and every array task exits 3 before touching the sampler unless `../GATE_RELEASED_cond_LOWZ_nc64` exists. **No agent submits this, and neither should you until the gate above is adjudicated.**
