# cond_LOWZ_nc256 — **CONDITIONAL, BLOCKED**

> **OPTIONAL -- NOT PART OF THE zeta = 0.35 CALIBRATION**
>
> As cond_LOWZ_nc64. The pre-registered kill criterion needs BOTH population sizes: drift at zeta = 0.10 greater than or equal to drift at zeta = 0.35 kills the guided-residual mechanism and revives Born-rarity reasoning. Release both or neither.

OPTIONAL. L = 64, T = 64, zeta = 0.10, lambda = 0.3032, N_c = 256, R = 48.

| | |
|---|---|
| L, T | 64, 64 |
| zeta | 0.1 |
| lambda | 0.3032 |
| N_c | [256] |
| R | 48 |
| tasks | 48 |
| seeds | 33620000–33620047 |
| core-hours | 9 (13 pessimistic) |
| slowest task | 0.19 h (0.27 h pessimistic) |
| partition / time / mem | cpu_med / 01:00:00 / 1G |

`run_preflight.sh` exits 3 while blocked, and every array task exits 3 before touching the sampler unless `../GATE_RELEASED_cond_LOWZ_nc256` exists. **No agent submits this, and neither should you until the gate above is adjudicated.**
