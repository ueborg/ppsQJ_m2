# cond_M96_nc2048 — **CONDITIONAL, BLOCKED**

> **CAMPAIGN C ADJUDICATION -- AND ONLY ONE OF THE TWO M96 ARMS**
>
> Release ONLY if campaign C identifies N_c = 2048 as the smallest N_c meeting the frozen production adequacy criterion at L = 96, or if it identifies none and the researcher accepts a scan at the largest calibrated rung. Never together with cond_M96_nc1024.

CONDITIONAL, STAGE 1. L = 96 mock-production scan over the frozen 9-point grid at N_c = 2048, R = 12.

| | |
|---|---|
| L, T | 96, 96 |
| zeta | 0.35 |
| lambda | 0.2032, 0.2182, 0.2232, 0.2282, 0.2332, 0.2382, 0.2432, 0.2482, 0.2632 |
| N_c | [2048] |
| R | 12 |
| tasks | 108 |
| seeds | 33540000–33548011 |
| core-hours | 702 (983 pessimistic) |
| slowest task | 7.34 h (10.27 h pessimistic) |
| partition / time / mem | cpu_long / 18:00:00 / 4G |

`run_preflight.sh` exits 3 while blocked, and every array task exits 3 before touching the sampler unless `../GATE_RELEASED_cond_M96_nc2048` exists. **No agent submits this, and neither should you until the gate above is adjudicated.**
