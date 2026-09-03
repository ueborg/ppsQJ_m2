# cond_M128_nc2048 — **CONDITIONAL, BLOCKED**

> **CAMPAIGN D ADJUDICATION -- STRONGLY GATED**
>
> Release ONLY if campaign D's N_c = 2048 rung PASSES the frozen adequacy screen at L = 128. If it fails, the conditional N_c = 4096 central rung comes first and this arm stays blocked. An adequate N_c must be identified BEFORE a 9-point scan at this L is run at all.

CONDITIONAL, STAGE 1. L = 128 mock-production scan over the frozen 9-point grid at N_c = 2048, R = 8.

| | |
|---|---|
| L, T | 128, 128 |
| zeta | 0.35 |
| lambda | 0.2032, 0.2182, 0.2232, 0.2282, 0.2332, 0.2382, 0.2432, 0.2482, 0.2632 |
| N_c | [2048] |
| R | 8 |
| tasks | 72 |
| seeds | 33560000–33568007 |
| core-hours | 1739 (2435 pessimistic) |
| slowest task | 27.27 h (38.18 h pessimistic) |
| partition / time / mem | cpu_long / 72:00:00 / 9G |

`run_preflight.sh` exits 3 while blocked, and every array task exits 3 before touching the sampler unless `../GATE_RELEASED_cond_M128_nc2048` exists. **No agent submits this, and neither should you until the gate above is adjudicated.**
