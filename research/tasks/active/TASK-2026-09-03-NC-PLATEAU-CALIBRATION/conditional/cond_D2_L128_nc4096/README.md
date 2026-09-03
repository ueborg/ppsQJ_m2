# cond_D2_L128_nc4096 — **CONDITIONAL, BLOCKED**

> **CAMPAIGN D ADJUDICATION**
>
> Recommend this arm if, on the L = 128 ladder completed by campaign D, EITHER |Delta_1024| = |I_2048 - I_1024| is resolved OUTSIDE the frozen material tolerance tau_I = 0.006 (i.e. the 95 % interval excludes [-tau_I, +tau_I]), OR no plateau criterion P1-P5 of ../SUCCESS_CRITERIA.yaml is satisfied at the top of that ladder. Do NOT recommend it because the observed Delta_1024 'looks large'.

CONDITIONAL. L = 128, T = 128, lambda = 0.3032, N_c = 4096, R = 8. One further rung on the hardest ladder in the programme. Read the wall-time line before releasing this: a single population is a multi-day job.

| | |
|---|---|
| L, T | 128, 128 |
| zeta | 0.35 |
| lambda | 0.3032 |
| N_c | [4096] |
| R | 8 |
| tasks | 8 |
| seeds | 33500000–33500007 |
| core-hours | 572 (801 pessimistic) |
| slowest task | 71.49 h (100.09 h pessimistic) |
| partition / time / mem | cpu_long / 168:00:00 / 26G |

`run_preflight.sh` exits 3 while blocked, and every array task exits 3 before touching the sampler unless `../GATE_RELEASED_cond_D2_L128_nc4096` exists. **No agent submits this, and neither should you until the gate above is adjudicated.**
