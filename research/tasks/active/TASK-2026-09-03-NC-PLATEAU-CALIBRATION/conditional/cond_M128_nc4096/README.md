# cond_M128_nc4096 — **CONDITIONAL, BLOCKED**

> **CAMPAIGN D AND cond_D2_L128_nc4096 ADJUDICATION**
>
> Release ONLY if N_c = 2048 FAILS the adequacy screen at L = 128 and the conditional N_c = 4096 central rung then PASSES it. Read the core-hour line before releasing: this is the most expensive object in the whole campaign by a wide margin and it should not be the first way the programme learns that L = 128 is unaffordable.

CONDITIONAL, STAGE 1. L = 128 mock-production scan over the frozen 9-point grid at N_c = 4096, R = 8.

| | |
|---|---|
| L, T | 128, 128 |
| zeta | 0.35 |
| lambda | 0.2032, 0.2182, 0.2232, 0.2282, 0.2332, 0.2382, 0.2432, 0.2482, 0.2632 |
| N_c | [4096] |
| R | 8 |
| tasks | 72 |
| seeds | 33580000–33588007 |
| core-hours | 3960 (5544 pessimistic) |
| slowest task | 62.09 h (86.93 h pessimistic) |
| partition / time / mem | cpu_long / 144:00:00 / 26G |

`run_preflight.sh` exits 3 while blocked, and every array task exits 3 before touching the sampler unless `../GATE_RELEASED_cond_M128_nc4096` exists. **No agent submits this, and neither should you until the gate above is adjudicated.**
