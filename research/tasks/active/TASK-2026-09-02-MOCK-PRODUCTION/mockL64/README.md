# mockL64 — TASK-2026-09-02-MOCK-PRODUCTION

**Main mock-production arm.**

MOCK-L64: the 13-point scan at L = 64, T = 64, N_c = 1024, MINUS the three lambdas already measured at R = 96 by TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA ARM B. Ten new lambdas, not thirteen. This is the campaign's wall-clock long pole.

| | |
|---|---|
| L | 64 |
| T | 64 |
| zeta | 0.35 |
| lambda | 0.2332, 0.2432, 0.2532, 0.2632, 0.2732, 0.2832, 0.3232, 0.3332, 0.3432, 0.3532 |
| lambdas | 10 of the frozen 13-point grid |
| N_c | 1024 |
| R per lambda | 24 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 314, 327, 341, 354, 368, 381, 435, 448, 462, 475 |
| array tasks | 240 |
| seeds | 31200000–31212023 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| rate | 4.850 ms/clone-window (`../COST_MODEL.md`) |
| slowest task | 0.66 h predicted, 0.92 h pessimistic |
| core-hours | 129.3 predicted, 181.0 pessimistic |
| elapsed at cap %64 | 2.32 h predicted, 3.25 h pessimistic |
| peak memory | 665 MB (requesting 2G) |
| partition | cpu_med (`--time=03:00:00`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, transfer back and analysis — is in `../RUCHE_RUNBOOK.md`.
`run_preflight.sh` must exit 0 first; it submits nothing and contains no
scheduler call.

No agent submits this. You do.
