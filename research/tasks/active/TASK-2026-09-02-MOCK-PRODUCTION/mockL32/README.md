# mockL32 — TASK-2026-09-02-MOCK-PRODUCTION

**Main mock-production arm.**

MOCK-L32: the full 13-point CMI(lambda) scan at L = 32, T = 32, N_c = 1024. The cheapest of the three curves and the one with no historical counterpart at any N_c.

| | |
|---|---|
| L | 32 |
| T | 32 |
| zeta | 0.35 |
| lambda | 0.2332, 0.2432, 0.2532, 0.2632, 0.2732, 0.2832, 0.2932, 0.3032, 0.3132, 0.3232, 0.3332, 0.3432, 0.3532 |
| lambdas | 13 of the frozen 13-point grid |
| N_c | 1024 |
| R per lambda | 24 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 78, 81, 84, 88, 91, 94, 97, 101, 104, 107, 111, 114, 117 |
| array tasks | 312 |
| seeds | 31000000–31012023 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| rate | 1.400 ms/clone-window (`../COST_MODEL.md`) |
| slowest task | 0.05 h predicted, 0.07 h pessimistic |
| core-hours | 12.1 predicted, 17.0 pessimistic |
| elapsed at cap %64 | 0.22 h predicted, 0.30 h pessimistic |
| peak memory | 262 MB (requesting 1G) |
| partition | cpu_short (`--time=01:00:00`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, transfer back and analysis — is in `../RUCHE_RUNBOOK.md`.
`run_preflight.sh` must exit 0 first; it submits nothing and contains no
scheduler call.

No agent submits this. You do.
