# mockL48 — TASK-2026-09-02-MOCK-PRODUCTION

**Main mock-production arm.**

MOCK-L48: the full 13-point CMI(lambda) scan at L = 48, T = 48, N_c = 1024.

| | |
|---|---|
| L | 48 |
| T | 48 |
| zeta | 0.35 |
| lambda | 0.2332, 0.2432, 0.2532, 0.2632, 0.2732, 0.2832, 0.2932, 0.3032, 0.3132, 0.3232, 0.3332, 0.3432, 0.3532 |
| lambdas | 13 of the frozen 13-point grid |
| N_c | 1024 |
| R per lambda | 24 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 176, 183, 191, 198, 206, 213, 221, 229, 236, 244, 251, 259, 266 |
| array tasks | 312 |
| seeds | 31100000–31112023 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| rate | 3.000 ms/clone-window (`../COST_MODEL.md`) |
| slowest task | 0.23 h predicted, 0.32 h pessimistic |
| core-hours | 58.8 predicted, 82.4 pessimistic |
| elapsed at cap %64 | 1.06 h predicted, 1.48 h pessimistic |
| peak memory | 430 MB (requesting 1G) |
| partition | cpu_med (`--time=02:00:00`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, transfer back and analysis — is in `../RUCHE_RUNBOOK.md`.
`run_preflight.sh` must exit 0 first; it submits nothing and contains no
scheduler call.

No agent submits this. You do.
