# lowlamL48 — TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION

LOWLAM-L48: the four new low-lambda points at L = 48, T = 48, N_c = 1024, R = 24. Completes the L = 48 curve to 17 points.

This arm computes **only the four new lambdas**. The thirteen already measured
at this exact cell by `TASK-2026-09-02-MOCK-PRODUCTION` are reused from
`../frozen_inputs/predecessor_nc1024_populations.csv` and are **not** recomputed.

| | |
|---|---|
| L | 48 |
| T | 48 |
| zeta | 0.35 |
| lambda (NEW) | 0.1932, 0.2032, 0.2132, 0.2232 |
| lambda (reused, not here) | 0.2332 … 0.3532, 13 points |
| grid | indices 0–3 of the frozen 17-point grid |
| N_c | 1024 |
| R per lambda | 24 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 146, 153, 161, 168 |
| array tasks | 96 |
| seeds | 32100000–32103023 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| cost model | `wall_s = 1.588743·n_steps + 286.09`, fitted to measured Ruche `wall_s` |
| slowest task | 9.6 min predicted, 13.5 min pessimistic |
| core-hours | 14.51 predicted, 20.32 pessimistic |
| elapsed at cap %64 | 19.2 min predicted, 26.9 min pessimistic (queue wait excluded) |
| peak memory | 430 MB (requesting 1G) |
| partition | **cpu_med** (`--time=00:45:00`) — see `../SCHEDULER_DECISION.md` |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, transfer back and analysis — is in `../RUCHE_RUNBOOK.md`.
`run_preflight.sh` must exit 0 first; it submits nothing and contains no
scheduler call.

No agent submits this. You do.
