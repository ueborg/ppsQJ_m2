# lowlamL64 — TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION

LOWLAM-L64: the four new low-lambda points at L = 64, T = 64, N_c = 1024, R = 24. Completes the L = 64 curve to 17 points and is this campaign's wall-clock long pole.

This arm computes **only the four new lambdas**. The thirteen already measured
at this exact cell by `TASK-2026-09-02-MOCK-PRODUCTION` are reused from
`../frozen_inputs/predecessor_nc1024_populations.csv` and are **not** recomputed.

| | |
|---|---|
| L | 64 |
| T | 64 |
| zeta | 0.35 |
| lambda (NEW) | 0.1932, 0.2032, 0.2132, 0.2232 |
| lambda (reused, not here) | 0.2332 … 0.3532, 13 points |
| grid | indices 0–3 of the frozen 17-point grid |
| N_c | 1024 |
| R per lambda | 24 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 260, 274, 287, 300 |
| array tasks | 96 |
| seeds | 32200000–32203023 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| cost model | `wall_s = 2.723572·n_steps + 850.23`, fitted to measured Ruche `wall_s` |
| slowest task | 28.3 min predicted, 39.6 min pessimistic |
| core-hours | 43.23 predicted, 60.52 pessimistic |
| elapsed at cap %64 | 56.6 min predicted, 79.2 min pessimistic (queue wait excluded) |
| peak memory | 665 MB (requesting 2G) |
| partition | **cpu_med** (`--time=02:00:00`) — see `../SCHEDULER_DECISION.md` |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, transfer back and analysis — is in `../RUCHE_RUNBOOK.md`.
`run_preflight.sh` must exit 0 first; it submits nothing and contains no
scheduler call.

No agent submits this. You do.
