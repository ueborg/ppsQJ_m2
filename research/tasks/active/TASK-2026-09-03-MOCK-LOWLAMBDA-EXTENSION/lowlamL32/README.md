# lowlamL32 — TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION

LOWLAM-L32: the four new low-lambda points at L = 32, T = 32, N_c = 1024, R = 24. Completes the L = 32 curve to 17 points.

This arm computes **only the four new lambdas**. The thirteen already measured
at this exact cell by `TASK-2026-09-02-MOCK-PRODUCTION` are reused from
`../frozen_inputs/predecessor_nc1024_populations.csv` and are **not** recomputed.

| | |
|---|---|
| L | 32 |
| T | 32 |
| zeta | 0.35 |
| lambda (NEW) | 0.1932, 0.2032, 0.2132, 0.2232 |
| lambda (reused, not here) | 0.2332 … 0.3532, 13 points |
| grid | indices 0–3 of the frozen 17-point grid |
| N_c | 1024 |
| R per lambda | 24 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 64, 68, 71, 74 |
| array tasks | 96 |
| seeds | 32000000–32003023 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| cost model | `wall_s = 0.815551·n_steps + 68.43`, fitted to measured Ruche `wall_s` |
| slowest task | 2.4 min predicted, 3.3 min pessimistic |
| core-hours | 3.57 predicted, 5.00 pessimistic |
| elapsed at cap %64 | 4.8 min predicted, 6.7 min pessimistic (queue wait excluded) |
| peak memory | 262 MB (requesting 1G) |
| partition | **cpu_med** (`--time=00:20:00`) — see `../SCHEDULER_DECISION.md` |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, transfer back and analysis — is in `../RUCHE_RUNBOOK.md`.
`run_preflight.sh` must exit 0 first; it submits nothing and contains no
scheduler call.

No agent submits this. You do.
