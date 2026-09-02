# armC — TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA

**Recommended for the overnight campaign.**

ARM C: the same stencil at L=128, NEIGHBOURING lambdas only. The central point is armA512's; it is not duplicated here.

| | |
|---|---|
| L | 128 |
| T | 128 |
| zeta | 0.35 |
| lambda | 0.2932, 0.3132 |
| N_c | 512 |
| R per lambda | 48 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 1589, 1698 |
| array tasks | 96 |
| seeds | 30400000–30401047 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| slowest task | 5.20 h predicted, 7.28 h pessimistic |
| core-hours | 482.9 predicted, 676.1 pessimistic |
| peak memory | 1202 MB (requesting 3G) |
| partition | cpu_long (`--time=12:00:00`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, rsync back and analysis — is in `../RUCHE_RUNBOOK.md`. `run_preflight.sh`
must exit 0 first; it submits nothing and contains no scheduler call.

No agent submits this. You do.
