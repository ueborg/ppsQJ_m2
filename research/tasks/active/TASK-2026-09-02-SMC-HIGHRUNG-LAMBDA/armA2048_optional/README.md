# armA2048_optional — TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA

**Optional extension. NOT recommended tonight — see `../NC2048_AUDIT.md`.**

OPTIONAL EXTENSION, NOT RECOMMENDED TONIGHT. See NC2048_AUDIT.md: a single task is ~20 h predicted (~28 h pessimistic) and cannot finish overnight, and whether it is worth running at all is exactly what Delta_512->1024 answers by morning.

| | |
|---|---|
| L | 128 |
| T | 128 |
| zeta | 0.35 |
| lambda | 0.3032 |
| N_c | 2048 |
| R per lambda | 16 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 1643 |
| array tasks | 16 |
| seeds | 30500000–30500015 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| slowest task | 20.12 h predicted, 28.16 h pessimistic |
| core-hours | 321.9 predicted, 450.6 pessimistic |
| peak memory | 4423 MB (requesting 9G) |
| partition | cpu_long (`--time=48:00:00`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, rsync back and analysis — is in `../RUCHE_RUNBOOK.md`. `run_preflight.sh`
must exit 0 first; it submits nothing and contains no scheduler call.

No agent submits this. You do.
