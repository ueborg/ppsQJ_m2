# armA1024 — TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA

**Recommended for the overnight campaign.**

ARM A rung 2: L=128, N_c=1024 at the central lambda. This is the wall-clock long pole of the whole campaign; submit it first.

| | |
|---|---|
| L | 128 |
| T | 128 |
| zeta | 0.35 |
| lambda | 0.3032 |
| N_c | 1024 |
| R per lambda | 32 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 1643 |
| array tasks | 32 |
| seeds | 30200000–30200031 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| slowest task | 10.06 h predicted, 14.08 h pessimistic |
| core-hours | 321.9 predicted, 450.6 pessimistic |
| peak memory | 2275 MB (requesting 5G) |
| partition | cpu_long (`--time=24:00:00`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, rsync back and analysis — is in `../RUCHE_RUNBOOK.md`. `run_preflight.sh`
must exit 0 first; it submits nothing and contains no scheduler call.

No agent submits this. You do.
