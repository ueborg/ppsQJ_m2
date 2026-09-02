# armA512 — TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA

**Recommended for the overnight campaign.**

ARM A rung 1: L=128, N_c=512 at the central lambda. It also supplies ARM C's central stencil point, which is therefore NOT recomputed in armC.

| | |
|---|---|
| L | 128 |
| T | 128 |
| zeta | 0.35 |
| lambda | 0.3032 |
| N_c | 512 |
| R per lambda | 48 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 1643 |
| array tasks | 48 |
| seeds | 30100000–30100047 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| slowest task | 5.03 h predicted, 7.04 h pessimistic |
| core-hours | 241.4 predicted, 338.0 pessimistic |
| peak memory | 1202 MB (requesting 3G) |
| partition | cpu_long (`--time=12:00:00`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, rsync back and analysis — is in `../RUCHE_RUNBOOK.md`. `run_preflight.sh`
must exit 0 first; it submits nothing and contains no scheduler call.

No agent submits this. You do.
