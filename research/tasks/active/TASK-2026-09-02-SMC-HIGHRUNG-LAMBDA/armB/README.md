# armB — TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA

**Recommended for the overnight campaign.**

ARM B: cheap-L, high-population three-point lambda stencil. Does a large enough population make CMI(lambda) a statistically coherent local curve?

| | |
|---|---|
| L | 64 |
| T | 64 |
| zeta | 0.35 |
| lambda | 0.2932, 0.3032, 0.3132 |
| N_c | 1024 |
| R per lambda | 96 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 395, 408, 421 |
| array tasks | 288 |
| seeds | 30300000–30302095 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| slowest task | 0.60 h predicted, 0.84 h pessimistic |
| core-hours | 167.1 predicted, 234.0 pessimistic |
| peak memory | 665 MB (requesting 2G) |
| partition | cpu_med (`--time=03:00:00`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, rsync back and analysis — is in `../RUCHE_RUNBOOK.md`. `run_preflight.sh`
must exit 0 first; it submits nothing and contains no scheduler call.

No agent submits this. You do.
