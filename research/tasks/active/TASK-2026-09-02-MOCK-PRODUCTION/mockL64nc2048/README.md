# mockL64nc2048 — TASK-2026-09-02-MOCK-PRODUCTION

**Main mock-production arm.**

MOCK-L64-NC2048: the shape check. Three lambdas at N_c = 2048 against ARM B's N_c = 1024 at exactly the same three lambdas, to measure whether the finite-population correction Delta_N(lambda) = I_2048 - I_1024 is a common shift, lambda-dependent, or unresolved. No 1/N_c law is fitted.

| | |
|---|---|
| L | 64 |
| T | 64 |
| zeta | 0.35 |
| lambda | 0.2932, 0.3032, 0.3132 |
| lambdas | 3 of the frozen 13-point grid |
| N_c | 2048 |
| R per lambda | 24 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 395, 408, 421 |
| array tasks | 72 |
| seeds | 31306000–31308023 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| rate | 5.820 ms/clone-window (`../COST_MODEL.md`) |
| slowest task | 1.39 h predicted, 1.95 h pessimistic |
| core-hours | 97.3 predicted, 136.2 pessimistic |
| elapsed at cap %64 | 1.75 h predicted, 2.45 h pessimistic |
| peak memory | 1202 MB (requesting 3G) |
| partition | cpu_med (`--time=04:00:00`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, transfer back and analysis — is in `../RUCHE_RUNBOOK.md`.
`run_preflight.sh` must exit 0 first; it submits nothing and contains no
scheduler call.

No agent submits this. You do.
