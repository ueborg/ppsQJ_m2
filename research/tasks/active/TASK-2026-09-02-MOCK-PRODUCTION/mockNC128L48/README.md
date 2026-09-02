# mockNC128L48 — TASK-2026-09-02-MOCK-PRODUCTION

**Matched low-`N_c` companion arm.** An addition to the task brief's arm list; the reason it exists is in `../NC128_COMPANION_RATIONALE.md`, and dropping it costs the campaign nothing except brief sections 9C and 12.

MATCHED LOW-N_c COMPANION at L = 48.

| | |
|---|---|
| L | 48 |
| T | 48 |
| zeta | 0.35 |
| lambda | 0.2332, 0.2432, 0.2532, 0.2632, 0.2732, 0.2832, 0.2932, 0.3032, 0.3132, 0.3232, 0.3332, 0.3432, 0.3532 |
| lambdas | 13 of the frozen 13-point grid |
| N_c | 128 |
| R per lambda | 48 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 176, 183, 191, 198, 206, 213, 221, 229, 236, 244, 251, 259, 266 |
| array tasks | 624 |
| seeds | 31500000–31512047 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| rate | 4.050 ms/clone-window (`../COST_MODEL.md`) |
| slowest task | 0.04 h predicted, 0.05 h pessimistic |
| core-hours | 19.9 predicted, 27.8 pessimistic |
| elapsed at cap %64 | 0.36 h predicted, 0.50 h pessimistic |
| peak memory | 166 MB (requesting 1G) |
| partition | cpu_short (`--time=01:00:00`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, transfer back and analysis — is in `../RUCHE_RUNBOOK.md`.
`run_preflight.sh` must exit 0 first; it submits nothing and contains no
scheduler call.

No agent submits this. You do.
