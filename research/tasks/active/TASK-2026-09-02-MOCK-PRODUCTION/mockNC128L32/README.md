# mockNC128L32 — TASK-2026-09-02-MOCK-PRODUCTION

**Matched low-`N_c` companion arm.** An addition to the task brief's arm list; the reason it exists is in `../NC128_COMPANION_RATIONALE.md`, and dropping it costs the campaign nothing except brief sections 9C and 12.

MATCHED LOW-N_c COMPANION at L = 32. Same grid, same dtau_mult = 6, same estimator, same code, only N_c differs. See ../NC128_COMPANION_RATIONALE.md: the historical N_c = 128 corpus is dtau_mult = 12 and shares NO exactly compatible cell with this campaign, so without this arm brief sections 9C and 12 have no matched comparison to make.

| | |
|---|---|
| L | 32 |
| T | 32 |
| zeta | 0.35 |
| lambda | 0.2332, 0.2432, 0.2532, 0.2632, 0.2732, 0.2832, 0.2932, 0.3032, 0.3132, 0.3232, 0.3332, 0.3432, 0.3532 |
| lambdas | 13 of the frozen 13-point grid |
| N_c | 128 |
| R per lambda | 48 |
| dtau_mult | 6 (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | 78, 81, 84, 88, 91, 94, 97, 101, 104, 107, 111, 114, 117 |
| array tasks | 624 |
| seeds | 31400000–31412047 (fresh and disjoint — see `../SEED_LEDGER.md`) |
| rate | 1.890 ms/clone-window (`../COST_MODEL.md`) |
| slowest task | 0.01 h predicted, 0.01 h pessimistic |
| core-hours | 4.1 predicted, 5.7 pessimistic |
| elapsed at cap %64 | 0.07 h predicted, 0.10 h pessimistic |
| peak memory | 145 MB (requesting 1G) |
| partition | cpu_short (`--time=01:00:00`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, transfer back and analysis — is in `../RUCHE_RUNBOOK.md`.
`run_preflight.sh` must exit 0 first; it submits nothing and contains no
scheduler call.

No agent submits this. You do.
