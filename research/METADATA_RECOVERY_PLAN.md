---
lifecycle: active
authoritative_for: metadata recovery procedure for simulation evidence
last_reviewed: 2026-08-10
---

# Metadata recovery plan

Scope: what the evidence schema requires for simulation evidence, where each
field can be recovered from, and the smallest safe catalogue script.

**Nothing in this plan modifies data.** All operations are read-only.

## 1. What the schema requires

For `raw_simulation_dataset` and `aggregated_dataset`, per
`audit/2026-08-10/EVIDENCE_SCHEMA.yaml`:

| field | why it is required |
|---|---|
| `L`, `zeta`, `lambda` | parameter coverage |
| `T` and **`T_over_L`** | the binding unrecorded quantity for every ν claim |
| `N_c`, `n_real`, `burn_in` | statistics adequacy |
| `seeds` | independence and reproducibility |
| `git_commit`, code version | which estimator generated it |
| SLURM job/config provenance | links a cell to its authorisation and logs |
| `observable_fields` | which observable IDs the data can support |
| `metadata_gaps` | explicit, may be empty, may not be omitted |

Missing values are recorded as `unknown_recoverable` when a recovery route
exists and `unrecoverable` only when it has been searched for and does not.

## 2. What a sample inspection established

Two files read on 2026-08-10, read-only.

**Per-realisation JSON**, `results/ruche_pull/pps/boundary/L80_z0.600_lam0.2324/real000.json`:

```json
{"L": 80, "lambda": 0.2324, "zeta": 0.6, "real": 0, "alpha": 0.2324,
 "w": 0.7676, "T": 80.0, "N_c": 128, "dtau": 0.3268, "theta_hat": -4.7425,
 "S_mean": 5.4457, "S_std": 0.1185, "eff_sample_size": 125.76,
 "n_T_mean": 0.0836, "n_distinct_ancestors": 1, "CMI_mean": 2.0730,
 "CMI_std": 0.2922, "B_L_mean": 10.8659, "B_L_std": 2.2694,
 "S_AB_mean": 5.2037, "wall_s": 883.31, "status": "ok"}
```

**Cut A summary JSON**, `results/ruche_pull/caseA_guided/summary_caseA_00796.json`:
carries `task_id`, `L`, `lam`, `zeta`, `T`, `N_c`, `n_real`, `n_workers`,
observables, `min_ess_frac_mean`, `n_collapses`, `wall_time`, `status`.

**Immediate scientific consequence.** The sampled L=80 cell has `T = 80.0`, so
**T/L = 1.0**. If that holds across the July campaign, then no dataset in the
project satisfies the T ≥ 2L condition, and the July campaign does not relieve
the ν data-adequacy problem. This is a one-line check that materially affects
`CB-NU-001` and must be run before any ν work.

## 3. Recovery routes, per field

| field | route | confidence |
|---|---|---|
| `L`, `lambda`, `zeta` | JSON keys, and redundantly the directory name `L{L}_z{zeta}_lam{lambda}` | high |
| `T`, `N_c` | JSON keys | high |
| `n_real` | count of `real*.json` per cell, and `n_real` in Cut A summaries | high |
| `alpha`, `w`, `dtau` | JSON keys | high |
| observables | JSON keys | high |
| `wall_s` | JSON key, gives the cost model for free | high |
| `burn_in` | **not in JSON.** Try worker defaults in `pps_qj/parallel/worker_*.py` and submit scripts | medium |
| `seeds` | **not in JSON.** Try `grid_pps.py` seed-offset config and submit-script env | medium |
| `git_commit` | **not in JSON.** Try `results/ruche_pull/logs/*.err|.out` headers, and correlate run dates against `git log` | low to medium |
| SLURM job id | log filenames encode it, e.g. `pps_1318092_45.err`. Mapping job-id to cell requires parsing log bodies | medium |
| code version | inferred from run date plus git log | low |

The redundancy between directory names and JSON keys is a genuine asset: it
gives a free consistency check on every cell.

## 4. The catalogue script (specified, not yet written)

`research/tools/catalogue_runs.py`, read-only, single pass.

- Walk a given results root, read every `real*.json` and `summary_*.json`.
- For each cell: parse the directory name, compare against the JSON keys, record
  any mismatch as a data-integrity finding.
- Emit one row per cell to `research/runs/_catalogue/<root>.csv` with
  L, zeta, lambda, T, T/L, N_c, n_real, dtau, mean wall_s, observable fields
  present, and a `consistent` flag.
- Emit a per-root summary of coverage and of which schema fields remain
  `unknown_recoverable`.
- Never open a file for writing under `results/`.

Cost estimate: roughly 10,400 small JSON reads. Single-threaded, a few minutes.
This is well inside the T0 read-only tier and needs no gate.

## 5. Order of work

1. **T/L check first.** One pass over the boundary tree extracting only `L` and
   `T`. Answers a live data-adequacy question in minutes.
2. Full catalogue as above.
3. Update `EV-DATA-RUCHEPULL-001` and `EV-DATA-BOUNDARYCSV-001` from the
   catalogue, downgrading `unknown_recoverable` to concrete values.
4. Only then attempt burn-in, seeds and git commit, which need source and log
   archaeology rather than data reading.
5. Record whatever remains genuinely lost as `unrecoverable`, with the search
   that was performed.

## 6. What is explicitly deferred

No 16,000-file reconstruction is performed in Phase 4. No `.tgz` archives are
unpacked. No aggregate is rebuilt. The catalogue is a read-only index, and
rebuilding aggregates is a separate task that would need its own evidence entry.
