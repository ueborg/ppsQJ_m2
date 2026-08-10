# PROPOSAL 2026-08-10-C — T/L recovery and the status of "T >= 2L is necessary"

Status: **PROPOSED. Not applied.**
Executed: `research/tools/catalogue_runs.py results/ruche_pull` (read-only, T0).
Output: `research/runs/_catalogue/ruche_pull_catalogue.csv`, 15139 rows.

## 1. Field recoverability, from per-realisation JSON

| field | recoverable | note |
|---|---|---|
| L | 15139 / 15139 (100%) | also redundantly in the directory name |
| T, zeta, N_c | 14339 (94.7%) | absent only from summary-type files |
| lambda, dtau, alpha, w, real | 13642 (90.1%) | |
| seed | 800 (5.3%) | **mostly missing** |
| n_real, task_id | 697 (4.6%) | Cut A summary files only |
| **burn_in** | **0 (0%)** | absent from every file |
| **git_commit** | **0 (0%)** | absent from every file |
| **job_id / array id** | **0 (0%)** | absent from every file; recoverable only by parsing `logs/*.err` filenames and bodies |

Directory-name versus JSON `L` mismatches: **0 / 15139.** Internal integrity is
good.

Reclassification: `burn_in`, `git_commit` and `job_id` move from
`unknown_recoverable` to **`unrecoverable` from the data**, pending a separate
attempt at worker defaults and SLURM logs. `seeds` stays `unknown_recoverable`.

## 2. T/L by campaign — the headline

| campaign | L | n | T/L | T >= 2L? |
|---|---|---|---|---|
| caseA_guided | 32, 48, 64 | 547 | **2.000** | **YES** |
| caseA_guided | 96 | 129 | 1.333 | no |
| caseA_guided | 128 | 21 | 1.000 | no |
| pps (boundary) | 64, 80, 96, 112, 128 | 5923 | **1.000** | no |
| refine | 64, 80, 96, 112, 128 | 4044 | **1.000** | no |
| refine_smallz | 96, 128, 160, **192** | 3675 | **1.000** | no |

Three findings:

**(a) T >= 2L runs DO exist.** Cut A at L = 32, 48, 64 sits exactly at T/L = 2.
My Stage 1 gap G2 ("no data anywhere satisfies T >= 2L") was **wrong as an
absolute statement** and is corrected here. It remains true for **every Cut B
campaign**.

**(b) T/L varies across campaigns and within Cut A.** Cut A steps down 2.0 →
1.333 → 1.0 as L rises, i.e. the largest Cut A systems got the least relaxation
relative to size. This reproduces the 2026-06-17 "T was capped for large L"
pattern in a campaign run after it was identified.

**(c) L = 192 exists** in `refine_smallz` (903 records). This size appears in no
project document. New evidence, not previously registered.

## 3. "T >= 2L is necessary" is a CLAIM, not a premise

Per instruction, this was not assumed. Its provenance:

- Origin: the 2026-06-17 observation that production T was capped at 128 for
  L >= 96, plus the argument that relaxation grows as L^z near criticality, so
  T < L must bias slopes and hence nu.
- **The tau_int pilot that would establish the actual T(L) rule has never been
  run.** It has been owed since 2026-06-17.
- The specific factor 2 has no derivation on record. It is a heuristic margin.

Proposed new claim, to be created rather than assumed:

```yaml
id: METH-TREQ-001
statement: "Reliable extraction of the Cut B correlation-length exponent requires
            T >= 2L, or a T set from a measured tau_int."
statement_class: conjecture
claim_kind: conjecture
epistemic_status: unsupported
confidence: unassessed
confidence_basis: "The L^z relaxation argument is sound in direction but supplies
  no factor. The pilot that would fix T(L) has never been run. Cut A at T/L = 2
  provides a natural control against Cut B at T/L = 1, but no comparison has been made."
falsifiers:
  - "A tau_int pilot showing observables and d_lambda I are T-stationary at T = L."
  - "Cut A results at T/L = 2 and T/L = 1 agreeing within error at matched L."
```

## 4. Consequence for canonical state

`CB-NU-001` (nu_B not measured, confidence set ~[1.5,3]) is **unaffected**. Its
basis is correction-model uncertainty, not T/L, so it does not depend on
METH-TREQ-001.

No claim currently asserts "T >= 2L is required" as canonical, so **no existing
claim needs revision.** The risk was that a future agent would treat the audit's
prose as canonical. Creating METH-TREQ-001 as `unsupported` closes that.

Proposed evidence updates: add the recovered parameter block and the corrected
`metadata_gaps` to `EV-DATA-RUCHEPULL-001`, and register
`EV-EXEC-CATALOGUE-001` for the catalogue run.

## 5. Cheap discriminating test this enables, for later

Cut A has matched L at T/L = 2 (L = 64) and T/L = 1 (L = 128), and Cut B has
T/L = 1 throughout. A read-only comparison of Cut A observables at L = 64
against L = 96 (T/L = 1.333) and L = 128 (T/L = 1.0) is the cheapest available
probe of whether T/L = 1 biases anything. **T0 tier, no new compute.**
