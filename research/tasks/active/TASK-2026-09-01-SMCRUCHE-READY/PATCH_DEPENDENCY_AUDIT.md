# PATCH_DEPENDENCY_AUDIT — verified from the repository, not from the plan

TASK-2026-09-01-SMCRUCHE-READY §2. Labels: `[E]` · `[I]` · `[C]` · `[J]`

**Conclusion: the expected atomic sequence is CORRECT. No stop condition fires.**

---

## 1. What each patch touches, read from the patch headers

| patch | files | new file? |
|---|---|---|
| `0001-add-statistical-diagnostics` | `pps_qj/cloning.py`, `pps_qj/gaussian_backend.py` | no — modifies |
| `0002-add-population-level-error-analysis` | `pps_qj/production/config.py`, `pps_qj/production/run.py` | no — modifies |
| `0003-add-statistics-planner` | `tools/plan_cloning_statistics.py` | **yes** |
| `0004-add-tests` | `tests/test_statistical_diagnostics.py` | **yes** |
| `0005-fix-lineage-ess-…-zeta-zero` | `pps_qj/cloning.py` | no — modifies |
| `0006-supersede-with-bias-aware-…-planner` | `tools/plan_cloning_statistics.py`, `tools/calibration/bias_calibration.json` | **yes (both)** |
| `0007-add-zeta-sector-regression-tests` | `tests/test_statistical_diagnostics.py` | no — modifies |
| `0008-add-bias-aware-planner-tests` | `tests/test_bias_aware_planner.py` | **yes** |

## 2. The four dependency facts, each verified directly

`[E]` **(a) `0005` is the repair to `0001`, and cannot exist without it.**
`0005`'s removed lines are exactly:

```
-            _lw = log_w if 'log_w' in dir() else np.zeros(N_c)
-            log_w_lineage += _lw
-            _c = log_w_lineage - float(log_w_lineage.max())
```

and `0001` **introduces** that first line at its own line 77
(`+            _lw = log_w if 'log_w' in dir() else np.zeros(N_c)`).
`[I]` So `0005` has no meaning against a tree without `0001`, and `0001` alone
lands the defective diagnostic. → **`0001 + 0005` is one logical change.**

`[E]` **(b) `0007` requires `0004`.** `0007` is a *modification*
(`index 201446f..41ba3a3 100644`, not a new file) whose single hunk is
`@@ -193,3 +193,57 @@` anchored on
`test_production_emits_per_clone_arrays_and_lineage_ess`, a function `0004`
creates. → **`0004 + 0007` is one logical change**, and `0007` carries the
regression coverage for the `0001` defect.

`[E]` **(c) `0003` and `0006` are mutually exclusive.** Both declare
`new file mode 100644` for the **same path** `tools/plan_cloning_statistics.py`
(`0003` index `0000000..fbf672b`, `0006` index `0000000..aaabc07`). Applying
both is impossible, not merely unwise. → **`0003` is rejected**, which matches
the SMCCERT verdict `SUPERSEDED`.

`[E]` **(d) `0008` depends on `0006` at RUNTIME, not at apply time.** It creates
a new test file that does
`sys.path.insert(0, os.path.join(ROOT, "tools")); import plan_cloning_statistics as P`
and reads `tools/calibration/bias_calibration.json`. `[I]` It will *apply*
without `0006` and its tests will *error* without it, so it must be ordered
after `0006` but need not be atomic with it.

## 3. The sequence, and why this order

```
commit 1   0001 + 0005     atomic: the defect never lands alone
commit 2   0002            independent of everything else
commit 3   0004 + 0007     atomic: 0007 modifies the file 0004 creates
commit 4   0006            0003 NOT applied
commit 5   0008            after 0006, which it imports
```

`[J]` Tests land **after** the code they test (commits 3 and 5 after 1 and 4) so
that a skipped fix fails loudly rather than silently. That is the SMCCERT plan's
stated rationale and the audit confirms it is achievable in this order.

## 4. Authority

`[E]` `TASK-2026-08-31-SMCCERT/GIT_INTEGRATION_PLAN.md` §1 and §6 are the final
integration recommendation and they state exactly this. `[E]` SMCCERT has **no**
`CORRECTIONS.md`; the parent's `CORRECTIONS.md` §10 concerns SMCSTAT candidate
statuses and does not bear on the patch set. `[E]` Nothing in GENCOL's artifacts
proposes a patch.
