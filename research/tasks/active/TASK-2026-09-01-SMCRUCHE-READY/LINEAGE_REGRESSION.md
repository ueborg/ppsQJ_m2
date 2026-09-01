# LINEAGE_REGRESSION — the ζ = 0 defect, reproduced and repaired

TASK-2026-09-01-SMCRUCHE-READY §5. Labels: `[E]` · `[I]` · `[C]` · `[J]`

**Result: old implementation → FAIL, repaired implementation → PASS.**

---

## 1. Protocol

`[E]` A throwaway worktree at the integration branch head (`e7cb73b`), with the
`0005` repair **reverse-applied** to recover the exact `0001`-as-proposed form,
confirmed by the reappearance of `_lw = log_w if 'log_w' in dir() else np.zeros(N_c)`
at `pps_qj/cloning.py:561`. Then re-applied. `[E]` A `conftest.py` shim strips
the venv's editable-install meta-path finder so the worktree tests **its own**
`pps_qj` and not the main checkout's; it is verification-only and is not part of
any patch.

`[E]` Probe: `L` = 8, `N_c` = 16, `T_total` = 2.0, `delta_tau` = 0.5, seed 11,
`backend="scalar"`, `jump_update_method="lowrank"`, `solver_method="brentq"`.

## 2. Measured values

### OLD — `0001` as proposed

| ζ | lineage ESS per window | min | instantaneous ESS (last) | surviving founders |
|---|---|---:|---:|---:|
| 0.30 | 15.9921, 15.7800, 15.0997, 14.3542 | 14.3542 | 15.643 | 13 / 16 |
| **0.00** | **16.0000, 16.0000, 16.0000, 16.0000** | **16.0000** | 12.000 | **4 / 16** |
| 1.00 | 16.0000, 16.0000, 16.0000, 16.0000 | 16.0000 | 16.000 | 16 / 16 |

`[E]` **At ζ = 0 the diagnostic reports a perfect, undegraded population of 16
at every window, on a run that has collapsed to 4 of 16 founders** and whose
instantaneous ESS has itself fallen to 12.

### REPAIRED — `0001 + 0005`

| ζ | lineage ESS per window | min | instantaneous ESS (last) | surviving founders |
|---|---|---:|---:|---:|
| 0.30 | **15.9921, 15.7800, 15.0997, 14.3542** | **14.3542** | 15.643 | 13 / 16 |
| **0.00** | **6.0000, 8.0000, 8.0000, 12.0000** | **6.0000** | 12.000 | 4 / 16 |
| 1.00 | 16.0000, 16.0000, 16.0000, 16.0000 | 16.0000 | 16.000 | 16 / 16 |

`[E]` **ζ = 0.30 is bit-identical, digit for digit, to the old implementation**,
and ζ = 1 is unchanged. `[I]` The repair therefore carries **no regression risk
in the production regime** — `log(weights) = log_w − lw_max` and the ESS is
invariant under a constant shift of the log weights, so the two forms are
algebraically the same wherever `log_w` was bound.

`[E]` At ζ = 0 the diagnostic now falls to **6.0 / 16** and tracks the collapse
it previously hid.

## 3. Test outcome

```
OLD:       2 failed, 2 passed, 11 deselected
             FAILED test_lineage_ess_is_not_saturated_at_zeta_zero
             FAILED test_lineage_ess_tracks_selection_pressure_across_zeta
REPAIRED:  4 passed, 11 deselected
```

`[E]` `test_lineage_ess_finite_and_defined_for_every_window` passes on both, as
designed — it checks shape and finiteness, not saturation, so it is not the
discriminating test and is not counted as one.

`[J]` The two discriminating tests fail on the old code and pass on the new. A
regression test that has never been observed to fail is a comment; these have
been observed to fail.

## 4. Why this matters beyond ζ = 0

`[J]` ζ = 0 is the fully post-selected no-click sector — a real limit of the ζ
family this project studies, and the **most** degenerate regime the sampler
supports. `[E]` The originally proposed test suite passed **12 / 12** while this
was broken, because `_run()` was only ever called at the intermediate production
ζ or at ζ = 1. `[J]` The failure was silent by construction: the
`'log_w' in dir()` guard turned an unbound name into zeros instead of a
`NameError`.
