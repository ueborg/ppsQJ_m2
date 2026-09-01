# CLEAN_CLONE_TEST — the check the previous readiness verdict lacked

TASK-2026-09-01-SMCRUCHE-PACKFIX §2. Labels: `[E]` · `[I]` · `[C]` · `[J]`

## The environment

`[E]` Built with `git archive HEAD | tar -x` plus the new package files, i.e.
**tracked content only**. 754 files. Under `research/tasks/active/` it contains
**only** `TASK-2026-09-01-SMCRUCHE-READY`; `TASK-2026-08-30-SMCSTAT` is **absent**,
exactly as in a Ruche clone.

## Control — the reported failure, reproduced

```
OLD import form -> ModuleNotFoundError: No module named 'instrumented'
```

`[J]` The environment is therefore a faithful reproduction of the failure, not a
guess at it.

## Both arms import

```
[env] python       .../.venv/bin/python3
[env] instrumented .../cleanclone/research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/support/instrumented.py
[env] pps_qj       .../cleanclone/pps_qj/__init__.py
```

`[E]` `instrumented` resolves to the **bundled** copy inside the clean tree, and
`pps_qj` to the clean tree's tracked package. Identical for `arm2`.

## A real `run_cell.py` smoke run, in the clean environment

`[E]` ARM 1 manifest row 0, executed for real:

```
[ok] idx=0 L=96 N_c=128 wall=351.82s -> ./results/ARM1_00000.json
```

| field | value |
|---|---|
| `L`, `T`, `N_c` | 96, 96.0, 128 |
| ζ, λ, `dtau_mult` | 0.35, 0.3032, 6.0 |
| `resample_scheme`, `seed` | systematic, 1168016 |
| `status` | **ok** |
| `n_steps` | 922 |
| `cmi_weighted_mean` | 0.3915025117013953 |
| `cmi_within_var` | 0.02765638645251927 |
| `n_nonfinite`, `brentq_fallbacks` | 0, 0 |
| `n_distinct_anc_final` | 1 (total founder collapse, as expected at this cell) |
| `per_clone_CMI` | 128 values stored |

`[E]` The seed matches the frozen stream: `1040000 + 1000·128 + 16 = 1168016`.

`[J]` This is the strongest available evidence that the package is
self-contained: not an import check, but a real row of the frozen experiment
produced end-to-end from tracked files only.

## One number worth knowing

`[E]` The cost model predicted `6.59e-3 × 128 × 922` ≈ **778 s** for this row;
the measured wall was **352 s**, i.e. **2.2× faster**.

`[I]` The rate anchors were measured during a contended overnight campaign, so
they are conservative on an idle machine. `[J]` The **62.2 / 194.1 core-hour**
figures are therefore an **upper bound on this hardware** and are unchanged —
no parameter or estimate was adjusted. A Ruche core is not this core, which is
why `RUCHE_RUNBOOK.md` still asks for a single interactive task before queueing.

## Negative controls on the preflight

| forced condition | preflight |
|---|---|
| `support/` hidden | **exit 1**, `ModuleNotFoundError` named explicitly |
| `--partition=cpu_short` vs a 4 h request | **exit 1**, "exceeds partition cpu_short MaxTime of 1 h" |
| `--partition` line deleted | **exit 1**, "the scheduler will pick a default (cpu_short…) and kill the job" |
| all restored | **exit 0**, `PREFLIGHT PASSED` |

`[J]` Each of the three failure modes that actually occurred — the missing
module, the wrong partition, the absent partition — now stops the preflight
before submission.
