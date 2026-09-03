# HUMAN_SUBMISSION — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

**Terminal state: `READY_FOR_HUMAN_SUBMISSION`.**

**No agent submitted anything and no agent may.**
`research/RESOURCE_POLICY.md` §4 forbids agent submission unconditionally — not
"until a gate", not "once approved", not "when HPC access returns". Nothing in
this package contains an executable scheduler call; every arm's preflight
asserts that about its own files and fails if it stops being true. The
researcher types the command.

`research/state/**` was not written. No predecessor task directory was modified.
`main` was not touched.

---

## 1. Classification

**READY FOR IMMEDIATE HUMAN SUBMISSION** — all seventeen arms below, validation
having passed (`VALIDATION.md`: 17/17 preflights, 16/16 negative controls,
bit-identical reproduction of predecessor populations, duplicate scan clean).

**BLOCKED** — the seven arms under `conditional/`. Separate directory, separate
gate headings in `CONDITIONAL_SUBMISSION.md`, hard interlock in every job
script. **No command in §5 of `RUCHE_RUNBOOK.md` reaches them.**

---

## 2. Every runnable arm

Columns: existing populations REUSED (never recomputed) · FRESH populations
required · manifest rows · fresh seed range · predicted and pessimistic
core-hours · expected wall-clock at `%64` excluding queue wait.

`<batch-submit>` below stands for the scheduler's batch-submission command. It is
written this way in this file on purpose: **an agent may not execute it**, and a
literal command string in an agent-authored file is exactly the thing the
resource policy's enforcement layer refuses. The runbook, which is written for
you, spells it out.

| campaign | directory | L | T | lambda | N_c | R | reuse | fresh | rows | fresh seeds | partition | `--time` | `--mem` | core-h | pess | elapsed h | dependency | command |
|---|---|--:|--:|---|--:|--:|--:|--:|--:|---|---|---|---|--:|--:|--:|---|---|
| A | `A_L64_nc2048_topup` | 64 | 64 | 0.3032 | 2048 | 48 | 24 | 24 | 24 | 33000024–33000047 | cpu_med | 03:00:00 | 3G | 32.1 | 45.0 | 1.34 | none | `cd A_L64_nc2048_topup && <batch-submit> submit.slurm` |
| A | `A_L64_nc4096` | 64 | 64 | 0.3032 | 4096 | 48 | 0 | 48 | 48 | 33020000–33020047 | cpu_long | 08:00:00 | 4G | 146.3 | 204.9 | 3.05 | none | `cd A_L64_nc4096 && <batch-submit> submit.slurm` |
| A | `A_L64_nc8192` | 64 | 64 | 0.3032 | 8192 | 48 | 0 | 48 | 48 | 33040000–33040047 | cpu_long | 18:00:00 | 7G | 333.2 | 466.5 | 6.94 | none | `cd A_L64_nc8192 && <batch-submit> submit.slurm` |
| B | `B_L64_cross_nc512` | 64 | 64 | 0.2182…0.2482 (7) | 512 | 48 | 0 | 336 | 336 | 33060000–33066047 | cpu_med | 01:00:00 | 1G | 90.8 | 127.2 | 1.73 | none | `cd B_L64_cross_nc512 && <batch-submit> submit.slurm` |
| B | `B_L64_cross_nc1024` | 64 | 64 | 0.2182…0.2482 (7) | 1024 | 48 | 72 | 264 | 264 | 33080000–33086047 | cpu_med | 02:00:00 | 2G | 136.0 | 190.4 | 2.74 | none | `cd B_L64_cross_nc1024 && <batch-submit> submit.slurm` |
| B | `B_L64_cross_nc2048` | 64 | 64 | 0.2182…0.2482 (7) | 2048 | 48 | 0 | 336 | 336 | 33100000–33106047 | cpu_med | 03:00:00 | 3G | 346.1 | 484.5 | 6.58 | none | `cd B_L64_cross_nc2048 && <batch-submit> submit.slurm` |
| B2 | `B2_L32_nc512` | 32 | 32 | 0.2182…0.2482 (7) | 512 | 48 | 0 | 336 | 336 | 33120000–33126047 | cpu_med | 01:00:00 | 1G | 7.5 | 10.5 | 0.14 | none | `cd B2_L32_nc512 && <batch-submit> submit.slurm` |
| B2 | `B2_L32_nc1024` | 32 | 32 | 0.2182…0.2482 (7) | 1024 | 48 | 72 | 264 | 264 | 33140000–33146047 | cpu_med | 01:00:00 | 1G | 11.2 | 15.7 | 0.23 | none | `cd B2_L32_nc1024 && <batch-submit> submit.slurm` |
| B2 | `B2_L32_nc2048` | 32 | 32 | 0.2182…0.2482 (7) | 2048 | 48 | 0 | 336 | 336 | 33160000–33166047 | cpu_med | 01:00:00 | 1G | 32.5 | 45.5 | 0.62 | none | `cd B2_L32_nc2048 && <batch-submit> submit.slurm` |
| B2 | `B2_L48_nc512` | 48 | 48 | 0.2182…0.2482 (7) | 512 | 48 | 0 | 336 | 336 | 33180000–33186047 | cpu_med | 01:00:00 | 1G | 30.5 | 42.7 | 0.58 | none | `cd B2_L48_nc512 && <batch-submit> submit.slurm` |
| B2 | `B2_L48_nc1024` | 48 | 48 | 0.2182…0.2482 (7) | 1024 | 48 | 72 | 264 | 264 | 33200000–33206047 | cpu_med | 01:00:00 | 2G | 45.7 | 63.9 | 0.92 | none | `cd B2_L48_nc1024 && <batch-submit> submit.slurm` |
| B2 | `B2_L48_nc2048` | 48 | 48 | 0.2182…0.2482 (7) | 2048 | 48 | 0 | 336 | 336 | 33220000–33226047 | cpu_med | 01:00:00 | 2G | 132.3 | 185.2 | 2.51 | none | `cd B2_L48_nc2048 && <batch-submit> submit.slurm` |
| C | `C_L96_nc1024` | 96 | 96 | 0.3032 | 1024 | 24 | 0 | 24 | 24 | 33240000–33240023 | cpu_long | 12:00:00 | 3G | 89.0 | 124.7 | 3.71 | none | `cd C_L96_nc1024 && <batch-submit> submit.slurm` |
| C | `C_L96_nc2048` | 96 | 96 | 0.3032 | 2048 | 24 | 0 | 24 | 24 | 33260000–33260023 | cpu_long | 24:00:00 | 4G | 202.7 | 283.8 | 8.45 | none | `cd C_L96_nc2048 && <batch-submit> submit.slurm` |
| D | `D_L128_nc2048` | 128 | 128 | 0.3032 | 2048 | 16 | 0 | 16 | 16 | 33280000–33280015 | cpu_long | 72:00:00 | 9G | 502.4 | 703.3 | 31.40 | none | `cd D_L128_nc2048 && <batch-submit> submit.slurm` |
| E | `E_L64_dtau_nc64` | 64 | 64 | 0.3032, dtau {3,6,12} | 64 | 48 | 0 | 144 | 144 | 33300000–33302047 | cpu_med | 01:00:00 | 1G | 9.1 | 12.8 | 0.33 | none | `cd E_L64_dtau_nc64 && <batch-submit> submit.slurm` |
| E | `E_L64_dtau_nc256` | 64 | 64 | 0.3032, dtau {3,6,12} | 256 | 48 | 0 | 144 | 144 | 33320000–33322047 | cpu_med | 01:00:00 | 1G | 32.3 | 45.3 | 1.16 | none | `cd E_L64_dtau_nc256 && <batch-submit> submit.slurm` |

**Totals: 3 280 fresh tasks · 240 populations reused · 2 180.0 core-hours
(3 051.9 pessimistic).**

## 3. The summary the gate asks for

| question | answer |
|---|---|
| total immediate core-hours | **2 180.0** |
| total pessimistic | **3 051.9** |
| longest expected single job | **31.40 h** — `D_L128_nc2048`, one population at `L = 128`, `N_c = 2048` |
| longest pessimistic single job | **43.96 h** (`--time=72:00:00`, 1.64× that) |
| which arrays can run concurrently | **all seventeen.** No arm consumes any other's output; the only couplings are scheduling ones |
| which campaign dominates cost | **none dominates.** B 26.3 % · A 23.5 % · D 23.0 % · C 13.4 % · B2 11.9 % · E 1.9 % |

**Cost per answer is a different ranking, and the more useful one.**
`D` is 502 core-hours for **16 tasks** and one screening number that
*cannot certify convergence*; `A`+`B` are 1 085 core-hours for 1 056 tasks and
carry the plateau and shape verdicts together. `E` is 1.9 % of the campaign and
is the only arm whose **both** outcomes kill a mechanism.

**Elapsed, two readings, because this is the number most often misread:**

- **if `%64` is granted per array** (192+ concurrent across arms): elapsed is the
  slowest arm, `D_L128_nc2048` at **31.4 h** (44.0 pessimistic). Everything else
  finishes inside ~9 h of compute.
- **if 64 slots are a shared total**: throughput-bound at `2180/64 = 34 h`
  (47.7 pessimistic) — still set by `D`, because `D` is one 31 h task either way.

Queue wait is excluded from every figure above and will dominate the short arms.
`RUCHE_RUNBOOK.md` §3 gives the accounting check to run **before** relying on
any of it.

## 4. What each campaign buys

| campaign | the one question it answers | what a negative answer means |
|---|---|---|
| **A** | is a high-`N_c` plateau OBSERVABLE at `L = 64`? | even `L = 64` is pre-asymptotic; no `L` has a calibrated `N_c`; no `I_inf`, no `B`, no `gamma` |
| **B** | does finite-`N_c` distort the SHAPE of `CMI(lambda)` where the locator sits? | transition-region production needs a higher `N_c` than the level test suggests |
| **B2** | *(with B)* does the CROSSING converge before the absolute level? | the locator inherits the level's problem, and `L = 128` production becomes unaffordable rather than merely expensive |
| **C** | does `L = 96` enter a simpler high-`N` regime? | `L = 96` is pre-asymptotic and is EXCLUDED from any cross-`L` `B` comparison |
| **D** | is the `1024 → 2048` change at `L = 128` still material? | recommend the conditional `N_c = 4096` rung — a ~71 h job at the partition's ceiling |
| **E** | does the drift depend on the window count `K`? | either outcome kills a mechanism; anything between is `INCONCLUSIVE` |

## 5. Three things to read before submitting

1. **`R = 48` in campaign A is not the brief's `R = 24`, and the reason is
   statistical.** At `R = 24` the top step's `Delta` half-width would be ~1.2×
   `tau_I`, so the arm could not satisfy P2 whatever the data did. The manifests
   are ordered so `--array=0-23` is a clean matched-`R`-24 sub-campaign if you
   want the literal design. `CAMPAIGN_DESIGN.md` §2A.
2. **Campaign B2 is an addition beyond the brief's literal §4.** It is 260
   core-hours (12 %) and it exists because without it §4B's load-bearing
   question cannot be answered — the low-`L` reference curves exist at
   `N_c = 1024` and nowhere else. It is the one arm droppable in a single line,
   and `SUBMISSION_DEPENDENCIES.md` §2 states exactly what is lost.
3. **The `--mem` model was a model quoted as a measurement, and it
   under-predicts.** `COST_MODEL.md` §4. `D_L128_nc2048` is the largest
   *modelled* rather than measured request in the campaign. **Please run the
   one `sacct … MaxRSS` line in `RUCHE_RUNBOOK.md` §7** — it is the first such
   measurement of this sampler on the cluster that would exist anywhere, and it
   settles the question for the whole programme.

## 6. After the results land

`RUCHE_RUNBOOK.md` §10 gives the single analysis command. Then write
`FALSIFICATION_RESULTS.md` against the **frozen** `FALSIFICATION_PLAN.md` and
walk `DECISION_TREE.md` from its entry condition.

**No result may be merged into `research/state/**` without red-team review and
the human gate. Completion is not adjudication.**
