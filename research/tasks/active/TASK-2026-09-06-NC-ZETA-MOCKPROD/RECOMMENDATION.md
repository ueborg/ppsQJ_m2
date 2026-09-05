# RECOMMENDATION — TASK-2026-09-06-NC-ZETA-MOCKPROD

Labels `[E]` `[I]` `[C]` `[J]`.

---

# Verdict: **Pursue**

> One verdict only. The alternative considered and NOT taken is discussed in
> section 4(a); the word appears there as a rejected option, not as a second
> verdict.

`[E]` The measurement package is at `READY_FOR_HUMAN_SUBMISSION`: 16 arms,
6 512 populations, 21/21 preflight checks on every production arm, 14/14
injected-fault controls, 13/13 smoke cases, the frozen predecessor untouched.
**`Pursue` does not mean "run it"** — it means the experiment is worth putting
in front of the human gate, and the deliverable is the package.

`[E]` The independent red team **killed all five candidates**, C1 on its
**analysis half only**; it could not break the measurement half and states that
no population needs to change. Its four prescribed repairs (R1–R4) have been
applied and are re-verified below. `[J]` The verdict is `Pursue` on the repaired
package, and the repairs are the most valuable thing this task produced.

---

## 1. Read this before submitting: what `R = 16` can and cannot see

`[E]` The red team required a minimum detectable effect beside every verdict.
Adding it exposed a limit that was invisible before. Measured from the real
`zeta = 0.35` corpus at `R = 16`, with the simulated `z_crit = 3.88`:

| `L` | `N_c` | median SEM/\|CMI\| | **MDE, % of CMI** |
|---:|---:|---:|---:|
| 32 | 512 / 1024 / 2048 | 0.0105 / 0.0078 / 0.0052 | **5.8 / 4.3 / 2.9** |
| 48 | 512 / 1024 / 2048 | 0.0157 / 0.0114 / 0.0078 | **8.6 / 6.2 / 4.3** |
| **64** | 512 / 1024 / 2048 | 0.0243 / 0.0188 / 0.0144 | **13.3 / 10.3 / 7.9** |

`[E]` And the rung changes actually measured at `zeta = 0.35`, `L = 64`, at
matched `lambda`: `1024 -> 2048` is **−0.13 %**, `2048 -> 4096` **−2.0 %**,
`4096 -> 8192` **+1.2 %**.

`[I]` **So at `L = 64` the top of the ladder cannot be resolved at `R = 16`.**
A rung that truly moves by 0.1–2 % will be classified `ROUGHLY_STABLE` because
nothing below ~8 % is visible there. Mapping this onto the brief's five
questions:

| the brief asks | answerable at `R = 16`? |
|---|---|
| is `N_c = 128` visibly inadequate | **yes** — the low-rung drift at `zeta = 0.35` was ~20 %, far above MDE at every `L` |
| does `256 -> 512` matter strongly | **yes at `L = 32, 48`**; marginal at `L = 64` |
| does `512 -> 1024` still visibly move | **yes at `L = 32, 48`** (MDE 4–6 %); only large effects at `L = 64` |
| does `1024 -> 2048` still visibly matter | **at `L = 32, 48` only** (MDE 2.9 / 4.3 %). **Not at `L = 64`.** |
| how does this change between low and high `zeta` | **yes**, subject to the above |

`[E]` **This is not fixable by tuning `R`.** Reaching a 3 % MDE at `L = 64`,
`N_c = 1024` needs `R ~ 190`; reaching 5 % needs `R ~ 68`. At `zeta = 0.70` the
latter alone is roughly 1 700 core-hours for one rung. `[J]` It is a structural
limit of a survey at this cost, not an oversight, and the honest thing is to
submit knowing it rather than to discover it in the output.

`[J]` **The recommendation stands** because the questions that actually size a
production campaign are the low-rung ones — nobody runs at `N_c = 128` — and
those are answerable at every `L`. The top-rung question is answered at `L = 32`
and `48` and is reported as unresolved at `L = 64`.

## 2. The repairs the red team required, and their state

| | repair | state |
|---|---|---|
| **R1** | Define the recommendation rule: smallest `N_c` such that **that rung and every higher rung** is `ROUGHLY_STABLE`; an `INCONCLUSIVE` rung breaks the chain | **applied.** It had no rule in the frozen spec at all, so the script had invented first-stable-wins and would have printed the cheapest `N_c` while the table above it said `STILL_CHANGING`. Specified as `amendment_2`; smoke cases `S10`–`S12` |
| **R2** | The `INCONCLUSIVE` gate's "or" is a **maximum** over `L`, evaluated on **both** rungs | **applied.** It was a minimum on the higher rung only, so an `L = 64` curve at six times the threshold read `ROUGHLY_STABLE`. Smoke case `S13` |
| **R3** | Print a minimum detectable effect beside every verdict | **applied**, and it is what §1 above is built from |
| **R4** | State `z_crit`'s real size | **applied.** `z_crit` is now **simulated at `R = 16`** rather than read off a normal quantile: the normal Šidák point 3.451 has a true null exceedance of **3.05 %, not 1 %**. Reproduced independently before changing anything. The simulated value is **3.88**. The family-wise rate across the 12 rung classifications is printed, not left to the reader |

`[E]` Two further code defects were fixed: the loader now **excludes `scratch/`**
(the red team's own synthetic files were being ingested as measurements) and
**deduplicates by seed** (requeued packs make duplicate files normal, and two
files with one seed are one population).

## 3. What the red team tried to break and could not

`[E]` The `N_c^0.1871` refutation — it attacked the comparison's sample-size and
`lambda` confound and found the refutation holds on `lambda`-matched medians
(4.67/4.78/4.91/4.72/4.60/4.75 ms over a 128× range in `N_c`). Grid coverage —
re-derived by hand, all six predictions and all three `phi` windows correct.
Seeds and duplication — 6 512 distinct, zero collisions against 54 external
manifests and 5 176 stored populations, zero overlap between 407 design cells and
134 stored cells. `--time` and `--mem` margins — adequate everywhere.

## 4. Three things the researcher should decide at the gate

`[E]` **(a) Whether the `L = 64` top-rung limit is acceptable.** §1. If it is
not, this survey should not be submitted in this form and the sharper question
is: *at which `L` and which rung is a 3 % finite-`N_c` drift worth resolving, and
what does that `R` cost at `zeta = 0.70`?* `[J]` That is the one place where the alternative gate outcome
would have applied instead, and it is the researcher's call, not mine.

`[E]` **(b) The `f_lam` probe.** One local timing run at `zeta = 0.10`,
`L = 64`, `N_c = 512`, `lambda = 0.040` — minutes of laptop time — would replace
this package's weakest input (a ~4× extrapolation) with a measurement. It needs
prior human approval under `RESOURCE_POLICY.md` §3 and was **not** run.

`[E]` **(c) A child task on `CMI(N_c) = c0 + c1/N_c`.** The red team fitted that
form to all six rungs at `zeta = 0.35`, `L = 64`, `lambda = 0.3032` with
`chi²/dof = 0.92` and `c0 = 0.34542 ± 0.00114`. `[I]` If that holds, this task's
standing premise — *the `N_c -> infinity` target is unknown, so the movement is
drift and never bias* — is **false at the one cell where it can be tested**, and
an extrapolated target would let the survey report bias rather than drift.
`[E]` One cell, one `L`, one `lambda`. It is **not** acted on here: it would
change a frozen premise on the strength of a single cell, and the correct route
is `research/tools/child_task.py propose`. `[J]` It is the most scientifically
interesting thing in the red team's report and it is deliberately parked.

## 5. Submission, in order

1. `[E]` Preflight: `bash shared/run_preflight.sh` → `ALL ARMS PASS`.
2. `[E]` Wave 1, `zeta = 0.10`, **102 core-h**. Analyse it before going on.
3. `[E]` Wave 2, `zeta = 0.20`, **221 core-h**.
4. `[E]` `M_z070_nc128` (67 core-h) → **read the wall times back**
   (`RUCHE_RUNBOOK.md` §2b) before committing the two large `zeta = 0.70` arms.
   `rho(0.70)` has no production-regime validation anywhere and is understated
   by 7–9 %.
5. `[E]` The rest of wave 3, then `E_dtau_z010` (analysable only after
   `M_z010_nc512` returns).
6. `[E]` **`conditional/M_z070_nc2048` — 861.6 core-h, 42.6 % of the full
   design — is NOT submitted with the rest.** `CONDITIONAL_SUBMISSION.md`.
   `[I]` Note in light of §1: at `zeta = 0.70`, `L = 64`, a `512 -> 1024` rung
   reading `ROUGHLY_STABLE` may mean "stable" or may mean "below a 10 % MDE".
   The analysis now prints the MDE beside that verdict, and the interlock should
   be read with it.

**Committed: 1 160.3 core-hours. Held: 861.6. Full design: 2 021.9.**

## 6. What this task did not do

`[E]` It wrote nothing to `research/state/**`; the fingerprint is unchanged.
It did not modify `TASK-2026-09-06-NC-ZETA-STAGE1`. It ran no simulation and
submitted nothing. It performed **no external prior-art search**, so external
novelty is `UNRESOLVED`, not favourable. It moves `DISP-PHI-001` and
`DISP-WINDOW-001` in neither direction, produces no `lambda_c(zeta)`, no
exponent, no `N_c^req(zeta)`, and certifies nothing.
