# CONDITIONAL_SUBMISSION — the BLOCKED group

**Every arm in `conditional/` is prepared, validated and BLOCKED.** None may be
submitted before the adjudication named in its own heading below. No agent may
submit any of them at any time, adjudicated or not
(`research/RESOURCE_POLICY.md` §4).

---

## How the block is enforced — three mechanisms, because a comment stops nothing

1. **Separate directory.** `conditional/` is not enumerated by any loop in
   `RUCHE_RUNBOOK.md`. The submission loop in §5 names the seventeen immediate
   arms explicitly rather than globbing `*/`, precisely so that adding a
   conditional arm cannot silently enrol it.
2. **A hard interlock in every job script.** Before importing the sampler, each
   array task checks for a release file named for that exact arm and **exits 3**
   if it is absent. Submitting a blocked arm by accident costs a few seconds of
   an array of no-ops — not core-hours, and not a result nobody is entitled to
   interpret.
3. **A refusing preflight.** `conditional/*/run_preflight.sh` exits 3 while the
   interlock is armed, so a "preflight everything" sweep reports these as
   `BLOCKED`, never as `READY`.

To release one, after its adjudication:

```bash
cd .../TASK-2026-09-03-NC-PLATEAU-CALIBRATION/conditional
echo "Released <date> by <who>: <which adjudication, and what it found>" \
    > GATE_RELEASED_<arm_name>
```

The content is not machine-checked and cannot be. It is a place to record which
adjudication released the arm; the point of the interlock is that a human wrote
it.

---

## GATE 1 — `cond_D2_L128_nc4096`

**Blocking condition: CAMPAIGN D ADJUDICATION.**

> Recommend this arm if, on the `L = 128` ladder completed by campaign D,
> **either** `|Delta_1024| = |I_2048 - I_1024|` is resolved OUTSIDE the frozen
> tolerance `tau_I = 0.006` (its 95 % interval excludes `[-tau_I, +tau_I]`),
> **or** no plateau criterion P1–P5 is satisfied at the top of that ladder.
> Do **not** recommend it because the observed `Delta_1024` "looks large".

`[E]` This trigger is fixed **now**, before the datum exists, and may not be made
to depend on the observed 2048 value after the fact.

| | |
|---|---|
| cell | `L = 128`, `T = 128`, `zeta = 0.35`, `lambda = 0.3032`, `N_c = 4096`, `R = 8` |
| cost | 572 core-hours (801 pessimistic), 8 tasks |
| **slowest task** | **71.5 h** predicted, **100.1 h pessimistic** |
| request | `cpu_long`, `--time=168:00:00`, `--mem=26G` |

`[!]` **Read this before releasing.** `--time=168:00:00` is `cpu_long`'s
`MaxTime` **exactly**. There is 1.68× headroom on the pessimistic estimate and
**no room to add any**. There is no checkpointing in this code path, so a task
that overruns is lost entirely after six days of wall-clock. `[I]` `L = 128` at
`N_c = 4096` is at the edge of what this cluster can run as a single job, and
`N_c = 8192` at `L = 128` is **not runnable at all** under this architecture.
`[J]` That, more than the core-hour figure, is the argument for taking the
locator route at `L = 128` instead.

`[E]` Note also that at `R = 8` this arm **cannot certify convergence** either —
it can only show whether the `2048 → 4096` step is again of the size the
previous ones were.

---

## GATE 2 — `cond_M96_nc1024` **or** `cond_M96_nc2048`, never both

**Blocking condition: CAMPAIGN C ADJUDICATION — and only one of the two.**

> Release `cond_M96_nc1024` only if campaign C identifies `N_c = 1024` as the
> smallest `N_c` meeting the frozen production adequacy criterion at `L = 96`.
> Release `cond_M96_nc2048` if it identifies 2048 instead, or if it identifies
> none and you accept a scan at the largest calibrated rung.

`[E]` **They are the same physical scan at two population sizes.** Running both
is duplicated compute, not a robustness check.

| arm | tasks | core-h | pess | slowest | request |
|---|---:|---:|---:|---:|---|
| `cond_M96_nc1024` | 108 | 308 | 432 | 3.2 h | `cpu_long` `08:00:00` `3G` |
| `cond_M96_nc2048` | 108 | 702 | 983 | 7.3 h | `cpu_long` `18:00:00` `4G` |

`[E]` Design: the frozen 9-point grid
`0.2032 0.2182 0.2232 0.2282 0.2332 0.2382 0.2432 0.2482 0.2632`, `R = 12`
(Stage 1). The centre is at `Delta lambda = 0.005`; the two outer guards reduce
the risk of a boundary-induced locator. `[E]` The grid comes from **observed
low-`L` locator behaviour only** and is not a critical-law assumption.

`[E]` **Stage 2 is not built and must not be improvised.** Topping up
crossing-bracketing cells to `R = 24` requires a *frozen rule* fixed before the
Stage-1 curves are seen. Cells must not be chosen because they look visually
attractive. That rule is a new task's job.

---

## GATE 3 — `cond_M128_nc2048` **or** `cond_M128_nc4096`, never both

**Blocking condition: CAMPAIGN D ADJUDICATION — STRONGLY GATED.**

> Release `cond_M128_nc2048` only if campaign D's `N_c = 2048` rung PASSES the
> frozen adequacy screen. If it fails, the conditional `N_c = 4096` central rung
> (GATE 1) comes first and both of these stay blocked. Release
> `cond_M128_nc4096` only if 2048 fails the screen **and** the 4096 central rung
> then passes it.

`[E]` **An adequate `N_c` must be identified BEFORE a 9-point scan at this `L`
is run at all.**

| arm | tasks | core-h | pess | slowest | request |
|---|---:|---:|---:|---:|---|
| `cond_M128_nc2048` | 72 | 1 739 | 2 435 | 27.3 h | `cpu_long` `72:00:00` `9G` |
| `cond_M128_nc4096` | 72 | **3 960** | 5 545 | **62.1 h** | `cpu_long` `144:00:00` `26G` |

`[!]` `cond_M128_nc4096` is **the most expensive object in the whole campaign by
a wide margin** — 1.8× the entire immediate group. `[J]` It should not be the
first way the programme learns that `L = 128` production is unaffordable. GATE 1
is 572 core-hours and answers that question first, which is the entire reason
GATE 3 sits behind it.

`[E]` Stage 1 uses `R = 8`, chosen against `R = 12` on an explicit
crossing-resolution-versus-cost basis: at `L = 128` the per-population spread is
so large that `R = 12` would widen the campaign by 50 % without moving any
crossing interval into a decisive range. `[J]` If `L = 128` production ever
happens, its `R` will be set by the calibration, not guessed here.

---

## GATE 4 — `cond_LOWZ_nc64` **and** `cond_LOWZ_nc256`, both or neither

**Blocking condition: OPTIONAL — not part of the `zeta = 0.35` calibration.**

`[E]` This is Design 2 of `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING`, prepared
here at its stated configuration and **deliberately not in the immediate
group**: the programme wants the `zeta = 0.35` calibration understood before it
spends anything on a second `zeta`. Release it only as an explicit decision to
buy that one test now.

| | |
|---|---|
| cell | `L = 64`, `T = 64`, **`zeta = 0.10`**, `lambda = 0.3032`, `N_c ∈ {64, 256}`, `R = 48` |
| cost | **11.8 core-hours** (16.5 pessimistic), 96 tasks, slowest 0.19 h |
| request | `cpu_med`, `--time=01:00:00`, `--mem=1G` |

`[E]` **Pre-registered kill criterion, from the parent task, unchanged**: drift
at `zeta ≈ 0.1` **greater than or equal to** drift at `zeta = 0.35` kills the
guided-residual mechanism and revives Born-rarity reasoning.

`[E]` **Release both or neither** — the criterion is a comparison across `N_c`
and needs both population sizes.

`[E]` **"Matched `lambda`" is read as THE SAME `lambda`**, not the same offset
from a putative `lambda_c`. `[J]` Matching on a critical-law offset would import
the very law the programme exists to measure, which the non-negotiable rules
forbid. This is a design decision made here and it should be checked rather than
assumed correct.

`[E]` **Constraint carried forward from the parent**: the known small-`zeta`
anomaly at `zeta <= 0.075`, `L >= 160` remains a **separate** open empirical
issue and may not be merged into this test's interpretation.

---

## What none of these may become

`[E]` A released arm is still a measurement, not an adjudication. No result from
any conditional arm may be merged into `research/state/**` without red-team
review and the human gate. `[E]` And no conditional arm is released
automatically by a script, by this task, or by any successor task: release is a
human writing a file.
