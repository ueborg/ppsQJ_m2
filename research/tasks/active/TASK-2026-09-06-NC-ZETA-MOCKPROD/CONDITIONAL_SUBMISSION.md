# CONDITIONAL_SUBMISSION — `conditional/M_z070_nc2048`

Labels `[E]` `[I]` `[C]` `[J]`.

---

## What is being held back, and what holding it saves

`[E]` One arm: `zeta = 0.70`, `N_c = 2048`, `L in {32, 48, 64}`, nine `lambda`,
`R = 16` — 432 populations, 432 array tasks, **861.6 core-hours (1 206.3
pessimistic)**, `--time=12:00:00`, `--mem=3G`, `cpu_long`, ~18.4 h elapsed at
`%64` excluding queue wait.

> **`861.6` core-hours = `42.6 %` of the full design, for one rung at one
> `zeta`. The committed campaign without it is `1 160.3` core-hours.**

`[E]` Brief §4 permits exactly this — making the high-`zeta` top rung
conditional — **only if the saved core-hours are reported clearly**. That is the
number above, and it is repeated in `COST_MODEL.md` §8, `HUMAN_SUBMISSION.md`
and `RECOMMENDATION.md`.

`[E]` The arm is **complete and preflight-clean** (24/24). It is held by
location, not by omission: it lives under `conditional/` so that a
submit-everything loop over `M_z*` in the task root cannot reach it by accident.

## Why this rung and no other

`[E]` `zeta = 0.70` is 84.0 % of the full design's cost. Within it, the `2048`
rung alone is more than the other four rungs combined, because cost scales with
`N_c` at a rate that is flat per clone-window — so doubling `N_c` doubles the
bill with no efficiency gain.

`[I]` And it is the rung whose answer is most nearly predictable in advance. The
tilt prefactor `(1-zeta)^2` says the sampler has *least* statistical work to do
at high `zeta`, so `zeta = 0.70` is where a small `N_c` is most likely to
suffice. `[C]` That is an inference from a prefactor, and
`NC-ZETA-CALIBRATION` established that the prefactor is **not** the whole `zeta`
dependence — accumulated weight variance is constant to 30 % over a 14x range in
`zeta` against 14–21x for the prefactor alone. `[J]` So the expectation is
weak enough that the rung is worth *holding* rather than *cutting*.

## The release condition

`[E]` Release `conditional/M_z070_nc2048` **if and only if**, after
`M_z070_nc512` and `M_z070_nc1024` have returned and
`analysis/mockprod_analysis.py` has run:

> the `512 -> 1024` rung at `zeta = 0.70` classifies **`CLEARLY_TOO_SMALL`** or
> **`STILL_CHANGING`**.

`[E]` **Do not release it if that rung classifies `ROUGHLY_STABLE`.** The survey
has then already answered its question at `zeta = 0.70` — the curves stopped
moving below 1024 — and 862 core-hours would buy a confirmation the survey does
not need.

`[E]` **Do not release it if that rung classifies `INCONCLUSIVE`.** An
inconclusive rung means `R = 16` cannot resolve movement at that `zeta`, and
adding a *larger* `N_c` at the *same* `R` cannot fix that. `[I]` The binding
budget is then `R`, exactly as it was at the `zeta = 0.35` anchor, and the
correct spend is more populations at 1024, not a new rung at 2048. `[J]` This is
the branch most likely to be misread as "we need more `N_c`", and it is the one
where more `N_c` is worth least.

## Summary of the four branches

| `512 -> 1024` at `zeta = 0.70` | action | why |
|---|---|---|
| `CLEARLY_TOO_SMALL` | **release** | movement is large at 1024; 2048 is needed to see where it stops, or that it does not |
| `STILL_CHANGING` | **release** | movement is reduced but visible; 2048 is the rung that decides |
| `ROUGHLY_STABLE` | **hold** | the question is answered below 1024; save 862 core-hours |
| `INCONCLUSIVE` | **hold** | `R` binds, not `N_c`; spend on `R` at 1024 instead |

`[E]` Whichever branch is taken, **record it**. A held arm is a decision with a
reason, and `FALSIFICATION_RESULTS.md` is where it goes. `[E]` Holding it is not
a negative result about the physics and must not be reported as one: it is a
budget decision, and the `zeta = 0.70` recommendation row then carries the
caveat "not tested above `N_c = 1024`".
