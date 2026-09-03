# PROBLEM_MEMO — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Charter Stage 1. Written **before** any arm was built, any cost computed or any
criterion frozen. Labels `[E]` `[I]` `[C]` `[J]`.

---

## 1. The observed problem

`[E]` At the hard cell `L = T = 128`, `zeta = 0.35`, `lambda = 0.3032`, the
conditional mutual information keeps moving as the population grows. Rebuilt
here from the raw per-population JSONs by `tools/reconstruct_inventory.py`:

| `N_c` | R | mean CMI | across-population SEM | `Delta_N` |
|---:|---:|---:|---:|---:|
| 64 | 64 | 0.51957 | 0.02494 | −0.09898 ± 0.03430 |
| 128 | 64 | 0.42059 | 0.02354 | −0.12127 ± 0.02892 |
| 256 | 64 | 0.29932 | 0.01679 | −0.04822 ± 0.02739 |
| 512 | 48 | 0.25109 | 0.02164 | −0.06021 ± 0.02343 |
| 1024 | 32 | 0.19088 | 0.00898 | — |

`[E]` The top measured step is resolved away from zero. `[E]` The observable has
fallen by a factor 2.7 across the ladder and shows no sign of stopping.

`[E]` Two things are **not** available to interpret that with.

First, `[E]` `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` established a
**structural obstruction**: the production mutation kernel's no-click branch is
deterministic, so `M^m(x, dy) <= beta M^m(x', dy)` fails for every finite `beta`
and every `m`, and with it the uniform-mixing hypotheses of every non-asymptotic
Feynman–Kac theory that could have supplied a constant. `[I]` Direct transfer is
**invalid**, not merely unverified. `[J]` It does **not** follow that no
`O(1/N_c)` bound exists — only that no controlled analytic `N_c_req` rule is
currently available.

Second, `[E]` a clean `I_N = I_inf + B/N` is rejected on the `L = 128` ladder.
Independently recomputed here from the raw files: `chi2 = 12.58` on 3 dof,
`p = 0.0056` — the same numbers that task reported, reached from the JSONs
rather than from its summary. `[E]` So the obvious empirical substitute for the
missing theory is also unavailable at the `L` where it is most needed.

`[E]` **A correction to the accepted framing.** The brief states that the `1/N`
description was rejected "at `L=96` and `L=128`". At `L = 128` this
reconstruction confirms it exactly. At `L = 96`, the production-geometry ladder
at `lambda = 0.3032` that can be rebuilt from raw files has only three rungs
(`N_c = 128, 256, 512`) and gives `chi2 = 1.90` on 1 dof, `p = 0.168` — **not
rejected**. `[I]` The predecessor's `L = 96` rejection (`chi2 = 10.54`, 3 dof)
must therefore rest on a four-rung `L = 96` ladder from a different cell or a
different corpus slice. `[J]` The `L = 96` half of that statement is not
reproducible from the raw `lambda = 0.3032` data and should not be leaned on
until its ladder is identified. This does not weaken the case for the campaign;
it strengthens it, because `L = 96` is then even less characterised than
assumed.

## 2. The smallest precise research question

> At `zeta = 0.35`, `T = L`, for the certified guided-cloning sampler and
> `OBS-CMI-001`: what is the smallest `N_c` at which `Delta_N = I_{2N} - I_N`
> satisfies the frozen criteria P1–P5, at each of `L = 64, 96, 128`; and does
> the inferred cross-`L` crossing location stabilise within `tau_lambda` at a
> smaller `N_c` than the absolute level does?

## 3. Why current approaches do not resolve it

`[E]` **The theory route is closed for now** (§1). `[E]` **The empirical route
has never been run to the top**: the largest `N_c` anywhere in the corpus is
2048, and it exists at exactly one `L` (64) and three `lambda`. Nothing
establishes that a plateau exists to be found.

`[E]` **The one place a plateau looks plausible is also the one place the
statistics are too weak to say so.** At `L = 64`, `lambda = 0.3032`, the
existing `1024 -> 2048` step is `Delta = +0.00235 ± 0.00528`. It is compatible
with zero — and its 95 % interval is `[−0.0080, +0.0127]`, which is 1.7× wider
than the material tolerance the programme needs. `[I]` **That step demonstrates
that `R` was too small to tell, not that the ladder converged**, and no increase
in `N_c` repairs it. `[J]` This distinction is the single most important thing
this campaign has to keep straight, and it is why `SUCCESS_CRITERIA.yaml` gives
it its own verdict label rather than folding it into "converged".

`[E]` **`R` and `N_c` have been conflated in the corpus's own design history.**
`N_c` controls the finite-particle approximation; `R` controls the uncertainty
of the finite-`N_c` population mean. The existing ladders vary both at once —
`R` runs 96, 64, 48, 32, 24 across rungs — so a step's `Delta` and its
half-width move together for reasons that have nothing to do with convergence.

## 4. Which decision changes with the answer

`[J]` Whether the programme can begin **rough production** at all, and at what
`N_c` per `L`. Every downstream quantity — `lambda_c(zeta, L)`, `nu`, `phi`, the
FSS form — is a functional of curves measured at some `N_c`. If that `N_c` is
pre-asymptotic and its drift is `L`-dependent, every one of them inherits an
`L`-dependent displacement that no amount of `R` removes and that looks exactly
like finite-size scaling.

`[J]` The campaign also decides where the money goes. A `L = 128`,
`N_c = 4096` central rung is ~572 core-hours for one point; a nine-`lambda`
`L = 128` scan at that `N_c` is ~3960. Getting the required `N_c` wrong by one
doubling at `L = 128` is a four-figure core-hour error.

## 5. Constraints and information structure

`[E]` `zeta = 0.35`, `T = L`, `dtau_mult = 6` certified, systematic resampling,
guided proposal with `proposal_c = zeta` and the exact RN compensator — fixed,
and this campaign changes none of them except `dtau_mult` in campaign E, where
the change is the experiment and the target measure is exactly invariant under
it. `[E]` Uncertainty comes from independent populations only. `[E]` Cost is
`rate(L, N_c) x N_c x K` with `K = ceil(2 lambda (L-1) T / dtau_mult)`, so it is
linear in `N_c` and in `lambda`, and roughly `L^2.2` in `L`. `[E]` Agents never
submit; the researcher does.

## 6. The strongest case that the problem matters

`[J]` The programme's stated purpose is to measure a phase boundary while
agnostic between `phi = 1/3`, `1/2` and `1`
(`NUMERICAL_CAMPAIGN_CHARTER.md` §0). `[E]` A finite-`N_c` displacement that
grows with `L` — and the corpus shows exactly that, `L = 64` nearly still at
`N_c = 1024` while `L = 128` is still moving by 0.06 — is **indistinguishable
from a finite-size effect by construction**. `[I]` A campaign that measures
crossings at a fixed `N_c` across several `L` therefore risks manufacturing an
exponent out of its own sampler, which is precisely the failure the numerical
charter says the next campaign must be unable to commit.

`[J]` And the cheap resolution may already be available: if the displacement is
common to both `L` it cancels exactly in `D = I_{L_1} - I_{L_2}` and does not
move the crossing at all. That would make production affordable at an `N_c`
where the absolute level is still visibly wrong. Nobody has tested it.

## 7. The strongest argument that the problem is artificial

`[J]` Written properly, because a straw man here makes the memo worthless.

**The argument.** The campaign is calibrating an *estimator's* convergence, and
the quantity anybody actually reports is a *crossing*, which is a ratio-like
object built from differences. Three independent things suggest the calibration
is over-specified:

1. `[E]` At `L = 64` the `1024 -> 2048` step is already `+0.0024`, i.e. 0.7 %
   of the value. Chasing that to `N_c = 8192` at 333 core-hours buys resolution
   on a number that is already small compared with the 5–15 % across-population
   spread any single measurement carries.
2. `[E]` The absolute-level tolerance `tau_I = 0.006` is a **worst-case**
   translation of the locator tolerance: it assumes the two curves' finite-`N_c`
   displacements do not cancel *at all*. `[I]` If they cancel even partially,
   `tau_I` is far stricter than the science needs, and the whole
   plateau-certification exercise at `L = 96` and `128` is calibrated against a
   tolerance nothing requires.
3. `[E]` The `R` required to certify `P2` at `L = 128` is ~2675 populations at
   the top step — about 13 000 core-hours at `N_c = 2048`. `[I]` A criterion
   that cannot be met at any affordable cost is arguably the wrong criterion
   rather than a discovery about the sampler.

**And a fourth, sharper form**: `[C]` if the drift were a pure additive constant
in `lambda` and in `L`, none of it would matter for anything the programme
reports, and the entire campaign would be measuring an offset that cancels.

## 8. What survives that criticism

`[J]` Points 1–3 are correct and they change the **design**, not the question.

- `[I]` Point 1 is answered by noting what the `L = 64` step actually
  establishes: nothing. Its interval is 1.7× the material tolerance, so it is
  consistent both with a plateau and with a drift twice the size of the
  tolerance. Campaign A exists because that ambiguity is unresolved, not
  because 0.7 % is intolerable. The fix is partly more `R`, not only more `N_c`,
  and the design says so.
- `[I]` Point 2 is not a reason to skip the measurement — it is the
  measurement. Campaigns B and B2 test the cancellation directly, and
  `SUCCESS_CRITERIA.yaml` explicitly permits defining production `N_c` from the
  crossing tolerance while the absolute level still fails `P2`. The criticism
  is absorbed into the design as its load-bearing hypothesis rather than
  argued against.
- `[E]` Point 3 is a genuine finding that this task should report as a result:
  **absolute-level plateau certification at `tau_I` is unreachable at `L = 128`
  with this estimator at any affordable `R`.** That is decision-relevant, it is
  derivable before any new datum, and it is why campaign D is honestly labelled
  a screening rung with no power to certify convergence.
- `[E]` The fourth form is falsifiable and is falsified in part already: the
  drift is **not** `L`-independent. `L = 64` moves by +0.002 from 1024 to 2048
  while `L = 128` moves by −0.060 from 512 to 1024. An `L`-dependent
  displacement does not cancel in a cross-`L` difference. `[I]` What remains
  open, and is exactly what campaigns B and B2 measure, is whether the
  `lambda`-dependence within one `L` is flat enough for the crossing to survive.

`[J]` What does not survive is any version of "run production now and check
later". The corpus contains one `L`-dependent, unbounded, unmodelled
displacement and no way to subtract it.
