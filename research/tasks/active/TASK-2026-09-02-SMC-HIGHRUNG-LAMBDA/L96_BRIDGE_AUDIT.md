# Should an L = 96 lambda stencil (ARM B2) be included?

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA, brief §13.

## Verdict

**Rejected. Not prepared.** The brief's test is "cheap relative to the total
**and** materially improves interpretation". It fails both halves.

## It is not cheap

At L = 96 the measured Ruche rate is 11.510 ms/clone-window and `n_steps` is
892 / 922 / 953 across the three stencil lambdas.

| N_c | R | tasks | slowest task | core-hours |
|---:|---:|---:|---:|---:|
| 512  | 64 | 192 | 1.56 h | 289.9 (405.8 pessimistic) |
| 1024 | 64 | 192 | 3.02 h | 578.4 (809.8 pessimistic) |

Even the cheaper version is 290 core-hours — **24 % of the entire recommended
1213 core-hour campaign**, and more than ARM A's first rung. That is not a
bridge, it is a fourth arm. The `N_c = 1024` version, which is the one that
would actually match ARM B's population, is 578 core-hours: it alone costs more
than ARM A entire.

## It does not materially improve interpretation

The scientific question ARM B and ARM C jointly ask is:

> If we pay for a sufficiently large population, does `CMI(lambda)` become a
> statistically coherent local curve, and does that survive at production L?

That is a **contrast between a cheap L where a large population is affordable
and an expensive L where it is not**. It is answered by the two endpoints. A
midpoint at L = 96 does not sharpen the contrast; it interpolates something the
campaign is not trying to fit. Specifically:

- There is no L-dependence *law* being measured here that a third point would
  constrain. The brief forbids inventing an `N_c(L, zeta, lambda)` law from
  three cells, and adding a fourth cell does not license one either.
- The S1–S4 smoothness verdict is per-L and categorical (pass/fail). A third L
  gives a third categorical answer, not a curve.
- L = 96 is also the *one* L where the campaign already has the most
  independent-population data (ARM1's full `N_c` ladder at
  `lambda = 0.3032`, 336 populations across five rungs). What L = 96 is short of
  is not more central-lambda statistics; it is neighbouring lambdas — and those
  are only interesting once the L = 64 and L = 128 stencils have said whether
  the approach works at all.

## What would change this

If ARM B passes S1–S4 cleanly and ARM C fails them, the natural follow-up is
exactly this bridge: find where between L = 64 and L = 128 the local curve stops
being coherent at affordable `N_c`. That is a *second* campaign with a real
question, priced against a known answer, not a speculative 290 core-hours spent
tonight alongside the arms that would tell us whether to want it.

Recorded here so the option is preserved rather than forgotten. It is not
prepared, and no seeds are reserved for it.
