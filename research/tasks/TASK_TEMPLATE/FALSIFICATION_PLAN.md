# FALSIFICATION_PLAN — <TASK-ID>      (Charter Stage 4, PRE-SPECIFIED)

**This file is written BEFORE the checks are run and FREEZES at
`stage_3_candidates`.** It says what will be attempted and what would kill the
candidate. It must contain **no outcomes** — those go to
`FALSIFICATION_RESULTS.md`. Validator check `F1` rejects a results/outcome/done
column here, because a plan that records its own answers is not a
pre-specification.

## Checks to attempt

| # | check | what would count as failing | which candidate it can kill |
|---|---|---|---|
| 1 | counterexample search | | |
| 2 | limiting and degenerate cases | | |
| 3 | smallest analytically transparent model | | |
| 4 | could the effect arise trivially? | | |
| 5 | strongest plausible baseline | | |
| 6 | regimes where the method SHOULD fail | | |
| 7 | artifact of a definition, normalization, or the simulator? | | |
| 8 | mechanism validation separated from performance | | |

## Pre-specified kill criteria

State these as commitments, not predictions. One line per candidate.

- **C1 dies if:**
- **C2 dies if:**

## Smallest decisive test

Complete the sentence: "if the observable takes value X in range R, H1 is
excluded; if X', H2 is excluded."

## What would make this test uninformative

Stated in advance, so that discovering it later is not reinterpreted as a result.
