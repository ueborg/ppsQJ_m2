# SEED_LEDGER — TASK-2026-09-06-NC-ZETA-MOCKPROD

Labels `[E]` `[I]` `[C]` `[J]`.

---

## 1. The rule

```
seed = 37 000 000 + 100 000 * arm_index + row_index
```

`[E]` `arm_index` is the arm's position in `tools/build_arms.py`'s deterministic
enumeration (`zeta` ascending, then `N_c` ascending, control last);
`row_index` is the row's position in that arm's `manifest.csv`. Regenerating the
package reproduces every seed exactly.

## 2. Why 37 000 000

`[E]` Highest seed in any existing manifest in the repository: **34 900 071**
(`TASK-2026-09-05-NC-ZETA-CALIBRATION`). `[E]` The block **36 000 000+** is
reserved by `TASK-2026-09-06-NC-ZETA-STAGE1` (`OPERATIONS.md` §1) and is not
used here even though that task has allocated no manifest — a reservation that
is only honoured when the reserver has already spent it is not a reservation.
`[E]` 35 000 000–35 999 999 is left as a gap. This task starts at **37 000 000**.

## 3. Allocation

| arm | index | seeds | count |
|---|---:|---|---:|
| `M_z010_nc128` | 0 | 37 000 000 – 37 000 431 | 432 |
| `M_z010_nc256` | 1 | 37 100 000 – 37 100 431 | 432 |
| `M_z010_nc512` | 2 | 37 200 000 – 37 200 431 | 432 |
| `M_z010_nc1024` | 3 | 37 300 000 – 37 300 431 | 432 |
| `M_z010_nc2048` | 4 | 37 400 000 – 37 400 431 | 432 |
| `M_z020_nc128` | 5 | 37 500 000 – 37 500 431 | 432 |
| `M_z020_nc256` | 6 | 37 600 000 – 37 600 431 | 432 |
| `M_z020_nc512` | 7 | 37 700 000 – 37 700 431 | 432 |
| `M_z020_nc1024` | 8 | 37 800 000 – 37 800 431 | 432 |
| `M_z020_nc2048` | 9 | 37 900 000 – 37 900 431 | 432 |
| `M_z070_nc128` | 10 | 38 000 000 – 38 000 431 | 432 |
| `M_z070_nc256` | 11 | 38 100 000 – 38 100 431 | 432 |
| `M_z070_nc512` | 12 | 38 200 000 – 38 200 431 | 432 |
| `M_z070_nc1024` | 13 | 38 300 000 – 38 300 431 | 432 |
| `M_z070_nc2048` **(conditional)** | 14 | 38 400 000 – 38 400 431 | 432 |
| `E_dtau_z010` | 15 | 38 500 000 – 38 500 031 | 32 |
| | | **total** | **6 512** |

`[E]` 6 512 seeds, all distinct, none colliding with any of the 69 other
manifests in the repository (24 216 existing seeds checked). Verified per arm by
preflight `P7`; negative control `N5` injects a collision with
`TASK-2026-09-02-MOCK-PRODUCTION`'s block and requires rejection.

## 4. Independence, and what a seed is and is not

`[E]` Every uncertainty this task reports is an **across-population** SEM over
the `R = 16` distinct seeds of a cell. `ESS`, `GESS`, `VIF` and founder counts
are diagnostics only — `DEC-MASTER-METRIC-001`, canonical.

`[I]` Distinct seeds give **independent populations**, not independent
trajectories within a population: clones inside one population share ancestry,
which is exactly why the within-clone spread may never be used as a standard
error. `VIF` is reported so that the gap between the two is visible, and never
so that it can be divided out.

`[E]` **`R` is matched at every cell that is compared**, in every arm, and
preflight `P1` fails otherwise. Negative control `N6` removes one population
from one cell and requires rejection.

## 5. The one cross-arm reuse

`[E]` The discretisation control reuses the `dtau_mult = 6` leg from
`M_z010_nc512` (seeds 37 200 000+) rather than recomputing it under
38 500 000+. `[J]` The saving is 16 populations and is not the reason: running
the same cell twice under two seed blocks and then comparing the two halves as
if they were one measurement is the error being avoided. `[E]` The dependency is
recorded in `ANALYSIS_SPEC.yaml` and enforced by preflight `P13`, which requires
the control arm to carry **only** `dtau_mult in {3, 12}`.
