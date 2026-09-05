# SOURCE_REGISTER — TASK-2026-09-06-NC-ZETA-MOCKPROD

Charter Stage 0, **task-scoped**. **Frozen at `stage_1_problem`**: this file
holds the *scope* — which sources are load-bearing and what their inspection
level was **before** any investigator ran. Sources inspected *during* the run go
to `SOURCE_INSPECTIONS.yaml`, which is append-only and never frozen.

Labels `[E]` `[I]` `[C]` `[J]`.

---

## 1. No external literature is load-bearing in this task

`[E]` **No external prior-art search is performed anywhere in this task**, by
instruction (brief §14: "Do not launch a literature review") and by
`research/RESOURCE_POLICY.md` §5.5. `[E]` Consequently **external novelty for
every candidate is `UNRESOLVED`, not favourable**, and no statement in this task
may be supported by the absence of a search.

`[J]` This is defensible here and would not be in a physics task: the question is
"how large a clone population does *this* sampler need on *this* cluster", which
no external source can answer. It is recorded as a limitation rather than
argued away.

## 2. Load-bearing canonical state

The sources whose content, if different from what is assumed, would change the
design. All were read in full.

| id | inspection level | why load-bearing |
|---|---|---|
| `OBS-CMI-001` | `fully_inspected` | defines the locator, the region convention, the `L % 4 == 0` guard and the `lambda = alpha/(alpha+w)` parameterization. If the region convention differed, every curve would be a different quantity. |
| `DISP-PHI-001` | `fully_inspected` | **open**, two positions. The `lambda` grids must span both without privileging either; this is the dispute that left-censored `NC-ZETA-CALIBRATION`'s brackets. |
| `CB-PHI-HALF-001` | `fully_inspected` | position `phi = 1/2`; one of the two grid-coverage anchors. |
| `CB-PHI-LINEAR-001` | `fully_inspected` | position `phi = 1`; the other. |
| `DEC-MASTER-METRIC-001` | `fully_inspected` | `ESS`, `GESS`, `VIF`, founder counts are **diagnostic only**; uncertainty must come from independent populations. Binds `ANALYSIS_SPEC.yaml`. |

`[E]` `L in {32, 48, 64}` all satisfy the `OBS-CMI-001` guard `L % 4 == 0`.

`[E]` The task moves `DISP-PHI-001` in **neither** direction and closes nothing.
Its grids are coverage brackets, not measurements of a boundary law.

## 3. Load-bearing execution-plane material (provenance, never support)

`[E]` These are read for orientation and for numbers that are re-derived here
from raw data. Under Appendix A.3 they are **provenance**; nothing below is cited
as support for a scientific claim.

| path | what is taken from it | what is re-derived here rather than trusted |
|---|---|---|
| `TASK-2026-09-02-MOCK-PRODUCTION/` | the arm layout, `run_cell.py`, the certified sampler bundle | nothing is read from its results; its `NC_FACTOR` table is **not** carried over |
| `TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION/` | the 17-point `zeta = 0.35` reference curves and the `L48-L64` crossing `0.23691` (`N_c = 1024`, `R = 24`) used as the **reference row only** | — |
| `TASK-2026-09-03-NC-PLATEAU-CALIBRATION/` | the measured-memory finding (inherited formula under-predicts peak RSS by up to 1.6×) and the `N_c^0.1871` rate law | **the rate law is re-fitted from raw `wall_s` and refuted** — see `COST_MODEL.md` §3 |
| `TASK-2026-09-05-NC-ZETA-CALIBRATION/` | the packing wrapper `run_pack.py`, the `rho(zeta)` construction, the `R`-limited anchor result | `rho(zeta)` is **independently re-derived** from `ALGRD/results/b0_L*.json` |
| `TASK-2026-09-06-NC-ZETA-STAGE1/` | **nothing is taken as design authority.** Its law-agnostic bracket *rule* is re-derived independently and its brackets are used only as a cross-check | the whole bracket derivation |
| `TASK-2026-08-11-ALGRD/results/b0_L{32,64,96,128}.json` | the only `zeta`-resolved timing anywhere in the repository | ratios recomputed from the raw JSON |

`[E]` **`TASK-2026-09-06-NC-ZETA-STAGE1` is not modified by this task.**
`frozen_inputs/STAGE1_BASELINE.sha256` records the SHA-256 of all 21 of its files
plus its zip at the moment this task opened; `tools/check_predecessor.py`
re-verifies them and exits non-zero on any difference.

## 4. Load-bearing code

| path | inspection | note |
|---|---|---|
| `support/instrumented.py` | sha256 `0a33c403…`, byte-identical to the file that produced every `zeta = 0.35` production population | the integrity check in `run_cell.py` refuses to run on any other bytes |
| `pps_qj/` | tracked production package | imported, not vendored |
| `analysis/anchor_scan.py` | **NOT USED** | known wrong, `EV-CODE-ANCHORSCAN-001`; blocked by the repository hook |

## 5. What is deliberately not inspected

`[E]` `theory/**`, `audit/**`, `continuousmeasurementslatex/**`,
`research/history/**` and the untracked `analysis/global_fss*.json` family. None
bears on the sizing question, and the last group has no provenance record and is
cited by nothing in `research/state/**`.

## 6. Honest limitation

`[E]` `research/state/sources/` is empty repository-wide, so no literature source
in this project carries an inspection level at all. `[J]` This task does not fix
that and does not need it fixed; it is recorded because the charter's Stage-0
completeness condition is genuinely unmet in this repository and a task that
silently omitted the fact would be misreporting its own foundation.
