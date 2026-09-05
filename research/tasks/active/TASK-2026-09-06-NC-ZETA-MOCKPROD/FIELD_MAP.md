# FIELD_MAP — TASK-2026-09-06-NC-ZETA-MOCKPROD

Charter Stage 2, **scoped to what this task actually touches**. Labels `[E]`
`[I]` `[C]` `[J]`.

`[E]` **This is a methodological sizing task, not a physics task.** No external
literature is load-bearing and none was searched (`SOURCE_REGISTER.md` §1), so
the map below is of *this programme's* dependency structure, not of a field.
`[E]` External novelty is therefore `UNRESOLVED`, not favourable.

---

## Nodes

**Observables and conventions**
- `OBS-CMI-001` — the locator. Active, definition verified, region convention
  audited 2026-08-10. Guard `L % 4 == 0`, satisfied by 32, 48, 64.
- `OBS-BL-001` — **retired**; one label over two quantities. Named here only so
  that nobody re-introduces it: this task uses CMI and never `B_L`.
- `DEC-MASTER-METRIC-001` — `ESS`/`GESS`/`VIF`/founder counts are diagnostics.
  Binds `ANALYSIS_SPEC.yaml`.

**Open disputes that constrain the design**
- `DISP-PHI-001` (**open**) — `phi = 1/2` (`CB-PHI-HALF-001`) versus `phi = 1`
  (`CB-PHI-LINEAR-001`). Constrains the `lambda` grids. Not moved by this task.
- `DISP-WINDOW-001` — window drift in fitted exponents. Reason this task fits
  **no** exponent and extracts **no** law.

**Methods and software**
- Guided cloning + systematic resampling + low-rank update, `dtau_mult = 6`, in
  `pps_qj`, driven through `support/instrumented.py` (sha256 `0a33c403…`),
  validated bitwise against production by `TASK-2026-08-30-SMCSTAT`.
- `run_cell.py` (per-row executor) and `run_pack.py` (scheduling wrapper), both
  byte-identical to the predecessors'.
- `analysis/anchor_scan.py` — **known wrong** (`EV-CODE-ANCHORSCAN-001`), not
  used, blocked by the repository hook.

**Data**
- The `zeta = 0.35` Ruche corpus: 5 176 populations, `N_c` 64…8192, `L` 32…128.
  Every production population this programme has.
- `TASK-2026-08-11-ALGRD/results/b0_L*.json` — the only `zeta`-resolved timing
  anywhere. Local, small, `lambda` on the `0.51*sqrt(zeta)` line.
- `results/boundary_aggregate.csv` — carries `zeta = 0.1, 0.2, 0.7` rows but has
  no `N_c`, `T`, `dtau_mult` or resampling column and a different `lambda` grid.
  **Not poolable.** Found by the investigator; recorded in
  `REUSE_AND_DEDUP_AUDIT.md`.

**Open bottlenecks this task sits on**
- No production data at any `zeta` other than 0.35.
- No Ruche `MaxRSS` measurement anywhere; all memory sizing is local `ru_maxrss`.
- No `zeta`-resolved *production* timing; `rho(zeta)` rests on local probes.
- `R`, not `N_c`, was the binding budget at the one calibrated cell.

## Relations

```
OBS-CMI-001            --defines-->        every curve in this task
DISP-PHI-001           --constrains-->     the three lambda grids (both positions bracketed)
DEC-MASTER-METRIC-001  --constrains-->     ANALYSIS_SPEC.yaml (SEM across populations only)
NC-PLATEAU-CALIBRATION --supplies-->       rate35(L, N_c)  [MEASURED]
NC-PLATEAU-CALIBRATION --asserts-->        rate ~ N_c^0.1871
   ^ this task         --REFUTES-->        that law, using that campaign's own later rungs
ALGRD b0_L*.json       --supplies-->       rho(zeta)       [RATIO, local]
MOCK-LOWLAMBDA-EXT     --supplies-->       the zeta=0.35 anchor crossing 0.23691 [reference row]
NC-ZETA-CALIBRATION    --supplies-->       run_pack.py, the packing discipline
NC-ZETA-STAGE1         --cross-checks-->   the law-agnostic bracket rule (re-derived, not copied)
this task              --produces-->       CMI(lambda) curves at zeta in {0.10, 0.20, 0.70}
                                           and a qualitative per-rung N_c status
this task              --produces-->       NOTHING about lambda_c(zeta), phi, or N_c^req(zeta)
```

## The one methodological transfer, and why it is not a bridge

`[E]` Sequential-Monte-Carlo population sizing is a standard problem in the
particle-filter literature, and "how many particles before the estimator stops
moving" is its standard form. `[J]` No `BRIDGE_AUDIT.md` is written, because this
task **makes no cross-field claim**: it measures a number for one sampler on one
cluster and does not assert that the answer transfers, generalises, or connects
two formalisms. `[C]` Whether the finite-population drift here matches the known
`O(1/N)` particle-filter bias structure is a real question and is **not** asked
in this task; it is parked in `PARKING_LOT.md`.
