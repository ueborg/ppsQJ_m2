# NOVELTY_GATE — TASK-2026-09-06-NC-ZETA-MOCKPROD

Duplicate gate, run **before** any candidate could be called new. Labels `[E]`
`[I]` `[C]` `[J]`.

**No candidate in this task is claimed to be novel.** The gate is run anyway,
because the rule is that novelty language requires a classification, not that a
classification is only needed when novelty is claimed.

---

## 1. Queries run

`.venv/bin/python3 research/tools/find_predecessors.py "<candidate statement>"`

| query | closest canonical record | score | classification |
|---|---|---:|---|
| minimum clone population `N_c` at which finite-population effects stop moving the CMI curves | `OBS-BLPROD-001` (observable, active) | 0.440 | **no predecessor found** in canonical state |
| finite-`N_c` drift of the cross-`L` CMI crossing depends on `zeta` | `CASEA-DRIFT-001` (claim, provisional) | 0.323 | **no predecessor found** in canonical state |
| discretisation `dtau_mult` dependence of CMI at low `zeta` | `CB-AMP-096-001` (claim, **withdrawn**) | 0.306 | **no predecessor found** in canonical state |
| per-clone-window runtime scales with `N_c` as a power law | `VR-CLOSE-001` (claim, provisional) | 0.214 | **no predecessor found** in canonical state |

`[E]` The top hits are observable definitions and unrelated `zeta`-drift claims;
none states anything about clone-population sizing. `[E]` Dead records were
boosted, not filtered, and three of the four top hits are dead records — which
is the tool working, not a match.

## 2. The gate's blind spot, stated plainly

`[E]` `find_predecessors.py` searches **canonical state only**. It cannot see
`research/tasks/`, `theory/`, `audit/` or `history/`. **This task's real
predecessors are all in the execution plane and the tool returns none of them.**
A "no predecessor found" from this tool is therefore evidence about canonical
state and nothing else.

## 3. The execution-plane predecessors, enumerated by hand

| predecessor | relation | classification |
|---|---|---|
| `TASK-2026-09-05-NC-ZETA-CALIBRATION` | same physical question, certification form; reached Gate A with verdict `Reformulate` because its brackets were left-censored by the historical `sqrt(zeta)` grid | **rediscovery of the question, not of the answer** — that task measured nothing |
| `TASK-2026-09-06-NC-ZETA-STAGE1` | the repaired certification design: law-agnostic brackets, TOST at `tau_lambda = 0.004`, `zeta in {0.05, 0.10, 0.20, 0.35}` | **strictly stronger question, not run.** This task is a deliberately weaker and cheaper sibling and says so in `CHARTER.md` §2 |
| `TASK-2026-09-03-NC-PLATEAU-CALIBRATION` | `N_c` ladder at `zeta = 0.35` only, to `N_c = 8192` | **corroboration and correction**: its data is reused to refute its own `N_c^0.1871` rate law (`COST_MODEL.md` §3) |
| `TASK-2026-09-02-MOCK-PRODUCTION` + `TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION` | the `zeta = 0.35` curves and the anchor crossing | **reference row**, quoted, never recomputed, never pooled |

`[J]` **The honest classification of this task as a whole is `replication at new
`zeta` of a measurement previously made only at `zeta = 0.35`, under a
deliberately weakened question.** It is not a discovery, it is not novel, and
nothing in it is presented as either.

## 4. What would make this a duplicate rather than a replication

`[E]` If any exact-compatible population already existed at
`zeta in {0.10, 0.20, 0.70}` for a cell in the design. `[E]` The whole-repository
scan finds **zero** (`REUSE_AND_DEDUP_AUDIT.md`), and preflight `P14` re-checks
it per arm against every stored result, with negative control `N12` proving the
check fires.

## 5. What "no predecessor found" means here, and what it never means

`[E]` **It is a statement about the searches actually performed, and about
nothing else.** In this task those searches were: four
`find_predecessors.py` queries over canonical state (§1), and the hand
enumeration of execution-plane predecessors (§3).

`[E]` **No external prior-art search was performed anywhere in this task.** Brief
§14 forbids a literature review and `research/RESOURCE_POLICY.md` §5.5 forbids
spawning a role because it exists; the `literature` worker was skipped and the
skip is recorded in `TASK_MANIFEST.yaml` and `RESOURCE_USAGE.md`.

`[E]` **Therefore external novelty for every candidate is `UNRESOLVED`, not
favourable.** Under charter §4.2 the absence of a search is evidence about the
search, never about the literature, and under §3 novelty is the researcher's
call and not an agent's. Nothing in this task is claimed to be new, and if it
were, this gate would not support the claim.

`[J]` The classification of the task as a whole remains what §3 says:
**replication at new `zeta`** of a measurement previously made only at
`zeta = 0.35`.

## 6. Closest predecessor, per candidate

`[E]` Required per candidate, not per task. The candidates are design options
(`CANDIDATES.md`), so the closest predecessor of each is the design that made
the same choice. All classifications below are made under the searches actually
performed (§5): canonical state via `find_predecessors.py`, plus the hand
enumeration of execution-plane predecessors. **No external search was
performed**, so no classification below is a statement about the literature.

| candidate | closest predecessor | classification |
|---|---|---|
| **C1** — the committed design | `TASK-2026-09-03-NC-PLATEAU-CALIBRATION`, the same `N_c` ladder and the same observable at `zeta = 0.35` | **replication** at three new `zeta`, under a deliberately weakened question |
| **C2** — `R = 24` | `TASK-2026-09-02-MOCK-PRODUCTION` and `TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION`, both `R = 24` at `zeta = 0.35` | **replication** of an established `R` choice; killed here on cost, not on correctness |
| **C3** — drop `zeta = 0.70` | `TASK-2026-09-05-NC-ZETA-CALIBRATION`, which scoped its high-`zeta` points as diagnostic cells rather than full screens for the same cost reason | **rediscovery** of that reasoning; killed as stated, its content preserved as the conditional interlock |
| **C4** — drop `L = 32` | `TASK-2026-09-03-NC-PLATEAU-CALIBRATION`'s `B2_L32_*` arms, which exist precisely as supporting information | **rediscovery**, and the predecessor's answer stands: `L = 32` is kept |
| **C5** — reuse the historical corpus | `TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION`, which faced the same temptation over `N_c = 2048` and refused it on the same grounds | **rediscovery** of a refusal; killed again here, on a metadata gap the predecessor did not face |

`[E]` **No candidate is classified `no predecessor found`**, and none is claimed
novel. `[J]` That three of five are rediscoveries of prior refusals is the
expected shape for a successor task, and is not a lesser outcome: confirming
that a design choice was already made and already justified prevents it being
re-litigated a third time.
