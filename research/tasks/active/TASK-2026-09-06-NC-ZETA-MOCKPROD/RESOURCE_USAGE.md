# RESOURCE_USAGE — TASK-2026-09-06-NC-ZETA-MOCKPROD

**Non-authoritative. Never scientific evidence.** Its purpose is to let routing
be scored empirically after several real tasks, rather than argued about in
advance. `research/RESOURCE_POLICY.md` §5.10–5.11.

---

## Workers

| role | dispatched | model | tier | why |
|---|---|---|---|---|
| `literature` | **skipped** | — | — | Brief §14 forbids a literature review. The question has no external answer, and `RESOURCE_POLICY.md` §5.5 says not to spawn a role because it exists. **Consequence recorded: external novelty is `UNRESOLVED`, not favourable.** |
| `theory` | **skipped** | — | — | Brief §14: no broad theory investigation, no `N_c` scaling law. There is no mechanism question in scope; the one that exists is parked. |
| `numerics` | yes | `sonnet` | 1 | Role default at posture `normal`. The work was archaeology and re-derivation from stored JSONs — the Tier-1 workhorse case exactly. |
| `red-team` | yes | `opus` | 2 | Role default. Mandatory for any substantive surviving candidate. |

`[E]` No collaboration round was opened: `COLLABORATION_NOT_NEEDED` recorded,
because only one investigator ran and there was no cross-role dependency.

## Model routing

```
Tier 1 (sonnet)  1 invocation   numerics investigator
Tier 2 (opus)    1 invocation   red team
Tier 3 (best)    0 invocations
posture          normal (default; the question declared none)
escalations      NONE
degradations     none
```

`[J]` **No escalation was made and none was needed.** The task's difficulty is
in bookkeeping and arithmetic discipline, not in inference. Putting the numerics
worker on Tier 2 would have been the 2026-08-10 failure — every worker on the
strongest model, buying nothing.

## Did the cheaper model buy anything?

`[E]` **`material_value: caught_error`**, for the Tier-1 numerics worker, on two
counts:

1. `[E]` It found that the lead's draft `A_LAM = -0.35` was chosen in the
   **less** conservative direction under a comment claiming the opposite. The
   correction to `-0.50` roughly doubles the low-`lambda` rate correction at
   `zeta = 0.10`, i.e. it moves `--time` in the safe direction on the arms where
   the model extrapolates furthest. `decision_at_stake`: whether the
   `zeta = 0.10` arms carry adequate wall-time margin.
2. `[E]` It found `results/boundary_aggregate.csv`, which carries
   `zeta = 0.1/0.2/0.7` rows and which the lead's scan of
   `research/tasks/**/results/` did not cover. It is not poolable, so the
   conclusion did not change — but the lead's reuse audit would have been
   incomplete and would have said so wrongly.

`[J]` **This is the routing result worth keeping**: a Tier-1 worker asked to
re-derive from raw data caught a Tier-2-authored reasoning error. The lesson is
not "use a stronger model"; it is that an independent implementation reading the
same raw data is cheap and finds sign errors that reasoning does not.

`[E]` Where it did **not** buy a change: its Q1 recommendation (carry the flat
plateau value down to `N_c = 128, 256` rather than applying a small-batch
penalty) was **not adopted**. The penalty errs upward, affects 36 core-hours,
and changes no `--time`. That disagreement is recorded in `COST_MODEL.md` §3
rather than resolved — `confirmed_existing` would be the wrong label, because
the lead did not confirm it; the lead declined it and said why.

## Local compute executed

`[E]` **No simulation and no production compute.** T0 read-only analysis only:

- a whole-repository scan of 5 185 stored populations (inventory, rates, seeds);
- `tools/cost_model.py` refits, run on every preflight;
- `tools/negative_controls.py`, 13 injected faults into temporary copies;
- `tools/smoke_test.py`, 8 synthetic cases in temporary directories;
- `analysis/mockprod_analysis.py` run against an empty corpus, to prove it
  degrades correctly before there is anything to analyse.

`[E]` Everything above is seconds of laptop time and well inside
`RESOURCE_POLICY.md` §3. `[E]` **Nothing was submitted and nothing could be**:
preflight `P12` verifies no executable scheduler call exists anywhere in the
package.

## Was any worker unnecessary in hindsight?

`[E]` **No.** The numerics worker caught an error that would have shipped. The
red team is mandatory. `[J]` Skipping `literature` and `theory` was correct and
would have been correct even without the brief's instruction: neither could have
moved a sizing decision.

## Token accounting

`[E]` Not exposed reliably per subagent by this runtime, and **not invented**.
Two subagent invocations, one Tier 1 and one Tier 2.
