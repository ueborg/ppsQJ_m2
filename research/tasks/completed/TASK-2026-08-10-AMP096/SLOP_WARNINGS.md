# SLOP WARNINGS — TASK-2026-08-10-AMP096           (Charter §6)

Explicit verdict on all twelve, assessed against the surviving output after
Stage 8. verdict: clear | flagged | fatal

| # | warning | verdict | reasoning |
|---|---|---|---|
| 1 | established method on a routine new dataset/model/topology/application | flagged | Applying the existing crossing estimator to `boundary_aggregate.csv` was routine, and the result was an artifact. |
| 2 | two known techniques combined with no nontrivial interaction | clear | No combination attempted. |
| 3 | metric is a monotone transform, weighted sum, or rename | clear | No new metric. C4 tested two existing locators against each other rather than minting a third. |
| 4 | another constraint on a familiar optimization, no conceptual change | clear | No optimization involved. |
| 5 | architecture swap for a small benchmark gain | clear | Not applicable to this project. |
| 6 | theorem whose assumptions largely encode the conclusion | **fatal** | C3 is `1^phi = 1`: the single-power premise encodes the conclusion, and that premise is denied by `CB-WINDOW-001`. Killed on A2. |
| 7 | regime constructed because it makes the method outperform | **fatal** | C5's window included edge cells where B_L had collapsed to numerical zero and where no crossing is defined. Restricting to the clean interior reproduces `CB-AMP-001` exactly. The apparent discrepancy *was* the window. |
| 8 | weak, obsolete, or informationally disadvantaged baseline | flagged | C1 and C5 compared an L-extrapolated historical number against a non-extrapolated modern one. The estimator mismatch was the comparison, not a finding about it. |
| 9 | computational scale treated as scientific depth | clear | T0 only. 450k subagent tokens produced one surviving scoped result, and that is reported plainly rather than dressed up. |
| 10 | runnable code treated as evidence a problem exists | clear | The opposite: the known-wrong scan script is on disk, still runs, and is blocked from producing evidence. |
| 11 | silo-breaking claimed from terminology alone | **fatal, historical** | The M1 manuscript's "twice the diffusive prefactor" is exactly this: a cross-field comparison resting on a symbol collision (`r_c`), where the cited paper contains no such statement. **Not committed by this task — detected by it.** |
| 12 | paper drafted around an artifact before the claim exists | flagged, historical | The 0.96 reached a submitted M1 report before the claim was stable. |

## Reformulation if flagged

Warnings 6, 7 and 8 killed candidates C3, C5 and C1 respectively; they are
recorded, not reformulated. Warning 11 is a **finding about existing material**,
and its reformulation is a proposed correction to `SRC-KMR-2023.invoked_for`,
filed in `proposed/`.
