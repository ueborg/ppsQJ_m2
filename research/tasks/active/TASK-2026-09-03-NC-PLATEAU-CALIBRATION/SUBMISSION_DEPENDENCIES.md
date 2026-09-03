# SUBMISSION_DEPENDENCIES — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

What may be queued together, what must wait for what, and why. **No agent
submits any of it** (`research/RESOURCE_POLICY.md` §4).

---

## 1. The immediate group has NO internal scientific dependencies

`[E]` All seventeen immediate arms are independent job arrays. Nothing in any of
them consumes the output of any other. They may be queued in any order and in
any combination, and a failure in one does not invalidate another.

`[E]` The only couplings are **scheduling** couplings:

| coupling | effect | what to do |
|---|---|---|
| `D_L128_nc2048` is the wall-clock long pole at ~31 h (~44 h pessimistic) | it decides when the campaign finishes | **queue it first**, before anything else |
| `A_L64_nc8192` is second at ~6.9 h (~9.7 h) | second-longest | queue second |
| the twelve `cpu_med` arms total 2 448 tasks | if `%64` is granted per-array they finish in hours; if it is a shared total they serialise | check the accounting first (runbook §2) |
| `%64` may be a per-array grant or an account-wide total | changes elapsed by an order of magnitude | **the `%N` cap is the only number safe to hand-edit** |

`[J]` If the allocation grants 64 slots in **total**, submit `D`, then `A`, then
`C`, then `B`, then `B2` and `E` — largest per-task first, so the long jobs are
not queued behind 2 448 short ones.

## 2. Analysis dependencies

`[E]` The frozen analysis runs on whatever is present and says what is missing.
But the **verdicts** have real prerequisites:

| verdict | needs |
|---|---|
| campaign A plateau P1–P5 | all three A arms complete; P5 additionally needs the reused `N_c = 1024` rung |
| shape H1/H2/H3 | all three B arms complete at ≥4 of the 7 `lambda` |
| **fully matched locator** | **B and B2 both complete at the same `N_c`** — this is the one real cross-arm dependency in the campaign |
| one-sided locator | B alone, plus the reused `N_c = 1024` low-`L` curves |
| `L = 96` / `L = 128` ladders | C / D complete |
| E1 vs E2 | both E arms complete; a single `N_c` gives half the answer and the analysis says so |

`[I]` **Consequence**: if the researcher drops `B2`, campaign B still returns
its shape answer and the **one-sided** locator diagnostic, but the fully matched
cross-`L` `N_c` comparison — decision branch L-1, the branch that could define
production `N_c` from the crossing tolerance — cannot be evaluated at all. That
is the whole cost of dropping B2, stated plainly.

## 3. The conditional group

`[E]` Every conditional arm is blocked by three independent mechanisms
(`CONDITIONAL_SUBMISSION.md`). Their dependencies:

```
campaign D returns
   |
   +-- D-1 (still drifting) --> cond_D2_L128_nc4096 becomes eligible
   |                                   |
   |                                   +-- if it then passes --> cond_M128_nc4096 eligible
   |
   +-- D-3 (small and inside tolerance) --> cond_M128_nc2048 eligible
   |
   +-- D-2 (small, wide interval) --> NOTHING becomes eligible. More R, or the locator route.

campaign C returns
   |
   +-- C-1 --> EXACTLY ONE of cond_M96_nc1024 / cond_M96_nc2048
   +-- C-2 or C-3 --> neither

cond_LOWZ_nc64 and cond_LOWZ_nc256
   |
   +-- no dependency on any of the above. Blocked by POLICY, not by data:
       the programme wants zeta = 0.35 understood first. Release BOTH or NEITHER
       -- the pre-registered kill criterion needs both population sizes.
```

`[E]` **Mutually exclusive pairs.** `cond_M96_nc1024` / `cond_M96_nc2048` and
`cond_M128_nc2048` / `cond_M128_nc4096` are each the same physical scan at two
population sizes. Running both members of a pair is duplicated compute, not a
robustness check, and each README says so.

## 4. What must NOT be treated as a dependency

`[E]` Campaign E does not gate anything in the immediate group and nothing gates
it. It is 1.9 % of the cost and both of its outcomes kill a mechanism, so it
should be queued with the rest rather than held back for a result it does not
need.

`[E]` Campaign B2 does not gate campaign B. B2's value is entirely in the joint
analysis.

`[E]` **No result from any arm may be merged into `research/state/**` without
passing red-team review and the human gate.** Completion is not adjudication.
