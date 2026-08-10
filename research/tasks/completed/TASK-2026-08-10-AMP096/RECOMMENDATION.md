# RECOMMENDATION — TASK-2026-08-10-AMP096

## Verdict

**Infrastructure first.**

## Basis

- **Stage 8 verdict: `killed`**, four of five candidates, six mandated attacks
  at `severity: fatal`. `REDTEAM.yaml` validates clean and reports
  `lead_summary_seen: false`.
- **A–H**: B (mechanistic contribution) is NONE and A (consequential
  bottleneck) is WEAK. No dimension is strong on the physics; G (informative
  failure) and H (infrastructure value) are strong, and both are properties of
  the *run*, not of a scientific result. Per §5 these do not compensate.
- **Slop warnings**: three fatal — 6 and 7 killed candidates C3 and C5; 11 is a
  fatal finding about existing manuscript material, detected rather than
  committed.
- **Kill criterion** (set in `CHARTER.md` before evidence): the withdrawal *does*
  have a traceable basis, so the task was not killed outright. But the only
  surviving outputs are metadata corrections and one single-dataset check.

Not `Stop`, because there is concrete, cheap, non-physics work with a named
artifact. Not `Reformulate`, because the sharper question that emerged
(is p = 1/2 still load-bearing for `CB-PHI-HALF-001`?) belongs to a task opened
deliberately against `DISP-PHI-001`, not to a re-run of this one. Not `Pursue`,
because nothing survived that would justify compute.

## What canonical state currently permits us to conclude

Referenced by ID; no numbers restated here that live in claim files.

- `CB-AMP-096-001` remains **`withdrawn`**. That verdict is correct and is
  unchanged by this task.
- The *reason* recorded on that claim is not supported by the primary document
  and should be softened to a two-part record: generated May 2026 as an
  L-extrapolated lambda_c fit, then reinterpreted as an r_c prefactor on
  2026-06-10. Both halves are true; the current single-cause note is not.
- `CB-AMP-001` remains **`provisional`**, and its falsifier **did not fire**.
  The apparent counter-evidence dissolved under attack.
- No Cut B amplitude can be promoted on current data, and no L-extrapolation is
  defensible at present L ranges.

## Disputes touched

- **`DISP-PHI-001`** — evidence leans **neither way**. C2 appeared to bear on it
  and was killed as a rediscovery of `METH-EXTRAP-001`. **NOT CLOSED.**
- **`DISP-WINDOW-001`** — reinforced in spirit (the window-edge artifact is a
  fresh instance) but no new discriminating evidence. **NOT CLOSED.**

Neither dispute moves. Recording that plainly is the point.

## Proposed state changes

Listed only. Files are in `proposed/`. **Not applied.** Human gate required.
None alters a scientific value, an exponent, an amplitude or a status.

1. `CB-AMP-096-001` — correct `evidence_note`; soften `confidence_basis`.
2. `OBS-BLKMR-001` — correct `recomputable_from`; record the partial
   single-dataset discharge of `open_task`.
3. `SRC-KMR-2023` — **withdraw the unsupported `invoked_for` statement**;
   upgrade the B_L definition from title-only to body-verified.
4. `SRC-FULGA-2012` — record the identification and `inspection_level`.

## Next human decision

Accept or reject the four proposals above, and decide whether to open a task on
whether p = 1/2 is still load-bearing for `CB-PHI-HALF-001`.

**STOPPED AT HUMAN GATE A.** No experiment was designed (nothing survived to
justify one), no compute was run, and `research/state/**` is unchanged.
