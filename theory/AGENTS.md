# ppsQJ_m2 — Chat Agent Protocol

This file tells any Claude chat working on the ppsQJ_m2 project how to
ground itself at the start of a session and how to keep the canonical
project state (`theory/HANDOFF.md`) up to date.

The Claude.ai project instructions for this project should point here.

---

## Start-of-chat protocol

Before doing substantive technical work in a new chat, Claude **must**
read the canonical handoff:

    /Users/catlover1337/Documents/ppsQJ_m2/theory/HANDOFF.md

Mechanism: `Desktop Commander:read_file`. If Desktop Commander is
unresponsive on the first try, retry once; if still unresponsive, fall
back to `project_knowledge_search` with query "HANDOFF current status"
and warn the user that the canonical on-disk file could not be reached
(in case it has been updated since the last project-knowledge upload).

After reading HANDOFF.md, follow its "File map" pointers as needed for
the specific question being asked. The most common follow-on reads are:

- `theory/SUMMARY_2026_05_22.md` — detailed theoretical state
- `theory/qj_pps_theory_summary.md` — long-form derivations
- `theory/NLSM_FRAMEWORK.md` — Case A/B distinction
- `theory/CASE_A_IMPLEMENTATION_SPEC.md` — Case A code spec

Read these only when the conversation needs that depth. Do not pre-load
everything.

## When to skip the start-of-chat read

If the opening is purely conversational ("hi", "are you around", "thanks
for earlier"), short clarification of a previous message, or anything
that clearly doesn't require project context, skip the read. The trigger
is whether the conversation is going to do real work.

If in doubt, read. The cost is a single tool call.

## End-of-chat protocol — handoff updates

Track during the conversation whether **substantial progress** has been
made. Substantial means anything a future chat would need to know to
avoid repeating work or making decisions on stale information.

### What counts as substantial progress

- Code committed to the repo (new modules, grids, workers, scripts)
- Jobs submitted, completed, or failed — with concrete counts, timings,
  and output locations
- New empirical findings from data analysis (e.g. measured crossings,
  variance scaling, cost models)
- Theoretical claims refined, refuted, or newly established
- Decisions about methodology, scope, or priority (including decisions
  to NOT do something, with reasons)
- New analysis files, theory documents, or prompts created
- Changes to the open-questions list

### What does NOT count as substantial

- Casual exploration or brainstorming that did not reach a conclusion
- Pure code edits with no conceptual change
- One-off computations or quick sanity checks
- Anything the user explicitly flags as "just thinking out loud"

### How to update

When substantial progress has accumulated:

1. **Before the conversation ends, or earlier if context is running
   long**, propose specific edits to `theory/HANDOFF.md` as concrete
   diffs. Prefer surgical `Desktop Commander:edit_block` operations on
   the affected subsections over rewriting whole sections.

2. **Wait for user confirmation.** Never edit silently. Show the
   proposed before/after text so the user can approve or revise.

3. **After confirmation, make the edits** and confirm what changed.

4. **Suggest the git commit + push command** so the on-disk file stays
   in sync with the GitHub repo:
   ```
   cd /Users/catlover1337/Documents/ppsQJ_m2
   git add theory/HANDOFF.md  # + any other modified files
   git commit -m "HANDOFF: <one-line summary of what changed>"
   git push origin main
   ```

5. **Update the "Last major update" date** in the HANDOFF header.

### Edit style guidelines

- Keep the existing HANDOFF structure: TL;DR → Theoretical synthesis →
  Numerics status → Open questions → Operational → References → File map.
- Surgical edits, not rewrites. The handoff is read top-to-bottom by
  future chats; section identity matters.
- Tables for status (job completions, data on disk, measured values);
  prose for context and reasoning.
- Be honest about what does NOT work or what was tried and failed —
  future chats need to know the dead ends.
- New code or analysis files added in the chat get a one-line entry
  in the "File map (code added this iteration, YYYY-MM-DD)" section
  at the bottom.

## What NOT to put in the handoff

- Anything already captured in user-level memories (general project
  facts, working style, etc).
- Detailed code listings — those live in the repo files themselves.
- Long mathematical derivations — link to `theory/*.md` instead.
- Ephemeral scratch ideas that have not been validated or committed.

## Honest limits

This protocol is reflexive, not autonomous. Claude cannot:

- Edit files between messages on its own initiative.
- Update HANDOFF without the user's confirmation of proposed diffs.
- Run `git push` from Habrok (the user has an SSH key issue on the
  cluster side; pushes must come from the Mac).
- Guarantee that Desktop Commander is reachable on every call — when
  it hangs, the chat must fall back to project_knowledge_search and
  warn the user.

The user remains the gatekeeper for what lands in HANDOFF and what
gets committed to git. The protocol's value is consistency: every
substantive chat reads the handoff, every chat with substantive
progress proposes a structured update.

---

## Verifying the setup works

In a new chat in this project, the first technical question should be
followed by Claude reading HANDOFF.md without being asked. If it does
not, the Claude.ai project instructions are not picking up this file —
check that the instructions reference `theory/AGENTS.md` correctly.
