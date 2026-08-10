# Proposed replacement project instructions

Audit 2026-08-10, Stage 3. Draft only. **Not installed.**

---

## Draft

> # ppsQJ_m2 — session protocol
>
> This project studies measurement-induced phase transitions in a monitored 1D
> Kitaev chain under quantum-jump unraveling with partial post-selection.
> Repository: `/Users/catlover1337/Documents/ppsQJ_m2/`.
>
> ## Authority hierarchy
>
> For scientific content, authority runs in exactly this order. There is never
> a need to judge which of two documents is newer.
>
> 1. `research/state/**` — claims, evidence, observables, sources, decisions,
>    disputes. **The only authoritative source of scientific state.**
> 2. `research/RESEARCH_CHARTER.md` — the epistemic rules. Binding on procedure.
> 3. `research/HANDOFF.md` — navigation and what is in flight. Contains no
>    scientific values, only claim IDs.
> 4. Everything else is **non-authoritative**.
>
> Explicitly non-authoritative, and never citable as evidence:
> project memory, this instruction file, `research/history/**`, `theory/**`,
> manuscripts and Overleaf drafts, prior conversations, and anything under
> `proposals/`, `tasks/` or `runs/` that has not been merged into `state/`.
>
> These may be read for orientation and cited as *provenance* ("memory asserts
> X", "the frozen HANDOFF claimed Y"). They may never be cited as *support*.
>
> ## Start of session
>
> For anything substantive, read in this order:
> 1. `research/RESEARCH_CHARTER.md`
> 2. `research/HANDOFF.md`
> 3. Only the specific claim and evidence files the task touches.
>
> Do not preload the state. Do not read `theory/**` unless a claim points there.
>
> Skip these reads only for genuinely conversational openings.
>
> ## Citing science
>
> Reference claims by ID. Do not restate numerical conclusions from memory, from
> a manuscript, or from a previous chat. If a number is needed, read the claim
> file and cite the ID alongside it.
>
> If asked a question whose answer is a claim, give the value **and** its status,
> scope and validity range. A bare number is an incomplete answer in this
> project. Exponents and amplitudes are meaningless without their observable,
> parameterization and fitting window.
>
> ## What may never be done
>
> - **Never write `research/state/**` directly.** State changes only through a
>   proposal that has passed review and human approval.
> - **Never launch an HPC job** without an approved `experiments/<EXP-ID>.yaml`.
>   Read-only analysis of existing data needs no gate and is encouraged.
> - **Never mark a claim verified** on the strength of memory, HANDOFF, a
>   manuscript, a chat, agent agreement, or a plausible-looking derivation.
>   Verification requires discriminating evidence that could have come out the
>   other way, with preserved artifacts.
> - **Never change a scientific definition silently.** Altering an observable,
>   estimator, rate convention or parameterization mints a new `OBS-ID` and is a
>   proposal, not an edit.
> - **Never resolve a documented dispute by picking the newer or more elegant
>   side.** Disputes are resolved by evidence or they stay open.
> - **Never suppress a contradiction.** If you find evidence against the claim
>   you were asked to support, register it.
> - **Never assume a path exists because a document mentions it.** Check.
>   A file that was suggested in a previous conversation was probably never
>   created.
>
> ## Before proposing a claim
>
> Check each of these. They are the failure modes this project has actually had.
>
> - Is the **parameterization** stated? (`λ_c` and `r_c` give different fitted
>   exponents over any finite window purely from the Jacobian.)
> - Is the **observable and estimator** identified by ID?
> - For an exponent or amplitude: is the **fitting window** stated, and has
>   window sensitivity been scanned across at least three windows?
> - Is the supporting evidence **discriminating**, or does it merely reproduce a
>   number that was already known?
> - Is a **proxy metric** standing in for the master metric?
> - Is this a **single cell, single L, or single pair** result?
> - Does the claim **depend on** a claim whose status has since moved?
> - Has this already been **killed**? Check `state/decisions/` and
>   `tasks/killed/` before proposing.
>
> ## End of session
>
> If the session produced a finding, **emit a proposal before it ends**. A
> result that exists only in the conversation is the specific failure this
> protocol was built to prevent. If there is no time for a full proposal, write
> a stub under `proposals/` with `type: chat_only_evidence`, the conversation
> reference, and a recovery priority.
>
> Propose diffs and wait for confirmation. Never edit canonical files silently.
>
> ## Operational
>
> - Desktop Commander reaches the Mac. `bash` reaches the container. Two
>   separate filesystems. Never confuse them.
> - Git from the Mac only, never from HPC. Stage specific files, never
>   `git add -A`.
> - Multi-line Python goes to a script file and runs under `.venv/bin/python3`.
>   Inline REPL Python gets mangled.
> - Long jobs: `nohup ... &`, then poll. Set timeouts to at least
>   (sleep_seconds + 15) × 1000 ms.
> - Prose style: direct, math-first, no filler, no em-dashes, no semicolons.
>   Pushback is expected. Flag bad reasoning immediately.

---

## Notes on the draft

**What changed relative to the current `AGENTS.md`.** The current protocol is
procedurally sound and its start-of-chat and end-of-chat discipline is worth
keeping. Three things break it: it names `HANDOFF.md` as canonical when HANDOFF
is a session log that contradicts itself, it points at three files that exist
only under `archive/`, and it makes HANDOFF the sole sink for every kind of
information. The draft above keeps the discipline and redirects the authority.

**The pre-proposal checklist is the load-bearing part.** Every item corresponds
to a documented failure in this project, not to a generic best practice. It is
deliberately phrased as questions an agent must answer rather than principles it
must hold.

**One honest risk.** These instructions are longer than the current ones and
add friction to every session. If the friction is not worth it on a given day
the temptation will be to bypass the state registries and work in prose, which
regenerates the current situation. The mitigation is that the mandatory surface
is small: read two files, cite IDs, emit a proposal. Everything else is
machinery that runs at merge time.
