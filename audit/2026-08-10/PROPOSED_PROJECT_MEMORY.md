# Proposed replacement project memory

Audit 2026-08-10, Stage 3. Draft only. **Not installed.**

## Design intent

The current memory is a ~2026-06-03 snapshot that has drifted for ten weeks and
now re-injects a corrected error into every new conversation. The replacement is
deliberately minimal: identity, locations, and pointers. **No amplitudes, no
exponents, no run statuses, no pending-action lists.** Anything that can change
belongs in the state registries, not here.

Rough target: under 250 words. The current memory is roughly 1,400.

---

## Draft

> **Project ppsQJ_m2.** Utku (MSc quantum engineering, supervisor Dganit Meidan,
> Ben-Gurion University) studies measurement-induced phase transitions in a
> monitored 1D Kitaev/Majorana chain under quantum-jump unraveling with partial
> post-selection parameter ζ. Two parameter cuts: Cut A (α+γ=1, w=0, self-dual,
> class D) and Cut B (γ=0, α+w=1, class DIII, the main campaign). The work
> extends KMR (SciPost Phys 14, 031) and LMR (PRX 15, 021020), which use quantum
> state diffusion rather than quantum jumps. Collaborators G. Kells and
> A. Romito, author list not finalised.
>
> **Canonical scientific state lives in `research/state/`**, not in memory, not
> in HANDOFF, not in any manuscript, and not in past conversations. Every
> scientific claim has an ID. Cite IDs, never restate numbers from memory.
>
> **Start any substantive session by reading `research/RESEARCH_CHARTER.md`,
> then `research/HANDOFF.md`.** Load individual claim and evidence files on
> demand. Do not preload the whole state.
>
> **Infrastructure.** Mac repository `/Users/catlover1337/Documents/ppsQJ_m2/`
> (GitHub `ueborg/ppsQJ_m2`, branch `main`), venv `.venv/bin/python3`. HPC is
> Ruche (Paris-Saclay, user `ercetinut`); Habrok (RUG) was retired 2026-07-07.
> Results and papers under `~/Downloads/01_M1_Internship/`. Desktop Commander
> reaches the Mac; `bash` reaches the container. These are separate filesystems
> and must never be confused. Git operations from the Mac only, never from HPC,
> and never `git add -A`.
>
> **Working style.** Direct, math-first. No filler, no em-dashes, no semicolons
> in prose. Explicit pushback welcomed and expected. Flag bad reasoning
> immediately. Epistemic tags [V]/[P]/[O]/[X] plus an evidence class are used
> throughout. Multi-line Python goes to a script file and is run with the venv
> interpreter rather than inlined. Long jobs launch with `nohup ... &` and are
> polled.
>
> **No agent may launch an HPC campaign without an approved experiment
> specification, and no agent writes `research/state/` directly.**

---

## What was deliberately removed, and why

| removed | reason |
|---|---|
| `A = 0.96 ± 0.05`, `φ = 0.502 ± 0.026` | wrong, and the inverse of the established correction |
| `y_λ = 1/2`, `y_ζ = 1`, `Δ_ζ = 1`, `ν = 2` | superseded or unmeasured; belongs in claims with status |
| `λ_c(1) ≈ 0.5 matches Carollo` | attribution known incorrect since 2026-06-06 |
| `λ_c ~ A√ζ` as a settled result | contested; there is a live φ = 1 alternative |
| "use 1/√L since ν = 2" | circular on an unmeasured ν |
| the eliminated-scenarios list | belongs in `state/decisions/` as kill records |
| the NLSM and θ₁ paragraphs | both rest on Δ_ζ = 1 |
| pending actions (Run A/B/C, Cut A implementation) | ~3 months stale; Cut A is in fact implemented and run |
| `paper/main.tex` | never existed. An agent's suggestion recorded as fact. |
| `clone_aggregate(1).pkl` as the main aggregate | path dead, superseded |
| the variance-reduction summary | belongs in claims and decisions |

## What was deliberately kept

Identity, supervisor, collaborators, the two cuts and their symmetry classes,
the KMR/LMR lineage, filesystem and HPC facts, the two-filesystem warning, git
hygiene, working style, and the two hard prohibitions. All of these are stable
on a timescale of months. The cut definitions are the one borderline case: they
are definitional rather than empirical, so they belong here, but if either cut
is ever redefined the memory must change with it.

## What memory got right and should survive in some form

The observation that **"König–Brouwer 2014" is a hallucinated citation** is
genuinely valuable and cost real effort to establish. It does not belong in
memory. It belongs in `state/sources/` as a source entry with
`attribution_verified: false` and a note, so that it is discoverable by the
literature agent rather than by luck.
