---
lifecycle: active
authoritative_for: staging plan for the 2026-08-10 migration commit
last_reviewed: 2026-08-10
---

# Git migration plan

**Nothing has been staged, committed or pushed.** This is a plan for the human.

State at capture: HEAD equals `origin/main` at `9b617fa` (2026-07-26), so nothing
committed is unpushed. Six tracked files modified, ~100 untracked paths. Full
manifest: `audit/2026-08-10/preservation/WORKTREE_STATE.md`.

**Never use `git add -A`.** The tree contains 122 untracked binary and data files
that should not enter version control.

## Category 1 — Source that should almost certainly be tracked

`analysis/var_reduction/` (19 `.py`) is the highest priority. It is the entire
2026-07-27 to 08-10 programme and currently exists in exactly one working tree.

```bash
git add analysis/var_reduction/*.py
git add analysis/_reanalyze_parity.py analysis/agg_guided.py analysis/aggregate_guided.py
git add analysis/caseA_diag.py analysis/crossings_B.py analysis/crossings_sanity.py
git add analysis/delta_B_diagnostic.py analysis/entropy_scaling_zeta0.py
git add analysis/fit_and_caseA.py analysis/fit_caseA.py analysis/fit_nu_zeta1.py
git add analysis/liouvillian/*.py
git add pps_qj/backward_pass_sector.py
git add scripts/aggregate.py scripts/make_benchmark_figures.py scripts/run_exact_benchmark.py
git add scripts/run_sweep_l4.py scripts/validate_cloning.py scripts/validate_jump_distribution.py
git add slurm/submit_nu_zeta1.sh
git add research/tools/validate_state.py
```

Review before staging the remaining loose `analysis/*.py`. Some are one-off
exploration (`analysis/tmp_explore.py`) and belong in category 5.

**`analysis/anchor_scan.py` is already tracked and is known wrong.** Do not
delete it. Add a refusal header in a separate, clearly-labelled commit so the
defect is visible in history. See `EV-CODE-ANCHORSCAN-001`.

## Category 2 — Research documentation that should be tracked

```bash
git add research/RESEARCH_CHARTER.md research/HANDOFF.md research/README.md
git add research/METADATA_RECOVERY_PLAN.md research/GIT_MIGRATION_PLAN.md
git add research/COWORK_AGENT_SPEC.md
git add research/state/                       # 59 YAML entities, text only
git add research/history/legacy/HANDOFF_pre_reconstruction_2026-08-10.md
git add theory/VARIANCE_REDUCTION.md          # cited as authority, never tracked
git add analysis/lambda_c_phi_analysis.md analysis/zeta0_benchmark_analysis.md
git add analysis/liouvillian/RESULTS.md
git add audit/2026-08-10/*.md audit/2026-08-10/*.yaml
git add audit/2026-08-10/scripts/*.py
git add audit/2026-08-10/preservation/WORKTREE_STATE.md
git add audit/2026-08-10/recovered_ephemeral/MANIFEST_TABLE.md
```

Decision needed on `audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/`
(93 files, 428 KB). It is small, it is the only surviving evidence for the
August programme, and the checksum manifest is useless without the files.
**Recommendation: track it**, on the grounds that 428 KB is cheap and the
alternative is a manifest pointing at nothing.

## Category 3 — Tests that should be tracked

```bash
git add tests/conftest.py tests/test_backward_pass_sector.py tests/test_exact_benchmark.py
```

`conftest.py` being untracked means the suite may not run from a clean clone.

## Category 4 — Should NOT normally be in Git

Do not stage. Add to `.gitignore` instead.

| pattern | count / note |
|---|---|
| `analysis/*.png` | ~40 loose figures with no provenance record |
| `scripts/validation_*.png`, `*.pdf` | generated |
| `results/ruche_pull/**` | 16,344 files, plus `.tgz` archives. Belongs in data backup, not git. |
| `results/local_boundary/**`, `results/L4/**` | generated |
| `results/ruche_pull.zip` | archive |
| `saturation_output/`, `logs/`, `outputs/` | generated |
| `analysis/parity_resolved_data.pkl` | already tracked; leave it, it is small and is cited evidence |
| `**/__pycache__/`, `.pytest_cache/`, `.DS_Store` | noise |

Proposed `.gitignore` additions:

```gitignore
results/ruche_pull/
results/ruche_pull.zip
results/local_boundary/
results/L4/
saturation_output/
logs/
analysis/*.png
scripts/*.png
scripts/*.pdf
**/__pycache__/
.DS_Store
```

`results/boundary_aggregate.csv` is a **judgement call**: 470 rows, small, and it
is the only aggregate of the July campaign. Recommendation: track it, since it is
cited by `EV-DATA-BOUNDARYCSV-001` and is a few tens of kilobytes.

## Category 5 — Uncertain, needs a human decision

| item | question |
|---|---|
| 9 `pps_qj/**/*.bak_{guided,spawn,prelowrank,prenewton}` | pre-refactor module copies sitting beside live modules. **Recommendation: move to `research/history/legacy/module_snapshots/` and track there**, so they stop shadowing live modules in grep and imports. Do not track in place. |
| `analysis/tmp_explore.py` | scratch. Delete or move to a scratch dir. |
| `notebooks/*.ipynb` | already tracked. Outputs bloat diffs. Consider `nbstripout`. |
| `analysis/fss_collapse_data.txt`, `analysis/*.json` | small analysis outputs. Track only those cited as evidence. |
| `theory/HANDOFF.md.bak_20260617`, `theory/archive/HANDOFF.md.bak*` | five superseded copies, two containing the boxed `A = 0.96`. **Recommendation: move under `research/history/legacy/handoff_backups/` and track**, so they are unambiguously historical rather than grep-reachable as current. |

## Six modified tracked files

Inspect each diff before staging. `pps_qj/parallel/worker_caseA.py` carries the
`MI_ends` observable implementation (`OBS-MIENDS-001`) and should be staged with
a message naming the observable ID.

```bash
git diff pps_qj/parallel/worker_caseA.py
git diff pps_qj/parallel/grid_caseA.py
git diff pps_qj/parallel/worker_clone_pps.py
git diff scripts/habrok/pps_scan/README.md
git diff scripts/habrok/pps_scan/submit_clone_guided_nu.sh
git diff theory/HANDOFF.md
```

`theory/HANDOFF.md` has +132 uncommitted lines (the August banners). Commit it
as-is to preserve the record, **then** freeze it per the lifecycle plan. Do not
edit it into agreement with the new state.

## Suggested commit sequence

Separate commits, so history stays readable:

1. `git commit -m "Preserve Aug 2026 variance-reduction programme: analysis/var_reduction + VARIANCE_REDUCTION.md"`
2. `git commit -m "Track missing tests (conftest, backward_pass_sector, exact_benchmark)"`
3. `git commit -m "HANDOFF: commit Aug 9/10 banners as-is before freezing"`
4. `git commit -m "Cut A: MI_ends end-to-end observable (OBS-MIENDS-001)"`
5. `git commit -m "Add 2026-08-10 reconstruction audit"`
6. `git commit -m "Add research/ knowledge plane: charter, handoff, state registries, validator"`
7. `git commit -m "gitignore: exclude generated results, figures, caches"`

Push from the Mac only. Never from Ruche.
