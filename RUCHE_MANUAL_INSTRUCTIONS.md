# Manual Ruche procedure — for the researcher

**These commands are for you to run by hand. No agent runs any of them.**

No part of this repository's automation has, or will be given, access to Ruche.
Agents never connect to the cluster, never query the scheduler, and never
submit, cancel or modify a job — at any stage, after any gate, after any
approved pilot, and regardless of what SSH configuration exists on this machine
(`research/RESOURCE_POLICY.md` §4; `CLAUDE.md` "Working rules").

Everything below was **prepared** locally and **executed by you**.

---

## Placeholders

Fill these in for your account; they are not established in the repository.

| placeholder | meaning | example |
|---|---|---|
| `<RUCHE_HOST>` | ssh alias or hostname | `ruche` or `ruche.mesocentre.universite-paris-saclay.fr` |
| `<RUCHE_USER>` | your login | `uerc` |
| `<RUCHE_CODE>` | git checkout on Ruche | `$HOME/ppsQJ_m2` |
| `<RUCHE_RESULTS>` | result tree(s) | `/gpfs/workdir/<RUCHE_USER>/pps_qj` |
| `<PRODUCTION_COMMIT>` | the commit frozen by this task | see `docs/PRODUCTION_ALGORITHM.md` |
| `<DATE>` | snapshot date | `2026-08-20` |

---

## Step 1 — update the Ruche checkout to the production commit

```bash
ssh <RUCHE_HOST>
cd <RUCHE_CODE>

# Inspect before you change anything.
git status --short
git log -n 3 --oneline

# Fetch and move onto the production commit.
git fetch origin
git checkout main
git pull --ff-only origin main
```

If `git status --short` showed local modifications you care about, stash or copy
them **before** pulling. If `pull --ff-only` refuses, the cluster checkout has
diverged; resolve that by hand rather than forcing it.

## Step 2 — confirm the commit hash

```bash
git rev-parse HEAD
git describe --tags --always --dirty
```

The hash must equal `<PRODUCTION_COMMIT>` exactly. If `describe` ends in
`-dirty`, the working tree differs from the commit — every run made from it
will be recorded with `git_dirty: true`, which is a permanent mark on that
result. Clean the tree first unless you intend that.

Confirm the production entry point resolves and agrees with itself:

```bash
python -m pps_qj.production.run \
    --L 32 --zeta 0.30 --lam 0.2793 --T 32 --Nc 64 --print-config
```

This only resolves and prints the configuration; it runs no simulation.
`deviations_from_certified` must be `[]`.

## Step 3 — copy the inventory collector to Ruche

Only needed if the checkout predates the collector; otherwise it is already at
`<RUCHE_CODE>/tools/ruche_inventory/` after step 1.

From **your laptop**:

```bash
cd /Users/catlover1337/Documents/ppsQJ_m2
scp -r tools/ruche_inventory <RUCHE_USER>@<RUCHE_HOST>:~/ruche_inventory
```

## Step 4 — run the collector

On Ruche, **on a login node**. It is read-only, needs no allocation, and
submits nothing.

```bash
ssh <RUCHE_HOST>
cd ~                       # the snapshot lands in the current directory

# Load whatever python you normally use, so package versions are recorded
# accurately. Activating the project venv gives the best .npz metadata.
module purge
module load Python/3.10.8-GCCcore-12.2.0
source "$HOME/venvs/pps_qj/bin/activate"   # if you have one

bash <RUCHE_CODE>/tools/ruche_inventory/collect_inventory.sh \
    --code-root    <RUCHE_CODE> \
    --results-root <RUCHE_RESULTS> \
    --tar
```

Add more `--results-root DIR` flags for each additional tree.

Before moving on, check its output against
`tools/ruche_inventory/expected_output.md`:

```bash
cat ruche_snapshot_<DATE>/warnings.txt
head -3 ruche_snapshot_<DATE>/result_inventory.tsv
du -sh ruche_snapshot_<DATE>
```

A `TRUNCATED` warning means the index is **not** a complete picture of the
cluster. Either raise the cap and re-run, or note it — the local importer will
otherwise present a partial index as if it were the whole story.

## Step 5 — (optional) scheduler accounting history

Only if you want the job history. This is a read-only accounting **query** over
your own completed jobs; it changes no queue state.

```bash
bash <RUCHE_CODE>/tools/ruche_inventory/collect_inventory.sh \
    --code-root    <RUCHE_CODE> \
    --results-root <RUCHE_RESULTS> \
    --slurm-history 365 \
    --tar
```

Equivalently, by hand:

```bash
sacct --user=$USER --starttime=now-365days \
      --format=JobID,JobName%40,Partition,State,ExitCode,Submit,Start,End,Elapsed,NCPUS,NNodes,ReqMem,MaxRSS,CPUTime,WorkDir%120 \
      --parsable2 --noconvert > slurm_history.txt
```

This is the one place the procedure touches the scheduler at all, it is a
`SELECT`-equivalent, and **you** are running it.

## Step 6 — package the snapshot

`--tar` already did this. To do it separately:

```bash
cd ~
tar -czf ruche_snapshot_<DATE>.tar.gz ruche_snapshot_<DATE>
ls -lh ruche_snapshot_<DATE>.tar.gz
```

Expect a few hundred KB to a few MB. If it is hundreds of MB, something bulk
was swept in — inspect before transferring, and do not just accept it.

## Step 7 — pull the bundle back to the laptop

From **your laptop** (pull, don't push — the collector never sends anything):

```bash
cd /Users/catlover1337/Documents/ppsQJ_m2
scp <RUCHE_USER>@<RUCHE_HOST>:~/ruche_snapshot_<DATE>.tar.gz \
    research/imports/ruche/
```

## Step 8 — place it and build the index

It is already in the right place. Build the index:

```bash
cd /Users/catlover1337/Documents/ppsQJ_m2
.venv/bin/python3 research/tools/import_ruche_snapshot.py \
    --all research/imports/ruche/
```

Writes `research/imports/ruche/RUCHE_DATA_INDEX.csv`. The bundle and the index
are gitignored: they are machine-local data with no provenance record, and are
not repository content.

Read the printed `reproducible fraction`. It is the share of cluster result
files that can be tied to the code that produced them.

---

## Benchmark runs (separate from the inventory)

Freezing the next numerical grid needs real Ruche timings from the production
commit, not laptop extrapolations (`NUMERICAL_CAMPAIGN_CHARTER.md` §"What is
still needed"). When you want them, run a **small** number of cells by hand:

```bash
cd <RUCHE_CODE>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m pps_qj.production.run \
    --config configs/production/benchmark_L64_z030.yaml \
    --output-dir /gpfs/workdir/$USER/pps_qj/bench_<DATE>
```

Then copy the small `.json` sidecars back the same way as the snapshot; they
carry `runtime_seconds`, `cpu_time_seconds`, the full environment block and the
genealogy diagnostics.

**No job-array script for a production campaign has been prepared, and none
should be until those benchmarks exist.** Preparing one is legitimate work for
a later task; submitting it is always yours alone.

---

## Summary of what is and is not automated

| step | who |
|---|---|
| writing the collector, importer, entry point, docs | prepared locally in this repo |
| connecting to Ruche | **you**, manually |
| updating the cluster checkout | **you**, manually |
| running the collector | **you**, manually |
| the accounting query | **you**, manually, opt-in |
| moving the bundle | **you**, manually |
| building the local index | local, offline, no cluster contact |
| submitting any job, ever | **you**, manually, always |
