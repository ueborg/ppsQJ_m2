#!/usr/bin/env bash
#
# ruche_inventory — READ-ONLY inventory collector for the Ruche HPC cluster.
#
# ============================ SAFETY CONTRACT ============================
# This script is run MANUALLY BY THE RESEARCHER, on Ruche, from a login node.
# It is read-only with respect to everything except its own output directory.
#
# It NEVER:
#   * submits, cancels or modifies a job (no sbatch/srun/scancel/scontrol)
#   * writes, moves or deletes anything outside $OUT_DIR
#   * modifies git state (no fetch/pull/checkout/commit; only read subcommands)
#   * installs packages
#   * copies files off Ruche (no scp/rsync/curl/wget; no network egress)
#   * reads credentials, keys, or the general environment block
#
# Everything it writes lands under ./ruche_snapshot_<date>/ in the CWD.
# =========================================================================
#
# Usage:
#   ./collect_inventory.sh [--code-root DIR] [--results-root DIR]... [--out DIR]
#                          [--max-depth N] [--tar] [--slurm-history [DAYS]]
#
# Defaults are placeholders; override them for your Ruche layout.

set -u   # deliberately NOT -e: a missing optional tool must not abort the run

VERSION="1.0"
DATE_TAG="$(date +%Y-%m-%d)"

CODE_ROOT="${RUCHE_CODE_ROOT:-$HOME/ppsQJ_m2}"
RESULTS_ROOTS=()
OUT_DIR=""
MAX_DEPTH=8
DO_TAR=0
DO_SLURM=0
SLURM_DAYS=180

# Cap on how many filesystem entries we will index, so a pathological tree
# cannot produce a multi-gigabyte inventory.
MAX_FILES="${RUCHE_MAX_FILES:-200000}"
# Cap on script/config entries, so the bundle stays small.
MAX_SCRIPTS="${RUCHE_MAX_SCRIPTS:-5000}"

while [ $# -gt 0 ]; do
  case "$1" in
    --code-root)     CODE_ROOT="$2"; shift 2 ;;
    --results-root)  RESULTS_ROOTS+=("$2"); shift 2 ;;
    --out)           OUT_DIR="$2"; shift 2 ;;
    --max-depth)     MAX_DEPTH="$2"; shift 2 ;;
    --max-files)     MAX_FILES="$2"; shift 2 ;;
    --max-scripts)   MAX_SCRIPTS="$2"; shift 2 ;;
    --tar)           DO_TAR=1; shift ;;
    --slurm-history)
      DO_SLURM=1
      if [ $# -ge 2 ] && printf '%s' "$2" | grep -qE '^[0-9]+$'; then
        SLURM_DAYS="$2"; shift 2
      else
        shift
      fi ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ ${#RESULTS_ROOTS[@]} -eq 0 ]; then
  # Placeholder defaults. Override with --results-root.
  RESULTS_ROOTS=(
    "${RUCHE_RESULTS_ROOT:-/gpfs/workdir/$USER}"
    "$HOME/pps_qj"
  )
fi

[ -n "$OUT_DIR" ] || OUT_DIR="ruche_snapshot_${DATE_TAG}"

mkdir -p "$OUT_DIR" || { echo "cannot create $OUT_DIR" >&2; exit 1; }
WARN="$OUT_DIR/warnings.txt"
: > "$WARN"

warn() { echo "WARNING: $*" | tee -a "$WARN" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

echo "ruche_inventory v$VERSION — read-only snapshot into $OUT_DIR"

# -------------------------------------------------------------------------
# README
# -------------------------------------------------------------------------
{
  echo "ruche_inventory snapshot"
  echo "========================"
  echo "collector_version : $VERSION"
  echo "collected_at_utc  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "collected_by      : ${USER:-unknown}"
  echo "hostname          : $(hostname 2>/dev/null)"
  echo "code_root         : $CODE_ROOT"
  echo "results_roots     : ${RESULTS_ROOTS[*]}"
  echo "max_depth         : $MAX_DEPTH"
  echo "max_files         : $MAX_FILES"
  echo "max_scripts       : $MAX_SCRIPTS"
  echo "slurm_history     : $DO_SLURM (days=$SLURM_DAYS)"
  echo
  echo "This snapshot is READ-ONLY metadata. It contains no simulation arrays,"
  echo "no credentials, and no general environment dump. It is NOT complete"
  echo "scientific provenance: it records what exists on the cluster, not what"
  echo "any of it means."
  echo
  echo "Files:"
  echo "  git_info.txt          code checkout state at code_root"
  echo "  environment.txt       host, python, packages, modules, CPU, SLURM env"
  echo "  file_inventory.tsv    every indexed file: path, size, mtime, type"
  echo "  result_inventory.tsv  cheap parameter metadata from recognised outputs"
  echo "  scripts_inventory.tsv checksums of scripts and configs under code_root"
  echo "  slurm_history.tsv     sacct export (only if --slurm-history was given)"
  echo "  warnings.txt          anything that could not be collected"
  echo "  checksums.txt         sha256 of every file in this snapshot"
} > "$OUT_DIR/README.txt"

# -------------------------------------------------------------------------
# Git / code state  (read-only subcommands only)
# -------------------------------------------------------------------------
{
  echo "# pwd at collection time"
  pwd
  echo
  echo "# code_root: $CODE_ROOT"
  if [ -d "$CODE_ROOT" ]; then
    if have git && git -C "$CODE_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
      echo "## git rev-parse HEAD";        git -C "$CODE_ROOT" rev-parse HEAD 2>&1
      echo "## git branch --show-current"; git -C "$CODE_ROOT" branch --show-current 2>&1
      echo "## git describe --tags --always --dirty"
      git -C "$CODE_ROOT" describe --tags --always --dirty 2>&1
      echo "## git status --short"
      git -C "$CODE_ROOT" status --short 2>&1 | head -300
      echo "## git log -n 20 --oneline --decorate"
      git -C "$CODE_ROOT" log -n 20 --oneline --decorate 2>&1
      echo "## git remote -v"
      git -C "$CODE_ROOT" remote -v 2>&1
    else
      echo "NOT A GIT CHECKOUT (or git unavailable)"
    fi
  else
    echo "CODE_ROOT DOES NOT EXIST"
  fi
} > "$OUT_DIR/git_info.txt" 2>&1
[ -d "$CODE_ROOT" ] || warn "code root $CODE_ROOT does not exist"

# -------------------------------------------------------------------------
# Environment  (no credentials; SLURM_* and thread vars only)
# -------------------------------------------------------------------------
{
  echo "## host"
  echo "hostname: $(hostname 2>/dev/null)"
  echo "uname:    $(uname -a 2>/dev/null)"
  [ -r /etc/os-release ] && { echo "## os-release"; cat /etc/os-release; }

  echo
  echo "## cpu"
  if have lscpu; then
    lscpu 2>/dev/null | grep -Ei 'model name|socket|core|thread|cpu\(s\)|mhz|cache' | head -20
  elif [ -r /proc/cpuinfo ]; then
    grep -m1 'model name' /proc/cpuinfo 2>/dev/null
    echo "cpu_count: $(grep -c ^processor /proc/cpuinfo 2>/dev/null)"
  else
    echo "no cpu info available"
  fi

  echo
  echo "## memory"
  [ -r /proc/meminfo ] && grep -E 'MemTotal|MemAvailable' /proc/meminfo 2>/dev/null

  echo
  echo "## python"
  for py in python3 python; do
    if have "$py"; then
      echo "$py: $(command -v $py)"
      "$py" -c 'import sys; print("version:", sys.version.split()[0])' 2>&1
      "$py" - <<'PYIN' 2>&1
mods = ["numpy", "scipy", "yaml", "pandas", "h5py", "numba", "tqdm"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"  {m}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"  {m}: NOT AVAILABLE ({type(exc).__name__})")
try:
    import numpy
    cfg = numpy.__config__.show(mode="dicts")
    blas = cfg.get("Build Dependencies", {}).get("blas", {})
    print("  numpy blas:", blas.get("name"), blas.get("version"))
except Exception:
    pass
PYIN
      break
    fi
  done

  echo
  echo "## virtualenv"
  echo "VIRTUAL_ENV: ${VIRTUAL_ENV:-<unset>}"

  echo
  echo "## modules"
  if have module; then
    module list 2>&1 | head -40
  elif have lmod; then
    lmod list 2>&1 | head -40
  else
    echo "no module system on PATH (expected on a login shell without it sourced)"
  fi
  echo "LOADEDMODULES: ${LOADEDMODULES:-<unset>}"

  echo
  echo "## scheduler identity (allow-listed variables only)"
  for v in SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_JOB_NAME \
           SLURM_JOB_PARTITION SLURM_CPUS_PER_TASK SLURM_SUBMIT_HOST \
           SLURM_CLUSTER_NAME; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && echo "$v=$val"
  done
  echo "(empty above is normal: the collector runs on a login node, not in a job)"

  echo
  echo "## thread pinning"
  for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
           NUMEXPR_NUM_THREADS; do
    eval "val=\${$v:-}"
    echo "$v=${val:-<unset>}"
  done

  echo
  echo "## scheduler availability (version query only — NOTHING is submitted)"
  if have sinfo; then sinfo --version 2>&1; else echo "sinfo: not on PATH"; fi
  if have sacct; then sacct --version 2>&1; else echo "sacct: not on PATH"; fi

  echo
  echo "## filesystem capacity at the inventoried roots"
  for r in "$CODE_ROOT" "${RESULTS_ROOTS[@]}"; do
    [ -d "$r" ] && df -h "$r" 2>/dev/null | tail -n +2 | sed "s|^|$r  |"
  done
} > "$OUT_DIR/environment.txt" 2>&1

# -------------------------------------------------------------------------
# Filesystem inventory
# -------------------------------------------------------------------------
FILE_TSV="$OUT_DIR/file_inventory.tsv"
printf 'root\trelative_path\tsize_bytes\tmtime_utc\textension\tdirectory\tlikely_campaign\n' > "$FILE_TSV"

# Portable mtime formatter: GNU stat, then BSD stat, then find -printf.
_stat_fmt() {
  if stat -c '%Y' "$1" >/dev/null 2>&1; then stat -c '%s|%Y' "$1"
  elif stat -f '%z|%m' "$1" >/dev/null 2>&1; then stat -f '%z|%m' "$1"
  else echo "|"; fi
}

guess_campaign() {
  # Heuristic only. Derived from the path, never from file contents.
  case "$1" in
    *caseA*|*case_a*)          echo "caseA" ;;
    *guided_prod*)             echo "guided_prod" ;;
    *guided_highL*|*highL*)    echo "guided_highL" ;;
    *guided_ladder*|*ladder*)  echo "guided_ladder" ;;
    *guided*)                  echo "guided" ;;
    *boundary*)                echo "boundary" ;;
    *doob*)                    echo "doob" ;;
    *dense*)                   echo "dense" ;;
    *refine_smallz*)           echo "refine_smallz" ;;
    *refine*)                  echo "refine" ;;
    *rescue*)                  echo "rescue" ;;
    *bench*)                   echo "benchmark" ;;
    *production*|*prod*)       echo "production" ;;
    *)                         echo "unknown" ;;
  esac
}

# GNU find supports -printf, which gives size+mtime in ONE pass with no
# per-file subprocess. On a scratch tree with 10^5 files that is the difference
# between seconds and an hour. Ruche is Linux, so this is the expected path;
# the portable fallback exists so the collector also runs on a Mac.
if find /dev/null -maxdepth 0 -printf '' >/dev/null 2>&1; then
  FAST_FIND=1
else
  FAST_FIND=0
  warn "find(1) lacks -printf (BSD find?); using the slow portable path"
fi

_stat_fmt() {
  if stat -c '%s|%Y' "$1" 2>/dev/null; then :
  elif stat -f '%z|%m' "$1" 2>/dev/null; then :
  else echo "|"; fi
}

for ROOT in "${RESULTS_ROOTS[@]}"; do
  if [ ! -d "$ROOT" ]; then
    warn "results root $ROOT does not exist -- skipped"
    continue
  fi
  echo "  indexing $ROOT ..."
  if [ "$FAST_FIND" -eq 1 ]; then
    # size \t epoch-mtime \t path, then annotate in one awk pass.
    find "$ROOT" -xdev -maxdepth "$MAX_DEPTH" -type f \
         ! -path '*/.git/*' ! -name '*.pyc' -printf '%s\t%T@\t%p\n' 2>/dev/null \
    | head -n "$MAX_FILES" \
    | awk -F'\t' -v root="$ROOT" 'BEGIN{OFS="\t"}
        {
          size=$1; mt=int($2); path=$3
          rel=path; sub("^" root "/", "", rel)
          n=split(rel,parts,"/"); base=parts[n]
          ext=""; if (base ~ /\./) { m=split(base,bp,"."); ext=bp[m] }
          dir=rel; if (index(dir,"/")>0) { sub("/[^/]*$","",dir) } else { dir="." }
          lc=tolower(path); camp="unknown"
          if (lc ~ /casea|case_a/)        camp="caseA"
          else if (lc ~ /guided_prod/)    camp="guided_prod"
          else if (lc ~ /guided_highl|highl/) camp="guided_highL"
          else if (lc ~ /guided_ladder|ladder/) camp="guided_ladder"
          else if (lc ~ /guided/)         camp="guided"
          else if (lc ~ /boundary/)       camp="boundary"
          else if (lc ~ /doob/)           camp="doob"
          else if (lc ~ /dense/)          camp="dense"
          else if (lc ~ /refine_smallz/)  camp="refine_smallz"
          else if (lc ~ /refine/)         camp="refine"
          else if (lc ~ /rescue/)         camp="rescue"
          else if (lc ~ /bench/)          camp="benchmark"
          else if (lc ~ /production|prod/) camp="production"
          print root, rel, size, strftime("%Y-%m-%dT%H:%M:%SZ", mt, 1), ext, dir, camp
        }' >> "$FILE_TSV"
  else
    n_indexed=0
    find "$ROOT" -xdev -maxdepth "$MAX_DEPTH" -type f \
         ! -path '*/.git/*' ! -name '*.pyc' 2>/dev/null |
    while IFS= read -r f; do
      n_indexed=$((n_indexed + 1))
      [ "$n_indexed" -gt "$MAX_FILES" ] && { warn "file cap $MAX_FILES reached under $ROOT -- inventory TRUNCATED"; break; }
      sm="$(_stat_fmt "$f")"; size="${sm%%|*}"; mt="${sm##*|}"
      if [ -n "$mt" ]; then
        mtime="$(date -u -d "@$mt" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
                 || date -u -r "${mt%%.*}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "$mt")"
      else
        mtime=""
      fi
      rel="${f#"$ROOT"/}"; base="$(basename -- "$f")"
      case "$base" in *.*) ext="${base##*.}" ;; *) ext="" ;; esac
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ROOT" "$rel" "$size" "$mtime" "$ext" "$(dirname -- "$rel")" \
        "$(guess_campaign "$f")"
    done >> "$FILE_TSV"
  fi
done

echo "  indexed $(($(wc -l < "$FILE_TSV") - 1)) files"

# -------------------------------------------------------------------------
# Scripts / configs checksums (under code_root only)
# -------------------------------------------------------------------------
SCRIPT_TSV="$OUT_DIR/scripts_inventory.tsv"

if have sha256sum;   then SHA="sha256sum"
elif have shasum;    then SHA="shasum -a 256"
else SHA=""; fi

PY=""
for c in python3 python; do have "$c" && { PY="$c"; break; }; done

if [ -d "$CODE_ROOT" ] && [ -n "$PY" ]; then
  # Hashing in Python avoids one subprocess per file; on a repo-sized tree
  # that is minutes versus seconds.
  "$PY" - "$CODE_ROOT" "$SCRIPT_TSV" "$MAX_SCRIPTS" <<'PYIN' 2>>"$WARN"
import hashlib, os, sys, time
root, out = sys.argv[1], sys.argv[2]
max_scripts = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
EXTS = (".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".cfg")
SKIP = {".git", ".venv", "venv", "site-packages", "__pycache__", "node_modules",
        ".pytest_cache", ".mypy_cache",
        # Bulk OUTPUT trees. Their .json/.yaml files are results, not code;
        # indexing them here would balloon the bundle and duplicate what
        # result_inventory.tsv already covers.
        "results", "outputs", "output", "data", "logs", "log", "figures",
        "analysis_output", "saturation_output", "validations", "notebooks"}
MAX_HASH_BYTES = 8 * 1024 * 1024   # do not hash a huge json blob
rows = []
truncated = False
root = os.path.abspath(root)
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in SKIP]
    if dirpath[len(root):].count(os.sep) >= 6:
        dirnames[:] = []
    if truncated:
        break
    for fn in filenames:
        if not fn.endswith(EXTS):
            continue
        full = os.path.join(dirpath, fn)
        try:
            st = os.stat(full)
        except OSError:
            continue
        digest = ""
        if st.st_size <= MAX_HASH_BYTES:
            try:
                h = hashlib.sha256()
                with open(full, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                digest = h.hexdigest()
            except OSError:
                digest = ""
        if len(rows) >= max_scripts:
            truncated = True
            break
        rows.append((
            os.path.relpath(full, root), st.st_size,
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
            digest,
        ))
rows.sort()
with open(out, "w") as fh:
    fh.write("relative_path\tsize_bytes\tmtime_utc\tsha256\n")
    for r in rows:
        fh.write("%s\t%d\t%s\t%s\n" % r)
print("  checksummed %d script/config file(s)%s"
      % (len(rows), " (TRUNCATED at cap)" if truncated else ""))
if truncated:
    print("WARNING: scripts_inventory truncated at %d entries" % max_scripts,
          file=sys.stderr)
PYIN
else
  printf 'relative_path\tsize_bytes\tmtime_utc\tsha256\n' > "$SCRIPT_TSV"
  [ -d "$CODE_ROOT" ] || warn "code root missing -- scripts_inventory.tsv is empty"
  [ -n "$PY" ] || warn "no python -- scripts_inventory.tsv is empty"
fi

# -------------------------------------------------------------------------
# Result metadata (cheap header reads only)
# -------------------------------------------------------------------------
PY=""
for c in python3 python; do have "$c" && { PY="$c"; break; }; done
HERE="$(cd "$(dirname -- "$0")" && pwd)"

if [ -n "$PY" ] && [ -f "$HERE/collect_results_metadata.py" ]; then
  echo "  extracting result metadata ..."
  "$PY" "$HERE/collect_results_metadata.py" \
      --file-inventory "$FILE_TSV" \
      --out "$OUT_DIR/result_inventory.tsv" \
      --warnings "$WARN" 2>>"$WARN"
else
  warn "python or collect_results_metadata.py unavailable — no result_inventory.tsv"
  printf 'root\trelative_path\tformat\tstatus\n' > "$OUT_DIR/result_inventory.tsv"
fi

# -------------------------------------------------------------------------
# SLURM accounting history — OPT-IN ONLY, and read-only
# -------------------------------------------------------------------------
if [ "$DO_SLURM" -eq 1 ]; then
  if have sacct; then
    echo "  exporting sacct history (last $SLURM_DAYS days, read-only query) ..."
    sacct --user="$USER" \
          --starttime="now-${SLURM_DAYS}days" \
          --format=JobID,JobName%40,Partition,State,ExitCode,Submit,Start,End,Elapsed,NCPUS,NNodes,ReqMem,MaxRSS,CPUTime,WorkDir%120 \
          --parsable2 --noconvert \
      > "$OUT_DIR/slurm_history.tsv" 2>>"$WARN" \
      || warn "sacct query failed"
    # --parsable2 emits '|' — normalise to TSV for consistency with the rest.
    if [ -s "$OUT_DIR/slurm_history.tsv" ]; then
      tr '|' '\t' < "$OUT_DIR/slurm_history.tsv" > "$OUT_DIR/slurm_history.tsv.tmp" \
        && mv "$OUT_DIR/slurm_history.tsv.tmp" "$OUT_DIR/slurm_history.tsv"
    fi
  else
    warn "sacct not on PATH — no slurm_history.tsv"
  fi
else
  echo "  (skipping slurm history; pass --slurm-history to include it)"
fi

# -------------------------------------------------------------------------
# Checksums of the snapshot itself
# -------------------------------------------------------------------------
(
  cd "$OUT_DIR" || exit 0
  if [ -n "$SHA" ]; then
    # shellcheck disable=SC2086
    find . -type f ! -name checksums.txt -print0 \
      | xargs -0 $SHA 2>/dev/null > checksums.txt
  else
    : > checksums.txt
  fi
)

# -------------------------------------------------------------------------
# Optional tarball
# -------------------------------------------------------------------------
if [ "$DO_TAR" -eq 1 ]; then
  if have tar; then
    tar -czf "${OUT_DIR}.tar.gz" "$OUT_DIR" 2>>"$WARN" \
      && echo "  wrote ${OUT_DIR}.tar.gz ($(du -h "${OUT_DIR}.tar.gz" 2>/dev/null | cut -f1))"
  else
    warn "tar not available"
  fi
fi

echo
echo "Done. Snapshot: $OUT_DIR"
echo "  size: $(du -sh "$OUT_DIR" 2>/dev/null | cut -f1)"
echo "  warnings: $(wc -l < "$WARN" | tr -d ' ') line(s) — see $WARN"
echo
echo "Nothing was submitted, cancelled, modified or transferred off this cluster."
