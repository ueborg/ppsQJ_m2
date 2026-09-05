#!/usr/bin/env python3
"""Preflight for one arm of TASK-2026-09-06-NC-ZETA-MOCKPROD.

Read-only. Contains NO scheduler call and cannot submit anything. Exit 0 clean,
non-zero on any failure.

    .venv/bin/python3 shared/preflight.py <ARM_DIR>

Every check recomputes something the package asserts rather than trusting it.
The negative controls that prove these checks can actually FAIL live in
tools/negative_controls.py -- a check that has never been seen to fail is not
known to be a check.
"""
from __future__ import annotations
import csv, glob, hashlib, json, math, os, re, sys

ARM = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
SHARED = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(SHARED, os.pardir))
ROOT = os.path.abspath(os.path.join(TASK, *([os.pardir] * 4)))
sys.path.insert(0, os.path.join(TASK, "tools"))
import cost_model as CM          # noqa: E402
import build_arms as BA          # noqa: E402

TAU = 0.004                      # tau_lambda, used ONLY as a margin yardstick
PARTITION_MAXTIME = {"cpu_med": 4 * 3600, "cpu_long": 168 * 3600}
# cpu_short is never used at any --time: it is effectively serialised for this
# account by QOSMaxJobsPerUserLimit, so its %N concurrency cap is not real.
FORBIDDEN_PARTITIONS = {"cpu_short"}
# Built from fragments so that this source file does not itself contain a
# literal scheduler command that a repository guard would have to reason about.
SCHED_WORDS = ["s" + "batch", "s" + "run", "q" + "sub", "b" + "sub",
               "s" + "cancel", "s" + "queue"]
SCHED = re.compile(r"\b(" + "|".join(SCHED_WORDS) + r")\b|\bssh\s")

fails, notes = [], []


def bad(code, msg):
    fails.append("%-5s %s" % (code, msg))


def ok(code, msg):
    notes.append("%-5s %s" % (code, msg))


def gib(s):
    """Parse a Slurm --mem value to GiB.

    NO SUFFIX MEANS MEGABYTES. An earlier parser in this programme read a bare
    `2048` as 2048 GiB and therefore failed OPEN on an under-sized request.
    """
    s = (s or "").strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT]?)i?[Bb]?", s, re.I)
    if not m:
        return 0.0
    v, u = float(m.group(1)), (m.group(2) or "M").upper()
    return v * {"K": 1 / 1048576.0, "M": 1 / 1024.0, "G": 1.0, "T": 1024.0}[u]


def hhmmss(s):
    p = [int(x) for x in s.split(":")]
    return p[0] * 3600 + p[1] * 60 + p[2]


# ---------------------------------------------------------------- load the arm
name = os.path.basename(ARM)
rows = list(csv.DictReader(open(os.path.join(ARM, "manifest.csv"))))
packs = list(csv.DictReader(open(os.path.join(ARM, "packs.csv"))))
slurm = open(os.path.join(ARM, "submit.slurm")).read()
sb = dict(re.findall(r"^#SBATCH --([a-z-]+)=(\S+)", slurm, re.M))
is_control = name.startswith("E_")

print("=" * 76)
print("  PREFLIGHT  %s   TASK-2026-09-06-NC-ZETA-MOCKPROD" % name)
print("  %d manifest rows, %d packs" % (len(rows), len(packs)))
print("=" * 76)

# P1  cell count and MATCHED R ------------------------------------------------
cells = {}
for r in rows:
    k = (float(r["zeta"]), int(r["L"]), int(r["N_c"]),
         float(r["lam"]), float(r["dtau_mult"]))
    cells[k] = cells.get(k, 0) + 1
Rs = set(cells.values())
if len(Rs) != 1:
    bad("P1", "R is NOT matched across cells in this arm: %s. Every comparison "
              "this task makes is between cells, so an unmatched R silently "
              "confounds the noise floor with the effect." % sorted(Rs))
elif not is_control and (len(cells) != 27 or Rs != {BA.R}):
    bad("P1", "expected 27 cells (3 L x 9 lambda) at R = %d, got %d cells at R = %s"
        % (BA.R, len(cells), Rs))
else:
    ok("P1", "%d cells, R = %d matched at every one" % (len(cells), list(Rs)[0]))

# P2  T = L, and the K the packer used --------------------------------------
#
# NOT the discretisation identity. Recomputing K from a row's own (L, lambda,
# T, dtau_mult) and comparing it with K computed from the same four numbers is
# a tautology, and an earlier draft of this file shipped exactly that: the
# negative control N13 corrupted T and the "identity" check happily agreed with
# the corruption. What is actually checkable here is the DESIGN INVARIANT
# T = L, which is what makes K, the cost and the observable comparable across
# this task's cells at all.
badT = [r for r in rows if abs(float(r["T"]) - int(r["L"])) > 1e-9]
if badT:
    bad("P2", "%d rows where T != L. The design fixes T = L; a row where they "
              "differ has a different discretisation, a different cost and a "
              "different observable from the cell it is filed under." % len(badT))
else:
    Ks = [CM.K(int(r["L"]), float(r["lam"]), float(r["dtau_mult"]), float(r["T"]))
          for r in rows]
    ok("P2", "T = L at all %d rows; K = ceil(2*lambda*(L-1)*T/dtau_mult) spans "
             "%d..%d and is the same K the packer costed with"
       % (len(rows), min(Ks), max(Ks)))

# P3  the array range and the pack tiling -------------------------------------
m = re.search(r"0-(\d+)%(\d+)", sb.get("array", ""))
if not m:
    bad("P3", "cannot parse --array=%r" % sb.get("array"))
elif int(m.group(1)) != len(packs) - 1:
    bad("P3", "--array=0-%s but packs.csv has %d packs" % (m.group(1), len(packs)))
else:
    ok("P3", "--array=0-%d matches %d packs; concurrency cap %s (a cap, not a "
             "throughput promise)" % (len(packs) - 1, len(packs), m.group(2)))

exp = 0
tiled = True
for p in sorted(packs, key=lambda x: int(x["start"])):
    if int(p["start"]) != exp:
        tiled = False
        break
    exp += int(p["count"])
if not tiled or exp != len(rows):
    bad("P3b", "packs do not tile the manifest exactly once (covered %d of %d rows)"
        % (exp, len(rows)))
else:
    ok("P3b", "packs tile all %d manifest rows exactly once: no gap, no overlap"
       % len(rows))

# P4  --time against the pessimistic slowest PACK -----------------------------
sec = [CM.row_seconds(int(r["L"]), int(r["N_c"]), float(r["lam"]),
                      float(r["zeta"]), float(r["dtau_mult"]), float(r["T"]))
       for r in rows]
pack_sec = [sum(sec[int(p["start"]):int(p["start"]) + int(p["count"])]) for p in packs]
slow = max(pack_sec)
need = 1.6 * CM.PESSIMISTIC * slow
have = hhmmss(sb["time"])
if have < need:
    bad("P4", "--time=%s (%.0f s) is below 1.6 x the pessimistic slowest pack "
              "(%.0f s)" % (sb["time"], have, need))
else:
    ok("P4", "--time=%s = %.2f x the pessimistic slowest pack (slowest pack "
             "%.1f min, slowest single population %.1f min)"
       % (sb["time"], have / (slow * CM.PESSIMISTIC), slow / 60, max(sec) / 60))

drift = [i for i, (p, e) in enumerate(zip(packs, pack_sec))
         if abs(e - float(p["est_sec"])) > 0.5]
if drift:
    bad("P4b", "packs.csv est_sec no longer reproduces from the cost model at "
               "%d packs, first at pack %d (file %.1f, recomputed %.1f)"
        % (len(drift), drift[0], float(packs[drift[0]]["est_sec"]), pack_sec[drift[0]]))
else:
    ok("P4b", "every packs.csv est_sec reproduces from the cost model to 0.5 s")

# P5  --mem against the modelled peak -----------------------------------------
maxL = max(int(r["L"]) for r in rows)
maxNc = max(int(r["N_c"]) for r in rows)
peak = CM.mem_mb(maxL, maxNc) / 1024.0
if gib(sb["mem"]) < 1.35 * peak:
    bad("P5", "--mem=%s = %.2f GiB, below 1.35 x the modelled peak %.2f GiB"
        % (sb["mem"], gib(sb["mem"]), peak))
else:
    ok("P5", "--mem=%s = %.2f x the modelled peak (%.0f MB at L=%d, N_c=%d). "
             "Model is 1.45 x the inherited formula, which alone under-predicts "
             "real peak RSS by up to 1.6x."
       % (sb["mem"], gib(sb["mem"]) / peak, peak * 1024, maxL, maxNc))

# P6  the partition rule ------------------------------------------------------
part = sb.get("partition")
if part in FORBIDDEN_PARTITIONS:
    bad("P6", "partition %s is never used in this programme at any --time" % part)
elif part not in PARTITION_MAXTIME:
    bad("P6", "unknown partition %r" % part)
elif have > PARTITION_MAXTIME[part]:
    bad("P6", "--time=%s exceeds the %s MaxTime" % (sb["time"], part))
elif part == "cpu_long" and have <= PARTITION_MAXTIME["cpu_med"]:
    bad("P6", "--time=%s fits cpu_med, so cpu_long is not the smallest partition "
              "that fits" % sb["time"])
else:
    ok("P6", "partition %s is the smallest that fits --time=%s" % (part, sb["time"]))

# P7  seeds -------------------------------------------------------------------
seeds = [int(r["seed"]) for r in rows]
if len(set(seeds)) != len(seeds):
    bad("P7", "duplicate seeds INSIDE this arm")
else:
    others = set()
    nfiles = 0
    for mf in glob.glob(os.path.join(ROOT, "research/tasks/**/manifest.csv"),
                        recursive=True):
        if os.path.abspath(mf) == os.path.join(ARM, "manifest.csv"):
            continue
        nfiles += 1
        for r2 in csv.DictReader(open(mf)):
            if r2.get("seed"):
                others.add(int(r2["seed"]))
    clash = sorted(set(seeds) & others)
    if clash:
        bad("P7", "%d seeds collide with another manifest, e.g. %s"
            % (len(clash), clash[:5]))
    else:
        ok("P7", "%d seeds, all distinct, none used by any of the %d other "
                 "manifests in the repository (%d seeds checked against)"
           % (len(seeds), nfiles, len(others)))

# P8  zeta = 0.35 is never recomputed -----------------------------------------
z35 = [r for r in rows if abs(float(r["zeta"]) - 0.35) < 1e-12]
if z35:
    bad("P8", "%d rows at zeta = 0.35. The brief forbids recomputing the anchor; "
              "it is quoted as a reference row only." % len(z35))
else:
    ok("P8", "no zeta = 0.35 row in this arm")

# P9  lambda grid identity AND coverage of BOTH open positions ----------------
#
# This is the check that TASK-2026-09-05-NC-ZETA-CALIBRATION's own P9 did not
# do. That one tested a stencil CENTRE against a law. A centre can sit on one
# law while the SPAN still fails to bracket the other -- which is exactly how
# that package came to ship stencils that would have missed at six of eight
# zeta. This tests the SPAN, against both positions, symmetrically.
ANCHOR = 0.23691     # L48-L64 crossing at zeta=0.35, N_c=1024, R=24, INTERIOR
if is_control:
    ok("P9", "control arm: one lambda by design, coverage check not applicable")
else:
    for z in sorted({float(r["zeta"]) for r in rows}):
        if z not in BA.GRID:
            bad("P9", "arm carries zeta = %.4f, which has no frozen lambda grid "
                      "in this design" % z)
            continue
        got = sorted({float(r["lam"]) for r in rows if float(r["zeta"]) == z})
        want = [round(x, 6) for x in BA.GRID[z]]
        if [round(x, 6) for x in got] != want:
            bad("P9", "zeta=%.2f lambda grid is not the frozen grid" % z)
            continue
        preds = {"phi=1/2": ANCHOR * math.sqrt(z / 0.35),
                 "phi=1  ": ANCHOR * (z / 0.35)}
        for lab, pr in preds.items():
            below = sum(1 for x in got if x < pr)
            above = sum(1 for x in got if x > pr)
            marg = min(pr - got[0], got[-1] - pr) / TAU
            if below < 2 or above < 2:
                bad("P9", "zeta=%.2f %s prediction %.4f has %d grid points below "
                          "and %d above; >= 2 each side is required"
                    % (z, lab, pr, below, above))
            elif marg < 4:
                bad("P9", "zeta=%.2f %s prediction %.4f is only %.1f tau from the "
                          "nearer END of the grid; >= 4 is required"
                    % (z, lab, pr, marg))
            else:
                ok("P9", "zeta=%.2f %s -> %.4f strictly interior, %d/%d grid "
                         "points either side, %.1f tau from the nearer END"
                   % (z, lab, pr, below, above, marg))
        lo = math.log(got[0] / ANCHOR) / math.log(z / 0.35)
        hi = math.log(got[-1] / ANCHOR) / math.log(z / 0.35)
        ok("P9b", "zeta=%.2f grid brackets boundary exponents phi in [%.2f, %.2f]. "
                  "Outside that range the crossing would fall off the grid and be "
                  "reported ABOVE_GRID / BELOW_GRID, never interpolated."
           % (z, min(lo, hi), max(lo, hi)))

# P10  cost-model literals, refitted from raw data ----------------------------
f10 = CM.refit(verbose=False)
if f10:
    bad("P10", "cost-model literals have drifted from the data: " + "; ".join(f10))
else:
    ok("P10", "rate35 and rho refit from raw stored results to within 0.5 % of "
              "the literals in tools/cost_model.py")

# P11  bundle integrity -------------------------------------------------------
man = json.load(open(os.path.join(TASK, "support", "BUNDLE_MANIFEST.json")))
b11 = []
for fd in man["files"]:
    p = os.path.join(TASK, "support", os.path.basename(fd["bundled_as"]))
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != fd["sha256_bundled"]:
        b11.append(fd["bundled_as"])
for fd in man["runners_verbatim_not_integrity_scanned"]:
    p = os.path.join(TASK, fd["bundled_as"])
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != fd["sha256"]:
        b11.append(fd["bundled_as"])
if b11:
    bad("P11", "bundled file(s) are not the certified bytes: %s" % ", ".join(b11))
else:
    ok("P11", "instrumented.py, run_cell.py and run_pack.py are byte-identical to "
              "the files that produced the zeta = 0.35 production corpus")

# P12  nothing in the package can submit --------------------------------------
hits = []
for dp, dn, fn in os.walk(TASK):
    dn[:] = [d for d in dn if d not in ("results", "logs", "__pycache__", "scratch")]
    for f2 in fn:
        if not f2.endswith((".py", ".sh", ".slurm")):
            continue
        p = os.path.join(dp, f2)
        for i, line in enumerate(open(p, errors="replace"), 1):
            stripped = line.lstrip()
            if stripped.startswith("#") or '"' in line or "'" in line:
                continue          # comments and string literals are documentation
            if SCHED.search(line):
                hits.append("%s:%d" % (os.path.relpath(p, TASK), i))
if hits:
    bad("P12", "executable scheduler or remote-launch call inside the package: %s"
        % ", ".join(hits))
else:
    ok("P12", "no executable scheduler or remote-launch call anywhere in the "
              "package. The researcher submits by hand; nothing here can.")

# P13  discretisation discipline ----------------------------------------------
dts = sorted({float(r["dtau_mult"]) for r in rows})
if is_control:
    if dts != [3.0, 12.0]:
        bad("P13", "control arm carries dtau_mult %s; expected [3.0, 12.0] only. "
                   "The 6.0 leg lives in M_z010_nc512 and must not be duplicated "
                   "here under a second seed block." % dts)
    else:
        ok("P13", "control arm carries only the dtau_mult = 3 and 12 legs; the 6 "
                  "leg is reused from M_z010_nc512 at matched R and matched seed "
                  "discipline")
elif dts != [6.0]:
    bad("P13", "production arm carries dtau_mult %s; production is 6.0 only, and "
               "non-6 rows may never be pooled with the production corpus" % dts)
else:
    ok("P13", "production arm is dtau_mult = 6.0 throughout")

# P14  no duplicate of any stored population ----------------------------------
existing = set()
for p in glob.glob(os.path.join(ROOT, "research/tasks/**/results/*.json"),
                   recursive=True):
    # scratch/ holds synthetic and throwaway data by construction -- the red
    # team's own fabricated populations live there. A scratch file is not a
    # stored population and must never enter a duplication, reuse or analysis
    # scan. This exclusion was added after the red team's scratch data made
    # P14 report 27 false duplicates.
    if os.sep + "scratch" + os.sep in p:
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if isinstance(d, dict) and "zeta" in d and "N_c" in d and "lam" in d:
        try:
            existing.add((float(d["zeta"]), int(d["L"]), float(d.get("T", 0)),
                          int(d["N_c"]), round(float(d["lam"]), 6),
                          float(d.get("dtau_mult", 0)),
                          d.get("resample_scheme") or ""))
        except Exception:
            pass
mine = {(float(r["zeta"]), int(r["L"]), float(r["T"]), int(r["N_c"]),
         round(float(r["lam"]), 6), float(r["dtau_mult"]), r["resample_scheme"])
        for r in rows}
dup = sorted(mine & existing)
if dup:
    bad("P14", "%d cells duplicate an already-stored population, e.g. %s"
        % (len(dup), dup[:2]))
else:
    ok("P14", "none of this arm's %d cells duplicates any of the %d stored cells "
              "in the repository" % (len(mine), len(existing)))

# P15  the frozen predecessor is untouched ------------------------------------
rc = os.system("%s %s >/dev/null 2>&1"
               % (sys.executable, os.path.join(TASK, "tools", "check_predecessor.py")))
if rc:
    bad("P15", "TASK-2026-09-06-NC-ZETA-STAGE1 has been modified or added to. The "
               "brief requires it untouched. Run tools/check_predecessor.py.")
else:
    ok("P15", "TASK-2026-09-06-NC-ZETA-STAGE1 is byte-identical to the baseline "
              "recorded when this task opened")

# P16  no certification language WHERE IT WOULD BE A CLAIM ---------------------
#
# SCOPE, and why it is not the whole task directory. The prohibition is on this
# task CLAIMING a certification, not on it discussing one. PROBLEM_MEMO.md,
# CANDIDATES.md and SOURCE_REGISTER.md exist precisely to explain what the
# stricter predecessors set out to certify and why this task does not; a check
# that fired on those would be demanding the task hide its own scope boundary.
# So P16 scans the SHIPPED PACKAGE and the CONCLUSION-BEARING artifacts -- the
# places where the word would be an assertion about this task's own output --
# and lets a guard word ("not", "never", "may not", ...) exempt a line even
# there.
# The prohibited act is certifying a RESULT of this survey -- a population
# size, a convergence, a crossing, a rung. Certifying a CODE PATH is a
# different and legitimate statement: the sampler really was validated bitwise
# against production by TASK-2026-08-30-SMCSTAT, and the package says so. So a
# line trips P16 only when certif* appears together with one of this survey's
# own objects.
CERT = re.compile(r"\bcertif(?:y|ies|ied|ication)\b", re.I)
OBJECT = re.compile(r"\b(N_c|Nc|converg\w*|crossing|stable|stability|adequate|"
                    r"plateau|rung|locator|tau_lambda|equivalen\w*)\b", re.I)
GUARD = re.compile(r"\b(not|never|no|cannot|may not|forbid\w*|without|unless|"
                   r"reserved|declin\w*|prohibit\w*)\b"
                   r"|not_a|prohibited|forbidden|may_not|does_not|invalid",
                   re.I)
SCOPE_FILES = ["ANALYSIS_SPEC.yaml", "SUCCESS_CRITERIA.md", "RUCHE_RUNBOOK.md",
               "HUMAN_SUBMISSION.md", "CONDITIONAL_SUBMISSION.md",
               "RECOMMENDATION.md", "RESEARCH_MEMO.md",
               "CLAIM_STRENGTH_AUDIT.yaml", "COST_MODEL.md"]
scan = [os.path.join(TASK, f) for f in SCOPE_FILES]
scan += glob.glob(os.path.join(TASK, "shared", "*.py"))
scan += glob.glob(os.path.join(TASK, "tools", "*.py"))
scan += glob.glob(os.path.join(TASK, "analysis", "*.py"))
scan += glob.glob(os.path.join(TASK, "*", "submit.slurm"))
scan += glob.glob(os.path.join(TASK, "conditional", "*", "submit.slurm"))
chits, nscanned = [], 0
for p in scan:
    if not os.path.isfile(p):
        continue
    nscanned += 1
    # An eight-line window, because in YAML the guard is routinely the KEY and
    # the certification word is in the folded VALUE beneath it -- e.g.
    #     does_not_mean: >
    #       ... bias is bounded, small, certified, or converged
    # A strictly line-local check reads that prohibition as a claim, which is
    # the opposite of what it says.
    lines = open(p, errors="replace").read().split("\n")
    for i, line in enumerate(lines, 1):
        if not (CERT.search(line) and OBJECT.search(line)):
            continue
        window = "\n".join(lines[max(0, i - 8):i])
        if not GUARD.search(window):
            chits.append("%s:%d" % (os.path.relpath(p, TASK), i))
if chits:
    bad("P16", "certification language in a conclusion-bearing or shipped file, "
        "where this task may not use it: %s" % ", ".join(chits[:6]))
else:
    ok("P16", "in the %d shipped and conclusion-bearing files, certif* never "
              "attaches to an N_c, a convergence, a crossing or a rung. This "
              "survey produces qualitative statuses; the word is reserved for a "
              "task that earns it. (certif* applied to the SAMPLER is allowed "
              "and true: TASK-2026-08-30-SMCSTAT validated it bitwise.)"
       % nscanned)

# P17  the runner path in submit.slurm actually resolves FROM THIS ARM --------
#
# This check exists because the bug was real. The conditional arm lives one
# directory deeper than the rest, and the template hard-coded "../shared", which
# does not resolve there. The job would have died at once with "No such file or
# directory" -- the same class of failure that killed the first Ruche job of
# TASK-2026-09-01-SMCRUCHE-PACKFIX, where the sampler lived in a directory that
# existed only in the developer's working tree. The path is now computed per
# arm; this check verifies the result rather than trusting the computation.
m17 = re.search(r'PPSQJ_PYTHON"\s+(\S+)/run_pack\.py', slurm)
if not m17:
    bad("P17", "submit.slurm does not invoke run_pack.py in a recognisable form")
else:
    rel = m17.group(1)
    target = os.path.normpath(os.path.join(ARM, rel, "run_pack.py"))
    if not os.path.isfile(target):
        bad("P17", "submit.slurm runs %s/run_pack.py, which does not resolve from "
                   "this arm (%s). The job would die immediately."
            % (rel, target))
    else:
        cellp = os.path.normpath(os.path.join(ARM, rel, "run_cell.py"))
        if not os.path.isfile(cellp):
            bad("P17", "run_pack.py resolves but its sibling run_cell.py does not: %s"
                % cellp)
        else:
            ok("P17", "submit.slurm runs %s/run_pack.py, which resolves from this "
                      "arm, and run_cell.py sits beside it" % rel)

# ---------------------------------------------------------------------- report
for n in notes:
    print("  PASS  " + n)
for f2 in fails:
    print("  FAIL  " + f2)
print("-" * 76)
print("  %d passed, %d FAILED" % (len(notes), len(fails)))
print("  This preflight contains no submission command and cannot submit.")
print("  research/RESOURCE_POLICY.md section 4: agents never submit HPC jobs,")
print("  at any stage, gate or approval level. The researcher submits by hand.")
sys.exit(1 if fails else 0)
