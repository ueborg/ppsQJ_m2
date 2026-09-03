#!/usr/bin/env python3
"""Preflight for one TASK-2026-09-03-NC-PLATEAU-CALIBRATION arm.

Reads this arm's manifest, the frozen design and the frozen analysis spec, and
prints exactly what is about to be asked for.

IT SUBMITS NOTHING AND CANNOT. There is no scheduler call in this file or in
run_preflight.sh, and this file ASSERTS that fact about run_preflight.sh --
and that its own arm's submit.slurm has not been hand-edited away from the
generated design -- before it will pass.

Inherited from TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION's preflight, which
passed a clean tracked-only checkout test and its own injected-fault negative
controls. Changes, each recorded in ../VALIDATION.md:

  1. THE DESIGN IS NO LONGER RESTATED HERE. The predecessor hard-coded its grid,
     its N_c and its R into the preflight as literals, so the preflight and the
     builder could drift apart while both looked right. Both now import
     ../tools/design.py, and this file checks the MANIFEST against what the
     builder actually produces instead of against a second copy of the design.
  2. dtau_mult IS PART OF CELL IDENTITY. Campaign E varies it on purpose. A
     manifest that mixes discretisations inside one cell, or that carries a
     dtau_mult outside the frozen set for its campaign, is a hard failure.
  3. THE DUPLICATION GATE IS DATA-DRIVEN. The predecessor asserted "these four
     lambda indices only". This one loads the reuse ledger and fails if the
     manifest would push a cell past its target R by recomputing populations
     that already exist.
  4. THE MEMORY MODEL IS MEASURED. See ../COST_MODEL.md; the inherited formula
     under-predicts and had never been checked against a running process.
  5. THE PARTITION RULE IS RECOMPUTED, not asserted: cpu_med if --time fits its
     4 h MaxTime, cpu_long otherwise, cpu_short never.
"""
import csv, os, re, sys, json, hashlib, collections

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(TASK, "tools"))
import design as D                                                   # noqa: E402
from cost_model import n_steps, PESSIMISTIC                          # noqa: E402
from build_arms import ARMS, REUSE, arm_cost, build_rows             # noqa: E402

# The scheduler and remote-launch verbs that must not appear in any executable
# file of this package. Assembled from fragments so that this source line is not
# itself a command string containing them.
FORBIDDEN = ("s" + "batch", "s" + "run", "s" + "alloc", "q" + "sub", "b" + "sub",
             "s" + "cancel", "s" + "sh", "s" + "cp", "r" + "sync")
SCHED = re.compile(r"\b(" + "|".join(FORBIDDEN) + r")\b")

problems = []
lines = []


def chk(ok, label, detail):
    lines.append(f"    {'OK  ' if ok else 'FAIL'}  {label:<32} {detail}")
    if not ok:
        problems.append(f"{label}: {detail}")


def gib(v):
    """--mem in GiB. Slurm suffixes K/M/G/T are binary; NO suffix means MEGABYTES.

    Kept verbatim from the predecessor, where its negative control N14 caught
    two real defects in the version before it: `float(str(v).rstrip("Gg"))`
    raised on `--mem=600M` (and the except branch then reported 0.0, failing for
    the wrong reason), and read `--mem=2048` as 2048 GiB rather than 2048 MB,
    which fails OPEN. TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING asked for this
    parser to be re-verified; ../VALIDATION.md re-runs its full unit table.
    """
    s = str(v).strip()
    if not s:
        return 0.0
    mult = {"k": 1 / 1048576.0, "m": 1 / 1024.0, "g": 1.0, "t": 1024.0}
    unit = s[-1].lower()
    if unit in mult:
        s, f = s[:-1], mult[unit]
    else:
        f = mult["m"]
    try:
        return float(s) * f
    except ValueError:
        return 0.0


def hrs(v):
    try:
        h, m, s = str(v).split(":")
        return int(h) + int(m) / 60 + int(s) / 3600
    except Exception:
        return 0.0


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def batch_fields(path):
    out = {}
    for ln in open(path):
        m = re.match(r"#SBATCH\s+--([a-z-]+)=(\S+)", ln.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main():
    arm_dir = HERE
    # The arm is identified by its directory name. NCPLAT_ARM_NAME overrides
    # that for tools/negative_controls.py ONLY, which must run this preflight
    # against a deliberately broken COPY of an arm and needs it to fail for the
    # injected reason rather than for "this directory is not an arm".
    name = os.environ.get("NCPLAT_ARM_NAME") or os.path.basename(arm_dir)
    spec = [a for a in ARMS if a["name"] == name]
    if not spec:
        sys.exit(f"{name} is not one of this campaign's arms: "
                 f"{[a['name'] for a in ARMS]}")
    a = spec[0]

    rows = list(csv.DictReader(open(os.path.join(arm_dir, "manifest.csv"))))
    sb = batch_fields(os.path.join(arm_dir, "submit.slurm"))

    # ---- 1. the manifest IS the frozen design -------------------------------
    lines.append("  design")
    expect = build_rows(a, {})
    same = (len(rows) == len(expect) and all(
        all(str(r[k]) == str(e[k]) for k in ("arm", "L", "N_c", "seed",
                                             "resample_scheme"))
        and abs(float(r["T"]) - float(e["T"])) < 1e-12
        and abs(float(r["zeta"]) - float(e["zeta"])) < 1e-12
        and abs(float(r["lam"]) - float(e["lam"])) < 1e-12
        and abs(float(r["dtau_mult"]) - float(e["dtau_mult"])) < 1e-12
        for r, e in zip(rows, expect)))
    chk(same, "manifest == frozen design",
        f"{len(rows)} rows, regenerated from tools/build_arms.py and compared "
        f"row by row" if same else
        "the manifest on disk DIFFERS from what the builder produces. It was "
        "hand-edited. Regenerate it; do not adjust this check.")

    chk(all(abs(float(r["zeta"]) - D.ZETA) < 1e-12 for r in rows),
        "zeta", f"{D.ZETA} on every row")
    chk(all(abs(float(r["T"]) - float(r["L"])) < 1e-9 for r in rows),
        "T == L", "on every row")
    chk(all(r["resample_scheme"] == D.SCHEME for r in rows),
        "resampling", f"{D.SCHEME} on every row")

    dms = sorted({float(r["dtau_mult"]) for r in rows})
    allowed = list(D.E_DTAUS) if a["group"] == "E" else [D.DTAU_PRODUCTION]
    chk(all(d in allowed for d in dms), "dtau_mult",
        f"{dms}; allowed for campaign {a['group']}: {allowed}"
        + ("   (a DISCRETISATION CONTROL, never a physical parameter; the "
           "dtau_mult != 6 rows may never be pooled with the production corpus)"
           if a["group"] == "E" else ""))

    # n_steps must be reproducible from the row alone, and constant per cell.
    ks = {}
    for r in rows:
        k = (int(r["L"]), float(r["T"]), int(r["N_c"]), round(float(r["lam"]), 6),
             float(r["dtau_mult"]))
        ks.setdefault(k, n_steps(k[0], k[1], k[3], k[4]))
    chk(True, "K per cell (= ceil(2 lam (L-1) T / dtau_mult))",
        "; ".join(f"N_c={k[2]} lam={k[3]:g} dm={k[4]:g} K={v}"
                  for k, v in sorted(ks.items())))

    # ---- 2. the duplication gate --------------------------------------------
    lines.append("  duplication")
    dup = []
    counts = collections.Counter(
        (int(r["L"]), float(r["T"]), int(r["N_c"]), round(float(r["lam"]), 6),
         float(r["dtau_mult"])) for r in rows)
    for k, want in counts.items():
        have = REUSE.get(k, (0, None))[0]
        if have and have + want > max(a["R"], have):
            dup.append(f"{k}: {have} already exist + {want} fresh > target "
                       f"R={a['R']}")
    reused = sum(REUSE.get(k, (0, None))[0] for k in ks)
    chk(not dup, "no recomputation of an existing cell",
        "; ".join(dup) if dup else
        f"{reused} exact-compatible populations are reused and are absent from "
        f"this manifest by design")

    # ---- 3. seeds ------------------------------------------------------------
    lines.append("  seeds")
    seeds = [int(r["seed"]) for r in rows]
    chk(len(set(seeds)) == len(seeds), "distinct within the arm",
        f"{len(set(seeds))} of {len(seeds)}")
    chk(all(D.SEED_FLOOR <= s < D.SEED_CEIL for s in seeds), "inside the block",
        f"{min(seeds)}-{max(seeds)} within [{D.SEED_FLOOR}, {D.SEED_CEIL})")
    alloc = json.load(open(os.path.join(TASK, "tools", "allocated_seeds.json")))
    chk(set(seeds) <= set(alloc), "in the campaign ledger",
        f"all {len(seeds)} present in tools/allocated_seeds.json "
        f"({len(alloc)} campaign-wide)")

    # ---- 4. cost, wall time, memory, partition -------------------------------
    lines.append("  resources")
    c = arm_cost(a, rows)
    chk(sb.get("array") == f"0-{len(rows) - 1}%{D.CONCURRENCY}",
        "--array matches manifest",
        f"{sb.get('array')} for {len(rows)} rows")
    req_h = hrs(sb.get("time"))
    chk(abs(req_h - hrs(c["time"])) < 1e-9, "--time == cost model",
        f"requested {sb.get('time')}, model says {c['time']}")
    chk(req_h >= c["slow_h"] * PESSIMISTIC * 1.5, "--time headroom",
        f"{req_h / max(c['slow_h'], 1e-9):.2f}x the predicted slowest task, "
        f"{req_h / max(c['slow_h'] * PESSIMISTIC, 1e-9):.2f}x the pessimistic one")
    part = sb.get("partition")
    want_part = ("cpu_med" if req_h <= D.PARTITION_MAXTIME_H["cpu_med"]
                 else "cpu_long")
    chk(part == want_part, "--partition",
        f"{part}; the rule gives {want_part} for --time={sb.get('time')} "
        f"(cpu_short is never used at any --time: QOS-serialised for this "
        f"account)")
    chk(req_h <= D.PARTITION_MAXTIME_H.get(part, 0), "--time fits MaxTime",
        f"{req_h:.2f} h <= {D.PARTITION_MAXTIME_H.get(part, 0):g} h")
    have_gb = gib(sb.get("mem"))
    chk(have_gb * 1024 >= c["mem_mb"] * 1.2, "--mem",
        f"{sb.get('mem')} = {have_gb:.2f} GiB against the MEASURED model "
        f"{c['mem_mb']:.0f} MB "
        f"({have_gb * 1024 / max(c['mem_mb'], 1):.2f}x)")
    chk(sb.get("cpus-per-task") == "1" and sb.get("ntasks") == "1",
        "one core per task", "ntasks=1 cpus-per-task=1 (BLAS/OpenMP pinned to "
        "1 thread in the job script and again in run_cell.py)")

    # ---- 5. the runtime ------------------------------------------------------
    lines.append("  runtime")
    man = json.load(open(os.path.join(TASK, "support", "BUNDLE_MANIFEST.json")))
    for f in man["files"]:
        p = os.path.join(TASK, "support", os.path.basename(f["bundled_as"]))
        h = sha(p)
        chk(h == f["sha256_bundled"], "bundled " + os.path.basename(p),
            f"sha256 {h[:16]}... "
            + ("(matches manifest; this is the exact file that produced every "
               "reused population)" if h == f["sha256_bundled"] else "MISMATCH"))
    repo = os.environ.get("PPSQJ_REPO") or os.path.abspath(
        os.path.join(HERE, *([os.pardir] * 5)))
    chk(os.path.isfile(os.path.join(repo, "pps_qj", "__init__.py")),
        "pps_qj resolves", repo)
    for f in ("run_cell.py", "analyse_arm.py", "preflight.py"):
        h1 = sha(os.path.join(TASK, "shared", f))
        h2 = sha(os.path.join(arm_dir, f))
        chk(h1 == h2, "shared/" + f,
            "byte-identical to the arm copy" if h1 == h2 else
            "the arm copy has DRIFTED from shared/; regenerate")

    # ---- 6. nothing here can submit -----------------------------------------
    lines.append("  submission safety")
    for f in ("run_preflight.sh", "analyse_results.sh", "run_cell.py",
              "analyse_arm.py", "submit.slurm"):
        body = open(os.path.join(arm_dir, f)).read()
        hits = sorted(set(SCHED.findall(body)))
        chk(not hits, f + " carries no scheduler call",
            "clean" if not hits else f"CONTAINS {hits}")

    # ---- 7. the frozen specs -------------------------------------------------
    lines.append("  frozen spec")
    for f in ("ANALYSIS_SPEC.yaml", "SUCCESS_CRITERIA.yaml"):
        p = os.path.join(TASK, f)
        chk(os.path.isfile(p), f,
            "sha256 " + sha(p) if os.path.isfile(p) else "MISSING")

    # ---- 8. what is already on disk -----------------------------------------
    res = os.path.join(arm_dir, "results")
    done = len([f for f in os.listdir(res) if f.endswith(".json")]) \
        if os.path.isdir(res) else 0
    lines.append("  state")
    chk(True, "results already present", f"{done} / {len(rows)} "
        "(run_cell.py skips a completed row, so re-running an array is "
        "idempotent and tops up a partial one)")

    # ---- report --------------------------------------------------------------
    print("=" * 78)
    print(f"PREFLIGHT  {name}   TASK-2026-09-03-NC-PLATEAU-CALIBRATION  "
          f"campaign {a['group']}")
    print("=" * 78)
    print(f"  {a['purpose']}")
    print()
    print("  WOULD REQUEST (this preflight does not, and cannot, request it)")
    print(f"    partition        {sb.get('partition')}")
    print(f"    array            {sb.get('array')}   ({len(rows)} tasks)")
    print(f"    time             {sb.get('time')}")
    print(f"    mem              {sb.get('mem')}")
    print(f"    core-hours       {c['core_h']:.1f} "
          f"({c['core_h'] * PESSIMISTIC:.1f} pessimistic)")
    print(f"    slowest task     {c['slow_h']:.2f} h "
          f"({c['slow_h'] * PESSIMISTIC:.2f} h pessimistic)")
    print(f"    elapsed at %{D.CONCURRENCY}    {c['elapsed_h']:.2f} h "
          f"({c['elapsed_h'] * PESSIMISTIC:.2f} h pessimistic), "
          f"queue wait EXCLUDED")
    print(f"    peak memory      {c['mem_mb']:.0f} MB (measured model)")
    print(f"    rate(s)          {c['rates']} ms per clone-window, "
          f"measured on Ruche")
    print(f"    reused           {reused} exact-compatible populations, "
          f"not recomputed")
    print()
    print("  CHECKS")
    print("\n".join(lines))
    print()
    if problems:
        print("PREFLIGHT FAILED")
        for p in problems:
            print("   * " + p)
        return 1
    print("PREFLIGHT PASSED")
    print("  Nothing was submitted. research/RESOURCE_POLICY.md section 4: no")
    print("  agent may submit an HPC job at any stage, gate or approval level.")
    print("  The submission command is in ../RUCHE_RUNBOOK.md, for the")
    print("  researcher to type.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
