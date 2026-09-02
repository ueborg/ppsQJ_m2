#!/usr/bin/env python3
"""Preflight for a SMC-HIGHRUNG-LAMBDA Ruche arm.

Reads this arm's manifest and the frozen spec and prints exactly what is about
to be asked for. IT NEVER SUBMITTED ANYTHING AND CANNOT: there is no scheduler
call in this file or in run_preflight.sh.

Inherited from TASK-2026-09-01-SMCRUCHE-PACKFIX's repaired preflight, with four
deliberate changes, each recorded in ../VALIDATION.md:

  1. RATE is MEASURED on Ruche from completed ARM1/ARM2 runs of this same code
     path. The predecessor's L=128 entry was a Mac-probe extrapolation it had to
     flag at +/-50 %; that extrapolation was low by 45 %.
  2. An --array range that does not match the manifest row count is now a HARD
     FAILURE. The predecessor printed "** MISMATCH **" and still exited 0.
  3. --mem and --time that are too tight for the prediction are HARD FAILURES,
     not "** TIGHT **" annotations.
  4. The frozen lambda stencil and the seed-disjointness block are checked here
     too, so a hand-edited manifest cannot reach the scheduler.
"""
import csv, os, re, sys, math, json, hashlib, subprocess, collections, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))

# --- seconds per clone-window, MEASURED on Ruche --------------------------
# Kept in step with tools/cost_model.py, which carries the full provenance.
# L=64 is derived by two independent routes that agree to 1.3 %; every other
# entry is the median of a completed rung of this identical production path.
RATE = {64: 5.000e-3, 96: 11.510e-3, 128: 21.522e-3}
PESSIMISTIC = 1.40

# Frozen design constants. A manifest that violates any of these was hand-edited.
LAM_STENCIL = (0.2932, 0.3032, 0.3132)
ZETA = 0.35
DTAU_MULT = 6.0
SEED_FLOOR = 30_000_000          # every predecessor seed is <= 20_384_063
SEED_CEIL = 31_000_000

RUCHE_PARTITIONS = {             # MaxTime in hours, as reported by Ruche
    "cpu_short": 1.0,
    "cpu_med": 4.0,
    "cpu_long": 7 * 24.0,
}


def n_steps(L, T, lam, dtau_mult):
    dtau = dtau_mult / max(2.0 * lam * (L - 1), 1e-12)
    return max(1, int(math.ceil(T / dtau)))


def mem_mb(L, N_c):
    per_clone = (2 * L) ** 2 * 8 + (2 * L) * L * 16
    return 128.0 + 2.0 * N_c * per_clone / 1e6


def _gb(v):
    try:
        return float(str(v).rstrip("Gg"))
    except Exception:
        return 0.0


def _hrs(v):
    try:
        h, m, s = str(v).split(":")
        return int(h) + int(m) / 60 + int(s) / 3600
    except Exception:
        return 0.0


def design_checks(rows):
    """The frozen design. A hand-edited manifest must not reach the scheduler."""
    problems, lines = [], []

    def chk(ok, label, detail):
        lines.append(f"    {'OK  ' if ok else 'FAIL'}  {label:<26} {detail}")
        if not ok:
            problems.append(f"{label}: {detail}")

    lams = sorted({round(float(r["lam"]), 6) for r in rows})
    chk(all(any(abs(l - s) < 1e-9 for s in LAM_STENCIL) for l in lams),
        "lambda on frozen stencil", f"{lams}  (frozen {list(LAM_STENCIL)})")
    zs = sorted({float(r["zeta"]) for r in rows})
    chk(zs == [ZETA], "zeta", f"{zs}  (frozen {ZETA})")
    dts = sorted({float(r["dtau_mult"]) for r in rows})
    chk(dts == [DTAU_MULT], "dtau_mult",
        f"{dts}  (CERTIFIED {DTAU_MULT}; the historical corpus used 12.0 and is "
        f"NOT poolable)")
    sch = sorted({r["resample_scheme"] for r in rows})
    chk(sch == ["systematic"], "resample_scheme", f"{sch}")

    seeds = [int(r["seed"]) for r in rows]
    chk(len(set(seeds)) == len(seeds), "seeds unique within arm",
        f"{len(set(seeds))} distinct of {len(seeds)}")
    chk(all(SEED_FLOOR <= s < SEED_CEIL for s in seeds),
        "seeds in the fresh block",
        f"{min(seeds)}-{max(seeds)}  (block [{SEED_FLOOR}, {SEED_CEIL}); every "
        f"predecessor seed is <= 20384063)")

    led = os.path.join(TASK, "tools", "existing_seeds.json")
    if os.path.isfile(led):
        prior = set(json.load(open(led)))
        overlap = sorted(set(seeds) & prior)
        chk(not overlap, "no overlap with predecessors",
            f"{len(prior)} predecessor seeds scanned, {len(overlap)} collisions"
            + (f": {overlap[:5]}" if overlap else ""))
    else:
        lines.append(f"    WARN  no-overlap ledger          {led} absent; "
                     f"the range check above still holds")

    # one lambda block must have identical R, or the stencil differences are
    # not comparable
    per = collections.Counter(round(float(r["lam"]), 6) for r in rows)
    chk(len(set(per.values())) == 1, "R equal across lambdas",
        ", ".join(f"{l:g}:{n}" for l, n in sorted(per.items())))
    return (not problems), lines, problems


def runtime_checks(sb, slowest_h, peak_mb, n_rows):
    """Anything that would stop run_cell.py starting, or mis-size the request."""
    problems, lines = [], []
    support = os.path.join(TASK, "support")
    repo = os.environ.get("PPSQJ_REPO") or os.path.abspath(
        os.path.join(HERE, *([os.pardir] * 5)))

    inst = os.path.join(support, "instrumented.py")
    man = os.path.join(support, "BUNDLE_MANIFEST.json")
    ppsqj = os.path.join(repo, "pps_qj", "__init__.py")
    for label, path in (("bundled instrumented.py", inst),
                        ("bundle manifest", man),
                        ("pps_qj package", ppsqj)):
        ok = os.path.isfile(path)
        lines.append(f"    {'OK  ' if ok else 'FAIL'}  {label:<26} {path}")
        if not ok:
            problems.append(f"{label} missing at {path}")

    if os.path.isfile(inst) and os.path.isfile(man):
        for f in json.load(open(man))["files"]:
            p = os.path.join(support, os.path.basename(f["bundled_as"]))
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            ok = (h == f["sha256_bundled"])
            lines.append(f"    {'OK  ' if ok else 'FAIL'}  bundle sha256              "
                         f"{h[:16]}...  ({'matches' if ok else 'DOES NOT MATCH'} manifest)")
            if not ok:
                problems.append(f"bundled {p} does not match its recorded sha256")

    # Import exactly what run_cell.py imports, in a subprocess, so a broken
    # import cannot poison this process. This is the check that would have
    # caught the predecessor's ModuleNotFoundError before submission.
    code = ("import sys;"
            f"sys.path.insert(0,{repo!r});sys.path.insert(0,{support!r});"
            "import instrumented, pps_qj, numpy;"
            "print(instrumented.__file__);print(pps_qj.__file__);"
            "print(numpy.__version__)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    ok = (r.returncode == 0)
    lines.append(f"    {'OK  ' if ok else 'FAIL'}  import instrumented+pps_qj+numpy")
    if ok:
        for ln in r.stdout.strip().splitlines():
            lines.append(f"          {ln}")
    else:
        tail = r.stderr.strip().splitlines()
        lines.append(f"          {tail[-1] if tail else '?'}")
        problems.append("run_cell.py's imports do not resolve")

    # --- array range: ONE task per manifest row, exactly. HARD failure. ------
    want = f"0-{n_rows - 1}"
    got = sb.get("array", "MISSING").split("%")[0]
    ok = (got == want)
    lines.append(f"    {'OK  ' if ok else 'FAIL'}  --array                    "
                 f"{sb.get('array', 'MISSING')}  (manifest has {n_rows} rows -> "
                 f"expect {want})")
    if not ok:
        problems.append(f"--array {sb.get('array')} does not match the manifest's "
                        f"{n_rows} rows; expected {want}")

    # --- partition -----------------------------------------------------------
    part = sb.get("partition")
    req_h = _hrs(sb.get("time"))
    if not part:
        problems.append("submit.slurm declares NO --partition; the scheduler would "
                        "pick a default (cpu_short, MaxTime 1 h) and kill the job")
        lines.append("    FAIL  --partition                 MISSING")
    else:
        maxh = RUCHE_PARTITIONS.get(part)
        if maxh is None:
            lines.append(f"    WARN  --partition                 {part} "
                         f"(unknown to this check; verify with `sinfo`)")
        else:
            ok = req_h <= maxh + 1e-9
            lines.append(f"    {'OK  ' if ok else 'FAIL'}  --partition                "
                         f"{part}  MaxTime {maxh:g} h  vs requested {req_h:g} h")
            if not ok:
                problems.append(f"--time={sb.get('time')} exceeds partition {part} "
                                f"MaxTime of {maxh:g} h")
            smaller = [p for p, m in sorted(RUCHE_PARTITIONS.items(),
                                            key=lambda kv: kv[1]) if m + 1e-9 >= req_h]
            if smaller and smaller[0] != part:
                lines.append(f"    NOTE  a smaller partition also fits "
                             f"{sb.get('time')}: {smaller[0]}")

    # --- time and memory sizing: HARD failures, not annotations ---------------
    pess_h = slowest_h * PESSIMISTIC
    ok = req_h >= pess_h
    lines.append(f"    {'OK  ' if ok else 'FAIL'}  --time vs pessimistic      "
                 f"requested {req_h:g} h  vs slowest {slowest_h:.2f} h "
                 f"({pess_h:.2f} h pessimistic)")
    if not ok:
        problems.append(f"--time={sb.get('time')} is below the pessimistic slowest "
                        f"task ({pess_h:.2f} h)")

    need_gb = peak_mb / 1024.0 * 1.5
    ok = _gb(sb.get("mem")) >= need_gb
    lines.append(f"    {'OK  ' if ok else 'FAIL'}  --mem vs 1.5x peak         "
                 f"requested {sb.get('mem', 'MISSING')}  vs peak {peak_mb:.0f} MB "
                 f"(need >= {need_gb:.2f} G)")
    if not ok:
        problems.append(f"--mem={sb.get('mem')} is below 1.5x the estimated "
                        f"{peak_mb:.0f} MB peak")

    for bad in ("sbatch", "srun", "salloc", "qsub"):
        if any(bad in ln for ln in open(os.path.join(HERE, "run_preflight.sh"))):
            problems.append(f"run_preflight.sh contains {bad!r}; a preflight must "
                            f"never be able to submit")
    lines.append("    OK    run_preflight.sh has no scheduler call")
    return (not problems), lines, problems


def read_spec_fields(spec_path):
    """PyYAML is OPTIONAL and used only to pretty-print three fields. The frozen
    analysis imports no yaml at all. Ruche's login node reports
    'No module named yaml'; the dependency-free fallback below reads the same
    fields so the human still sees the question and the decision rule."""
    try:
        import yaml
        s = yaml.safe_load(open(spec_path))
        return "PyYAML present", s
    except Exception as e:
        return f"PyYAML absent ({e}); using the built-in fallback reader", None


def _field(txt, name):
    m = re.search(rf"^\s*{name}:\s*([>|][-+]?)?\s*$", txt, re.M)
    if m:
        out, indent = [], None
        for ln in txt[m.end():].splitlines()[1:]:
            if not ln.strip():
                continue
            ind = len(ln) - len(ln.lstrip())
            if indent is None:
                indent = ind
            if ind < indent:
                break
            out.append(ln.strip())
        return " ".join(out)
    m = re.search(rf"^\s*{name}:\s*(\S.*?)\s*$", txt, re.M)
    return m.group(1).strip("'\"") if m else ""


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "manifest.csv"))))
    arm = rows[0]["arm"]
    spec_path = os.path.join(TASK, "analysis_spec.yaml")
    spec_hash = hashlib.sha256(open(spec_path, "rb").read()).hexdigest()
    yaml_status, spec = read_spec_fields(spec_path)
    txt = open(spec_path).read()
    if spec:
        entry = next((a for a in spec["arms"] if a["id"] == arm), spec["arms"][0])
        question = " ".join(entry["question"].split())
        primary = " ".join(str(entry.get("primary_statistic", "")).split())
        rule = " ".join(str(entry.get("decision_rule", "")).split())
    else:
        question, primary, rule = (_field(txt, "question"),
                                   _field(txt, "primary_statistic"),
                                   _field(txt, "decision_rule"))

    Ls = sorted({int(r["L"]) for r in rows})
    lms = sorted({float(r["lam"]) for r in rows})
    ladder = collections.Counter(int(r["N_c"]) for r in rows)

    tot = slowest = peak = 0.0
    for r in rows:
        L, T, N = int(r["L"]), float(r["T"]), int(r["N_c"])
        s = RATE[L] * N * n_steps(L, T, float(r["lam"]), float(r["dtau_mult"]))
        tot += s
        slowest = max(slowest, s)
        peak = max(peak, mem_mb(L, N))

    sb = {}
    sp = os.path.join(HERE, "submit.slurm")
    if os.path.isfile(sp):
        for line in open(sp):
            m = re.match(r"#SBATCH\s+--(\S+?)=(\S+)", line.strip())
            if m:
                sb[m.group(1)] = m.group(2)

    W = 34

    def p(k, v):
        print(f"  {k:<{W}} {v}")

    def block(label, text):
        for i, ln in enumerate(textwrap.wrap(text, width=76 - W) or [""]):
            print(f"  {label if i == 0 else '':<{W}}{ln}")

    print("=" * 78)
    print(f"  PREFLIGHT — {arm}    (this script does NOT submit anything)")
    print("=" * 78)
    p("task", "TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA")
    p("arm", arm)
    block("scientific question", question)
    p("manifest rows", len(rows))
    p("L", ", ".join(map(str, Ls)))
    p("T", ", ".join(f"{t:g}" for t in sorted({float(r['T']) for r in rows})))
    p("zeta", ", ".join(f"{z:g}" for z in sorted({float(r['zeta']) for r in rows})))
    p("lambda", ", ".join(f"{l:g}" for l in lms))
    p("dtau_mult", ", ".join(f"{d:g}" for d in sorted({float(r['dtau_mult']) for r in rows})))
    p("resample_scheme", ", ".join(sorted({r["resample_scheme"] for r in rows})))
    p("N_c (this arm)", ", ".join(str(n) for n in sorted(ladder)))
    p("R per (N_c, lambda)", ", ".join(
        f"{n}:{ladder[n] // max(len(lms), 1)}" for n in sorted(ladder)))
    p("n_steps per run", ", ".join(str(x) for x in sorted(
        {n_steps(int(r["L"]), float(r["T"]), float(r["lam"]), float(r["dtau_mult"]))
         for r in rows})))
    p("seed range", f"{min(int(r['seed']) for r in rows)}–"
                    f"{max(int(r['seed']) for r in rows)}")
    print("  " + "-" * 74)
    p("expected core-hours", f"{tot / 3600:.1f}  "
                             f"({tot / 3600 * PESSIMISTIC:.1f} pessimistic)")
    p("slowest single task", f"{slowest / 3600:.2f} h  "
                             f"({slowest / 3600 * PESSIMISTIC:.2f} h pessimistic)")
    p("peak memory per task", f"{peak:.0f} MB")
    p("submit.slurm --array", sb.get("array", "MISSING"))
    p("submit.slurm --partition", sb.get("partition", "MISSING"))
    p("submit.slurm --time", sb.get("time", "MISSING"))
    p("submit.slurm --mem", sb.get("mem", "MISSING"))
    p("analysis-spec sha256", spec_hash)
    p("PyYAML", yaml_status)
    print("  " + "-" * 74)
    block("primary statistic", primary)
    block("decision rule", rule)
    resdir = os.path.join(HERE, "results")
    done = len([f for f in os.listdir(resdir) if f.endswith(".json")]) \
        if os.path.isdir(resdir) else 0
    print("  " + "-" * 74)
    print(f"  results already present: {done} / {len(rows)}"
          f"{'   (a resubmission will SKIP these)' if done else ''}")
    print(f"  PPSQJ_REPO = {os.environ.get('PPSQJ_REPO', '(unset — derived from the package location, which is fine)')}")

    print("  " + "-" * 74)
    print("  FROZEN DESIGN")
    ok1, lines, prob1 = design_checks(rows)
    for ln in lines:
        print(ln)
    print("  " + "-" * 74)
    print("  RUNTIME SELF-CONTAINMENT, ARRAY AND PARTITION")
    ok2, lines, prob2 = runtime_checks(sb, slowest / 3600.0, peak, len(rows))
    for ln in lines:
        print(ln)
    print("=" * 78)
    problems = prob1 + prob2
    if problems:
        print("  PREFLIGHT FAILED — this package must not be queued as it stands:")
        for pr in problems:
            print(f"    * {pr}")
        print("=" * 78)
        return 1
    print("  PREFLIGHT PASSED. NOTHING WAS SUBMITTED AND NOTHING HERE CAN SUBMIT.")
    print("  To submit, read ../RUCHE_RUNBOOK.md and type the command yourself.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
