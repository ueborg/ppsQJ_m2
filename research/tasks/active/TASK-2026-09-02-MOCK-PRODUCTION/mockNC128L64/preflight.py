#!/usr/bin/env python3
"""Preflight for a MOCK-PRODUCTION Ruche arm.

Reads this arm's manifest and the frozen spec and prints exactly what is about
to be asked for. IT NEVER SUBMITTED ANYTHING AND CANNOT: there is no scheduler
call in this file or in run_preflight.sh, and this file asserts that fact about
run_preflight.sh before passing.

Inherited from TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA's preflight, which is the
version that passed a clean tracked-only checkout test and ten injected-fault
negative controls. Four deliberate changes, each recorded in ../VALIDATION.md:

  1. The rate model is now rate_ms(L, N_c) = BASE_MS[L] * NC_FACTOR[N_c]. The
     predecessor held a rate per L only and extrapolated FLAT in N_c; its own
     returned ARM A JSONs show the rate rises 30 % from N_c=256 to N_c=1024 at
     L=128, so the flat model understated armA1024 by 30 %. Provenance for
     every entry: ../COST_MODEL.md.
  2. The frozen lambda check is against a 13-POINT GRID, not a 3-point stencil.
  3. N_c is checked against the frozen set {128, 1024, 2048}. A manifest at any
     other population was hand-edited.
  4. The seed block moved to [31e6, 32e6). Every seed anywhere in the task tree,
     INCLUDING the predecessor's 480 freshly allocated ones, is <= 30_500_015.
"""
import csv, os, re, sys, math, json, hashlib, subprocess, collections, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))

# --- ms per clone-window. Kept in step with tools/cost_model.py, which carries
# the full provenance. L=64 and L=128 are MEASURED on Ruche from completed runs
# of this identical production path; L=32 and L=48 are derived by downward
# L-scaling at fixed N_c with an exponent BELOW every measured exponent, then
# rounded up. NC_FACTOR is measured from the L=128 rung ladder.
BASE_MS = {32: 1.400, 48: 3.000, 64: 4.850, 80: 8.550, 128: 27.898}
NC_FACTOR = {128: 1.35, 1024: 1.00, 2048: 1.20}
PESSIMISTIC = 1.40
PACKING = 1.15                   # calibrated on ARM B: 288 tasks, %64, 2.76 h

# Frozen design constants. A manifest that violates any of these was hand-edited.
LAM_GRID = tuple(round(0.2332 + 0.010 * i, 4) for i in range(13))
ZETA = 0.35
DTAU_MULT = 6.0
ALLOWED_NC = (128, 1024, 2048)
SEED_FLOOR = 31_000_000          # every seed anywhere else is <= 30_500_015
SEED_CEIL = 32_000_000

RUCHE_PARTITIONS = {             # MaxTime in hours, as reported by Ruche
    "cpu_short": 1.0,
    "cpu_med": 4.0,
    "cpu_long": 7 * 24.0,
}


def rate_s(L, N_c):
    return BASE_MS[L] * NC_FACTOR[N_c] * 1e-3


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
    off = [l for l in lams if not any(abs(l - g) < 1e-9 for g in LAM_GRID)]
    chk(not off, "lambda on frozen grid",
        f"{len(lams)} of the 13 frozen grid points"
        + (f"; OFF-GRID: {off}" if off else ""))
    zs = sorted({float(r["zeta"]) for r in rows})
    chk(zs == [ZETA], "zeta", f"{zs}  (frozen {ZETA})")
    dts = sorted({float(r["dtau_mult"]) for r in rows})
    chk(dts == [DTAU_MULT], "dtau_mult",
        f"{dts}  (CERTIFIED {DTAU_MULT}; the historical corpus used 12.0 and is "
        f"NOT poolable)")
    ncs = sorted({int(r["N_c"]) for r in rows})
    chk(all(n in ALLOWED_NC for n in ncs), "N_c on frozen set",
        f"{ncs}  (frozen {list(ALLOWED_NC)})")
    ts = sorted({(int(r["L"]), float(r["T"])) for r in rows})
    chk(all(abs(t - L) < 1e-9 for L, t in ts), "T == L",
        ", ".join(f"L={L}:T={t:g}" for L, t in ts))
    sch = sorted({r["resample_scheme"] for r in rows})
    chk(sch == ["systematic"], "resample_scheme", f"{sch}")

    seeds = [int(r["seed"]) for r in rows]
    chk(len(set(seeds)) == len(seeds), "seeds unique within arm",
        f"{len(set(seeds))} distinct of {len(seeds)}")
    chk(all(SEED_FLOOR <= s < SEED_CEIL for s in seeds),
        "seeds in the fresh block",
        f"{min(seeds)}-{max(seeds)}  (block [{SEED_FLOOR}, {SEED_CEIL}); every "
        f"seed anywhere else in the task tree is <= 30500015)")

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

    # every lambda block in an arm must have identical R, or the curve's
    # point-to-point comparisons carry silently different weights
    per = collections.Counter(round(float(r["lam"]), 6) for r in rows)
    chk(len(set(per.values())) == 1, "R equal across lambdas",
        f"R = {sorted(set(per.values()))} over {len(per)} lambdas")

    # mockL64 deliberately omits the three reused lambdas. Any OTHER arm at
    # (L=64, N_c=1024) that DID include them would be duplicating ARM B.
    L0, N0 = int(rows[0]["L"]), int(rows[0]["N_c"])
    if (L0, N0) == (64, 1024):
        dup = [l for l in lams if l in (0.2932, 0.3032, 0.3132)]
        chk(not dup, "no ARM-B duplication",
            "the three ARM-B lambdas are absent, as designed"
            if not dup else f"DUPLICATES ARM B at {dup}")
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
    # caught the ModuleNotFoundError that killed the first ARM 1 Ruche job.
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
                         f"(unknown to this check; verify against the live cluster)")
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

    conc = 64
    sp = os.path.join(HERE, "submit.slurm")
    sb = {}
    if os.path.isfile(sp):
        for line in open(sp):
            m = re.match(r"#SBATCH\s+--(\S+?)=(\S+)", line.strip())
            if m:
                sb[m.group(1)] = m.group(2)
        if "%" in sb.get("array", ""):
            try:
                conc = int(sb["array"].split("%")[1])
            except Exception:
                pass

    # The cost loop indexes BASE_MS and NC_FACTOR, so an (L, N_c) off the frozen
    # sets would raise KeyError and abort with a traceback instead of a
    # diagnosis. The exit code would still be non-zero -- it fails closed -- but
    # a traceback is not a report, so the membership check happens FIRST.
    unknown = sorted({(int(r["L"]), int(r["N_c"])) for r in rows}
                     - {(L, n) for L in BASE_MS for n in NC_FACTOR})
    if unknown:
        print("=" * 78)
        print(f"  PREFLIGHT — {arm}    (this script does NOT submit anything)")
        print("=" * 78)
        print("  PREFLIGHT FAILED — this package must not be queued as it stands:")
        print(f"    * manifest contains (L, N_c) outside the frozen sets: {unknown}")
        print(f"      L must be one of {sorted(BASE_MS)} and "
              f"N_c one of {sorted(NC_FACTOR)}.")
        print("      This manifest was hand-edited. Regenerate it with "
              "tools/build_arms.py.")
        print("=" * 78)
        return 1

    tot = slowest = peak = 0.0
    for r in rows:
        L, T, N = int(r["L"]), float(r["T"]), int(r["N_c"])
        s = rate_s(L, N) * N * n_steps(L, T, float(r["lam"]), float(r["dtau_mult"]))
        tot += s
        slowest = max(slowest, s)
        peak = max(peak, mem_mb(L, N))
    core_h = tot / 3600.0
    slow_h = slowest / 3600.0
    elapsed = max(core_h / conc * PACKING, slow_h)

    W = 34

    def p(k, v):
        print(f"  {k:<{W}} {v}")

    def block(label, text):
        for i, ln in enumerate(textwrap.wrap(text, width=76 - W) or [""]):
            print(f"  {label if i == 0 else '':<{W}}{ln}")

    print("=" * 78)
    print(f"  PREFLIGHT — {arm}    (this script does NOT submit anything)")
    print("=" * 78)
    p("task", "TASK-2026-09-02-MOCK-PRODUCTION")
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
    p("rate ms/clone-window", ", ".join(
        f"{L}/{n}:{BASE_MS[L] * NC_FACTOR[n]:.3f}" for L in Ls for n in sorted(ladder)))
    p("n_steps per run", ", ".join(str(x) for x in sorted(
        {n_steps(int(r["L"]), float(r["T"]), float(r["lam"]), float(r["dtau_mult"]))
         for r in rows})))
    p("seed range", f"{min(int(r['seed']) for r in rows)}–"
                    f"{max(int(r['seed']) for r in rows)}")
    print("  " + "-" * 74)
    p("expected core-hours", f"{core_h:.1f}  ({core_h * PESSIMISTIC:.1f} pessimistic)")
    p("slowest single task", f"{slow_h:.2f} h  ({slow_h * PESSIMISTIC:.2f} h pessimistic)")
    p(f"elapsed at the cap %{conc}", f"{elapsed:.2f} h  "
                                     f"({elapsed * PESSIMISTIC:.2f} h pessimistic)")
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
    ok2, lines, prob2 = runtime_checks(sb, slow_h, peak, len(rows))
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
