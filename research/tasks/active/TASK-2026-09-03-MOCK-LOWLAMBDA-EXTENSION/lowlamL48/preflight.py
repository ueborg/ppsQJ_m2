#!/usr/bin/env python3
"""Preflight for a LOWLAMBDA-EXTENSION Ruche arm.

Reads this arm's manifest and the frozen spec and prints exactly what is about
to be asked for. IT SUBMITS NOTHING AND CANNOT: there is no scheduler call in
this file or in run_preflight.sh, and this file asserts that fact about
run_preflight.sh before passing.

Inherited from TASK-2026-09-02-MOCK-PRODUCTION's preflight -- the version that
passed a clean tracked-only checkout test and its own injected-fault negative
controls. Six deliberate changes, each recorded in ../VALIDATION.md:

  1. THE COST MODEL IS NOW FITTED, NOT ASSUMED. The predecessor carried a
     per-clone-window rate table (BASE_MS x NC_FACTOR) with the L=32 and L=48
     entries DERIVED by L-scaling. That campaign has since returned 1152
     completed N_c=1024 populations, so this preflight imports the affine fit
     to those measurements from ../tools/cost_model.py and additionally
     REFITS it from the frozen data at run time, failing on drift.
  2. The frozen lambda check is against a 17-POINT GRID, and separately against
     the FOUR indices this task is allowed to compute. A manifest carrying any
     already-measured lambda would be duplicating the predecessor and is a hard
     failure, not a note.
  3. N_c is checked against the single frozen value 1024. The predecessor's
     {128, 1024, 2048} is gone with the arms that needed it.
  4. The seed block moved to [32e6, 33e6). Every seed anywhere else in the task
     tree, INCLUDING the predecessor's 2808 allocated ones, is <= 31,612,047.
  5. THE PARTITION RULE IS INVERTED. The predecessor required the SMALLEST
     partition that fits --time. This task requires cpu_med on all three arms
     regardless, on measured queue evidence (../SCHEDULER_DECISION.md), so the
     check is now "is it cpu_med, and does --time fit cpu_med's MaxTime".
  6. The elapsed-time model uses the two-wave floor for a 96-task array at %64,
     not the many-wave throughput average.
"""
import csv, os, re, sys, math, json, hashlib, subprocess, collections, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(TASK, "tools"))
from cost_model import (n_steps, wall_s, wall_s_affine, wall_s_maxrate, mem_mb,
                        elapsed_h, fit_from_frozen, AFFINE, RATE_MAX_MS,
                        FIT_RANGE, PESSIMISTIC, DTAU_MULT, NC)

# Frozen design constants. A manifest that violates any of these was hand-edited.
GRID = tuple(round(0.1932 + 0.010 * i, 4) for i in range(17))
NEW_IDX = (0, 1, 2, 3)
NEW_LAMS = tuple(GRID[i] for i in NEW_IDX)
REUSED_LAMS = tuple(GRID[i] for i in range(4, 17))
ZETA = 0.35
ALLOWED_NC = (NC,)
R_EXPECTED = 24
SEED_FLOOR = 32_000_000      # every seed anywhere else is <= 31,612,047
SEED_CEIL = 33_000_000
REQUIRED_PARTITION = "cpu_med"
FIT_TOL = 0.005              # 0.5 % drift between the literals and a refit

RUCHE_PARTITIONS = {         # MaxTime in hours, as reported by Ruche
    "cpu_short": 1.0,
    "cpu_med": 4.0,
    "cpu_long": 7 * 24.0,
}


def _gb(v):
    """--mem, in GiB. Understands Slurm's suffixes and its default unit.

    The predecessor's version was `float(str(v).rstrip("Gg"))`, which is wrong
    in two ways this task's own negative control N14 exposed:

      * `--mem=600M` is a perfectly ordinary Slurm request. rstrip("Gg") leaves
        "600M", float() raises, and the except branch returned 0.0 -- so the
        arm failed the memory check for a parse error while REPORTING an
        under-request. It failed closed, which is why nothing broke, but the
        reason it printed was not the reason it failed.
      * `--mem=2048` means 2048 MEGABYTES to Slurm, not 2048 gigabytes. The old
        parser read it as 2048 GiB and would have waved through an arm asking
        for a third of what it needs. That one fails OPEN.

    Slurm suffixes: K, M, G, T (binary). No suffix means MB.
    """
    s = str(v).strip()
    if not s:
        return 0.0
    mult = {"k": 1 / (1024.0 * 1024.0), "m": 1 / 1024.0, "g": 1.0, "t": 1024.0}
    unit = s[-1].lower()
    if unit in mult:
        s, f = s[:-1], mult[unit]
    else:
        f = mult["m"]                      # Slurm's default unit is megabytes
    try:
        return float(s) * f
    except ValueError:
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
    off = [l for l in lams if not any(abs(l - g) < 1e-9 for g in GRID)]
    chk(not off, "lambda on frozen grid",
        f"{len(lams)} of the 17 frozen grid points"
        + (f"; OFF-GRID: {off}" if off else ""))

    # THE DUPLICATION GATE. This task may compute grid indices 0-3 and nothing
    # else. A manifest carrying one of the predecessor's thirteen would burn
    # core-hours reproducing a measurement this package already ships in
    # frozen_inputs/, and would then be silently averaged into it.
    dup = [l for l in lams if any(abs(l - g) < 1e-9 for g in REUSED_LAMS)]
    chk(not dup, "no predecessor duplication",
        "the 13 already-measured lambdas are absent, as designed"
        if not dup else f"DUPLICATES TASK-2026-09-02-MOCK-PRODUCTION at {dup}")
    missing = [l for l in NEW_LAMS if not any(abs(l - g) < 1e-9 for g in lams)]
    chk(not missing, "all four new lambdas present",
        f"{[f'{l:g}' for l in NEW_LAMS]}"
        + (f"; MISSING {missing}" if missing else ""))

    zs = sorted({float(r["zeta"]) for r in rows})
    chk(zs == [ZETA], "zeta", f"{zs}  (frozen {ZETA})")
    dts = sorted({float(r["dtau_mult"]) for r in rows})
    chk(dts == [DTAU_MULT], "dtau_mult",
        f"{dts}  (CERTIFIED {DTAU_MULT}; the historical corpus used 12.0 and is "
        f"NOT poolable, and neither would a mixed-dtau extension be)")
    ncs = sorted({int(r["N_c"]) for r in rows})
    chk(ncs == list(ALLOWED_NC), "N_c frozen",
        f"{ncs}  (frozen {list(ALLOWED_NC)}; the predecessor's 128 and 2048 "
        f"arms have no counterpart here)")
    ts = sorted({(int(r["L"]), float(r["T"])) for r in rows})
    chk(all(abs(t - L) < 1e-9 for L, t in ts), "T == L",
        ", ".join(f"L={L}:T={t:g}" for L, t in ts))
    Ls = sorted({int(r["L"]) for r in rows})
    chk(len(Ls) == 1 and Ls[0] in (32, 48, 64), "single L in {32,48,64}", f"{Ls}")
    sch = sorted({r["resample_scheme"] for r in rows})
    chk(sch == ["systematic"], "resample_scheme", f"{sch}")

    seeds = [int(r["seed"]) for r in rows]
    chk(len(set(seeds)) == len(seeds), "seeds unique within arm",
        f"{len(set(seeds))} distinct of {len(seeds)}")
    chk(all(SEED_FLOOR <= s < SEED_CEIL for s in seeds),
        "seeds in the fresh block",
        f"{min(seeds)}-{max(seeds)}  (block [{SEED_FLOOR}, {SEED_CEIL}); every "
        f"seed anywhere else in the task tree is <= 31612047)")

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

    # Every lambda block must have identical R, and it must be the R the
    # reused half of the grid was measured at -- otherwise the 17-point curve
    # would carry silently heteroscedastic error bars across its own join.
    per = collections.Counter(round(float(r["lam"]), 6) for r in rows)
    chk(len(set(per.values())) == 1, "R equal across lambdas",
        f"R = {sorted(set(per.values()))} over {len(per)} lambdas")
    chk(set(per.values()) == {R_EXPECTED}, "R matches the reused half",
        f"R = {sorted(set(per.values()))}  (the predecessor's primary "
        f"matched-R block is {R_EXPECTED}; a different R here would make the "
        f"join a change of precision as well as a change of lambda)")
    return (not problems), lines, problems


def cost_model_checks():
    """The literals in tools/cost_model.py must still match the frozen data."""
    problems, lines = [], []
    try:
        refit = fit_from_frozen()
    except Exception as e:
        return False, [f"    FAIL  refit cost model           {e}"], \
            [f"the cost model cannot be refitted from the frozen snapshot: {e}"]
    for L in sorted(AFFINE):
        if L not in refit:
            problems.append(f"no frozen data at L={L} to refit the cost model")
            lines.append(f"    FAIL  refit L={L:<21} no frozen rows")
            continue
        a, b, sd, n, rmax, span = refit[L]
        la, lb = AFFINE[L]
        # compare the models where they are USED, not coefficient by
        # coefficient: a slope/intercept trade can leave both far off while the
        # prediction is fine, and only the prediction is load-bearing.
        worst = 0.0
        for lam in NEW_LAMS:
            ns = n_steps(L, float(L), lam)
            worst = max(worst, abs((a * ns + b) - (la * ns + lb)) / (la * ns + lb))
        ok = (worst <= FIT_TOL and abs(rmax - RATE_MAX_MS[L]) <= 1e-3
              and span == FIT_RANGE[L])
        lines.append(
            f"    {'OK  ' if ok else 'FAIL'}  cost model L={L:<15} "
            f"refit {a:.6f}*n+{b:.2f} (n={n}, resid sd {sd:.1f} s) vs literal "
            f"{la:.6f}*n+{lb:.2f}; max prediction drift {100 * worst:.3f} % "
            f"(tol {100 * FIT_TOL:g} %)")
        if not ok:
            problems.append(f"cost model at L={L} has drifted from the frozen "
                            f"data it claims to be fitted to")
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
    frozen = os.path.join(TASK, "frozen_inputs",
                          "predecessor_nc1024_populations.csv")
    for label, path in (("bundled instrumented.py", inst),
                        ("bundle manifest", man),
                        ("pps_qj package", ppsqj),
                        ("frozen predecessor data", frozen)):
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
            # The bundle is not merely self-containment here: it is the evidence
            # that the four new lambdas come out of the SAME sampler as the
            # thirteen reused ones, which is what licenses one 17-point curve.
            pred = os.path.join(TASK, os.pardir,
                                "TASK-2026-09-02-MOCK-PRODUCTION", "support",
                                "instrumented.py")
            if os.path.isfile(pred):
                ph = hashlib.sha256(open(pred, "rb").read()).hexdigest()
                ok2 = (ph == h)
                verdict = ("byte-identical" if ok2 else
                           "DIFFERS -- the old and new lambdas would not be "
                           "one curve")
                lines.append("    %s  sampler == predecessor's   %s"
                             % ("OK  " if ok2 else "FAIL", verdict))
                if not ok2:
                    problems.append("the bundled sampler is not the file that "
                                    "produced the reused populations")
            else:
                lines.append("    NOTE  sampler == predecessor's   predecessor "
                             "archive not present (clean-checkout run); the "
                             "sha256 above is the recorded identity")

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

    # --- partition: cpu_med, REQUIRED -- see change 5 in the module docstring
    part = sb.get("partition")
    req_h = _hrs(sb.get("time"))
    if not part:
        problems.append("submit.slurm declares NO --partition; the scheduler "
                        "would pick a default (cpu_short, MaxTime 1 h)")
        lines.append("    FAIL  --partition                 MISSING")
    else:
        ok = (part == REQUIRED_PARTITION)
        lines.append(f"    {'OK  ' if ok else 'FAIL'}  --partition                "
                     f"{part}  (this task REQUIRES {REQUIRED_PARTITION} on all "
                     f"three arms -- ../SCHEDULER_DECISION.md)")
        if not ok:
            problems.append(f"--partition={part}; this task requires "
                            f"{REQUIRED_PARTITION}. cpu_short is serialised for "
                            f"this account by QOSMaxJobsPerUserLimit and a "
                            f"'smallest partition that fits' rule would send two "
                            f"of these three arms there.")
        maxh = RUCHE_PARTITIONS.get(part)
        if maxh is not None:
            ok = req_h <= maxh + 1e-9
            lines.append(f"    {'OK  ' if ok else 'FAIL'}  --time vs MaxTime          "
                         f"{part} MaxTime {maxh:g} h vs requested {req_h:g} h")
            if not ok:
                problems.append(f"--time={sb.get('time')} exceeds partition "
                                f"{part} MaxTime of {maxh:g} h")

    # --- time and memory sizing: HARD failures, not annotations ---------------
    pess_h = slowest_h * PESSIMISTIC
    ok = req_h >= pess_h
    lines.append(f"    {'OK  ' if ok else 'FAIL'}  --time vs pessimistic      "
                 f"requested {req_h:g} h  vs slowest {slowest_h * 60:.1f} min "
                 f"({pess_h * 60:.1f} min pessimistic) -> margin "
                 f"{req_h / pess_h if pess_h else float('inf'):.1f}x")
    if not ok:
        problems.append(f"--time={sb.get('time')} is below the pessimistic "
                        f"slowest task ({pess_h * 60:.1f} min)")

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

    # The cost loop indexes AFFINE and RATE_MAX_MS, so an L off the frozen set
    # would raise KeyError and abort with a traceback instead of a diagnosis.
    # The exit code would still be non-zero -- it fails closed -- but a
    # traceback is not a report, so the membership check happens FIRST.
    unknown = sorted({(int(r["L"]), int(r["N_c"])) for r in rows}
                     - {(L, NC) for L in AFFINE})
    if unknown:
        print("=" * 78)
        print(f"  PREFLIGHT — {arm}    (this script does NOT submit anything)")
        print("=" * 78)
        print("  PREFLIGHT FAILED — this package must not be queued as it stands:")
        print(f"    * manifest contains (L, N_c) outside the frozen sets: {unknown}")
        print(f"      L must be one of {sorted(AFFINE)} and N_c must be {NC}.")
        print("      This manifest was hand-edited. Regenerate it with "
              "tools/build_arms.py.")
        print("=" * 78)
        return 1

    tot = slowest = peak = 0.0
    for r in rows:
        L, T = int(r["L"]), float(r["T"])
        s = wall_s(L, T, float(r["lam"]), int(r["N_c"]), float(r["dtau_mult"]))
        tot += s
        slowest = max(slowest, s)
        peak = max(peak, mem_mb(L, int(r["N_c"])))
    core_h = tot / 3600.0
    slow_h = slowest / 3600.0
    elapsed = elapsed_h(len(rows), core_h, slow_h, conc)

    W = 34

    def p(k, v):
        print(f"  {k:<{W}} {v}")

    def block(label, text):
        for i, ln in enumerate(textwrap.wrap(text, width=76 - W) or [""]):
            print(f"  {label if i == 0 else '':<{W}}{ln}")

    L0 = Ls[0]
    print("=" * 78)
    print(f"  PREFLIGHT — {arm}    (this script does NOT submit anything)")
    print("=" * 78)
    p("task", "TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION")
    p("parent", "TASK-2026-09-02-MOCK-PRODUCTION (complete, READ ONLY)")
    p("arm", arm)
    block("scientific question", question)
    p("manifest rows", len(rows))
    p("L", ", ".join(map(str, Ls)))
    p("T", ", ".join(f"{t:g}" for t in sorted({float(r['T']) for r in rows})))
    p("zeta", ", ".join(f"{z:g}" for z in sorted({float(r['zeta']) for r in rows})))
    p("lambda (NEW, computed here)", ", ".join(f"{l:g}" for l in lms))
    p("lambda (reused, NOT here)", ", ".join(f"{l:g}" for l in REUSED_LAMS))
    p("dtau_mult", ", ".join(f"{d:g}" for d in sorted({float(r['dtau_mult']) for r in rows})))
    p("resample_scheme", ", ".join(sorted({r["resample_scheme"] for r in rows})))
    p("N_c", ", ".join(str(n) for n in sorted({int(r["N_c"]) for r in rows})))
    p("R per lambda", len(rows) // max(len(lms), 1))
    p("n_steps per run", ", ".join(str(n_steps(L0, float(L0), l)) for l in lms))
    p("cost-model fit range n_steps", f"{FIT_RANGE[L0][0]}-{FIT_RANGE[L0][1]}  "
                                      f"(this arm uses "
                                      f"{n_steps(L0, float(L0), min(lms))}-"
                                      f"{n_steps(L0, float(L0), max(lms))}, "
                                      f"an extrapolation of "
                                      f"{n_steps(L0, float(L0), min(lms)) / FIT_RANGE[L0][0]:.2f}x "
                                      f"below the fitted floor)")
    p("wall model (affine, measured)",
      f"{AFFINE[L0][0]:.6f}*n_steps + {AFFINE[L0][1]:.2f} s")
    p("wall model (worst rate seen)", f"{RATE_MAX_MS[L0]:.3f} ms/clone-window")
    p("adopted per-lambda wall_s",
      ", ".join(f"{wall_s(L0, float(L0), l):.0f}" for l in lms)
      + "   (larger of the two)")
    p("seed range", f"{min(int(r['seed']) for r in rows)}–"
                    f"{max(int(r['seed']) for r in rows)}")
    print("  " + "-" * 74)
    p("expected core-hours", f"{core_h:.2f}  ({core_h * PESSIMISTIC:.2f} pessimistic)")
    p("slowest single task", f"{slow_h * 60:.1f} min  "
                             f"({slow_h * PESSIMISTIC * 60:.1f} min pessimistic)")
    p(f"elapsed at the cap %{conc}", f"{elapsed * 60:.1f} min  "
                                     f"({elapsed * PESSIMISTIC * 60:.1f} min "
                                     f"pessimistic), QUEUE WAIT EXCLUDED")
    p("waves at this cap", f"{math.ceil(len(rows) / conc)}  "
                           f"({len(rows)} tasks / {conc})")
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
    print("  COST MODEL vs THE DATA IT CLAIMS TO BE FITTED TO")
    ok3, lines, prob3 = cost_model_checks()
    for ln in lines:
        print(ln)
    print("  " + "-" * 74)
    print("  RUNTIME SELF-CONTAINMENT, ARRAY AND PARTITION")
    ok2, lines, prob2 = runtime_checks(sb, slow_h, peak, len(rows))
    for ln in lines:
        print(ln)
    print("=" * 78)
    problems = prob1 + prob3 + prob2
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
