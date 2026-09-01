#!/usr/bin/env python3
"""Preflight for a SMCCERT Ruche arm. Reads the manifest and the frozen spec and
prints what is about to be asked for. IT NEVER SUBMITTED ANYTHING AND CANNOT:
there is no sbatch call anywhere in this file or in run_preflight.sh."""
import csv, os, re, sys, math, hashlib, collections, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))

# Seconds per clone-window by L. MEASURED, not modelled:
#   L = 16,24,32,64 from the SMCCERT probe (scratch/probe.jsonl)
#   L = 96          from the SMCSTAT B-T96 block wall clock
#   L = 128         DERIVED: the SMCSTAT timing probe gives 2.68 ms/clone-window
#                   at L=96 and 6.03 at L=128, a ratio of 2.250, applied to the
#                   measured L=96 rate. No L=128 run exists in the programme, so
#                   the ARM2 cost is the least reliable number here: treat +/-50%.
RATE = {16: 3.99e-4, 24: 4.92e-4, 32: 9.07e-4, 64: 2.83e-3, 96: 6.59e-3, 128: 1.483e-2}

def n_steps(L, T, lam, dtau_mult):
    dtau = dtau_mult / max(2.0 * lam * (L - 1), 1e-12)
    return int(math.ceil(T / dtau))

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
        h, m, sec = str(v).split(":")
        return int(h) + int(m) / 60 + int(sec) / 3600
    except Exception:
        return 0.0


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "manifest.csv"))))
    arm = rows[0]["arm"]
    spec_path = os.path.join(HERE, "analysis_spec.yaml")
    spec_hash = hashlib.sha256(open(spec_path, "rb").read()).hexdigest()
    question = ""
    for line in open(spec_path):
        if line.strip().startswith("question:"):
            question = "(see analysis_spec.yaml `question`)"
    try:
        import yaml
        entry = yaml.safe_load(open(spec_path))["arms"][0]
        question = " ".join(entry["question"].split())
        rule = " ".join(entry["decision_rule"].split())
        primary = entry.get("primary_statistic", "")
    except Exception as e:                       # yaml is optional on a login node
        rule = primary = f"(PyYAML unavailable: {e})"

    Ls  = sorted({int(r["L"]) for r in rows})
    Ts  = sorted({float(r["T"]) for r in rows})
    zs  = sorted({float(r["zeta"]) for r in rows})
    lms = sorted({float(r["lam"]) for r in rows})
    dts = sorted({float(r["dtau_mult"]) for r in rows})
    sch = sorted({r["resample_scheme"] for r in rows})
    ladder = collections.Counter(int(r["N_c"]) for r in rows)

    tot = 0.0; peak = 0.0; slowest = 0.0
    for r in rows:
        L, T, N, lam, dm = int(r["L"]), float(r["T"]), int(r["N_c"]), float(r["lam"]), float(r["dtau_mult"])
        ns = n_steps(L, T, lam, dm)
        s = RATE[L] * N * ns
        tot += s; slowest = max(slowest, s); peak = max(peak, mem_mb(L, N))
    mem_req = int(math.ceil(peak / 1024.0)) * 2          # ~2x headroom, whole GB
    time_req = max(2, int(math.ceil(slowest / 3600.0 * 4)))  # 4x the slowest task

    W = 34
    def p(k, v): print(f"  {k:<{W}} {v}")
    print("=" * 78)
    print(f"  PREFLIGHT — {arm}    (this script does NOT submit anything)")
    print("=" * 78)
    p("arm", arm)
    def block(label, text):
        lines = textwrap.wrap(text, width=76 - W) or [""]
        print(f"  {label:<{W}}{lines[0]}")
        for ln in lines[1:]:
            print(f"  {'':<{W}}{ln}")
    block("scientific question", question)
    p("manifest rows", len(rows))
    p("L", ", ".join(map(str, Ls)))
    p("T", ", ".join(f"{t:g}" for t in Ts))
    p("zeta", ", ".join(f"{z:g}" for z in zs))
    p("lambda", ", ".join(f"{l:g}" for l in lms))
    p("dtau_mult", ", ".join(f"{d:g}" for d in dts))
    p("resample_scheme", ", ".join(sch))
    p("N_c ladder (this arm)", ", ".join(str(n) for n in sorted(ladder)))
    p("independent populations R per N_c", ", ".join(f"{n}:{ladder[n]}" for n in sorted(ladder)))
    p("n_steps per run", ", ".join(str(x) for x in sorted(
        {n_steps(int(r["L"]), float(r["T"]), float(r["lam"]), float(r["dtau_mult"])) for r in rows})))
    print("  " + "-" * 74)
    # Read what submit.slurm ACTUALLY requests and cross-check it. A preflight
    # that prints its own suggestion rather than the batch script's real request
    # tells you nothing about what you are about to queue.
    sb = {}
    sp = os.path.join(HERE, "submit.slurm")
    if os.path.isfile(sp):
        for line in open(sp):
            m = re.match(r"#SBATCH\s+--(\S+?)=(\S+)", line.strip())
            if m:
                sb[m.group(1)] = m.group(2)
    want_array = f"0-{len(rows) - 1}"
    got_array = sb.get("array", "?").split("%")[0]
    ok_array = (got_array == want_array)
    p("expected array size", f"{want_array}   ({len(rows)} tasks)")
    p("submit.slurm --array", f"{sb.get('array','MISSING')}"
                              f"{'   OK' if ok_array else '   ** MISMATCH, expected ' + want_array + ' **'}")
    p("expected core-hours", f"{tot / 3600:.1f}")
    p("slowest single task", f"{slowest / 3600:.2f} core-h")
    p("peak memory per task", f"{peak:.0f} MB")
    p("memory: computed need", f"{peak:.0f} MB  (suggest {mem_req}G)")
    p("submit.slurm --mem", f"{sb.get('mem','MISSING')}"
                            f"{'' if _gb(sb.get('mem')) >= peak/1024*1.5 else '   ** TIGHT **'}")
    p("time: computed need", f"{slowest/3600:.2f} h  (suggest {time_req:02d}:00:00)")
    p("submit.slurm --time", f"{sb.get('time','MISSING')}"
                             f"{'' if _hrs(sb.get('time')) >= slowest/3600*2 else '   ** TIGHT **'}")
    p("output directory", os.path.join(HERE, "results"))
    p("analysis-spec sha256", spec_hash)
    print("  " + "-" * 74)
    block("primary statistic", primary)
    block("decision rule", rule)
    print("=" * 78)
    done = len([f for f in os.listdir(os.path.join(HERE, "results"))
                if f.endswith(".json")]) if os.path.isdir(os.path.join(HERE, "results")) else 0
    print(f"  results already present: {done} / {len(rows)}"
          f"{'   (a resubmission will SKIP these)' if done else ''}")
    print(f"  PPSQJ_REPO = {os.environ.get('PPSQJ_REPO', '** NOT SET — run_cell.py will refuse **')}")
    print("=" * 78)
    print("  NOTHING WAS SUBMITTED. To submit, read RUCHE_RUNBOOK.md and type the")
    print("  sbatch command yourself.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
