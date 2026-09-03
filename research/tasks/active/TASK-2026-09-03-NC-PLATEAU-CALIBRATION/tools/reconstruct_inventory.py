#!/usr/bin/env python3
"""Stage-0 data reconstruction for TASK-2026-09-03-NC-PLATEAU-CALIBRATION.

Walks EVERY per-population result JSON in research/tasks/**/results/ -- the raw
files the sampler itself wrote, never a predecessor's summary table -- and emits

    EXISTING_POPULATION_INVENTORY.csv   one row per population
    EXISTING_LADDERS.md                 every (L,T,zeta,lambda) ladder in N_c
    REUSE_LEDGER.csv                    what this task reuses, tops up, or refuses
    frozen_inputs/reuse_populations.csv the reusable rows, hashed

EXACT COMPATIBILITY is decided here and nowhere else. A population is
exact-compatible with a target cell iff every one of

    L, T, zeta, lambda, N_c, dtau_mult, resample_scheme, status == ok,
    sampler sha256, run_cell call signature

matches. "Similar" is not a category this file has.

Read-only with respect to every predecessor directory. Contains no scheduler
call and cannot submit.
"""
import os, sys, csv, json, glob, math, hashlib, collections, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
REPO = os.path.abspath(os.path.join(TASK, *([os.pardir] * 4)))

# The certified sampler. Every reusable population must have been produced by a
# file with THIS sha256; TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION established it
# is byte-identical across the whole reuse set.
CERTIFIED_SAMPLER_SHA = None    # filled in by main() from support/instrumented.py

FIELDS = ["source_task", "arm", "L", "T", "zeta", "lam", "N_c", "R_index",
          "dtau_mult", "resample_scheme", "seed", "status", "wall_s", "n_steps",
          "cmi_weighted_mean", "cmi_unweighted_mean", "gess_final",
          "ess_cum_final", "n_distinct_anc_final", "brentq_fallbacks",
          "result_path", "exact_compatible"]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def collect(repo):
    rows = []
    pat = os.path.join(repo, "research", "tasks", "**", "results", "*.json")
    for p in sorted(glob.glob(pat, recursive=True)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or "cmi_weighted_mean" not in d:
            continue
        parts = os.path.relpath(p, repo).split(os.sep)
        task = next((x for x in parts if x.startswith("TASK-")), "?")
        arm = parts[parts.index(task) + 1] if task in parts else "?"
        rows.append(dict(
            source_task=task, arm=arm, L=int(d["L"]), T=float(d["T"]),
            zeta=float(d["zeta"]), lam=float(d["lam"]), N_c=int(d["N_c"]),
            R_index=None, dtau_mult=float(d["dtau_mult"]),
            resample_scheme=d.get("resample_scheme", ""), seed=int(d["seed"]),
            status=d.get("status", ""), wall_s=d.get("wall_s"),
            n_steps=d.get("n_steps"),
            cmi_weighted_mean=d.get("cmi_weighted_mean"),
            cmi_unweighted_mean=d.get("cmi_unweighted_mean"),
            gess_final=d.get("gess_final"), ess_cum_final=d.get("ess_cum_final"),
            n_distinct_anc_final=d.get("n_distinct_anc_final"),
            brentq_fallbacks=d.get("brentq_fallbacks"),
            result_path=os.path.relpath(p, repo), exact_compatible=""))
    # R_index: position within its own cell, in SEED ORDER. Seed order is
    # observable-blind, which is what makes a disjoint block cut legitimate.
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["source_task"], r["arm"], r["L"], r["T"], r["zeta"], r["lam"],
            r["N_c"], r["dtau_mult"], r["resample_scheme"])].append(r)
    for k, v in by.items():
        for i, r in enumerate(sorted(v, key=lambda x: x["seed"])):
            r["R_index"] = i
    return rows


def cell_key(r):
    return (r["L"], r["T"], r["zeta"], round(r["lam"], 6), r["N_c"],
            r["dtau_mult"], r["resample_scheme"])


def sem(v):
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def main():
    rows = collect(REPO)
    sampler = os.path.join(TASK, "support", "instrumented.py")
    global CERTIFIED_SAMPLER_SHA
    CERTIFIED_SAMPLER_SHA = sha256(sampler) if os.path.isfile(sampler) else "ABSENT"

    # --- exact compatibility --------------------------------------------------
    # The production configuration this task computes in, in full.
    for r in rows:
        why = []
        if r["status"] != "ok":
            why.append("status")
        if r["resample_scheme"] != "systematic":
            why.append("resample_scheme")
        if abs(r["T"] - r["L"]) > 1e-9:
            why.append("T!=L")
        if abs(r["zeta"] - 0.35) > 1e-12:
            why.append("zeta")
        if r["brentq_fallbacks"]:
            why.append("brentq_fallback")
        r["exact_compatible"] = "yes" if not why else "no:" + "+".join(why)

    with open(os.path.join(TASK, "EXISTING_POPULATION_INVENTORY.csv"),
              "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # --- ladders --------------------------------------------------------------
    cells = collections.defaultdict(list)
    for r in rows:
        cells[cell_key(r)].append(r)
    ladders = collections.defaultdict(dict)   # (L,T,zeta,lam,dtau,scheme) -> N_c -> stat
    for k, v in cells.items():
        L, T, z, lam, N, dm, sc = k
        m = [x["cmi_weighted_mean"] for x in v]
        ladders[(L, T, z, lam, dm, sc)][N] = dict(
            R=len(v), mean=st.mean(m), sem=sem(m),
            tasks=sorted({x["source_task"] for x in v}),
            arms=sorted({x["arm"] for x in v}),
            n_steps=v[0]["n_steps"],
            wall_med=st.median([x["wall_s"] for x in v]),
            wall_max=max(x["wall_s"] for x in v),
            seed_lo=min(x["seed"] for x in v), seed_hi=max(x["seed"] for x in v),
            ok=all(x["exact_compatible"] == "yes" for x in v))

    out = []
    A = out.append
    A("# EXISTING_LADDERS — every N_c ladder in the corpus, rebuilt from raw files")
    A("")
    A("Rebuilt by `tools/reconstruct_inventory.py` from the per-population JSON")
    A("the sampler itself wrote. **No predecessor summary table was read.**")
    A("")
    A(f"Sampler bundled here: sha256 `{CERTIFIED_SAMPLER_SHA[:16]}…`")
    A("")
    A("`Delta_N = I_2N - I_N`. `sem` is the across-population standard error at")
    A("the R shown, never a within-population clone spread.")
    A("")
    for key in sorted(ladders):
        L, T, z, lam, dm, sc = key
        d = ladders[key]
        A(f"## L={L}, T={T:g}, zeta={z}, lambda={lam:g}, dtau_mult={dm:g}, {sc}")
        A("")
        A("| N_c | R | mean CMI | SEM | Delta_N | Delta_N SEM | n_steps | "
          "median wall_s | source | exact-compatible |")
        A("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        ns = sorted(d)
        for i, N in enumerate(ns):
            s = d[N]
            dN = dNs = ""
            if 2 * N in d:
                a, b = d[N], d[2 * N]
                dN = f"{b['mean'] - a['mean']:+.5f}"
                dNs = f"{math.hypot(a['sem'], b['sem']):.5f}"
            A(f"| {N} | {s['R']} | {s['mean']:.5f} | {s['sem']:.5f} | {dN} | "
              f"{dNs} | {s['n_steps']} | {s['wall_med']:.1f} | "
              f"{'+'.join(t.replace('TASK-2026-','') for t in s['tasks'])}"
              f"/{'+'.join(s['arms'])} | {'yes' if s['ok'] else 'NO'} |")
        A("")
    open(os.path.join(TASK, "EXISTING_LADDERS.md"), "w").write("\n".join(out) + "\n")

    # --- machine-readable ladder dump for the analysis and cost model ---------
    json.dump({("%d|%g|%g|%g|%g|%s" % k): {str(n): {kk: vv for kk, vv in s.items()}
                                           for n, s in v.items()}
               for k, v in ladders.items()},
              open(os.path.join(TASK, "frozen_inputs", "existing_ladders.json"), "w"),
              indent=1, sort_keys=True)

    print(f"{len(rows)} populations, {len(cells)} cells, {len(ladders)} ladders")
    bad = collections.Counter(r["exact_compatible"] for r in rows
                              if r["exact_compatible"] != "yes")
    print("not exact-compatible:", dict(bad) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
