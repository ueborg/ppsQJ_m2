#!/usr/bin/env python
"""Comprehensive parallel diagnostic suite for the cloning codebase.

Runs five diagnostics (D1-D5) on the Habrok interactive node in <= ~3h wall,
using a self-calibrating scheduler that caps per-task cost and trims by
priority to fit a wall-time budget.  Designed to settle the open questions
left by the entropy_stride / PPS_BACKEND changes:

  D1  backend N_c crossover    -- is 'batched' ever faster, and at what N_c?
  D2  entropy_stride safety    -- are stride=1 and stride=4 statistically equal?
  D3  backend equivalence      -- do scalar/batched give the same physics?
  D4  hot-path cost profile    -- is orbitals_from_covariance ~37% on Habrok?
  D5  N_c variance scaling      -- does rel-err ~ 1/sqrt(N_c)? (MPI viability)

Usage:
    cd ~/pps_qj && source ~/venvs/pps_qj/bin/activate
    python analysis/diagnostic_suite.py [--outdir outputs/diagnostics] \
        [--budget-hours 2.5] [--max-task-min 40] [--workers 24]

The suite prints a time-budget estimate (from a quick calibration) BEFORE
dispatching, prints live progress as each task finishes, and writes results
to JSON incrementally so a dropped terminal does not lose completed work.

D1 and D4 are highest priority; D2/D3 medium; D5 low (trimmed first).
"""
from __future__ import annotations

# BLAS must be single-threaded: parallelism is across processes, one core each.
# These MUST be set before numpy is imported (here or inherited from the shell).
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import cProfile
import io
import json
import math
import pstats
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# Fixed physical point for all diagnostics: moderate lambda near criticality.
LAM_DEFAULT = 0.31
ZETA_DEFAULT = 0.30


# ---------------------------------------------------------------------------
# Top-level task workers (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _one_run(params: dict) -> dict:
    """Run a single run_cloning and return timing + (optional) observables.

    params keys: L, lam, zeta, T, N_c, seed, backend, entropy_stride,
                 record_entropy, compute_bl.
    Imports are done inside so the worker is robust under both fork and spawn.
    """
    import numpy as _np
    from pps_qj.cloning import run_cloning, CloningCollapse
    from pps_qj.gaussian_backend import build_gaussian_chain_model

    L    = int(params["L"]);   lam = float(params["lam"])
    zeta = float(params["zeta"]); T = float(params["T"])
    N_c  = int(params["N_c"]); seed = int(params["seed"])
    backend        = params.get("backend", "scalar")
    entropy_stride = int(params.get("entropy_stride", 1))
    record_entropy = bool(params.get("record_entropy", True))
    compute_bl     = bool(params.get("compute_bl", False))

    alpha, w = lam, 1.0 - lam
    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    rng = _np.random.default_rng(seed)

    out = {"ok": True, "wall": float("nan")}
    t0 = time.perf_counter()
    try:
        res = run_cloning(
            model, zeta=zeta, T_total=T, N_c=N_c, rng=rng,
            show_progress=False, record_entropy=record_entropy,
            entropy_stride=entropy_stride, backend=backend,
        )
    except CloningCollapse as exc:
        return {"ok": False, "error": f"collapse: {exc}",
                "wall": time.perf_counter() - t0}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc),
                "wall": time.perf_counter() - t0, "tb": traceback.format_exc()}
    out["wall"] = time.perf_counter() - t0

    out.update({
        "theta_hat": float(res.theta_hat),
        "S_mean": float(res.S_mean), "S_std": float(res.S_std),
        "S_var": float(res.S_var), "covar_Sk": float(res.covar_Sk),
        "chi_k": float(res.chi_k), "n_T_mean": float(res.n_T_mean),
        "min_ess_frac": float(res.min_ess_frac_postburnin),
        "n_distinct_ancestors": int(res.n_distinct_ancestors),
        "n_steps": int(res.S_history.shape[0]) if hasattr(res.S_history, "shape")
                   else len(res.S_history),
    })
    if compute_bl:
        from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L
        comps = _batched_compute_B_L(res.final_covs, L)
        bl, cmi = comps["B_L"], comps["CMI"]
        fmb, fmc = _np.isfinite(bl), _np.isfinite(cmi)
        out["B_L_mean"]  = float(_np.mean(bl[fmb]))  if fmb.any() else float("nan")
        out["CMI_mean"]  = float(_np.mean(cmi[fmc])) if fmc.any() else float("nan")
    return out


def _profile_run(params: dict) -> dict:
    """cProfile a single scalar run; return tottime fractions per hot function."""
    import numpy as _np
    from pps_qj.cloning import run_cloning
    from pps_qj.gaussian_backend import build_gaussian_chain_model

    L = int(params["L"]); N_c = int(params["N_c"]); T = float(params["T"])
    alpha, w = float(params["lam"]), 1.0 - float(params["lam"])
    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    rng = _np.random.default_rng(int(params.get("seed", 7)))

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    run_cloning(model, zeta=float(params["zeta"]), T_total=T, N_c=N_c, rng=rng,
                show_progress=False, record_entropy=True, backend="scalar")
    pr.disable()
    wall = time.perf_counter() - t0

    st = pstats.Stats(pr)
    total = sum(v[2] for v in st.stats.values()) or 1.0  # sum tottime
    # Bucket tottime by hot-function name (substring match on the func name).
    buckets = {"orbitals_from_covariance": 0.0, "branch_norm/brentq": 0.0,
               "batched_entanglement_entropy": 0.0, "eigh/eigvalsh": 0.0,
               "qr": 0.0, "matmul/dot": 0.0, "covariance_from_orbitals": 0.0,
               "apply_projective_jump": 0.0}
    for (fname, _line, func), v in st.stats.items():
        tt = v[2]; name = f"{func}"
        if "orbitals_from_covariance" in name: buckets["orbitals_from_covariance"] += tt
        elif "branch_norm" in name or "brentq" in name: buckets["branch_norm/brentq"] += tt
        elif "batched_entanglement_entropy" in name: buckets["batched_entanglement_entropy"] += tt
        elif "eigh" in name or "eigvalsh" in name: buckets["eigh/eigvalsh"] += tt
        elif name == "qr" or "qr" == name.split(".")[-1]: buckets["qr"] += tt
        elif "covariance_from_orbitals" in name: buckets["covariance_from_orbitals"] += tt
        elif "apply_projective_jump" in name: buckets["apply_projective_jump"] += tt
        elif name in ("dot", "matmul") or "matmul" in name: buckets["matmul/dot"] += tt
    fracs = {k: float(v / total) for k, v in buckets.items()}

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(20)
    return {"ok": True, "L": L, "wall": wall, "total_tottime": float(total),
            "fractions": fracs, "top20": s.getvalue()}


# ---------------------------------------------------------------------------
# Cost calibration: fit cost(L) ~ coeff * L^p (per backend), linear in N_c, T.
# ---------------------------------------------------------------------------

def calibrate(workers: int) -> dict:
    """Quick two-point calibration per backend. Returns a cost-model dict.

    Runs at small L with N_c=20, T=2 (a handful of seconds each).  The two
    L points are timed in parallel.  Linear scaling in N_c and T is assumed
    (true for this algorithm: n_steps ~ T, work ~ N_c per step).
    """
    cal_L = (32, 48)
    cfgs = [dict(L=L, lam=LAM_DEFAULT, zeta=ZETA_DEFAULT, T=2.0, N_c=20,
                 seed=11 + i, backend=b, record_entropy=True)
            for b in ("scalar", "batched") for i, L in enumerate(cal_L)]
    print(f"  calibrating cost model ({len(cfgs)} short runs) ...", flush=True)
    times: dict[tuple, float] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(cfgs))) as ex:
        futs = {ex.submit(_one_run, c): (c["backend"], c["L"]) for c in cfgs}
        for fut in as_completed(futs):
            b, L = futs[fut]
            r = fut.result()
            times[(b, L)] = r["wall"] if r.get("ok") else float("nan")
            print(f"    cal {b:7s} L={L}: {times[(b, L)]:.2f}s", flush=True)

    model = {"cal_Nc": 20, "cal_T": 2.0, "per": {}}
    for b in ("scalar", "batched"):
        t1, t2 = times.get((b, cal_L[0])), times.get((b, cal_L[1]))
        if t1 and t2 and t1 > 0 and t2 > 0 and math.isfinite(t1) and math.isfinite(t2):
            p = math.log(t2 / t1) / math.log(cal_L[1] / cal_L[0])
            coeff = t1 / (cal_L[0] ** p)
            if not (1.5 < p < 7.0):  # implausible fit -> safe fallback
                p, coeff = 4.0, t1 / (cal_L[0] ** 4.0)
        else:
            p, coeff = 4.0, 5e-6  # very rough fallback
        model["per"][b] = {"coeff": coeff, "p": p}
    print(f"    fitted exponents: scalar p={model['per']['scalar']['p']:.2f}, "
          f"batched p={model['per']['batched']['p']:.2f}", flush=True)
    return model


def est_cost(model: dict, L: int, N_c: int, T: float,
             backend: str = "scalar", profile: bool = False) -> float:
    """Estimated wall-seconds for one run, from the calibrated model."""
    per = model["per"].get(backend, model["per"]["scalar"])
    base = per["coeff"] * (L ** per["p"])
    cost = base * (N_c / model["cal_Nc"]) * (T / model["cal_T"])
    return cost * (2.5 if profile else 1.0)  # cProfile overhead factor


# ---------------------------------------------------------------------------
# Task builders.  Each task: dict(diag, key, kind, priority, params).
#   priority: 1 = highest (trim last).  D1,D4=1 ; D2,D3=2 ; D5=3 (trim first).
# ---------------------------------------------------------------------------

def build_tasks(args) -> list[dict]:
    tasks: list[dict] = []

    # D1 -- backend N_c crossover (record_entropy=False isolates the backend).
    for L in (64, 96):
        for N_c in (40, 100, 200, 250, 400):
            for backend in ("scalar", "batched"):
                for rep in range(args.d1_reps):
                    tasks.append(dict(
                        diag="D1", kind="run", priority=1,
                        key=f"L{L}_Nc{N_c}_{backend}_r{rep}",
                        params=dict(L=L, lam=LAM_DEFAULT, zeta=ZETA_DEFAULT,
                                    T=args.d1_T, N_c=N_c, seed=100 + rep,
                                    backend=backend, record_entropy=False)))

    # D2 -- entropy stride safety (scalar; stride logic is backend-independent).
    for L in (32, 64):
        for stride in (1, 4):
            for rep in range(args.d2_reps):
                tasks.append(dict(
                    diag="D2", kind="run", priority=2,
                    key=f"L{L}_stride{stride}_r{rep}",
                    params=dict(L=L, lam=LAM_DEFAULT, zeta=ZETA_DEFAULT,
                                T=args.d2_T, N_c=100, seed=200 + rep,
                                backend="scalar", entropy_stride=stride,
                                record_entropy=True)))

    # D3 -- scalar vs batched equivalence (paired on seed).
    for seed in range(args.d3_seeds):
        for backend in ("scalar", "batched"):
            tasks.append(dict(
                diag="D3", kind="run", priority=2,
                key=f"seed{seed}_{backend}",
                params=dict(L=64, lam=LAM_DEFAULT, zeta=ZETA_DEFAULT,
                            T=args.d3_T, N_c=100, seed=300 + seed,
                            backend=backend, record_entropy=True,
                            compute_bl=True)))

    # D4 -- hot-path profile (scalar, cProfile).
    for L in (64, 96, 128):
        tasks.append(dict(
            diag="D4", kind="profile", priority=1, key=f"L{L}",
            params=dict(L=L, lam=LAM_DEFAULT, zeta=ZETA_DEFAULT,
                        T=args.d4_T, N_c=30, seed=7)))

    # D5 -- N_c variance scaling (B_L/CMI rel-err vs N_c).
    for N_c in (50, 100, 200, 400):
        for rep in range(args.d5_nreal):
            tasks.append(dict(
                diag="D5", kind="run", priority=3,
                key=f"Nc{N_c}_r{rep}",
                params=dict(L=64, lam=LAM_DEFAULT, zeta=ZETA_DEFAULT,
                            T=args.d5_T, N_c=N_c, seed=500 + rep,
                            backend="scalar", record_entropy=True,
                            compute_bl=True)))
    return tasks


# ---------------------------------------------------------------------------
# Scheduler: estimate cost, cap per-task, trim by priority to fit the budget.
# ---------------------------------------------------------------------------

def schedule(tasks: list[dict], model: dict, workers: int,
             budget_s: float, max_task_s: float) -> tuple[list[dict], list[dict]]:
    for t in tasks:
        p = t["params"]
        t["est"] = est_cost(model, p["L"], p["N_c"], p["T"],
                            p.get("backend", "scalar"), profile=(t["kind"] == "profile"))

    kept, trimmed = [], []
    # 1) Hard per-task cap: no single serial run may dominate the wall.
    for t in tasks:
        (trimmed if t["est"] > max_task_s else kept).append(t)
    for t in trimmed:
        t["trim_reason"] = f"est {t['est']/60:.0f}min > cap {max_task_s/60:.0f}min"

    def est_wall(ts: list[dict]) -> float:
        if not ts:
            return 0.0
        total = sum(t["est"] for t in ts)
        return max(max(t["est"] for t in ts), total / workers) * 1.15  # +overhead

    # 2) Trim by priority (drop highest-priority-number, then highest-cost).
    while kept and est_wall(kept) > budget_s:
        victim = max(kept, key=lambda t: (t["priority"], t["est"]))
        victim["trim_reason"] = f"budget: wall est > {budget_s/3600:.1f}h"
        kept.remove(victim)
        trimmed.append(victim)
    return kept, trimmed


def print_budget(kept: list[dict], trimmed: list[dict], workers: int,
                 budget_s: float) -> None:
    print("\n" + "=" * 72)
    print("TIME BUDGET ESTIMATE (from calibration)")
    print("=" * 72)
    diags = ("D1", "D2", "D3", "D4", "D5")
    print(f"{'diag':>5} {'kept':>5} {'trim':>5} {'serial_h':>10} {'maxtask_min':>12}")
    for d in diags:
        k = [t for t in kept if t["diag"] == d]
        tr = [t for t in trimmed if t["diag"] == d]
        serial = sum(t["est"] for t in k) / 3600.0
        mx = max((t["est"] for t in k), default=0.0) / 60.0
        print(f"{d:>5} {len(k):>5} {len(tr):>5} {serial:>10.2f} {mx:>12.1f}")
    total = sum(t["est"] for t in kept)
    wall = max(max((t["est"] for t in kept), default=0.0), total / workers) * 1.15
    print("-" * 72)
    print(f"  total serial work : {total/3600:.2f} h")
    print(f"  workers           : {workers}")
    print(f"  estimated wall    : {wall/3600:.2f} h   (budget {budget_s/3600:.1f} h)")
    if trimmed:
        print(f"\n  TRIMMED {len(trimmed)} tasks to fit budget:")
        for t in trimmed:
            print(f"    - {t['diag']} {t['key']}: {t.get('trim_reason', '?')}")
    print("=" * 72, flush=True)


# ---------------------------------------------------------------------------
# Aggregators -> verdicts.  `R` maps (diag, key) -> merged result dict
# (task params + payload).  Each returns a JSON-able summary dict.
# ---------------------------------------------------------------------------

def _mse(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), (float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1
                             else float("nan")), int(a.size)


def agg_D1(R):
    rows, by_L = {}, {}
    for (diag, key), r in R.items():
        if diag != "D1" or not r.get("ok"):
            continue
        L, Nc, b = r["params"]["L"], r["params"]["N_c"], r["params"]["backend"]
        rows.setdefault((L, Nc, b), []).append(r["wall"])
    out = {"per_point": [], "crossover": {}, "prod_estimate_Nc250": {}}
    for L in (64, 96):
        for Nc in (40, 100, 200, 250, 400):
            sc = rows.get((L, Nc, "scalar")); ba = rows.get((L, Nc, "batched"))
            if not sc or not ba:
                continue
            sm, bm = float(np.mean(sc)), float(np.mean(ba))
            sp = sm / bm if bm > 0 else float("nan")
            out["per_point"].append(dict(L=L, N_c=Nc, scalar_s=sm, batched_s=bm,
                                         speedup=sp))
            by_L.setdefault(L, []).append((Nc, sp))
    for L, pts in by_L.items():
        pts.sort()
        cross = next((Nc for Nc, sp in pts if sp >= 1.0), None)
        out["crossover"][L] = cross
        near = min(pts, key=lambda x: abs(x[0] - 250), default=(None, None))
        out["prod_estimate_Nc250"][L] = near[1]
    return out


def agg_D2(R):
    g = {}
    for (diag, key), r in R.items():
        if diag != "D2" or not r.get("ok"):
            continue
        L, s = r["params"]["L"], r["params"]["entropy_stride"]
        g.setdefault((L, s), {"S_mean": [], "S_var": [], "covar_Sk": [],
                              "S_std": [], "wall": []})
        for k in ("S_mean", "S_var", "covar_Sk", "S_std", "wall"):
            g[(L, s)][k].append(r.get(k))
    out = {"per_L": [], "any_nan": False, "safe": True}
    for L in (32, 64):
        a1, a4 = g.get((L, 1)), g.get((L, 4))
        if not a1 or not a4:
            continue
        rec = {"L": L}
        for obs in ("S_mean", "S_var", "covar_Sk"):
            m1, e1, _ = _mse(a1[obs]); m4, e4, _ = _mse(a4[obs])
            if not (np.isfinite(m1) and np.isfinite(m4)):
                out["any_nan"] = True; out["safe"] = False
                rec[obs] = dict(stride1=m1, stride4=m4, z=float("nan"), nan=True)
                continue
            denom = math.sqrt((e1 or 0) ** 2 + (e4 or 0) ** 2) or float("nan")
            z = abs(m1 - m4) / denom if denom and np.isfinite(denom) else float("nan")
            if np.isfinite(z) and z > 3.0:
                out["safe"] = False
            rec[obs] = dict(stride1=m1, stride4=m4, z=z, nan=False)
        w1 = float(np.mean(a1["wall"])); w4 = float(np.mean(a4["wall"]))
        rec["wall_saving_pct"] = 100.0 * (1 - w4 / w1) if w1 > 0 else float("nan")
        out["per_L"].append(rec)
    return out


def agg_D3(R):
    pair = {}
    for (diag, key), r in R.items():
        if diag != "D3" or not r.get("ok"):
            continue
        seed = r["params"]["seed"]; b = r["params"]["backend"]
        pair.setdefault(seed, {})[b] = r
    obs = ("theta_hat", "S_mean", "B_L_mean", "CMI_mean")
    diffs = {o: [] for o in obs}
    for seed, d in pair.items():
        if "scalar" not in d or "batched" not in d:
            continue
        for o in obs:
            sv, bv = d["scalar"].get(o), d["batched"].get(o)
            if sv is not None and bv is not None and np.isfinite(sv) and np.isfinite(bv):
                diffs[o].append(bv - sv)
    out = {"per_obs": {}, "pass": True}
    for o in obs:
        a = np.asarray(diffs[o], float)
        if a.size == 0:
            out["per_obs"][o] = dict(n=0); continue
        md, sd = float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else float("nan")
        se = sd / math.sqrt(a.size) if a.size > 1 else float("nan")
        # consistent if mean paired diff is within ~3 SE of zero (no systematic bias)
        z = abs(md) / se if se and np.isfinite(se) and se > 0 else 0.0
        ok = (not np.isfinite(z)) or z < 3.0
        out["per_obs"][o] = dict(n=int(a.size), mean_diff=md, sd=sd, z=float(z), ok=bool(ok))
        if not ok:
            out["pass"] = False
    return out


def agg_D4(R):
    out = {"per_L": [], "dominant": None}
    best = 0.0
    for (diag, key), r in R.items():
        if diag != "D4" or not r.get("ok"):
            continue
        fr = r["fractions"]; L = r["L"]
        out["per_L"].append(dict(L=L, wall=r["wall"], fractions=fr,
                                 orbitals_frac=fr.get("orbitals_from_covariance", 0.0)))
        if L == 128:
            best = fr.get("orbitals_from_covariance", 0.0)
    out["per_L"].sort(key=lambda x: x["L"])
    out["orbitals_frac_L128"] = best
    out["dominant"] = ("orbitals_from_covariance" if best > 0.30 else "other")
    return out


def agg_D5(R):
    g = {}
    for (diag, key), r in R.items():
        if diag != "D5" or not r.get("ok"):
            continue
        Nc = r["params"]["N_c"]
        g.setdefault(Nc, {"B_L_mean": [], "CMI_mean": []})
        g[Nc]["B_L_mean"].append(r.get("B_L_mean"))
        g[Nc]["CMI_mean"].append(r.get("CMI_mean"))
    rows = []
    for Nc in sorted(g):
        bm, be, n = _mse(g[Nc]["B_L_mean"]); cm, ce, _ = _mse(g[Nc]["CMI_mean"])
        rows.append(dict(N_c=Nc, n=n,
                         BL_relErr=abs(be / bm) if bm else float("nan"),
                         CMI_relErr=abs(ce / cm) if cm else float("nan")))
    out = {"rows": rows}
    valid = [(r["N_c"], r["BL_relErr"]) for r in rows
             if np.isfinite(r["BL_relErr"]) and r["BL_relErr"] > 0]
    if len(valid) >= 3:
        Ncs, errs = zip(*valid)
        p = float(np.polyfit(np.log(Ncs), np.log(errs), 1)[0])
        out["fitted_exponent"] = p
        out["verdict"] = ("1/sqrt(N_c): MPI justified" if abs(p + 0.5) < 0.15
                          else "flatter than 1/sqrt(N_c): genealogical degeneracy"
                          if p > -0.3 else f"intermediate ({p:.2f})")
    else:
        out["fitted_exponent"] = None
        out["verdict"] = "insufficient valid points (collapse?)"
    return out


# ---------------------------------------------------------------------------
# Main: calibrate -> schedule -> dispatch (live) -> aggregate -> verdicts.
# ---------------------------------------------------------------------------

def _parse_args(argv):
    ap = argparse.ArgumentParser(description="Parallel cloning diagnostic suite")
    ap.add_argument("--outdir", type=str, default="outputs/diagnostics")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--budget-hours", type=float, default=2.5)
    ap.add_argument("--max-task-min", type=float, default=40.0)
    ap.add_argument("--d1-reps", type=int, default=3)
    ap.add_argument("--d1-T", type=float, default=5.0)
    ap.add_argument("--d2-reps", type=int, default=10)
    ap.add_argument("--d2-T", type=float, default=10.0)
    ap.add_argument("--d3-seeds", type=int, default=20)
    ap.add_argument("--d3-T", type=float, default=10.0)
    ap.add_argument("--d4-T", type=float, default=2.0)
    ap.add_argument("--d5-nreal", type=int, default=5)
    ap.add_argument("--d5-T", type=float, default=20.0)
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip calibration; use a crude fallback cost model")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, os.cpu_count() or 1))
    budget_s = args.budget_hours * 3600.0
    max_task_s = args.max_task_min * 60.0

    print("=" * 72)
    print("CLONING DIAGNOSTIC SUITE")
    print(f"  outdir={outdir}  workers={workers}  budget={args.budget_hours}h  "
          f"max_task={args.max_task_min}min")
    print(f"  BLAS threads pinned to {os.environ.get('OMP_NUM_THREADS')}")
    print("=" * 72, flush=True)

    if args.no_calibrate:
        model = {"cal_Nc": 20, "cal_T": 2.0,
                 "per": {"scalar": {"coeff": 5e-6, "p": 4.0},
                         "batched": {"coeff": 7e-6, "p": 4.0}}}
    else:
        model = calibrate(workers)

    tasks = build_tasks(args)
    kept, trimmed = schedule(tasks, model, workers, budget_s, max_task_s)
    print_budget(kept, trimmed, workers, budget_s)
    if not kept:
        print("No tasks fit the budget; relax --budget-hours/--max-task-min.")
        return 1


    # ---- Dispatch with live progress + incremental JSON ----
    ctx = None
    try:
        import multiprocessing as _mp
        ctx = _mp.get_context("fork")  # cheap on Linux; children inherit imports
    except (ValueError, RuntimeError):
        ctx = None

    R: dict[tuple, dict] = {}
    raw_path = outdir / "diagnostics_raw.json"
    t_start = time.time()
    n_done = 0; n_total = len(kept)
    print(f"\nDispatching {n_total} tasks on {workers} workers ...\n", flush=True)

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        fut_meta = {}
        for t in kept:
            fn = _profile_run if t["kind"] == "profile" else _one_run
            fut_meta[ex.submit(fn, t["params"])] = t
        for fut in as_completed(fut_meta):
            t = fut_meta[fut]
            try:
                payload = fut.result()
            except Exception as exc:  # noqa: BLE001
                payload = {"ok": False, "error": repr(exc)}
            rec = {"diag": t["diag"], "key": t["key"], "kind": t["kind"],
                   "params": t["params"], **payload}
            R[(t["diag"], t["key"])] = rec
            n_done += 1
            el = time.time() - t_start
            status = "ok" if payload.get("ok") else f"FAIL({payload.get('error','?')[:40]})"
            print(f"[{n_done:>3}/{n_total}] {t['diag']} {t['key']:<26} "
                  f"{payload.get('wall', float('nan')):7.1f}s  {status}  "
                  f"(elapsed {el/60:.1f}min)", flush=True)
            if n_done % 5 == 0 or n_done == n_total:
                raw_path.write_text(json.dumps(list(R.values()), indent=1, default=float))

    raw_path.write_text(json.dumps(list(R.values()), indent=1, default=float))

    # ---- Aggregate + verdicts ----
    summary = {"D1": agg_D1(R), "D2": agg_D2(R), "D3": agg_D3(R),
               "D4": agg_D4(R), "D5": agg_D5(R),
               "meta": {"workers": workers, "wall_min": (time.time() - t_start) / 60.0,
                        "n_done": n_done, "n_trimmed": len(trimmed),
                        "trimmed": [f"{t['diag']}:{t['key']}" for t in trimmed]}}
    (outdir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    _print_verdicts(summary)
    print(f"\nRaw -> {raw_path}\nSummary -> {outdir/'diagnostics_summary.json'}",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# Verdict printer -- the explicit decision lines requested in the spec.
# ---------------------------------------------------------------------------

def _print_verdicts(s: dict) -> None:
    print("\n" + "=" * 72)
    print("VERDICTS")
    print("=" * 72)

    # D1 -> BACKEND_DEFAULT
    d1 = s["D1"]; cross = d1.get("crossover", {})
    parts = []
    for L in (64, 96):
        c = cross.get(L)
        parts.append(f"L={L}: " + (f"batched wins at N_c>={c}" if c else "batched never wins"))
    p250 = d1.get("prod_estimate_Nc250", {})
    sp = p250.get(128, p250.get(96, p250.get(64)))
    any_win = any(cross.get(L) for L in (64, 96))
    rec = ("'scalar' (batched shows no win in tested range)" if not any_win
           else "'scalar' until verified at the exact production (L,N_c); "
                f"batched crossover seen at {cross}")
    print(f"BACKEND_DEFAULT: {rec}")
    print(f"    {' | '.join(parts)};  nearest-N_c=250 speedup est: {sp}")

    # D2 -> ENTROPY_STRIDE
    d2 = s["D2"]
    verdict = "UNSAFE (NaN -> burn-in indexing bug present)" if d2.get("any_nan") \
        else ("SAFE (stride=4 statistically consistent with stride=1)" if d2.get("safe")
              else "UNSAFE (stride=4 differs > 3 sigma from stride=1)")
    print(f"ENTROPY_STRIDE: {verdict}")
    for rec in d2.get("per_L", []):
        zs = {k: round(rec[k]["z"], 2) for k in ("S_mean", "S_var", "covar_Sk") if k in rec}
        print(f"    L={rec['L']}: z={zs}  wall_saving={rec.get('wall_saving_pct', float('nan')):.1f}%")

    # D3 -> STATISTICAL_EQUIV
    d3 = s["D3"]
    print(f"STATISTICAL_EQUIV: {'PASS' if d3.get('pass') else 'FAIL'}")
    for o, v in d3.get("per_obs", {}).items():
        if v.get("n"):
            print(f"    {o}: mean(batched-scalar)={v['mean_diff']:.3e} "
                  f"z={v['z']:.2f} {'ok' if v['ok'] else 'DIVERGENT'}")

    # D4 -> HOT_PATH
    d4 = s["D4"]
    is_dom = d4.get("dominant") == "orbitals_from_covariance"
    print(f"HOT_PATH: orbitals_from_covariance {'IS' if is_dom else 'is NOT'} dominant "
          f"(L=128 frac={d4.get('orbitals_frac_L128', 0.0):.1%})")
    for rec in d4.get("per_L", []):
        print(f"    L={rec['L']}: orbitals={rec['orbitals_frac']:.1%} "
              f"brentq={rec['fractions'].get('branch_norm/brentq', 0):.1%} "
              f"entropy={rec['fractions'].get('batched_entanglement_entropy', 0):.1%} "
              f"eigh={rec['fractions'].get('eigh/eigvalsh', 0):.1%}")

    # D5 -> MPI_VIABILITY
    d5 = s["D5"]; exp = d5.get("fitted_exponent")
    exp_s = f"{exp:.3f}" if isinstance(exp, (int, float)) else "n/a"
    print(f"MPI_VIABILITY: rel-err ~ N_c^({exp_s})  ->  {d5.get('verdict')}")
    for r in d5.get("rows", []):
        print(f"    N_c={r['N_c']:>4}: BL_relErr={r['BL_relErr']:.1%} "
              f"CMI_relErr={r['CMI_relErr']:.1%}  (n={r['n']})")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    sys.exit(main())
