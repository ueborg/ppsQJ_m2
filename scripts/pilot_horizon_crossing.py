#!/usr/bin/env python3
"""Production-equivalent horizon/crossing pilot for low-zeta Cut B.

This is deliberately a wrapper around scripts/exp_adaptive_cloning.py. It does
not reimplement the QJ dynamics or the resampler. The production-equivalence
gate from that file runs before any study work.

Question:
    Does the cross-L transition locator move between T=L and T=2L at production-
    relevant L, and does tail-averaged CMI reduce crossing multiplicity?

Important:
    - lambda values are ABSOLUTE and explicitly supplied.
    - no sqrt(zeta) law is assumed.
    - "never" is an exact-target self-normalized importance-sampling arm, but
      normalized observable estimates can have finite-N importance-sampling bias.
    - agents do not submit Ruche jobs.
"""
from __future__ import annotations
import os, sys, json, time, argparse, traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from exp_adaptive_cloning import run_adaptive, gate
from omnibus_observables import observables
from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.cloning import CloningCollapse


def seed_of(L: int, lam: float, zeta: float, real: int) -> int:
    # Deliberately independent of Tmult and mode: T=2L reuses the same random
    # prefix as T=L when the effective cloning-window duration is kept fixed.
    return (
        91_000_000
        + int(L) * 1_000_000
        + int(round(zeta * 10_000)) * 10_000
        + int(round(lam * 10_000)) * 100
        + int(real)
    )


def fixed_window_dt(L: int, lam: float) -> float:
    """Base-T effective window duration, reused for all horizon multiples.

    Production requests dtau_mult=12. We first resolve the T=L run's integer
    number of windows, then use its effective dt for every Tmult. Therefore
    T=2L has exactly twice as many windows of the same duration, making the
    T=L path a true random-number prefix of the T=2L path within each arm.
    """
    requested = 12.0 / max(2.0 * lam * (L - 1), 1e-12)
    n0 = max(1, int(np.ceil(float(L) / requested)))
    return float(L) / n0


def mode_cfg(tag: str):
    if tag == "always":
        return "always", 0.0
    if tag == "ess0.9":
        return "ess", 0.9
    if tag == "ess0.5":
        return "ess", 0.5
    if tag == "never":
        return "never", 0.0
    raise ValueError("unknown mode %r" % tag)


def checkpoint(outdir, L, Tmult, zeta, lam, mode, real):
    d = Path(outdir) / ("z%.3f" % zeta) / ("L%d_T%s" % (L, str(Tmult)))
    return d / ("lam%.4f_%s_real%03d.json" % (lam, mode, real))


def run_one(t):
    path = checkpoint(
        t["outdir"], t["L"], t["Tmult"], t["zeta"], t["lam"],
        t["mode"], t["real"]
    )
    if path.exists():
        return "skip"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        L, lam, zeta = int(t["L"]), float(t["lam"]), float(t["zeta"])
        Tmult, Nc, real = float(t["Tmult"]), int(t["Nc"]), int(t["real"])
        mode, tau = mode_cfg(t["mode"])
        dt = fixed_window_dt(L, lam)
        model = build_gaussian_chain_model(L, 1.0 - lam, lam)
        seed = seed_of(L, lam, zeta, real)

        t0 = time.time()
        res = run_adaptive(
            model, zeta, Tmult * L, Nc, np.random.default_rng(seed), dt,
            mode=mode, ess_tau=tau, entropy_stride=4,
            jump_update_method="lowrank", refresh_every=100,
            solver_method="newton",
        )
        wall = time.time() - t0

        per = [observables(np.asarray(G, float), L) for G in res["final_covs"]]
        w = np.asarray(res["final_weights"], float)
        w = w / w.sum()

        rec = dict(
            status="ok", L=L, lambda_=lam, zeta=zeta, real=real,
            mode=t["mode"], Tmult=Tmult, T=float(Tmult * L),
            Nc=Nc, seed=seed, delta_tau_effective=dt, wall_s=wall,
            theta=float(res["theta_hat"]),
            CMI_tavg50=float(res["CMI_tavg50"]),
            CMI_tavg75=float(res["CMI_tavg75"]),
            n_events=int(res["n_resampling_events"]),
            ess_final=float(res["ess_final_frac"]),
            min_ess=float(res["min_ess_frac"]),
            gess_root=float(res["gess_root"]),
            gess_recent=float(res["gess_recent"]),
        )
        rec["lambda"] = rec.pop("lambda_")
        for key in ("CMI", "B_L", "c_eff", "S_AB"):
            vals = np.asarray([p[key] for p in per], float)
            rec[key] = float(np.dot(w, vals))

        with open(path, "w") as f:
            json.dump(rec, f)
        return "ok"
    except CloningCollapse as e:
        with open(path, "w") as f:
            json.dump(dict(status="collapse", error=str(e), **t), f)
        return "collapse"
    except Exception as e:
        with open(path, "w") as f:
            json.dump(dict(status="error", error=str(e),
                           traceback=traceback.format_exc(), **t), f)
        return "error"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--Tmult", type=float, required=True)
    p.add_argument("--zeta", type=float, default=0.20)
    p.add_argument("--lams", default="0.14,0.18,0.22,0.26,0.30")
    p.add_argument("--modes", default="always,ess0.9,never")
    p.add_argument("--nreal", type=int, default=12)
    p.add_argument("--Nc", type=int, default=128)
    p.add_argument("--nworkers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if not gate(a):
        raise SystemExit("production-equivalence gate failed")

    lams = [float(x) for x in a.lams.split(",") if x.strip()]
    modes = [x.strip() for x in a.modes.split(",") if x.strip()]
    tasks = [
        dict(outdir=a.outdir, L=a.L, Tmult=a.Tmult, zeta=a.zeta,
             lam=lam, mode=mode, real=r, Nc=a.Nc)
        for lam in lams for mode in modes for r in range(a.nreal)
    ]
    todo = [
        t for t in tasks
        if not checkpoint(
            t["outdir"], t["L"], t["Tmult"], t["zeta"], t["lam"],
            t["mode"], t["real"]
        ).exists()
    ]
    print("HORIZON-CROSSING:",
          "L", a.L, "Tmult", a.Tmult, "zeta", a.zeta,
          "lams", lams, "modes", modes,
          "tasks", len(tasks), "remaining", len(todo), flush=True)
    if a.dry_run:
        return

    if a.nworkers <= 1:
        status = [run_one(t) for t in todo]
    else:
        import multiprocessing as mp
        with mp.Pool(a.nworkers) as pool:
            status = pool.map(run_one, todo, chunksize=1)

    counts = {}
    for s in status:
        counts[s] = counts.get(s, 0) + 1
    print("DONE", counts, flush=True)


if __name__ == "__main__":
    main()
