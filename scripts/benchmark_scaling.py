#!/usr/bin/env python3
"""Cost model for the QJ-PPS cloning simulator.  Timing only.

SCOPE.  DEC-MASTER-METRIC-001 marks wall time diagnostic_only: the accepted
production metric is t_wall * sigma^2(lambda_c).  Nothing here may support a
claim that one configuration is BETTER.  This measures how cost scales, so a
campaign can be budgeted and an L ladder chosen.  That is all it is for.

WHY.  The Phase 3 budget rests on wall ~ L^4.10, fitted on L = 64..128 in the
production data and then extrapolated DOWN to L = 32.  If per-call overhead
dominates at small L that extrapolation is wrong and the L ladder changes.
Nobody has measured it.

DESIGN.  Every sweep varies ONE axis with the rest pinned, at a SHORT horizon so
the whole thing is cheap; the T sweep then measures whether cost is linear in T,
which is what licenses extrapolating these numbers to production T = L.

Reports wall_s and two normalisations:
  s_per_clone_step = wall / (N_c * n_steps)
  s_per_clone_step_L2 = that / L^2      (flat => per-step work is O(L^2))
"""
import os, sys, json, time, math, argparse, platform

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

BASE = dict(L=64, N_c=64, zeta=0.50, lam=0.35, T=8.0,
            dtau_mult=12.0, solver="newton", jump="lowrank", stride=4)


def sweeps(which):
    S = {}
    S["L"]      = [dict(BASE, L=L) for L in (16, 24, 32, 48, 64, 96, 128)]
    S["N_c"]    = [dict(BASE, N_c=n) for n in (32, 64, 128, 256, 512)]
    S["zeta"]   = [dict(BASE, zeta=z) for z in (0.10, 0.30, 0.50, 0.70, 0.90, 1.00)]
    S["dtau"]   = [dict(BASE, dtau_mult=m) for m in (2.0, 4.0, 6.0, 8.0, 12.0)]
    S["T"]      = [dict(BASE, T=t) for t in (2.0, 4.0, 8.0, 16.0, 32.0)]
    S["lam"]    = [dict(BASE, lam=x) for x in (0.15, 0.25, 0.35, 0.45, 0.55)]
    S["solver"] = [dict(BASE, solver=s, jump=j, stride=st)
                   for s in ("brentq", "newton") for j in ("eigh", "lowrank")
                   for st in (1, 4)]
    # the L ladder at production horizon T = L, the number the budget rests on
    S["Lprod"]  = [dict(BASE, L=L, T=float(L), N_c=128) for L in (24, 32, 48, 64, 96)]
    if which == "all":
        return [(k, c) for k in S for c in S[k]]
    return [(which, c) for c in S[which]]


def one(cfg, rep):
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    from pps_qj.cloning import run_cloning, CloningCollapse
    L, lam = cfg["L"], cfg["lam"]
    alpha, w = float(lam), float(1.0 - lam)
    dtau = cfg["dtau_mult"] / max(2.0 * alpha * (L - 1), 1e-6)
    n_steps = cfg["T"] / dtau
    model = build_gaussian_chain_model(L, w, alpha)
    rng = np.random.default_rng(90210 + rep)
    t0 = time.time()
    try:
        res = run_cloning(model, cfg["zeta"], cfg["T"], cfg["N_c"], rng,
                          delta_tau=dtau, record_entropy=True,
                          entropy_stride=cfg["stride"], proposal_c=cfg["zeta"],
                          jump_update_method=cfg["jump"], refresh_every=100,
                          solver_method=cfg["solver"])
    except CloningCollapse as e:
        return dict(cfg, rep=rep, status="collapse", error=str(e))
    wall = time.time() - t0
    denom = max(cfg["N_c"] * n_steps, 1e-9)
    return dict(cfg, rep=rep, status="ok", wall_s=wall, n_steps=n_steps,
                s_per_clone_step=wall / denom,
                s_per_clone_step_L2=wall / denom / (L ** 2),
                n_T_mean=float(res.n_T_mean),
                n_distinct_ancestors=int(res.n_distinct_ancestors),
                n_resampling_events=int(getattr(res, "n_resampling_events", -1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--sweep", default="all",
                   help="all | L | N_c | zeta | dtau | T | lam | solver | Lprod")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    todo = sweeps(a.sweep)
    print("[bench] host=%s python=%s numpy=%s" % (platform.node(),
          platform.python_version(), np.__version__), flush=True)
    print("[bench] sweep=%s points=%d reps=%d" % (a.sweep, len(todo), a.reps), flush=True)
    if a.dry_run:
        for k, c in todo:
            print("   %-7s L=%-4d N_c=%-4d zeta=%.2f T=%-5.1f dtau_mult=%-4.1f %s/%s/s%d"
                  % (k, c["L"], c["N_c"], c["zeta"], c["T"], c["dtau_mult"],
                     c["solver"], c["jump"], c["stride"]))
        return

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    rows = []
    for k, c in todo:
        for rep in range(a.reps):
            r = one(c, rep); r["sweep"] = k
            rows.append(r)
            if r["status"] == "ok":
                print("[%-7s] L=%-4d N_c=%-4d zeta=%.2f T=%-5.1f %s/%s/s%d "
                      "wall=%8.3f s  s/clone-step=%.3e  /L^2=%.3e"
                      % (k, c["L"], c["N_c"], c["zeta"], c["T"], c["solver"],
                         c["jump"], c["stride"], r["wall_s"],
                         r["s_per_clone_step"], r["s_per_clone_step_L2"]), flush=True)
            else:
                print("[%-7s] L=%d COLLAPSE" % (k, c["L"]), flush=True)
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)
    print("[bench] wrote %s (%d rows)" % (a.out, len(rows)), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
