#!/usr/bin/env python3
"""N_c LADDER -- is the finite population BIASING the locator, or only adding noise?

THE POINT.  Write the estimator as

    lambda_c_hat(N_c) = lambda_c + b(N_c) + eps.

More realisations kill eps like N_real^-1/2 and do NOTHING to b(N_c).  So until
b is bounded, buying trajectories may buy a more precise wrong answer.  There is
already evidence b != 0: TASK-2026-08-11-ARCH's red team measured a finite-N_c
shift in the locator observable at z = 3.4 (mean B_L 1.2015 +- 0.0763 at N_c=44
against 0.9161 +- 0.0348 at N_c=350).

SCORED BY THE LOCATOR, NOT BY ESS.  DEC-MASTER-METRIC-001 lists ESS and GESS as
diagnostic_only.  If doubling N_c heals the genealogy dramatically but moves
lambda_c by less than 0.003, the genealogy is ugly and harmless.  If lambda_c
marches 0.350 -> 0.365 -> 0.374, that is the bottleneck and no amount of extra
realisations will help.

DESIGN.  Narrow bracket only -- this is not a boundary scan.  The omnibus at
zeta = 0.55 puts the locators at CMI 0.3373, c_eff 0.3507, B_L 0.3752, so the
default bracket is 5 points over [0.31, 0.41], which contains all three with
room either side.  L in {32,48,64} because the omnibus measured CMI's L-scramble
ratio there as 11.96 against 2.4 on L in {64..128}.

N_c IS IN THE CHECKPOINT PATH, unlike the omnibus, so ladder rungs never collide.

Cost is exactly linear in N_c (benchmark: wall ~ N_c^1.00 +- 0.02), so the
64/128/256/512 ladder costs 7.5x one rung at N_c=128.  About 63 core-hours for
the default grid.

NO AGENT SUBMITS THIS.
"""
import os, sys, json, time, argparse, traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np


def seed_of(L, lam, zeta, real):
    base = int(L * 10_000_000 + round(lam * 1e4) * 1_000 + round(zeta * 1_000))
    return base * 101 + real


def ckpt(outdir, Nc, L, lam, zeta, real):
    d = os.path.join(outdir, "Nc%d" % Nc, "L%d_z%.3f_lam%.4f" % (L, zeta, lam))
    return d, os.path.join(d, "real%03d.json" % real)


def git_commit():
    try:
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        h = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        d = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            text=True).strip()
        return h, bool(d)
    except Exception:
        return "unknown", None


def run_one(t):
    from omnibus_observables import observables
    Nc, L, lam, zeta, real = t["Nc"], t["L"], t["lam"], t["zeta"], t["real"]
    d, path = ckpt(t["outdir"], Nc, L, lam, zeta, real)
    if os.path.exists(path):
        return "skip"
    os.makedirs(d, exist_ok=True)
    try:
        from pps_qj.gaussian_backend import build_gaussian_chain_model
        from pps_qj.cloning import run_cloning, CloningCollapse
        alpha, w = float(lam), float(1.0 - lam)
        T = float(t["Tmult"] * L)
        dtau = t["dtau_mult"] / max(2.0 * alpha * (L - 1), 1e-6)
        model = build_gaussian_chain_model(L, w, alpha)
        sd = seed_of(L, lam, zeta, real)
        commit, dirty = git_commit()
        t0 = time.time()
        try:
            res = run_cloning(model, zeta, T, Nc, np.random.default_rng(sd),
                              delta_tau=dtau, record_entropy=True,
                              entropy_stride=t["stride"], proposal_c=zeta,
                              jump_update_method="lowrank", refresh_every=100,
                              solver_method=t["solver"])
        except CloningCollapse as e:
            json.dump({"N_c": Nc, "L": L, "lambda": lam, "zeta": zeta, "real": real,
                       "status": "collapse", "error": str(e)}, open(path, "w"))
            return "collapse"
        wall = time.time() - t0
        per = [observables(np.asarray(G, dtype=np.float64), L) for G in res.final_covs]
        rec = {"N_c": int(Nc), "L": L, "lambda": lam, "zeta": zeta, "real": real,
               "seed": sd, "git_commit": commit, "git_dirty": dirty,
               "alpha": alpha, "w": w, "T": T, "dtau": dtau,
               "theta_hat": float(res.theta_hat),
               "eff_sample_size": float(res.eff_sample_size),
               "n_T_mean": float(res.n_T_mean),
               "n_distinct_ancestors": int(res.n_distinct_ancestors),
               "n_resampling_events": int(getattr(res, "n_resampling_events", -1)),
               "wall_traj_s": wall, "status": "ok"}
        anc = np.asarray(getattr(res, "ancestor_ids_final", np.asarray([], dtype=np.intp)))
        if anc.size:
            cnt = np.bincount(anc, minlength=int(Nc)).astype(float)
            nz = cnt[cnt > 0]
            rec["genealogical_ess"] = float(nz.sum() ** 2 / np.sum(nz ** 2))
            # the diagnostic that matters: diversity as a FRACTION of population
            rec["gen_ess_frac"] = rec["genealogical_ess"] / float(Nc)
        for k in ("CMI", "B_L", "S_AB", "I3", "MI_ends", "varN", "c_eff"):
            v = np.array([p[k] for p in per], dtype=float)
            rec[k + "_mean"] = float(np.nanmean(v))
            rec[k + "_std"] = float(np.nanstd(v))
        json.dump(rec, open(path, "w"))
        return "ok"
    except Exception as e:
        json.dump({"N_c": Nc, "L": L, "lambda": lam, "zeta": zeta, "real": real,
                   "status": "error", "error": str(e),
                   "traceback": traceback.format_exc()}, open(path, "w"))
        return "error"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--zeta", type=float, default=0.55)
    p.add_argument("--Ls", default="32,48,64")
    p.add_argument("--Ncs", default="64,128,256,512")
    p.add_argument("--lams", default="0.31,0.335,0.36,0.385,0.41")
    p.add_argument("--nreal", type=int, default=12)
    p.add_argument("--Tmult", type=float, default=1.0)
    p.add_argument("--dtau-mult", type=float, default=12.0, dest="dtau_mult")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--solver", default="newton")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--nworkers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    tasks = []
    for Nc in [int(x) for x in a.Ncs.split(",")]:
        for L in [int(x) for x in a.Ls.split(",")]:
            for lam in [float(x) for x in a.lams.split(",")]:
                for r in range(a.nreal):
                    tasks.append(dict(Nc=Nc, L=L, lam=lam, zeta=a.zeta, real=r,
                                      outdir=a.outdir, Tmult=a.Tmult,
                                      dtau_mult=a.dtau_mult, stride=a.stride,
                                      solver=a.solver))
    # heaviest first, so a wall-clock kill loses the cheapest work
    tasks.sort(key=lambda t: -(t["Nc"] * t["L"] ** 4))
    mine = [t for i, t in enumerate(tasks) if i % a.nshards == a.shard]
    todo = [t for t in mine if not os.path.exists(
        ckpt(t["outdir"], t["Nc"], t["L"], t["lam"], t["zeta"], t["real"])[1])]
    commit, dirty = git_commit()
    print("[ncladder] commit=%s dirty=%s" % (commit[:12], dirty), flush=True)
    print("[ncladder] zeta=%.3f Ls=%s Ncs=%s nlam=%d nreal=%d"
          % (a.zeta, a.Ls, a.Ncs, len(a.lams.split(",")), a.nreal), flush=True)
    print("[ncladder] shard %d/%d : %d of %d, %d remaining"
          % (a.shard, a.nshards, len(mine), len(tasks), len(todo)), flush=True)
    if a.dry_run:
        return
    t0 = time.time()
    if a.nworkers <= 1:
        out = [run_one(t) for t in todo]
    else:
        import multiprocessing as mp
        with mp.Pool(a.nworkers) as pool:
            out = pool.map(run_one, todo, chunksize=1)
    n = {}
    for s in out:
        n[s] = n.get(s, 0) + 1
    print("[ncladder] done %.1f s : %s" % (time.time() - t0, n), flush=True)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    sys.path.insert(0, here)
    main()
