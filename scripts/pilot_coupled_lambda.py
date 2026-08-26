#!/usr/bin/env python3
"""PHASE 2 PILOT -- does coupling lambda across a scan fix the multiplicity?

THE MEASURED PROBLEM.  Re-running TASK-2026-08-12-LAMC's frozen gate on the
complete 2026-08-26 export: at zeta = 0.40..0.70, ZERO of ten L-pairs yield a
valid crossing, and the gate rejects 8-10 of 10 on sign_change_multiplicity,
not on curve_collapse.  The curves are not merging; they weave.  Going from 7
to 15 lambda points and from 2 to 12 realisations at zeta = 0.55..0.70 changed
n_valid from 0 to 0.  So this is not a data-volume problem and it is not a
lambda-resolution problem -- more lambda points give more chances for a noise
zero.  It is a SMOOTHNESS problem in D(lambda) = O_L2(lambda) - O_L1(lambda).

THE INTERVENTION.  Production seeds on (L, lam, zeta, real), so every lambda
gets an independent stream and D(lambda) inherits fresh noise at every point.
Drop lam from the seed and one realisation walks the whole lambda scan on a
common stream, so D(lambda) becomes a smooth curve per realisation.

WHY THIS IS NOT CHEATING.  At each lambda the arm is still an unbiased draw
from the same distribution -- only the CORRELATION between lambda points
changes.  Cell means are unchanged in expectation.  The bootstrap resamples
REALISATIONS, which is the independent unit in both arms, so error bars stay
honest.  What shrinks is Var[D(lambda1) - D(lambda2)], i.e. the shape noise.

THE CAVEAT, STATED UP FRONT.  Different lambda have different jump rates, so
the streams desynchronise once the jump counts diverge.  This buys correlation
at early times and loses it later; how much survives to t = T is exactly what
this pilot measures.  The exact version needs uniformization with a dominating
rate held fixed across the window, whose implementation is NOT validated
(TASK-2026-08-11-ARCH sec 7.2).  Test the free version first: if partial
coupling already collapses the multiplicity, the uniformization work is moot.

PRE-REGISTERED SUCCESS CRITERION, fixed before any run:
  coupled arm attains a UNIQUE sign change in >= 8 of 10 bootstrap resamples
  for >= 2 of the 3 L-pairs, at a zeta where the independent arm attains 0.
Anything less and the cheap coupling is dead and the decision moves to
uniformization.

NO AGENT SUBMITS THIS.
"""
import os, sys, json, time, argparse, traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

# Absolute lambda bracket.  NOT a multiple of 0.5*sqrt(zeta): the whole point is
# to stop the scan support from carrying the hypothesis (LAMC sec 10, and the
# 2026-08-26 finding that fitted phi equals the grid's own phi within errors).
DEFAULT_LAMS = [0.22, 0.25, 0.28, 0.31, 0.34, 0.37, 0.40, 0.43, 0.46, 0.49, 0.52]


def seed_of(L, lam, zeta, real, mode):
    """mode='independent' reproduces run_local_boundary._seed exactly.
       mode='coupled' drops lam, so one realisation shares a stream across the
       whole lambda scan at fixed (L, zeta)."""
    lam_key = 0.0 if mode == "coupled" else lam
    base = int(L * 10_000_000 + round(lam_key * 1e4) * 1_000 + round(zeta * 1_000))
    return base * 101 + real


def ckpt(outdir, mode, L, lam, zeta, real):
    d = os.path.join(outdir, mode, "L%d_z%.3f_lam%.4f" % (L, zeta, lam))
    return d, os.path.join(d, "real%03d.json" % real)


def git_commit():
    try:
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        h = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            text=True).strip()
        return h, bool(dirty)
    except Exception:
        return "unknown", None


def run_one(t):
    mode, L, lam, zeta, real = t["mode"], t["L"], t["lam"], t["zeta"], t["real"]
    d, path = ckpt(t["outdir"], mode, L, lam, zeta, real)
    if os.path.exists(path):
        return "skip"
    os.makedirs(d, exist_ok=True)
    try:
        from pps_qj.gaussian_backend import build_gaussian_chain_model
        from pps_qj.cloning import run_cloning, CloningCollapse
        from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L
        alpha, w = float(lam), float(1.0 - lam)
        T = float(t["Tmult"] * L)
        dtau = t["dtau_mult"] / max(2.0 * alpha * (L - 1), 1e-6)
        model = build_gaussian_chain_model(L, w, alpha)
        sd = seed_of(L, lam, zeta, real, mode)
        commit, dirty = git_commit()
        t0 = time.time()
        try:
            res = run_cloning(model, zeta, T, t["Nc"], np.random.default_rng(sd),
                              delta_tau=dtau, record_entropy=True,
                              entropy_stride=t["stride"], proposal_c=zeta,
                              jump_update_method="lowrank", refresh_every=100,
                              solver_method=t["solver"])
        except CloningCollapse as e:
            json.dump({"mode": mode, "L": L, "lambda": lam, "zeta": zeta, "real": real,
                       "status": "collapse", "error": str(e)}, open(path, "w"))
            return "collapse"
        wall = time.time() - t0
        c = _batched_compute_B_L(res.final_covs, L)
        anc = np.asarray(getattr(res, "ancestor_ids_final", np.asarray([], dtype=np.intp)))
        if anc.size:
            cnt = np.bincount(anc, minlength=int(t["Nc"])).astype(float)
            nz = cnt[cnt > 0]
            gess, nanc = float(nz.sum() ** 2 / np.sum(nz ** 2)), int(nz.size)
        else:
            gess, nanc = float("nan"), int(res.n_distinct_ancestors)
        rec = {"mode": mode, "L": L, "lambda": lam, "zeta": zeta, "real": real,
               "seed": sd, "git_commit": commit, "git_dirty": dirty,
               "alpha": alpha, "w": w, "T": T, "N_c": int(t["Nc"]), "dtau": dtau,
               "theta_hat": float(res.theta_hat), "S_mean": float(res.S_mean),
               "S_std": float(res.S_std), "eff_sample_size": float(res.eff_sample_size),
               "n_T_mean": float(res.n_T_mean), "n_distinct_ancestors": nanc,
               "genealogical_ess": gess,
               "n_resampling_events": int(getattr(res, "n_resampling_events", -1)),
               "CMI_mean": float(np.nanmean(c["CMI"])), "CMI_std": float(np.nanstd(c["CMI"])),
               "B_L_mean": float(np.nanmean(c["B_L"])), "B_L_std": float(np.nanstd(c["B_L"])),
               "S_AB_mean": float(np.nanmean(c["S_AB"])),
               "wall_s": wall, "status": "ok"}
        json.dump(rec, open(path, "w"))
        return "ok"
    except Exception as e:
        json.dump({"mode": mode, "L": L, "lambda": lam, "zeta": zeta, "real": real,
                   "status": "error", "error": str(e),
                   "traceback": traceback.format_exc()}, open(path, "w"))
        return "error"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--zeta", type=float, default=0.55)
    p.add_argument("--Ls", default="32,64,96")
    p.add_argument("--lams", default=",".join(str(x) for x in DEFAULT_LAMS))
    p.add_argument("--modes", default="independent,coupled")
    p.add_argument("--nreal", type=int, default=16)
    p.add_argument("--Nc", type=int, default=128)
    p.add_argument("--Tmult", type=float, default=1.0)
    p.add_argument("--dtau-mult", type=float, default=12.0, dest="dtau_mult")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--solver", default="newton")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--nworkers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    Ls = [int(x) for x in a.Ls.split(",")]
    lams = [float(x) for x in a.lams.split(",")]
    tasks = [dict(mode=m, L=L, lam=l, zeta=a.zeta, real=r, outdir=a.outdir,
                  Nc=a.Nc, Tmult=a.Tmult, dtau_mult=a.dtau_mult,
                  stride=a.stride, solver=a.solver)
             for m in a.modes.split(",") for L in Ls for l in lams
             for r in range(a.nreal)]
    mine = [t for i, t in enumerate(tasks) if i % a.nshards == a.shard]
    todo = [t for t in mine
            if not os.path.exists(ckpt(t["outdir"], t["mode"], t["L"], t["lam"],
                                       t["zeta"], t["real"])[1])]
    commit, dirty = git_commit()
    print("[pilot] commit=%s dirty=%s" % (commit[:12], dirty), flush=True)
    print("[pilot] zeta=%.3f Ls=%s nlam=%d nreal=%d modes=%s"
          % (a.zeta, Ls, len(lams), a.nreal, a.modes), flush=True)
    print("[pilot] shard %d/%d : %d of %d, %d remaining"
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
    print("[pilot] done %.1f s : %s" % (time.time() - t0, n), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
