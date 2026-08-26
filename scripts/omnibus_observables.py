#!/usr/bin/env python3
"""OMNIBUS OBSERVABLE COMPARISON -- which locator resolves the transition?

WHY THIS, AND WHY NOW.  Two independent estimators now agree that the mid-zeta
L-signal is absent from the existing data:

  * LAMC's pre-registered crossing gate rejects 8-10 of 10 pairs at
    zeta = 0.40..0.70 on sign_change_multiplicity, and sixfold more data at
    zeta = 0.55..0.70 left n_valid at exactly 0.
  * A global FSS collapse, controlled by permuting the L labels within each
    lambda column, beats that destroyed control by only 1.2-2.6x at
    zeta = 0.35..0.60.  A lambda-label scramble looked spectacular (42-1293x)
    but only tests smoothness in lambda, which was never in question.

The collapse also measured WHY: nu_eff = 3-5 across mid-zeta, so 1/nu ~ 0.25 and
the scaling variable L^(1/nu) varies by only 1.20 across L = 64..128.  The
finite-size lever in the variable that matters is nearly absent.  Even the wide
L = 32..96 ladder only lifts that to ~1.36.  So the L ladder ALONE cannot fix
this, and the observable becomes the main line of attack rather than a
refinement.

WHAT THIS RUNS.  Identical trajectories, seven locators read off the SAME final
covariance matrices, so the comparison is free of sampling differences and the
only variable is the observable.  All entropies use worker_clone_pps's
convention exactly (log2, Majorana nu-spectrum of the (1j*Gamma) subblock).

  CMI          incumbent, OBS-CMI-001
  B_L          incumbent, OBS-BLPROD-001 (carries an explicit ln L via S_AB)
  S_AB         half-chain entropy
  I3           tripartite information, built to cancel ln L and boundary terms
  MI_ends      antipodal MI, OBS-MIENDS-001 convention (A = first L/4,
               C = last L/4), power law at criticality, exponential off it
  varN         bipartite number-like fluctuation: a QUADRATIC functional of
               Gamma, so it self-averages far better than any entropy
  c_eff        fitted coefficient of the Calabrese-Cardy profile
               S(l) = (c/3) ln[(L/pi) sin(pi l / L)] + const.
               THE STRUCTURALLY DIFFERENT ONE: an order parameter from a SINGLE
               system size.  c > 0 in the log phase, c -> 0 in the area law.
               It needs no cross-L difference at all, which is exactly the
               quantity that has failed.

RANKING RULE, fixed before the run.  For each observable: (i) the L-scramble
ratio of a global collapse, (ii) the width of the bootstrap CI on lambda_c,
(iii) nu_eff, (iv) cost.  The winner is the one with the largest L-scramble
ratio at fixed core-hours.  c_eff is scored differently and deliberately: by
the width of the CI on the lambda where c(lambda) crosses zero, using ONE L.

GROUND TRUTH ON BOTH ENDS.  Run at zeta = 1.00 and zeta = 0.30, the two slices
that survived the L-scramble control (ratios 3.3 and 5.2, lambda_c = 0.4364 and
0.2326), plus zeta = 0.55, which did not (2.4).  An observable that reproduces
the two anchors AND resolves 0.55 is the one to build the campaign on.

NO AGENT SUBMITS THIS.
"""
import os, sys, json, time, argparse, traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np


def _ent(sub):
    """worker_clone_pps._batch_entropy, single-matrix form. log2 convention."""
    ell = sub.shape[-1] // 2
    eigs = np.linalg.eigvalsh((1j * sub).astype(np.complex128))
    nus = np.clip(np.abs(eigs[ell:]), 0.0, 1.0)
    pp = np.clip(0.5 * (1.0 + nus), 1e-15, 1.0 - 1e-15)
    pm = np.clip(0.5 * (1.0 - nus), 1e-15, 1.0 - 1e-15)
    return float(-np.sum(pp * np.log2(pp) + pm * np.log2(pm)))


def _profile(G, L, lmin=2):
    """S(l) for contiguous blocks of l sites, l = lmin .. L/2 (Majorana idx 2l)."""
    ls, Ss = [], []
    for l in range(lmin, L // 2 + 1):
        ls.append(l)
        Ss.append(_ent(G[:2 * l, :2 * l]))
    return np.array(ls, float), np.array(Ss)


def _c_eff(ls, Ss, L):
    """Fit S(l) = (c/3) ln[(L/pi) sin(pi l / L)] + b.  Returns c."""
    x = np.log((L / np.pi) * np.sin(np.pi * ls / L))
    ok = np.isfinite(x) & np.isfinite(Ss)
    if ok.sum() < 4:
        return np.nan
    A = np.vstack([x[ok], np.ones(ok.sum())]).T
    b, *_ = np.linalg.lstsq(A, Ss[ok], rcond=None)
    return float(3.0 * b[0])


def observables(G, L):
    """Every locator, from one covariance matrix."""
    hL, qL = L // 2, L // 4
    o = {}
    S_AB = _ent(G[:L, :L])
    S_BC = _ent(G[hL:hL + L, hL:hL + L])
    S_B = _ent(G[hL:L, hL:L])
    S_ABC = _ent(G[:3 * L // 2, :3 * L // 2])
    o["S_AB"] = S_AB
    o["CMI"] = S_AB + S_BC - S_B - S_ABC
    o["B_L"] = o["CMI"] * S_AB

    # I3 = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC on four contiguous quarters
    idx = [slice(0, 2 * qL), slice(2 * qL, 4 * qL), slice(4 * qL, 6 * qL)]
    S_A = _ent(G[idx[0], idx[0]]); S_C = _ent(G[idx[2], idx[2]])
    pick = np.r_[np.arange(0, 2 * qL), np.arange(4 * qL, 6 * qL)]
    S_AC = _ent(G[np.ix_(pick, pick)])
    o["I3"] = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

    # antipodal MI, OBS-MIENDS-001 q4 convention: A = sites [0,L/4), C = [3L/4,L)
    a = np.arange(0, 2 * qL); c = np.arange(2 * (L - qL), 2 * L)
    both = np.r_[a, c]
    o["MI_ends"] = (_ent(G[np.ix_(a, a)]) + _ent(G[np.ix_(c, c)])
                    - _ent(G[np.ix_(both, both)]))

    # quadratic functional: Majorana two-point weight in the half-chain block.
    # Var-like, self-averages far better than any entropy.
    o["varN"] = float(np.sum(G[:L, :L] ** 2) / (2.0 * L))

    ls, Ss = _profile(G, L)
    o["c_eff"] = _c_eff(ls, Ss, L)
    o["S_prof_l"] = ls.tolist()
    o["S_prof_S"] = Ss.tolist()
    return o


def seed_of(L, lam, zeta, real):
    """Byte-identical to scripts/run_local_boundary.py:_seed."""
    base = int(L * 10_000_000 + round(lam * 1e4) * 1_000 + round(zeta * 1_000))
    return base * 101 + real


def ckpt(outdir, L, lam, zeta, real):
    d = os.path.join(outdir, "L%d_z%.3f_lam%.4f" % (L, zeta, lam))
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
    L, lam, zeta, real = t["L"], t["lam"], t["zeta"], t["real"]
    d, path = ckpt(t["outdir"], L, lam, zeta, real)
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
            res = run_cloning(model, zeta, T, t["Nc"], np.random.default_rng(sd),
                              delta_tau=dtau, record_entropy=True,
                              entropy_stride=t["stride"], proposal_c=zeta,
                              jump_update_method="lowrank", refresh_every=100,
                              solver_method=t["solver"])
        except CloningCollapse as e:
            json.dump({"L": L, "lambda": lam, "zeta": zeta, "real": real,
                       "status": "collapse", "error": str(e)}, open(path, "w"))
            return "collapse"
        wall_traj = time.time() - t0

        t1 = time.time()
        per = [observables(np.asarray(G, dtype=np.float64), L) for G in res.final_covs]
        wall_obs = time.time() - t1

        rec = {"L": L, "lambda": lam, "zeta": zeta, "real": real, "seed": sd,
               "git_commit": commit, "git_dirty": dirty,
               "alpha": alpha, "w": w, "T": T, "N_c": int(t["Nc"]), "dtau": dtau,
               "theta_hat": float(res.theta_hat),
               "eff_sample_size": float(res.eff_sample_size),
               "n_T_mean": float(res.n_T_mean),
               "n_distinct_ancestors": int(res.n_distinct_ancestors),
               "n_resampling_events": int(getattr(res, "n_resampling_events", -1)),
               "wall_traj_s": wall_traj, "wall_obs_s": wall_obs, "status": "ok"}
        anc = np.asarray(getattr(res, "ancestor_ids_final", np.asarray([], dtype=np.intp)))
        if anc.size:
            cnt = np.bincount(anc, minlength=int(t["Nc"])).astype(float)
            nz = cnt[cnt > 0]
            rec["genealogical_ess"] = float(nz.sum() ** 2 / np.sum(nz ** 2))
        for k in ("CMI", "B_L", "S_AB", "I3", "MI_ends", "varN", "c_eff"):
            v = np.array([p[k] for p in per], dtype=float)
            rec[k + "_mean"] = float(np.nanmean(v))
            rec[k + "_std"] = float(np.nanstd(v))
        # population-mean entropy profile, so c_eff can be refitted downstream
        P = np.array([p["S_prof_S"] for p in per], dtype=float)
        rec["S_prof_l"] = per[0]["S_prof_l"]
        rec["S_prof_S_mean"] = np.nanmean(P, axis=0).tolist()
        json.dump(rec, open(path, "w"))
        return "ok"
    except Exception as e:
        json.dump({"L": L, "lambda": lam, "zeta": zeta, "real": real,
                   "status": "error", "error": str(e),
                   "traceback": traceback.format_exc()}, open(path, "w"))
        return "error"


# Absolute lambda windows, NOT multiples of 0.5*sqrt(zeta).  Centred on the two
# collapse anchors that survived the L-scramble control (0.2326 at zeta=0.30,
# 0.4364 at zeta=1.00) and on the midpoint for the slice that did not (0.55).
# zeta=0.55 REPOSITIONED 2026-08-26 after the first attempt failed to bracket:
# every locator crowded the lower edge (MI_ends and varN pinned exactly at
# 0.3300, CMI 0.3424, c_eff 0.3507) and the c_eff slope profile was
# + - - - - - - - - - -, i.e. one point on the positive side.  (0.22,0.44)
# shares six of its eleven points with (0.33,0.55), so the completed cells at
# 0.33, 0.352, 0.374, 0.396, 0.418 and 0.44 are reused by the checkpointing
# and only 180 of 396 runs are new.
WINDOWS = {0.30: (0.16, 0.32), 0.55: (0.22, 0.44), 1.00: (0.35, 0.55)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--zetas", default="0.30,0.55,1.00")
    p.add_argument("--Ls", default="32,48,64,96")
    p.add_argument("--nlam", type=int, default=11)
    p.add_argument("--nreal", type=int, default=12)
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

    tasks = []
    for z in [float(x) for x in a.zetas.split(",")]:
        lo, hi = WINDOWS.get(round(z, 2), (0.20, 0.60))
        for lam in np.round(np.linspace(lo, hi, a.nlam), 4):
            for L in [int(x) for x in a.Ls.split(",")]:
                for r in range(a.nreal):
                    tasks.append(dict(L=L, lam=float(lam), zeta=z, real=r,
                                      outdir=a.outdir, Nc=a.Nc, Tmult=a.Tmult,
                                      dtau_mult=a.dtau_mult, stride=a.stride,
                                      solver=a.solver))
    mine = [t for i, t in enumerate(tasks) if i % a.nshards == a.shard]
    todo = [t for t in mine
            if not os.path.exists(ckpt(t["outdir"], t["L"], t["lam"], t["zeta"], t["real"])[1])]
    commit, dirty = git_commit()
    print("[omnibus] commit=%s dirty=%s" % (commit[:12], dirty), flush=True)
    print("[omnibus] zetas=%s Ls=%s nlam=%d nreal=%d Nc=%d"
          % (a.zetas, a.Ls, a.nlam, a.nreal, a.Nc), flush=True)
    print("[omnibus] shard %d/%d : %d of %d, %d remaining"
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
    print("[omnibus] done %.1f s : %s" % (time.time() - t0, n), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
