#!/usr/bin/env python3
"""Paired-seed validation of the production QJ-PPS configuration.

WHY THIS EXISTS
---------------
TASK-2026-08-10-SAMPLER, verdict "Infrastructure first", named the blocker:
the entire Cut B production campaign ran OUTSIDE its recorded validation range.

  PPS_DTAU_MULT = 12.0 in 222 log entries, at zeta down to 0.05 and L up to 128.
      Recorded safe: "zeta >= 0.3 (mult=2.0) and zeta >= 0.5 (mult=3.0) AT L=32"
      (worker_clone_pps.py:85-87).  Driver recommends 6 (grid_pps.py:1443).
      chunkcert swept mult 1,2,4,8 at L=32 only.  12.0 is beyond the sweep.

  SOLVER = newton in all 225 campaign logs.  The code describes newton as a
      STATISTICAL, not bit-exact, change requiring paired-seed validation
      (worker_clone_pps.py:136-139).  No evidence item provides one.

Until both are certified, every number derived from EV-DATA-RUCHEPULL-001 rests
on an undocumented configuration.  This script produces that certification.

PAIRED SEEDS ARE AUTOMATIC.  run_local_boundary._seed depends only on
(L, lam, zeta, real), never on dtau_mult or solver, so arm A and arm B at the
same (cell, real) consume the same seed stream by construction.  Differences
between arms are therefore attributable to the configuration, not to sampling.

ARMS (factorial, each isolating one knob against A):
  A_production   dtau 12, newton, lowrank, stride 4   <- exactly what production ran
  D_solver_only  dtau 12, brentq, lowrank, stride 4   <- A vs D isolates the SOLVER
  C_recommended  dtau  6, brentq, lowrank, stride 4   |  D vs C vs B is the
  B_certified    dtau  2, brentq, lowrank, stride 4   |  dtau ladder at fixed solver
  E_stride1      dtau 12, newton, lowrank, stride 1   <- A vs E isolates ENTROPY STRIDE
  F_eigh         dtau 12, newton, eigh,    stride 4   <- A vs F is the SPEED benchmark

METRIC.  DEC-MASTER-METRIC-001 makes t_wall * sigma^2(lambda_c) the only metric
that may support a production claim; ESS and GESS are diagnostic_only.  This
script records wall time per realisation AND the per-realisation observables, so
both factors are recoverable downstream.  It does not itself decide anything.

NO AGENT RUNS THIS.  Prepared locally, submitted by the researcher.
"""
import os, sys, json, time, argparse, math, traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ARMS = {
    "A_production":  dict(dtau_mult=12.0, solver="newton", jump="lowrank", stride=4),
    "D_solver_only": dict(dtau_mult=12.0, solver="brentq", jump="lowrank", stride=4),
    "C_recommended": dict(dtau_mult=6.0,  solver="brentq", jump="lowrank", stride=4),
    "B_certified":   dict(dtau_mult=2.0,  solver="brentq", jump="lowrank", stride=4),
    "E_stride1":     dict(dtau_mult=12.0, solver="newton", jump="lowrank", stride=1),
    "F_eigh":        dict(dtau_mult=12.0, solver="newton", jump="eigh",    stride=4),
}

# (L, zeta).  SAMPLER specified zeta = 0.9 (ties to every prior VR measurement)
# and zeta = 0.2 (where R was largest and nothing was ever certified); L = 32
# (the only size with an existing certification) and L = 96 (production scale).
CELLS = [(32, 0.20), (32, 0.90), (96, 0.20), (96, 0.90)]

# lambda at the production anchor.  For a CONFIGURATION comparison the value is
# irrelevant so long as every arm shares it; the anchor keeps continuity with
# the campaign being certified.  It is NOT a claim about where lambda_c is.
def anchor_lambda(zeta):
    return 0.5 * math.sqrt(zeta)


def seed_of(L, lam, zeta, real):
    """Byte-identical to scripts/run_local_boundary.py:_seed."""
    base = int(L * 10_000_000 + round(lam * 1e4) * 1_000 + round(zeta * 1_000))
    return base * 101 + real


def ckpt_path(outdir, arm, L, lam, zeta, real):
    d = os.path.join(outdir, arm, "L%d_z%.3f_lam%.4f" % (L, zeta, lam))
    return d, os.path.join(d, "real%03d.json" % real)


def _atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def git_commit():
    try:
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        h = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip()
        return h, bool(dirty)
    except Exception:
        return "unknown", None


def run_one(task):
    arm, L, zeta, real = task["arm"], task["L"], task["zeta"], task["real"]
    cfg, outdir = ARMS[arm], task["outdir"]
    lam = anchor_lambda(zeta)
    d, path = ckpt_path(outdir, arm, L, lam, zeta, real)
    if os.path.exists(path):
        return ("skip", path)
    os.makedirs(d, exist_ok=True)
    try:
        from pps_qj.gaussian_backend import build_gaussian_chain_model
        from pps_qj.cloning import run_cloning, CloningCollapse
        from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L

        alpha, w = float(lam), float(1.0 - lam)
        T = float(task["Tmult"] * L)
        dtau = cfg["dtau_mult"] / max(2.0 * alpha * (L - 1), 1e-6)
        model = build_gaussian_chain_model(L, w, alpha)
        sd = seed_of(L, lam, zeta, real)
        rng = np.random.default_rng(sd)
        commit, dirty = git_commit()

        t0 = time.time()
        try:
            res = run_cloning(
                model, zeta, T, task["Nc"], rng, delta_tau=dtau,
                record_entropy=True, entropy_stride=cfg["stride"],
                proposal_c=zeta, jump_update_method=cfg["jump"],
                refresh_every=100, solver_method=cfg["solver"],
            )
        except CloningCollapse as e:
            _atomic_write(path, {"arm": arm, "L": L, "lambda": lam, "zeta": zeta,
                                 "real": real, "status": "collapse", "error": str(e),
                                 "seed": sd, "git_commit": commit})
            return ("collapse", path)
        wall = time.time() - t0

        comps = _batched_compute_B_L(res.final_covs, L)
        # Genealogy.  ancestor_ids_final / n_resampling_events exist only from
        # commit 5ce8502 onward; getattr keeps this runnable on older checkouts.
        anc_ids = getattr(res, "ancestor_ids_final", np.asarray([], dtype=np.intp))
        anc_ids = np.asarray(anc_ids)
        if anc_ids.size:
            counts = np.bincount(anc_ids, minlength=int(task["Nc"])).astype(float)
            nz = counts[counts > 0]
            gen_ess = float(nz.sum() ** 2 / np.sum(nz ** 2))
            n_anc = int(nz.size)
        else:
            gen_ess, n_anc = float("nan"), int(res.n_distinct_ancestors)

        rec = {
            "arm": arm, "dtau_mult": cfg["dtau_mult"], "solver": cfg["solver"],
            "jump_update_method": cfg["jump"], "entropy_stride": cfg["stride"],
            "L": L, "lambda": lam, "zeta": zeta, "real": real,
            "alpha": alpha, "w": w, "T": T, "N_c": int(task["Nc"]), "dtau": dtau,
            "seed": sd, "git_commit": commit, "git_dirty": dirty,
            "theta_hat": float(res.theta_hat),
            "S_mean": float(res.S_mean), "S_std": float(res.S_std),
            "eff_sample_size": float(res.eff_sample_size),
            "n_T_mean": float(res.n_T_mean),
            "n_distinct_ancestors": n_anc,
            "genealogical_ess": gen_ess,
            "n_resampling_events": int(getattr(res, "n_resampling_events", -1)),
            "CMI_mean": float(np.nanmean(comps["CMI"])),
            "CMI_std": float(np.nanstd(comps["CMI"])),
            "B_L_mean": float(np.nanmean(comps["B_L"])),
            "B_L_std": float(np.nanstd(comps["B_L"])),
            "S_AB_mean": float(np.nanmean(comps["S_AB"])),
            "wall_s": wall, "status": "ok",
        }
        _atomic_write(path, rec)
        return ("ok", path)
    except Exception as e:
        _atomic_write(path, {"arm": arm, "L": L, "lambda": lam, "zeta": zeta,
                             "real": real, "status": "error", "error": str(e),
                             "traceback": traceback.format_exc()})
        return ("error", path)


def build_tasks(args):
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit("unknown arm %r; known: %s" % (a, ", ".join(ARMS)))
    cells = CELLS
    if args.cells:
        cells = []
        for tok in args.cells.split(","):
            Ls, zs = tok.split(":")
            cells.append((int(Ls), float(zs)))
    tasks = []
    for arm in arms:
        for (L, zeta) in cells:
            for real in range(args.nreal):
                tasks.append(dict(arm=arm, L=L, zeta=zeta, real=real,
                                  outdir=args.outdir, Nc=args.Nc, Tmult=args.Tmult))
    return tasks


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", required=True)
    p.add_argument("--arms", default="A_production,D_solver_only,C_recommended,B_certified")
    p.add_argument("--cells", default="", help="e.g. 32:0.2,96:0.9 (default: all four)")
    p.add_argument("--nreal", type=int, default=40, help="paired seeds per arm per cell")
    p.add_argument("--Nc", type=int, default=128)
    p.add_argument("--Tmult", type=float, default=1.0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--nworkers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tasks = build_tasks(args)
    mine = [t for i, t in enumerate(tasks) if i % args.nshards == args.shard]
    todo = [t for t in mine
            if not os.path.exists(ckpt_path(t["outdir"], t["arm"], t["L"],
                                            anchor_lambda(t["zeta"]), t["zeta"], t["real"])[1])]
    commit, dirty = git_commit()
    print("[validate] commit=%s dirty=%s" % (commit[:12], dirty), flush=True)
    print("[validate] shard %d/%d : %d of %d tasks, %d remaining"
          % (args.shard, args.nshards, len(mine), len(tasks), len(todo)), flush=True)
    print("[validate] arms=%s Nc=%d Tmult=%s nreal=%d workers=%d"
          % (args.arms, args.Nc, args.Tmult, args.nreal, args.nworkers), flush=True)
    if dirty:
        print("[validate] WARNING: working tree is dirty; every record is marked "
              "git_dirty=true and is not reproducible from a commit.", flush=True)
    if args.dry_run:
        for t in todo[:20]:
            print("   ", t["arm"], "L=%d" % t["L"], "zeta=%.2f" % t["zeta"], "real=%d" % t["real"])
        print("    ... %d total" % len(todo))
        return

    t0 = time.time()
    if args.nworkers <= 1:
        results = [run_one(t) for t in todo]
    else:
        import multiprocessing as mp
        with mp.Pool(processes=args.nworkers) as pool:
            results = pool.map(run_one, todo, chunksize=1)
    n = {}
    for st, _ in results:
        n[st] = n.get(st, 0) + 1
    print("[validate] done in %.1f s : %s" % (time.time() - t0, n), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
