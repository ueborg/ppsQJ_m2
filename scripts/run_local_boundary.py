#!/usr/bin/env python3
"""
Local checkpointed guided-cloning boundary spot-check runner (low-rank path).

For LOCAL (no-HPC) use: resumable, BLAS-pinned, one process per realization via
a small pool.  Each (L, lambda, zeta, realization) writes an ATOMIC JSON
checkpoint immediately, so the run can be killed (Ctrl-C / closed laptop) and
resumed by re-running the same command -- finished checkpoints are skipped.

Uses the validated low-rank projective-jump update (jump_update_method=lowrank,
~2.2x faster than eigh, bit-identical observables) and guided cloning
(proposal_c=zeta).

Targeted campaign (per local-strategy memo):
    zeta in {0.25, 0.30}, L in {64, 96, 128, (160)},
    lambda around lambda_c(zeta) = 0.5*sqrt(zeta),
    analyse lambda_cross/zeta vs L*zeta^2.

RUN:
  python scripts/run_local_boundary.py run --outdir results/local_boundary \
      --zetas 0.25 0.30 --Ls 64 96 128 \
      --lam-mults 0.85 0.925 1.0 1.075 1.15 \
      --nreal 12 --Nc 128 --Tmult 1.0 --dtau-mult 12 --nworkers 3 --entropy-stride 4

DRY-RUN (list grid + rough cost, runs nothing):
  python scripts/run_local_boundary.py run --outdir results/local_boundary ... --dry-run

AGGREGATE finished checkpoints -> CSV (+ crossing estimate):
  python scripts/run_local_boundary.py aggregate --outdir results/local_boundary
"""
import os, sys, json, argparse, time, math, traceback

# Pin BLAS BEFORE importing numpy.  Children re-import this module under the
# 'spawn' start method, so they inherit these too -> no thread oversubscription.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def lam_c(zeta):
    return 0.5 * math.sqrt(zeta)


def _seed(L, lam, zeta, real):
    base = int(L * 10_000_000 + round(lam * 1e4) * 1_000 + round(zeta * 1_000))
    return base * 101 + real


def ckpt_path(outdir, L, lam, zeta, real):
    d = os.path.join(outdir, "L%d_z%.3f_lam%.4f" % (L, zeta, lam))
    return d, os.path.join(d, "real%03d.json" % real)


def run_one(task):
    L, lam, zeta, real = task["L"], task["lam"], task["zeta"], task["real"]
    outdir = task["outdir"]
    d, path = ckpt_path(outdir, L, lam, zeta, real)
    if os.path.exists(path):
        return ("skip", path)
    os.makedirs(d, exist_ok=True)
    try:
        from pps_qj.gaussian_backend import build_gaussian_chain_model
        from pps_qj.cloning import run_cloning, CloningCollapse
        from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L
        alpha, w = float(lam), float(1.0 - lam)
        T = float(task["Tmult"] * L)
        dtau = task["dtau_mult"] / max(2.0 * alpha * (L - 1), 1e-6)
        model = build_gaussian_chain_model(L, w, alpha)
        rng = np.random.default_rng(_seed(L, lam, zeta, real))
        t0 = time.time()
        try:
            res = run_cloning(model, zeta, T, task["Nc"], rng, delta_tau=dtau,
                              record_entropy=True, entropy_stride=task["entropy_stride"],
                              proposal_c=zeta, jump_update_method="lowrank",
                              refresh_every=100, solver_method=task.get("solver","brentq"))
        except CloningCollapse as e:
            rec = {"L": L, "lambda": lam, "zeta": zeta, "real": real,
                   "status": "collapse", "error": str(e)}
            _atomic_write(path, rec)
            return ("collapse", path)
        comps = _batched_compute_B_L(res.final_covs, L)
        rec = {
            "L": L, "lambda": lam, "zeta": zeta, "real": real,
            "alpha": alpha, "w": w, "T": T, "N_c": task["Nc"], "dtau": dtau,
            "theta_hat": float(res.theta_hat),
            "S_mean": float(res.S_mean), "S_std": float(res.S_std),
            "eff_sample_size": float(res.eff_sample_size),
            "n_T_mean": float(res.n_T_mean),
            "n_distinct_ancestors": int(res.n_distinct_ancestors),
            "CMI_mean": float(np.nanmean(comps["CMI"])),
            "CMI_std": float(np.nanstd(comps["CMI"])),
            "B_L_mean": float(np.nanmean(comps["B_L"])),
            "B_L_std": float(np.nanstd(comps["B_L"])),
            "S_AB_mean": float(np.nanmean(comps["S_AB"])),
            "wall_s": time.time() - t0, "status": "ok",
        }
    except Exception as e:
        rec = {"L": L, "lambda": lam, "zeta": zeta, "real": real,
               "status": "error", "error": repr(e), "traceback": traceback.format_exc()}
    _atomic_write(path, rec)
    return (rec.get("status", "ok"), path)


def _atomic_write(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2, default=float)
    os.replace(tmp, path)


def build_tasks(args):
    tasks = []
    for L in args.Ls:
        for zeta in args.zetas:
            lc = lam_c(zeta)
            for m in args.lam_mults:
                lam = round(lc * m, 4)
                for real in range(args.nreal):
                    tasks.append(dict(L=L, lam=lam, zeta=zeta, real=real,
                                      outdir=args.outdir, Nc=args.Nc, Tmult=args.Tmult,
                                      dtau_mult=args.dtau_mult, entropy_stride=args.entropy_stride,
                                      solver=args.solver))
    return tasks


def cmd_run(args):
    os.makedirs(args.outdir, exist_ok=True)
    tasks = build_tasks(args)
    if getattr(args, 'nshards', 1) > 1:
        tasks = [t for i, t in enumerate(tasks) if i % args.nshards == args.shard]
        print('[run] shard %d/%d -> %d realizations this task'
              % (args.shard, args.nshards, len(tasks)), flush=True)
    todo = [t for t in tasks if not os.path.exists(
        ckpt_path(t["outdir"], t["L"], t["lam"], t["zeta"], t["real"])[1])]
    print("[run] grid: %d realizations | %d remaining | %d done"
          % (len(tasks), len(todo), len(tasks) - len(todo)), flush=True)
    print("[run] Ls=%s zetas=%s lam_mults=%s nreal=%d Nc=%d Tmult=%s dtau_mult=%s workers=%d"
          % (args.Ls, args.zetas, args.lam_mults, args.nreal, args.Nc, args.Tmult,
             args.dtau_mult, args.nworkers), flush=True)
    # rough per-(L) cost model from a light empirical fit; just a planning aid.
    est = sum(_cost_est(t["L"], t["Nc"], t["Tmult"], t["dtau_mult"]) for t in todo)
    print("[run] VERY rough serial estimate ~%.1f h => ~%.1f h on %d workers"
          % (est / 3600, est / 3600 / max(args.nworkers, 1), args.nworkers), flush=True)
    if args.dry_run:
        print("[run] --dry-run: not launching.", flush=True); return
    if not todo:
        print("[run] nothing to do.", flush=True); return
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    t_start = time.time(); done = 0
    with ctx.Pool(args.nworkers) as pool:
        for status, path in pool.imap_unordered(run_one, todo):
            done += 1
            el = time.time() - t_start; rate = done / max(el, 1e-9)
            eta = (len(todo) - done) / max(rate, 1e-9)
            print("[%d/%d] %-8s %s | elapsed %.1fm ETA %.1fm"
                  % (done, len(todo), status, os.path.relpath(path, args.outdir),
                     el / 60, eta / 60), flush=True)
    print("[run] complete: %d realizations in %.1f min" % (done, (time.time() - t_start) / 60), flush=True)


def _cost_est(L, Nc, Tmult, dtau_mult):
    # crude: per-realization wall ~ c * Nc * (Tmult*L) * L^2 / dtau_mult-ish.
    # Calibrated very loosely to ~14s at L=96,Nc=16,Tmult~0.19. Planning only.
    # ~Nc * Tmult * L^4 (per-jump O(L^2) x n_jumps ~ L^2 Tmult), calibrated to
    # one smoke task; a rough planning aid only -- trust the live ETA once running.
    return 2.15e-8 * Nc * Tmult * (L ** 4)


def cmd_aggregate(args):
    import glob
    from collections import defaultdict
    rows = []
    for p in glob.glob(os.path.join(args.outdir, "**", "real*.json"), recursive=True):
        try:
            r = json.load(open(p))
            if r.get("status") == "ok":
                rows.append(r)
        except Exception:
            pass
    if not rows:
        print("[aggregate] no completed 'ok' checkpoints."); return
    g = defaultdict(list)
    for r in rows:
        g[(r["L"], r["zeta"], r["lambda"])].append(r)
    agg = []
    for (L, z, lam), rs in sorted(g.items()):
        cmi = np.array([x["CMI_mean"] for x in rs]); bl = np.array([x["B_L_mean"] for x in rs])
        n = len(rs)
        agg.append({"L": L, "zeta": z, "lambda": lam, "nreal": n,
                    "CMI": float(np.nanmean(cmi)), "CMI_se": float(np.nanstd(cmi) / max(math.sqrt(n), 1)),
                    "B_L": float(np.nanmean(bl)), "B_L_se": float(np.nanstd(bl) / max(math.sqrt(n), 1))})
    import csv
    csv_path = os.path.join(args.outdir, "aggregate.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(agg[0].keys())); wtr.writeheader(); wtr.writerows(agg)
    print("[aggregate] %d ok realizations -> %d (L,zeta,lambda) points -> %s" % (len(rows), len(agg), csv_path))
    # crossing estimate per zeta between consecutive L (linear interp on B_L vs lambda)
    print("[aggregate] B_L crossings lambda_x (consecutive L), and lambda_x/zeta vs L*zeta^2:")
    for z in sorted(set(a["zeta"] for a in agg)):
        sub = [a for a in agg if a["zeta"] == z]
        Ls = sorted(set(a["L"] for a in sub))
        curves = {L: sorted([(a["lambda"], a["B_L"]) for a in sub if a["L"] == L]) for L in Ls}
        for L1, L2 in zip(Ls[:-1], Ls[1:]):
            lx = _cross(curves[L1], curves[L2])
            if lx is not None:
                Lg = math.sqrt(L1 * L2)
                print("  zeta=%.3f (L=%d,%d): lambda_x=%.4f  lambda_x/zeta=%.3f  L*zeta^2=%.2f"
                      % (z, L1, L2, lx, lx / z, Lg * z * z))
            else:
                print("  zeta=%.3f (L=%d,%d): no bracketed crossing in lambda window" % (z, L1, L2))


def _cross(c1, c2):
    """Linear-interp crossing of two (lambda, B_L) curves on common lambda support."""
    if len(c1) < 2 or len(c2) < 2:
        return None
    import numpy as _np
    x1, y1 = _np.array([p[0] for p in c1]), _np.array([p[1] for p in c1])
    x2, y2 = _np.array([p[0] for p in c2]), _np.array([p[1] for p in c2])
    lo, hi = max(x1.min(), x2.min()), min(x1.max(), x2.max())
    xs = _np.linspace(lo, hi, 200)
    d = _np.interp(xs, x1, y1) - _np.interp(xs, x2, y2)
    s = _np.where(_np.diff(_np.sign(d)) != 0)[0]
    if len(s) == 0:
        return None
    i = s[0]
    # linear root between xs[i], xs[i+1]
    x_a, x_b, d_a, d_b = xs[i], xs[i + 1], d[i], d[i + 1]
    return float(x_a - d_a * (x_b - x_a) / (d_b - d_a))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "aggregate"):
        p = sub.add_parser(name)
        p.add_argument("--outdir", required=True)
        if name == "run":
            p.add_argument("--zetas", type=float, nargs="+", default=[0.25, 0.30])
            p.add_argument("--Ls", type=int, nargs="+", default=[64, 96, 128])
            p.add_argument("--lam-mults", type=float, nargs="+", default=[0.85, 0.925, 1.0, 1.075, 1.15])
            p.add_argument("--nreal", type=int, default=12)
            p.add_argument("--Nc", type=int, default=128)
            p.add_argument("--Tmult", type=float, default=1.0)
            p.add_argument("--dtau-mult", type=float, default=12.0, dest="dtau_mult")
            p.add_argument("--nworkers", type=int, default=3)
            p.add_argument("--entropy-stride", type=int, default=4, dest="entropy_stride")
            p.add_argument("--solver", choices=["brentq","newton"], default="brentq",
                           help="newton = ~1.27x faster cloning, statistical (not bit-exact); validate first")
            p.add_argument("--shard", type=int, default=0,
                           help="this task index (0..nshards-1); for Slurm arrays use $SLURM_ARRAY_TASK_ID")
            p.add_argument("--nshards", type=int, default=1,
                           help="total shards; round-robin splits the realization list disjointly across array tasks")
            p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "aggregate":
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
