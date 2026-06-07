"""
worker_opdim_pps.py -- measure the bond-operator scaling dimension Delta_B at
criticality (lambda_c, zeta=1) for the QJ Case-B Kitaev MIPT, together with the
Jian/Foster multifractal exponents and the entanglement central charge.

GOAL (the y_zeta determination):
    y_zeta = 2 - Delta_B(lambda_c),   phi = y_lambda / y_zeta = 1/(2 y_zeta).
The project measured Delta_B ~ 1 at the NO-CLICK critical point (a free/UV
anchor); this worker measures Delta_B at the actual MIPT (lambda_c, zeta=1),
i.e. the IR value, which is what enters phi.  Prediction (Foster y_lambda=1/2 +
the boundary data): Delta_B(lambda_c) ~ 1.1-1.4.

WHY zeta=1 needs NO cloning: the PPS weight zeta^{N_clicks}=1, so Born-rule
averages are plain trajectory averages.  We sample exact continuous-time
Born-rule quantum-jump trajectories (gaussian_born_rule_trajectory) -- no
population dynamics, no reweighting, no Trotter dt -- which is the cheapest
correct estimator at the Born point.

PER-TRAJECTORY OBSERVABLES (from the final-time Majorana covariance Gamma,
2L x 2L, convention Gamma_{ab} = (i/2)<[g_a,g_b]>, so <g_a g_b> = -i Gamma_ab):
  b[x]   = Gamma[2x, 2x+3] = <B_x>_Q,  B_x = i g_{2x} g_{2x+3}  (the measured bond)
           -> trajectory-COVARIANCE C_sc(r) = Cov_traj(b_x, b_{x+r}) ~ r^{-2 Delta_B}
              (the single-copy-mass / nonlinear-in-rho correlator that sets y_zeta).
  g[r]   = mean_p Gamma[p,p+r]^2   (squared Majorana correlator, transl.-averaged)
           -> X1 from <g>, X_typ from exp<log g>, x2 from Var(log g)  [Jian Table I].
  cq[r]  = mean_x ( Gamma[2x,2y+3]Gamma[2x+3,2y] - Gamma[2x,2y]Gamma[2x+3,2y+3] ),
           y = x+r : per-trajectory quantum-connected bond correlator (diagnostic;
           linear-in-rho, expected ~trivial at long times -- contrast with C_sc).
  S_half = entanglement_entropy(Gamma, L//2)  -> c_ent from S_half vs log L.

Translation averages use a BULK window of half-width bulk_frac*L to avoid OBC
Friedel/boundary contamination (we reuse the validated OBC backend rather than
risk a hand-rolled PBC generator; the trajectory covariance subtracts the mean
profile, so it is much less Friedel-prone than a single-state correlator).

Grid: task_id indexes (L, lam) in L_LIST x LAM_LIST.  Env overrides:
  PPS_N_TRAJ    (default 2000)   trajectories per (L,lam)
  PPS_T_MULT    (default 3.0)    evolution time T = T_MULT * L
  PPS_N_WORKERS (default 1)      multiprocessing pool size (= cpus-per-task)
  PPS_BULK_FRAC (default 0.25)
  PPS_L_LIST    (default "64,96,128")
  PPS_LAM_LIST  (default "0.46,0.48,0.50,0.52,0.54")
  PPS_SEED0     (default 20260607)
  PPS_FORCE_RERUN (default 0; 1 to overwrite an existing .npz)

Usage:  python -m pps_qj.parallel.worker_opdim_pps <task_id> <output_dir>
Output: <output_dir>/opdim_{task_id:04d}.npz
"""
from __future__ import annotations
import os, sys, time
import numpy as np
from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L  # identical CMI/B_L tripartition

from pps_qj.gaussian_backend import (
    build_gaussian_chain_model,
    gaussian_born_rule_trajectory,
    entanglement_entropy,
)

# ---- module globals for the worker pool (set by _pool_init via fork) ----
_MODEL = None
_T = None


def _pool_init(L, w, alpha, T):
    global _MODEL, _T
    _MODEL = build_gaussian_chain_model(int(L), float(w), float(alpha))
    _T = float(T)


def _observables_from_cov(Gamma, L, bulk_frac):
    """Return (b, g, cq, S_half, S_AB, S_BC, S_B, S_ABC, CMI, B_L) for one cov."""
    w0 = int(round(bulk_frac * L))
    n2 = 2 * L

    # bond expectations b[x] = Gamma[2x, 2x+3], x = 0..L-2
    xs = np.arange(L - 1)
    b = Gamma[2 * xs, 2 * xs + 3].astype(np.float64)              # (L-1,)

    # squared Majorana correlator g[r], translation-averaged over the bulk
    p_lo, p_hi = 2 * w0, n2 - 2 * w0                              # Majorana bulk [p_lo,p_hi)
    r_max_g = max((p_hi - p_lo) - 1, 0)
    g = np.full(r_max_g, np.nan)
    for r in range(1, r_max_g + 1):
        ps = np.arange(p_lo, p_hi - r)
        if ps.size:
            g[r - 1] = float(np.mean(Gamma[ps, ps + r] ** 2))

    # quantum-connected bond correlator cq[r] (bond separation r), diagnostic
    b_lo, b_hi = w0, (L - 1) - w0                                 # bond bulk [b_lo,b_hi)
    r_max_b = max((b_hi - b_lo) - 1, 0)
    cq = np.full(r_max_b, np.nan)
    for r in range(1, r_max_b + 1):
        x = np.arange(b_lo, b_hi - r)
        y = x + r
        a, bb, c, d = 2 * x, 2 * x + 3, 2 * y, 2 * y + 3
        cq[r - 1] = float(np.mean(Gamma[a, d] * Gamma[bb, c]
                                  - Gamma[a, c] * Gamma[bb, d]))

    S_half = float(entanglement_entropy(Gamma, L // 2))

    # crossing observables: quarter-chain tripartition CMI / B_L, IDENTICAL to
    # the clone pipeline (reuse _batched_compute_B_L on a batch of one).
    # A=[0,L/2) B=[L/2,L) C=[L,3L/2) (Majorana idx); CMI=S_AB+S_BC-S_B-S_ABC,
    # B_L = CMI * S_AB. NaN when L % 4 != 0.
    _bl = _batched_compute_B_L([Gamma], L)
    return (b, g, cq, S_half,
            float(_bl["S_AB"][0]), float(_bl["S_BC"][0]),
            float(_bl["S_B"][0]),  float(_bl["S_ABC"][0]),
            float(_bl["CMI"][0]),  float(_bl["B_L"][0]))


def _run_one(args):
    seed_seq, bulk_frac, L = args
    rng = np.random.default_rng(seed_seq)
    res = gaussian_born_rule_trajectory(_MODEL, _T, rng)
    Gamma = np.asarray(res.final_covariance, dtype=np.float64)
    return _observables_from_cov(Gamma, L, bulk_frac)


def _grid():
    L_list = [int(x) for x in os.environ.get("PPS_L_LIST", "64,96,128").split(",")]
    lam_list = [float(x) for x in
                os.environ.get("PPS_LAM_LIST", "0.46,0.48,0.50,0.52,0.54").split(",")]
    return [(L, lam) for L in L_list for lam in lam_list]


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m pps_qj.parallel.worker_opdim_pps "
                         "<task_id> <output_dir>")
    task_id = int(sys.argv[1])
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    grid = _grid()
    if not (0 <= task_id < len(grid)):
        raise SystemExit(f"task_id {task_id} out of range 0..{len(grid) - 1} "
                         f"(grid size {len(grid)})")
    L, lam = grid[task_id]
    w, alpha = 1.0 - lam, lam

    N_traj = int(os.environ.get("PPS_N_TRAJ", "2000"))
    T = float(os.environ.get("PPS_T_MULT", "3.0")) * L
    n_workers = int(os.environ.get("PPS_N_WORKERS", "1"))
    bulk_frac = float(os.environ.get("PPS_BULK_FRAC", "0.25"))
    seed0 = int(os.environ.get("PPS_SEED0", "20260607"))
    force = os.environ.get("PPS_FORCE_RERUN", "0") not in ("0", "", "false", "False")

    out_file = os.path.join(out_dir, f"opdim_{task_id:04d}.npz")
    if os.path.exists(out_file) and not force:
        print(f"[skip] {out_file} exists (set PPS_FORCE_RERUN=1 to overwrite)")
        return

    children = np.random.SeedSequence(seed0 + 1_000_003 * task_id).spawn(N_traj)
    args = [(children[i], bulk_frac, L) for i in range(N_traj)]

    t0 = time.time()
    print(f"[opdim] task {task_id}: L={L} lam={lam} w={w:.3f} T={T} "
          f"N_traj={N_traj} workers={n_workers} bulk_frac={bulk_frac}", flush=True)

    if n_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers, initializer=_pool_init,
                      initargs=(L, w, alpha, T)) as pool:
            results = pool.map(_run_one, args,
                               chunksize=max(1, N_traj // (n_workers * 8)))
    else:
        _pool_init(L, w, alpha, T)
        results = [_run_one(a) for a in args]

    bs  = np.stack([r[0] for r in results], axis=0)              # (N, L-1)
    gs  = np.stack([r[1] for r in results], axis=0)              # (N, r_max_g)
    cqs = np.stack([r[2] for r in results], axis=0)              # (N, r_max_b)
    Sh  = np.asarray([r[3] for r in results], dtype=np.float64)  # (N,)
    S_AB_t  = np.asarray([r[4] for r in results], dtype=np.float64)
    S_BC_t  = np.asarray([r[5] for r in results], dtype=np.float64)
    S_B_t   = np.asarray([r[6] for r in results], dtype=np.float64)
    S_ABC_t = np.asarray([r[7] for r in results], dtype=np.float64)
    CMI_t   = np.asarray([r[8] for r in results], dtype=np.float64)
    B_L_t   = np.asarray([r[9] for r in results], dtype=np.float64)

    # block over trajectories -> per-block means (analog of per-realisation
    # means across clones); spread across blocks gives the crossing error bar.
    n_blocks = max(1, min(int(os.environ.get("PPS_N_BLOCKS", "20")), N_traj))
    idx_blocks = np.array_split(np.arange(N_traj), n_blocks)
    def _bmeans(a):
        return np.array([np.nanmean(a[ix]) for ix in idx_blocks], dtype=np.float64)
    def _msem(a):
        a = a[np.isfinite(a)]; n = a.size
        return ((float(np.mean(a)) if n else float("nan")),
                (float(np.std(a) / np.sqrt(n)) if n else float("nan")))
    B_L_means_all   = _bmeans(B_L_t)
    CMI_means_all   = _bmeans(CMI_t)
    S_AB_means_all  = _bmeans(S_AB_t)
    S_BC_means_all  = _bmeans(S_BC_t)
    S_B_means_all   = _bmeans(S_B_t)
    S_ABC_means_all = _bmeans(S_ABC_t)
    S_means_all     = _bmeans(Sh)
    B_L_mean,  B_L_err   = _msem(B_L_means_all)
    CMI_mean,  CMI_err   = _msem(CMI_means_all)
    S_AB_mean, S_AB_err  = _msem(S_AB_means_all)
    S_BC_mean, S_BC_err  = _msem(S_BC_means_all)
    S_B_mean,  S_B_err   = _msem(S_B_means_all)
    S_ABC_mean,S_ABC_err = _msem(S_ABC_means_all)
    S_mean,    S_err     = _msem(S_means_all)

    np.savez_compressed(
        out_file,
        L=L, lam=lam, w=w, alpha=alpha, zeta=1.0, T=T, N_c=1,
        N_traj=N_traj, n_blocks=n_blocks, bulk_frac=bulk_frac,
        seed0=seed0, task_id=task_id,
        b=bs, g=gs, cq=cqs, S_half=Sh,
        # crossing observables (quarter-chain tripartition; identical to dense)
        B_L_mean=B_L_mean, B_L_err=B_L_err, B_L_means_all=B_L_means_all,
        CMI_mean=CMI_mean, CMI_err=CMI_err, CMI_means_all=CMI_means_all,
        S_AB_mean=S_AB_mean, S_AB_err=S_AB_err, S_AB_means_all=S_AB_means_all,
        S_BC_mean=S_BC_mean, S_BC_err=S_BC_err, S_BC_means_all=S_BC_means_all,
        S_B_mean=S_B_mean, S_B_err=S_B_err, S_B_means_all=S_B_means_all,
        S_ABC_mean=S_ABC_mean, S_ABC_err=S_ABC_err, S_ABC_means_all=S_ABC_means_all,
        S_mean=S_mean, S_err=S_err, S_means_all=S_means_all,
        wall_time=time.time() - t0,
    )
    print(f"[opdim] task {task_id} done in {time.time() - t0:.1f}s -> {out_file}",
          flush=True)


if __name__ == "__main__":
    main()
