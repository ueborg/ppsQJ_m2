"""
worker_areaphase_pps.py -- GATE 2: measure the area-phase correlation length
xi(zeta, lambda) just above lambda_c, to discriminate the boundary exponent phi.

PREDICTIONS (HANDOFF gate 2):
    xi ~ zeta^{-1/2}, lambda-FLAT   <=>  phi = 1/2  (saturated-defect window law)
    xi ~ zeta^{-1}                  <=>  phi = 1    (coherent / perturbative channel)
    essential form                  <=>  marginal asymptote

METHOD.  At zeta<1 the PPS average needs cloning (population reweighting).  We run
the same cloning backend as worker_clone_pps, take the final clone covariances
Gamma (2L x 2L, Gamma_{ab}=(i/2)<[g_a,g_b]>), and form the SINGLE-COPY-MASS bond
correlator exactly as worker_opdim does at zeta=1, but now ACROSS THE CLONE
POPULATION instead of across independent Born trajectories:

    b[x]      = Gamma[2x, 2x+3]                      (the measured bond <B_x>)
    C_sc(r)   = Cov_clones(b_x, b_{x+r}), bulk-averaged over x

In the AREA phase (lambda > lambda_c) C_sc(r) decays exponentially, C_sc ~ e^{-r/xi};
we fit log C_sc(r) vs r to extract xi.

EVEN-r ONLY (load-bearing, the 2026-06-10 opdim pre-run fix): odd-r C_sc are
exact zeros by the two-chain decoupling (b_x lives on chain A for even x, B for
odd x). We fit on even r and report the odd-r residual as a free null test.

CAVEAT [honest]: clone-population Cov carries genealogical (finite-N_c) bias --
the clones share ancestry. For a LENGTH SCALE in the area phase this is far less
sensitive than a critical amplitude (the decoupling-coalescence is r-independent
to leading order), but the N_c-dependence of xi MUST be checked on at least one
rung before banking phi. The grid includes an N_c knob for exactly this.

Grid (env-overridable, area-phase auto-placement lambda = lambda_c(zeta)+offset
with lambda_c ~ 0.5*sqrt(zeta)):
  PPS_L_LIST       (default "64,96")
  PPS_ZETA_LIST    (default "0.10,0.20,0.30,0.50,0.80")
  PPS_LAM_OFFSETS  (default "0.06,0.10,0.14")   offsets ABOVE lambda_c into area phase
  PPS_NC           (default 250)
  PPS_T_MULT       (default 3.0)   T = T_MULT * L
  PPS_BULK_FRAC    (default 0.25)
  PPS_N_WORKERS    (default 1)     realisation-pool size
  PPS_SEED0        (default 20260610)
  PPS_FORCE_RERUN  (default 0)

Usage:  python -m pps_qj.parallel.worker_areaphase_pps <task_id> <output_dir>
Output: <output_dir>/areaphase_{task_id:04d}.npz
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
import numpy as np

from pps_qj.cloning import CloningCollapse, CloningResult, run_cloning
from pps_qj.gaussian_backend import build_gaussian_chain_model

N_REAL = 5
_LAMBDA_C = lambda z: 0.5 * np.sqrt(z)   # small-zeta corner law (placeholder center)


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
def _grid():
    L_list   = [int(x)   for x in os.environ.get("PPS_L_LIST", "64,96").split(",")]
    z_list   = [float(x) for x in os.environ.get("PPS_ZETA_LIST",
                                                  "0.10,0.20,0.30,0.50,0.80").split(",")]
    offsets  = [float(x) for x in os.environ.get("PPS_LAM_OFFSETS",
                                                  "0.06,0.10,0.14").split(",")]
    g = []
    for L in L_list:
        for z in z_list:
            for off in offsets:
                lam = float(min(0.99, _LAMBDA_C(z) + off))
                g.append((L, lam, z))
    return g


# ---------------------------------------------------------------------------
# bond correlator + xi fit
# ---------------------------------------------------------------------------
def _bond_profiles(final_covs, L):
    """Return (N_c, L-1) array of b[x]=Gamma[2x,2x+3] for each clone covariance."""
    xs = np.arange(L - 1)
    out = np.empty((len(final_covs), L - 1), dtype=np.float64)
    for i, G in enumerate(final_covs):
        G = np.asarray(G, dtype=np.float64)
        out[i] = G[2 * xs, 2 * xs + 3]
    return out


def _C_sc(bprofiles, L, bulk_frac):
    """Translation-averaged clone-covariance of the bond profile.
    Returns r (1..r_max) and C_sc(r). C_sc(r)=mean_x Cov_clones(b_x,b_{x+r})."""
    w0 = int(round(bulk_frac * L))
    x_lo, x_hi = w0, (L - 1) - w0
    r_max = max((x_hi - x_lo) - 1, 0)
    # center per-site across clones
    bmean = bprofiles.mean(axis=0, keepdims=True)
    bc = bprofiles - bmean                                  # (N_c, L-1)
    rs = np.arange(1, r_max + 1)
    C = np.full(r_max, np.nan)
    Nc = bprofiles.shape[0]
    for k, r in enumerate(rs):
        x = np.arange(x_lo, x_hi - r)
        if x.size == 0:
            continue
        # Cov_clones(b_x, b_{x+r}) averaged over x and clones
        prod = bc[:, x] * bc[:, x + r]                      # (N_c, |x|)
        C[k] = float(prod.sum() / (Nc * x.size))            # population cov (bias ok for length)
    return rs, C


def _fit_xi(rs, C):
    """Fit xi from C_sc(r) ~ e^{-r/xi} on EVEN r (odd r are decoupling zeros).
    Returns (xi, R2, n_pts, odd_null) where odd_null = rms(C on odd r)/rms(C on even r)."""
    even = (rs % 2 == 0)
    odd = ~even
    Ce, re = C[even], rs[even]
    Co = C[odd]
    # odd-r null metric
    rms = lambda a: float(np.sqrt(np.nanmean(a ** 2))) if np.isfinite(a).any() else np.nan
    odd_null = rms(Co) / rms(Ce) if rms(Ce) and np.isfinite(rms(Ce)) else np.nan
    # keep positive, above a relative noise floor, contiguous from small r
    good = np.isfinite(Ce) & (Ce > 0)
    if good.sum() < 3:
        return np.nan, np.nan, int(good.sum()), odd_null
    Cg, rg = Ce[good], re[good]
    floor = 1e-3 * Cg[0]
    keep = Cg > floor
    Cg, rg = Cg[keep], rg[keep]
    if Cg.size < 3:
        return np.nan, np.nan, int(Cg.size), odd_null
    y = np.log(Cg)
    A = np.vstack([rg, np.ones_like(rg)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = coef[0]
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    xi = (-1.0 / slope) if slope < 0 else np.nan      # decaying -> slope<0 -> xi>0
    return float(xi), float(R2), int(Cg.size), float(odd_null)


# ---------------------------------------------------------------------------
# one realisation
# ---------------------------------------------------------------------------
def _run_one(args):
    L, w, alpha, zeta, T, N_c, seed, bulk_frac = args
    rng = np.random.default_rng(seed)
    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    try:
        res: CloningResult = run_cloning(model, zeta=zeta, T_total=T, N_c=N_c,
                                         rng=rng, show_progress=False)
    except CloningCollapse as exc:
        return {"ok": False, "error": f"collapse: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    bprof = _bond_profiles(res.final_covs, L)
    rs, C = _C_sc(bprof, L, bulk_frac)
    xi, R2, npts, odd_null = _fit_xi(rs, C)
    return {"ok": True, "xi": xi, "R2": R2, "n_fit": npts,
            "odd_null": odd_null, "ess": float(res.eff_sample_size),
            "rs": rs.astype(np.float64), "C_sc": C.astype(np.float64)}


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m pps_qj.parallel.worker_areaphase_pps "
                         "<task_id> <output_dir>")
    task_id = int(sys.argv[1]); out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)

    grid = _grid()
    if not (0 <= task_id < len(grid)):
        raise SystemExit(f"task_id {task_id} out of range 0..{len(grid)-1} (grid {len(grid)})")
    L, lam, zeta = grid[task_id]
    w, alpha = 1.0 - lam, lam
    N_c   = int(os.environ.get("PPS_NC", "250"))
    T     = float(os.environ.get("PPS_T_MULT", "3.0")) * L
    bulk  = float(os.environ.get("PPS_BULK_FRAC", "0.25"))
    seed0 = int(os.environ.get("PPS_SEED0", "20260610"))
    nwork = int(os.environ.get("PPS_N_WORKERS", "1"))
    force = os.environ.get("PPS_FORCE_RERUN", "0") not in ("0", "", "false", "False")

    out_file = out_dir / f"areaphase_{task_id:04d}.npz"
    if out_file.exists() and not force:
        print(f"[areaphase] task {task_id}: exists, skip"); return 0

    t0 = time.time()
    print(f"[areaphase] task {task_id}: L={L} lam={lam:.4f} zeta={zeta:.2f} "
          f"(lam_c~{_LAMBDA_C(zeta):.4f}) N_c={N_c} T={T} workers={nwork}", flush=True)

    args = [(L, w, alpha, zeta, T, N_c, seed0 + 7919 * task_id + 997 * r, bulk)
            for r in range(N_REAL)]
    if nwork > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(nwork, N_REAL)) as pool:
            results = pool.map(_run_one, args)
    else:
        results = [_run_one(a) for a in args]

    xis  = np.array([r["xi"]       for r in results if r.get("ok")], dtype=np.float64)
    R2s  = np.array([r["R2"]       for r in results if r.get("ok")], dtype=np.float64)
    odds = np.array([r["odd_null"] for r in results if r.get("ok")], dtype=np.float64)
    esss = np.array([r["ess"]      for r in results if r.get("ok")], dtype=np.float64)
    ok_results = [r for r in results if r.get("ok")]
    nfail = sum(1 for r in results if not r.get("ok"))

    def _ms(a):
        a = a[np.isfinite(a)]
        return ((float(np.mean(a)), float(np.std(a) / max(1, a.size) ** 0.5)) if a.size
                else (float("nan"), float("nan")))
    xi_mean, xi_err = _ms(xis)

    np.savez_compressed(
        out_file,
        task_id=task_id, L=L, lam=lam, zeta=zeta, w=w, alpha=alpha,
        lam_c_center=float(_LAMBDA_C(zeta)), N_c=N_c, T=T, n_real=N_REAL,
        bulk_frac=bulk, n_fail=nfail,
        xi_mean=xi_mean, xi_err=xi_err, xis_all=xis,
        R2_mean=float(np.nanmean(R2s)) if R2s.size else float("nan"),
        odd_null_mean=float(np.nanmean(odds)) if odds.size else float("nan"),
        ess_mean=float(np.nanmean(esss)) if esss.size else float("nan"),
        # keep the first realisation's raw C_sc(r) for inspection
        rs=(ok_results[0]["rs"] if ok_results else np.asarray([])),
        C_sc=(ok_results[0]["C_sc"] if ok_results else np.asarray([])),
        wall_time=time.time() - t0,
    )
    print(f"[areaphase] task {task_id} done: xi={xi_mean:.3f}+-{xi_err:.3f} "
          f"R2={np.nanmean(R2s) if R2s.size else float('nan'):.3f} "
          f"odd_null={np.nanmean(odds) if odds.size else float('nan'):.3f} "
          f"nfail={nfail} t={time.time()-t0:.1f}s -> {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
