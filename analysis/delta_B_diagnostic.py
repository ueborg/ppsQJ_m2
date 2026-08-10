#!/usr/bin/env python3
"""
delta_B_diagnostic.py -- settle CRITICAL vs GAPPED for the zeta=0 no-click state.

The gate-1 result (Delta_B growing with L and lambda) is the fingerprint of a
correlator decaying FASTER than a power law -> the state may be GAPPED, not
critical, consistent with the short xi_state (~0.4-0.6) from anchor_scan.

Prints the ACTUAL decay shapes and fits BOTH a power law and an exponential, so
the verdict is not inferred from a single forced power-law fit:

  G2(r) = sqrt(mean_p Gamma[p,p+r]^2)   -- raw Majorana two-point envelope
          critical free fermion -> r^{-1};   gapped -> e^{-r/xi}
  cq(r) -- connected bond correlator (as in delta_B_zeta0)

Also checks convergence (T vs ~2T). Run on Mac/Habrok (needs the backend).
  python analysis/delta_B_diagnostic.py
"""
import numpy as np
from delta_B_zeta0 import steady_cov, cq_connected


def g2_envelope(G, L, bulk_frac=0.25):
    G = np.asarray(G, dtype=np.float64)
    w0 = int(round(bulk_frac * L))
    p_lo, p_hi = 2 * w0, 2 * ((L - 1) - w0)
    r_max = max((p_hi - p_lo) - 1, 0)
    rs = np.arange(1, r_max + 1)
    g = np.full(r_max, np.nan)
    for k, r in enumerate(rs):
        p = np.arange(p_lo, p_hi - r)
        if p.size:
            g[k] = float(np.sqrt(np.mean(G[p, p + r] ** 2)))
    return rs, g


def fit_both(rs, y):
    """Fit y~r^-a and y~e^{-r/xi} on positive points. Returns (a,R2p,xi,R2e)."""
    good = np.isfinite(y) & (y > 1e-6 * np.nanmax(y))
    rs, y = rs[good], y[good]
    if rs.size < 5:
        return np.nan, np.nan, np.nan, np.nan
    ly = np.log(y)
    cp = np.polyfit(np.log(rs), ly, 1); a = -cp[0]
    r2p = 1 - np.sum((ly - np.polyval(cp, np.log(rs))) ** 2) / np.sum((ly - ly.mean()) ** 2)
    ce = np.polyfit(rs, ly, 1); xi = (-1.0 / ce[0]) if ce[0] < 0 else np.nan
    r2e = 1 - np.sum((ly - np.polyval(ce, rs)) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return float(a), float(r2p), float(xi), float(r2e)


def main():
    L, lam = 256, 0.30
    print(f"diagnostic at L={L}, lam={lam} (mid critical segment)\n")
    G = steady_cov(L, lam)
    G_long = steady_cov(L, lam, T_mult=30.0)
    rs, g = g2_envelope(G, L); _, g_b = g2_envelope(G_long, L)
    rc, cq = cq_connected(G, L)

    print("raw two-point envelope G2(r)=sqrt(<Gamma[p,p+r]^2>):")
    print("  r :  " + " ".join(f"{r:7d}" for r in rs[:16:2]))
    print("  G2:  " + " ".join(f"{v:7.4f}" for v in g[:16:2]))
    a, r2p, xi, r2e = fit_both(rs, g)
    print(f"  power r^-a : a={a:.3f}  R2={r2p:.3f}   (critical free fermion: a~1)")
    print(f"  exponential: xi={xi:.3f} R2={r2e:.3f}   (gapped: finite xi)")
    print(f"  VERDICT: {'GAPPED (exp fits better)' if r2e > r2p else 'CRITICAL (power fits better)'}")

    conv = np.nanmax(np.abs(g - g_b) / (np.abs(g) + 1e-12))
    print(f"\nconvergence (T vs ~2T) max rel change in G2: {conv:.2e} "
          f"({'OK' if conv < 0.05 else 'NOT CONVERGED -- raise T_mult'})")

    print("\nconnected bond cq(r):")
    print("  r :  " + " ".join(f"{r:9d}" for r in rc[:16:2]))
    print("  cq:  " + " ".join(f"{v:9.2e}" for v in cq[:16:2]))
    a2, r2p2, xi2, r2e2 = fit_both(rc, np.abs(cq))
    print(f"  power a={a2:.3f} R2={r2p2:.3f} | exp xi={xi2:.3f} R2={r2e2:.3f}"
          f"  -> {'GAPPED' if r2e2 > r2p2 else 'CRITICAL'}")


if __name__ == "__main__":
    main()
