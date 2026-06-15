#!/usr/bin/env python3
# WARNING (2026-06-15): the E_analytic kernel in this file is WRONG. It drops the
# hopping w from the measured bond (it uses t1 = -i*kappa), so its Fermi-step /
# lambda*=4/5 / nu0 output is a self-consistency check of an incorrect symbol, NOT
# a test of the real model. The correct no-click physics (area-law for lambda>0,
# critical only at lambda=0, xi_nc ~ lambda^-2) is in delta_B_zeta0.py and
# exponent_noclick.py. See theory/HANDOFF.md (zeta=0 anchor, CORRECTED 2026-06-15).
# Do not trust this file.
"""
anchor_scan.py  --  GATE 1 (single-particle band level): numerical test of the
analytical zeta=0 (no-click) anchor for the Case B Kitaev subchain (ppsQJ_m2).
REAL project conventions: alpha = lambda, w = 1 - lambda, kappa = alpha/4
(so the exceptional point sits at lambda* = 4/5).

Confirms the [V] foundation of the phi=1/2 derivation and the REFUTATION of
xi_ps ~ lambda^-2.  Three single-particle, convention-independent claims; all
three pass (container run 2026-06-10):

  [1] Fermi step   : Im E(q) flips sign at q = +/- pi/2 for all lambda < lambda*.
  [2] state xi     : sinh(1/xi) = (w^2-kappa^2)/(2 kappa w); SHORT at small lambda
                     (~1/ln(1/lambda)), NOT lambda^-2; diverges only at the EP.
  [3] EP exponent  : xi ~ |lambda-lambda*|^{-nu0} as lambda -> 4/5; nu0 ~ 1.
  [4] Delta_B (HOOK): model claims <B0 Br>_c ~ r^{-2} (Delta_B=1, matches measured
                     1.009). NOT computed here -- needs the REAL no-click Majorana
                     covariance (worker_zeta0_pps.py produces it; this script only
                     does the single-particle band structure). B_x = i g_{2x} g_{2x+3},
                     b[x] = Gamma[2x, 2x+3]; the 1/r^2 lives in the inter-sublattice
                     block. See delta_B_hook().

This is the SSH-anchor self-consistency check.  To verify the REDUCTION (that the
real Case B no-click dynamics IS this anchor), extend worker_zeta0_pps.py to dump
the steady-state covariance and run delta_B on it (still pending; see HANDOFF
gate 1 items (a)+(b)).

Runtime: seconds.  No cluster needed.
"""
import numpy as np
import argparse

def E_analytic(q, w, kappa):
    return np.sqrt(w**2 - kappa**2 - 2j * kappa * w * np.cos(q))

def conv(lam):                       # REAL project conventions
    return (1.0 - lam, lam / 4.0)    # (w, kappa)

LAMBDA_STAR = 4.0 / 5.0              # EP: kappa = w  <=>  lambda/4 = 1-lambda

def fermi_step(w, kappa, nq=4000):
    qs = np.linspace(-np.pi, np.pi, nq, endpoint=False)
    s = np.sign(E_analytic(qs, w, kappa).imag)
    return qs[np.where(np.diff(s) != 0)[0]] / np.pi

def xi_state(w, kappa):
    return 1.0 / np.arcsinh((w**2 - kappa**2) / (2 * kappa * w))

def delta_B_hook():
    """
    Compute Delta_B with the real Majorana covariance, not the single-particle
    band structure here. Steps:
      1. Get the no-click steady-state covariance Gamma (extend worker_zeta0_pps:
         it already evolves to steady state; return covariance_from_orbitals).
      2. b[x] = Gamma[2x, 2x+3] (the measured bond), C_sc(r)=Cov_x(b_x,b_{x+r}).
      3. Fit C_sc(r) ~ r^{-2 Delta_B} on EVEN r (odd r are exact zeros by the
         two-chain decoupling). Expect Delta_B ~ 1.
    """
    raise NotImplementedError("run on real Majorana covariance from worker_zeta0_pps")

def main():
    argparse.ArgumentParser().parse_args()
    print("=" * 62)
    print("anchor_scan.py   (alpha=lambda, w=1-lambda, kappa=lambda/4)")
    print("EP at lambda* = 4/5 = 0.800")
    print("=" * 62)

    print("\n[1]+[2] Fermi step location and state xi  (lambda < lambda*):")
    print("    %7s %6s %7s %8s %12s" % ("lambda", "w", "kappa", "xi", "flips q/pi"))
    for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.79]:
        w, k = conv(lam)
        f = fermi_step(w, k)
        print("    %7.2f %6.2f %7.4f %8.4f %12s" % (lam, w, k, xi_state(w, k), np.round(f, 2)))
    print("    -> Fermi step pinned at q=+/-pi/2; xi short at small lambda,")
    print("       diverging only as lambda -> 0.8 (EP).  NOT lambda^-2.")

    print("\n[3] EP exponent nu0  (xi ~ |lambda-4/5|^{-nu0}, expect ~1):")
    ds = np.array([0.05, 0.02, 0.01, 0.005, 0.002])
    xis = np.array([xi_state(*conv(LAMBDA_STAR - d)) for d in ds])
    print("    nu0 = %.3f" % (-np.polyfit(np.log(ds), np.log(xis), 1)[0]))

    print("\n[4] Delta_B: NOT computed -- run delta_B_hook on the real no-click")
    print("    covariance from (an extended) worker_zeta0_pps.py.")

if __name__ == "__main__":
    main()
