#!/usr/bin/env python3
"""plan_cloning_statistics.py - budget planner for guided-cloning campaigns.

TASK-2026-08-31-SMCCERT. Task-local. NOT canonical. This SUPERSEDES the
SMCSTAT prototype (proposed_git/0003-add-statistics-planner.patch), which must
NOT be merged as written.

WHY THE PROTOTYPE HAD TO CHANGE
    The prototype's docstring already wrote the right model,
        MSE(R, N_c) = A/(R*N_c) + (B/N_c)^2
    but `plan()` implemented only the first term. `B` appears nowhere in its
    code. Concretely, `plan()` sets
        N_c = max(N_pre, 32);  R = ceil(M / N_c)
    so N_c is pinned at the pre-asymptotic floor and EVERY remaining core-hour
    goes to R. A bias term is invariant under R. The prototype therefore does
    exactly the thing that must never happen: it buys an arbitrarily tight
    interval around a systematically displaced number, and the tightening looks
    like convergence.

    Its one bias-adjacent guard, N_c >= 5*VIF (epsilon = VIF/N_c <= 0.2), is
    MEASURED HERE TO BE INSUFFICIENT, not merely unproven. At the A-HV cell,
    N_c = 256 gives VIF = 18.9, so epsilon = 0.074 and the prototype certifies
    the configuration - while the measured bias there is +0.0348 against an
    across-population SEM of 0.0083 at R = 48. The guard passes a cell whose
    bias is FOUR TIMES its own error bar.

    And epsilon cannot be repaired by tightening its threshold, because VIF does
    not predict bias. S-CRIT32 (L=32, lambda=0.2793, VIF 2.4-5.9) and A-MV
    (L=32, lambda=0.35, VIF 3.3-4.1) have OVERLAPPING VIF and differ in B by a
    factor of ~24. See BIAS_CALIBRATION below and MATCHED_CELL_TESTS.md.

WHAT IT DOES
    Given a physical cell (L, T, zeta, lambda) and a target - either a CMI
    standard error or a crossing precision - it returns a recommended
    (N_c, R), the predicted uncertainty, the estimated core-hours and memory,
    and the uncertainty OF THAT PREDICTION.

WHAT IT REFUSES TO DO
    Extrapolate silently. Every quantity it reports carries a calibration
    status, and outside the calibrated box it returns the literal token

        CALIBRATION_REQUIRED

    rather than a number. This is the whole point: the project has been burned
    by plausible-looking output from an uncalibrated model
    (EV-CODE-ANCHORSCAN-001), and a planner that guesses is worse than no
    planner because it is trusted.

THE MODEL
    Var(population mean of f) = A_f(L,T,zeta,lam) / N_c,  A_f = VIF * tau^2
    Var(mean over R populations) = A_f / (R * N_c)
    bias                          = B_f / N_c,  B_f CALIBRATED PER CELL, never
                                    inferred from VIF (see above)
    MSE(R, N_c)                   = A/(R*N_c) + (B/N_c)^2
    cost                          = R * (c0 + c1 * N_c),  c1 = n_steps * c(L,zeta)

    At fixed clone budget M = R*N_c the VARIANCE term is allocation-invariant.
    The bias term is not: it depends on N_c ALONE. The binding constraints are
    therefore, in this order:
        N_c >= |B_f| / bias_tol   (THE BIAS FLOOR - the constraint the prototype
                                   did not have. R cannot buy it down.)
        N_c >= 5 * VIF            (the epsilon <= 0.2 pre-asymptotic criterion,
                                   RETAINED but demoted: necessary, not sufficient)
        R   >= R_min(zeta, L)     (so the SEM is usable and the interval covers)
    Budget is spent on N_c until the bias floor is met, and only then on R.
    If the constraints cannot be met inside the budget, the planner says the
    budget is insufficient. It does not shave R, and it does not shave N_c.

    Where B_f is not calibrated for the requested cell, the planner returns
    CALIBRATION_REQUIRED. It NEVER substitutes a VIF-based guess.

CALIBRATION PROVENANCE
    Every calibration constant below is tagged with where it came from. Numbers
    tagged `corpus` are from TV-CORPUS-001 (N_c = 128, T = L, dtau_mult = 12);
    numbers tagged `local` are from this task's runs at dtau_mult = 6 on the
    certified `lowrank` path. They are NOT interchangeable and the tag travels
    with the answer.
"""
from __future__ import annotations
import argparse, json, math, os, sys

CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"

# ---------------------------------------------------------------------------
# Calibration tables. Replaced by tools/calibrate_cloning.py output when
# LOCAL_VALIDATION.md lands; the shipped defaults are what this task measured.
# ---------------------------------------------------------------------------

# VIF(zeta) at L = 128, from TV-CORPUS-001 (dtau_mult = 12, T = L, N_c = 128).
# Median over the scanned lambda grid. Provenance: corpus.
VIF_ZETA_L128 = {
    1.00: 0.97, 0.95: 1.82, 0.90: 3.98, 0.80: 9.0, 0.70: 19.4, 0.60: 33.0,
    0.50: 47.3, 0.45: 60.0, 0.40: 65.4, 0.35: 62.3, 0.30: 55.0, 0.25: 45.0,
    0.20: 31.1, 0.15: 22.0, 0.10: 13.5, 0.075: 8.0, 0.05: 4.62,
}
# VIF vs L at zeta in [0.3, 0.6], T = L, corpus. NOTE: this is NOT a power law -
# a window scan gives exponents spanning 0.93-3.16 and the growth flattens
# between L = 112 and L = 128. Interpolation is therefore done on the TABLE,
# never through a fitted exponent, and outside [64, 128] the planner refuses.
VIF_L_TABLE = {64: 20.8, 80: 34.9, 96: 54.5, 112: 75.2, 128: 81.8}

# Per-clone-per-window wall cost, milliseconds, `lowrank` path, this machine.
# Provenance: local, measured 2026-08-30 at zeta = 0.35, lambda = 0.30.
MS_PER_CLONE_WINDOW = {16: 0.25, 32: 0.37, 48: 0.68, 64: 1.16, 96: 2.68, 128: 6.03}
# Cost rises steeply as zeta -> 1 because the guided proposal intensity is
# c = zeta: at zeta = 1 the proposal is unthinned, so there are ~1/zeta times as
# many jumps per window and each jump costs a low-rank update plus a QR.
# Provenance: local, measured this task. Below zeta ~ 0.3 the no-jump segments
# dominate and the ratio flattens.
COST_ZETA_FACTOR = {1.00: 7.4, 0.90: 6.2, 0.70: 4.4, 0.50: 2.8, 0.35: 1.0,
                    0.30: 0.95, 0.20: 0.85, 0.10: 0.78, 0.05: 0.75}

# ---------------------------------------------------------------------------
# BIAS CALIBRATION. TASK-2026-08-31-SMCCERT.
#
# B_f is the coefficient of the finite-population bias in the POINT ESTIMATE:
#     E[Ihat(N_c)] = I_inf + B_f / N_c + o(1/N_c)
# It is NOT reducible by adding independent populations R.
#
# These entries come from measured N_c ladders, each with a >=3-window scan and
# a bootstrap CI over independent populations. The table is deliberately SMALL
# and its domain is deliberately narrow: a bias rule invented from two points
# would be exactly the failure this task exists to prevent. Outside the listed
# cells the planner returns CALIBRATION_REQUIRED.
#
# Loaded from calibration/bias_calibration.json when that file is present, so
# the numbers travel with their provenance and can be regenerated. The inline
# dict is the fallback and must stay in sync.
_BIAS_CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "calibration", "bias_calibration.json")

BIAS_CALIBRATION = {}     # populated below; key: (L, T, zeta, lam, scheme)


def _load_bias_calibration():
    """Read the calibrated bias table. Returns (table, provenance_note)."""
    if os.path.exists(_BIAS_CAL_PATH):
        with open(_BIAS_CAL_PATH) as fh:
            doc = json.load(fh)
        tab = {}
        for e in doc["cells"]:
            tab[(e["L"], e["T"], e["zeta"], e["lam"],
                 e.get("resample_scheme", "systematic"))] = e
        return tab, doc.get("provenance", "")
    return {}, "NO CALIBRATION FILE FOUND"


BIAS_CALIBRATION, BIAS_PROVENANCE = _load_bias_calibration()


def estimate_bias_coefficient(L, T, zeta, lam, scheme="systematic"):
    """B_f with an explicit calibration status, and its CI.

    Returns (B, B_ci, status, notes). There is NO interpolation and NO fallback
    to a VIF proxy: either the exact cell is calibrated or the answer is
    CALIBRATION_REQUIRED. The measured evidence for refusing to interpolate is
    that two cells with OVERLAPPING VIF differ in B by a factor of ~24 when only
    lambda moves, so no smooth surface through the calibrated points is
    defensible on this many points.
    """
    key = (L, T, zeta, lam, scheme)
    if key in BIAS_CALIBRATION:
        e = BIAS_CALIBRATION[key]
        return (e["B"], e["B_ci"], "calibrated",
                [f"B_f calibrated at this exact cell: {e.get('provenance','')}",
                 f"ladder {e.get('ladder')}, R={e.get('R')}, "
                 f"MDE|B|={e.get('mde_B')}"])
    near = [k for k in BIAS_CALIBRATION
            if k[0] == L and abs(k[2] - zeta) < 1e-9 and k[4] == scheme]
    notes = [f"cell (L={L}, T={T}, zeta={zeta}, lam={lam}, {scheme}) is NOT in "
             f"the bias calibration table"]
    if near:
        notes.append("calibrated cells at this L and zeta: "
                     + ", ".join(f"lam={k[3]} (T={k[1]})" for k in near)
                     + " - but B is NOT interpolated in lambda: the measured "
                       "lambda-dependence of B is far too steep for that")
    notes.append("no VIF-based substitute is used: VIF does not predict B")
    return None, None, CALIBRATION_REQUIRED, notes


# Minimum number of independent populations, by regime.
# Provenance: local + corpus (Route B's disjoint-split coverage measurement).
# The deficit at intermediate zeta and large L is a SKEWNESS effect, so it
# shrinks only like R^{-1/2} and is REDUCED, not removed, by raising R.
R_MIN_DEFAULT = 12
R_MIN_HARD = 8          # below this the SE of the SE is unusable at all


def _interp_table(table, x, name):
    ks = sorted(table)
    if x < ks[0] or x > ks[-1]:
        return None, f"{name}={x} outside calibrated range [{ks[0]}, {ks[-1]}]"
    if x in table:
        return table[x], None
    lo = max(k for k in ks if k <= x); hi = min(k for k in ks if k >= x)
    t = (x - lo) / (hi - lo)
    return table[lo] * (1 - t) + table[hi] * t, None


def n_steps(L, T, lam, dtau_mult=6.0):
    dtau = dtau_mult / max(2.0 * lam * (L - 1), 1e-12)
    return int(math.ceil(T / dtau))


def estimate_vif(L, zeta, lam=None, T=None, scheme="systematic"):
    """VIF estimate with an explicit calibration status.

    Returns (value, status, notes). `status` is 'measured', 'calibrated',
    'interpolated' or CALIBRATION_REQUIRED.

    A DIRECTLY MEASURED VIF at the exact cell beats the corpus table and is
    used first. The table is a fallback for cells that have a corpus VIF but no
    local ladder, and it is T = L only.
    """
    notes = []
    key = (L, T, zeta, lam, scheme)
    if key in BIAS_CALIBRATION and "vif_measured" in BIAS_CALIBRATION[key]:
        e = BIAS_CALIBRATION[key]
        notes.append(f"VIF MEASURED at this exact cell over the ladder "
                     f"{e.get('ladder')} at R={e.get('R')}: median {e['vif_measured']:.2f}, "
                     f"range {e['vif_min']:.2f}-{e['vif_max']:.2f}. "
                     f"Provenance: {e.get('provenance','')}")
        return float(e["vif_measured"]), "measured", notes
    vz, e1 = _interp_table(VIF_ZETA_L128, zeta, "zeta")
    vl, e2 = _interp_table(VIF_L_TABLE, L, "L")
    if e1 or e2:
        return None, CALIBRATION_REQUIRED, [e for e in (e1, e2) if e]
    # The corpus is T = L. Away from T = L the planner has no calibration for
    # the T-dependence and says so rather than assuming separability.
    if T is not None and abs(T - L) > 1e-9:
        notes.append(f"T={T} != L={L}: the corpus calibration is T=L only; "
                     f"the T-dependence is NOT calibrated")
        return None, CALIBRATION_REQUIRED, notes
    # Combine: scale the zeta profile (anchored at L=128) by the L table.
    vif = vz * (vl / VIF_L_TABLE[128])
    notes.append("provenance: corpus (N_c=128, T=L, dtau_mult=12); "
                 "GENCOL E4 supports dtau_mult-invariance of VIF, so the "
                 "transfer to dtau_mult=6 is supported but not verified here")
    notes.append("VIF vs L is interpolated on a TABLE, never through a fitted "
                 "exponent: a window scan gives 0.93-3.16 and the growth "
                 "flattens at L=112->128, so no power law is used")
    status = "calibrated" if (L in VIF_L_TABLE and zeta in VIF_ZETA_L128) else "interpolated"
    return vif, status, notes


def cost_core_seconds(L, T, zeta, lam, N_c, R, dtau_mult=6.0):
    ms, e = _interp_table(MS_PER_CLONE_WINDOW, L, "L")
    if e:
        return None, CALIBRATION_REQUIRED, [e]
    zf, e2 = _interp_table(COST_ZETA_FACTOR, zeta, "zeta")
    if e2:
        return None, CALIBRATION_REQUIRED, [e2]
    ns = n_steps(L, T, lam, dtau_mult)
    c1 = ns * ms * zf / 1e3                     # seconds per clone
    c0 = 2.0                                    # per-population overhead, seconds
    return R * (c0 + c1 * N_c), "calibrated", [
        f"n_steps={ns}, {ms:.2f} ms/clone-window at L={L}, "
        f"zeta cost factor {zf:.2f}",
        "provenance: local, this machine, `lowrank` path. NOT valid for a "
        "different machine, a different jump path, or L > 128."]


def memory_mb(L, N_c):
    per_clone = (2 * L) ** 2 * 8 + (2 * L) * L * 16
    return 128.0 + 2.0 * N_c * per_clone / 1e6   # x2: systematic resampling copies


def plan(L, T, zeta, lam, target_sem=None, target_dlam=None, budget_core_h=None,
         tau2=None, dD=None, dtau_mult=6.0, R_min=None, bias_tol=None,
         resample_scheme='systematic'):
    out = {"cell": dict(L=L, T=T, zeta=zeta, lam=lam, dtau_mult=dtau_mult),
           "notes": [], "status": "ok"}
    # Calibration gaps are COLLECTED, not short-circuited. The prototype
    # returned on the first gap, so a caller never learned that the bias
    # calibration was missing too - and "add T=L VIF data" is a very different
    # instruction from "add a bias ladder at this cell".
    gaps = []
    vif, vst, vnotes = estimate_vif(L, zeta, lam, T, resample_scheme)
    out["notes"] += vnotes
    if vif is None:
        out["VIF"] = CALIBRATION_REQUIRED
        gaps.append("VIF: " + "; ".join(vnotes[-1:] or ["outside the calibrated box"]))
    else:
        out["VIF"] = round(vif, 1); out["VIF_status"] = vst
    Bf0, B_ci0, bst0, bnotes0 = estimate_bias_coefficient(L, T, zeta, lam, resample_scheme)
    if Bf0 is None:
        out["B_f"] = CALIBRATION_REQUIRED
        out["B_f_status"] = bst0
        out["notes"] += bnotes0
        gaps.append("B_f: no calibrated bias ladder at this exact cell")
    if gaps:
        out["status"] = CALIBRATION_REQUIRED
        out["calibration_gaps"] = gaps
        out["required_input"] = [
            g.split(":")[0] + " calibration at this cell" for g in gaps]
        out["why"] = (
            "finite-N_c bias is invariant under R. Sizing from a variance model "
            "alone produces a tight interval around a displaced number, and the "
            "tightening is indistinguishable from convergence. Every gap above "
            "must be closed before this cell can be sized.")
        return out

    _bk = (L, T, zeta, lam, resample_scheme)
    if tau2 is None and _bk in BIAS_CALIBRATION and "tau2_measured" in BIAS_CALIBRATION[_bk]:
        tau2 = float(BIAS_CALIBRATION[_bk]["tau2_measured"])
        out["notes"].append(
            f"tau^2 taken from the measured ladder at this exact cell: {tau2:.5g} "
            f"(median within-population across-clone variance of CMI)")
    if tau2 is None:
        out["notes"].append(
            "tau^2 (the WITHIN-population across-clone variance of CMI) was not "
            "supplied. It is cell-specific, it is cheap to measure from a single "
            "pilot population, and the planner will NOT invent it.")
        out["status"] = CALIBRATION_REQUIRED
        out["required_input"] = "tau2 (per-clone CMI variance from one pilot population)"
        return out

    A = vif * tau2
    out["A_f"] = A

    # --- constraints ------------------------------------------------------
    N_pre = int(math.ceil(5.0 * vif))       # epsilon = VIF/N_c <= 0.2
    Rmin = R_min if R_min is not None else (
        24 if (0.25 <= zeta < 0.9 and L >= 96) else R_MIN_DEFAULT)
    out["constraints"] = {
        "N_c_min_preasymptotic": N_pre,
        "N_c_min_reason": "epsilon = VIF/N_c <= 0.2; below this the 1/N_c "
                          "expansion is not controlled and a finite-N_c bias "
                          "is predicted rather than excluded",
        "R_min": Rmin,
        "R_min_reason": ("intermediate zeta with L >= 96: Student-t coverage is "
                         "0.914-0.930 at nominal 0.95 and the deficit is a "
                         "SKEWNESS effect, so raising R reduces it only as "
                         "R^-1/2 and does not remove it"
                         if Rmin == 24 else
                         "R >= 12 for a usable SE of the SE; R >= 8 is the hard floor"),
        "R_hard_floor": R_MIN_HARD,
    }

    # --- required total clone budget --------------------------------------
    M = None
    if target_sem is not None:
        M = A / (target_sem ** 2)
        out["target"] = {"kind": "CMI SEM", "value": target_sem}
    elif target_dlam is not None:
        if dD is None:
            out["notes"].append(
                "A crossing target needs |D'| = |d/dlam (CMI_L2 - CMI_L1)| at "
                "the crossing, which is cell-specific and is NOT invented here.")
            out["status"] = CALIBRATION_REQUIRED
            out["required_input"] = "dD (local slope of the CMI curve difference)"
            return out
        # Var(lam_c) = [(1-th)^2 V(D_j) + th^2 V(D_{j+1})] / D'^2, worst case th=1,
        # V(D) = 2A/(R N_c) for two independent size curves of similar A.
        M = 2.0 * A / (dD ** 2 * target_dlam ** 2)
        out["target"] = {"kind": "crossing SE", "value": target_dlam, "dD": dD}
        out["notes"].append(
            "Crossing budget uses the worst-case bracket position theta=1 and "
            "assumes the two size curves are INDEPENDENT - measured, not "
            "assumed: cross-L split-half noise correlation has mean -0.0011 and "
            "max |r| = 0.076 over 132 corpus pairs.")
    if M is None:
        out["status"] = "no target given"
        return out
    out["required_total_clones_M"] = int(math.ceil(M))

    # --- THE BIAS FLOOR ---------------------------------------------------
    # This block is the whole reason this file supersedes the prototype. It runs
    # BEFORE the allocation, because the bias constraint binds N_c alone and no
    # amount of R can relax it.
    Bf, B_ci, bst, bnotes = estimate_bias_coefficient(L, T, zeta, lam, resample_scheme)
    out["notes"] += bnotes
    out["B_f_status"] = bst
    if Bf is None:
        out["status"] = CALIBRATION_REQUIRED
        out["B_f"] = CALIBRATION_REQUIRED
        out["required_input"] = (
            "a measured N_c ladder at this exact cell (>=4 rungs, R>=32) to "
            "calibrate B_f. The planner will NOT size a high-difficulty cell "
            "from a variance model alone, and will NOT infer B_f from VIF.")
        out["why"] = (
            "finite-N_c bias is invariant under R. Sizing from variance alone "
            "produces a tight interval around a displaced number, and the "
            "tightening is indistinguishable from convergence.")
        return out
    out["B_f"] = Bf
    out["B_f_ci"] = list(B_ci)

    if bias_tol is None:
        if target_sem is None:
            out["notes"].append(
                "no --bias-tol given and no CMI SEM target to derive one from")
            out["status"] = CALIBRATION_REQUIRED
            out["required_input"] = "bias_tol (absolute, in CMI units)"
            return out
        bias_tol = 0.5 * target_sem
        out["notes"].append(
            f"bias_tol defaulted to 0.5 x target SEM = {bias_tol:.5g}. At that "
            f"ratio the bias contributes 20% of the MSE. It is a DEFAULT, not a "
            f"derived requirement; pass --bias-tol to set it from the science.")
    out["bias_tolerance"] = bias_tol

    # The floor uses the CONSERVATIVE end of B's OWN confidence interval, so a
    # poorly determined B produces a larger N_c rather than a confident answer.
    B_worst = max(abs(B_ci[0]), abs(B_ci[1]))
    N_bias = int(math.ceil(B_worst / bias_tol))
    out["constraints"]["N_c_min_bias"] = N_bias
    out["constraints"]["N_c_min_bias_reason"] = (
        f"|B_f| <= {B_worst:.3f} at the upper end of its 95% CI, so "
        f"N_c >= {B_worst:.3f}/{bias_tol:.5g} = {N_bias} is required for the "
        f"bias to sit under the tolerance. R CANNOT relax this.")

    # --- allocate: N_c first (bias), then R (variance) --------------------
    N_c = max(N_bias, N_pre, 32)
    R = int(math.ceil(M / N_c))
    if R < Rmin:
        R = Rmin
    # NOTE the difference from the prototype: when R is raised to its floor the
    # prototype RECOMPUTED N_c downward from M/R. That is exactly the move that
    # trades N_c for R and it is removed. N_c only ever goes up.
    binding = ("bias" if N_bias >= max(N_pre, 32) else
               "pre-asymptotic epsilon" if N_pre >= 32 else "floor of 32")
    out["recommended"] = {"N_c": int(N_c), "R": int(R),
                          "total_clones": int(N_c * R),
                          "N_c_binding_constraint": binding}
    sem = math.sqrt(A / (N_c * R))
    bias = Bf / N_c
    out["predicted_sem_CMI"] = sem
    out["predicted_bias_CMI"] = bias
    out["predicted_bias_ci"] = [B_ci[0] / N_c, B_ci[1] / N_c]
    out["predicted_rmse_CMI"] = math.sqrt(sem ** 2 + bias ** 2)
    out["bias_over_sem"] = abs(bias) / sem if sem > 0 else float("inf")
    out["regime"] = ("BIAS_LIMITED" if abs(bias) > sem else
                     "VARIANCE_LIMITED" if sem > 2 * abs(bias) else "BALANCED")
    if abs(bias) > bias_tol * 1.0000001:
        out["status"] = "bias tolerance not met"
        out["verdict"] = (
            f"REFUSING: even at N_c = {N_c} the predicted bias {bias:.5g} "
            f"exceeds the tolerance {bias_tol:.5g}. Raising R would shrink the "
            f"interval and leave the displacement. Raise N_c or relax the "
            f"tolerance.")
    if target_dlam is not None:
        out["predicted_crossing_se"] = math.sqrt(2 * A / (N_c * R)) / dD
        out["notes"].append(
            "A crossing locator differences two CMI curves. A bias COMMON to "
            "both sizes cancels in the difference to first order; a bias that "
            "differs between them does not. Both sizes must be bias-controlled "
            "separately, which is what the N_c floor above enforces.")

    cs, cst, cnotes = cost_core_seconds(L, T, zeta, lam, N_c, R, dtau_mult)
    out["notes"] += cnotes
    if cs is None:
        out["cost_core_hours"] = CALIBRATION_REQUIRED
    else:
        out["cost_core_hours"] = round(cs / 3600.0, 2)
        out["cost_status"] = cst
    out["memory_mb_per_population"] = round(memory_mb(L, N_c), 0)

    if budget_core_h is not None and isinstance(out.get("cost_core_hours"), float):
        if out["cost_core_hours"] > budget_core_h:
            out["verdict"] = (
                f"BUDGET INSUFFICIENT: the target needs "
                f"{out['cost_core_hours']:.1f} core-h but {budget_core_h:.1f} "
                f"were allowed. The planner does NOT shave R below {Rmin} or "
                f"N_c below {N_pre} to fit; either relax the target, or accept "
                f"a larger SEM, and re-plan.")
        else:
            out["verdict"] = "within budget"

    # --- honest uncertainty ON THE PREDICTION -----------------------------
    out["prediction_uncertainty"] = {
        "VIF": "roughly a factor 1.5 either way: the corpus VIF is a median "
               "over a lambda grid and varies across it; the L-interpolation "
               "is on a table whose spacing is 16",
        "cost": "roughly a factor 1.3: the zeta cost factor is calibrated at "
                "six zeta values on one machine and one thermal state",
        "sem": "scales as sqrt(VIF), so roughly a factor 1.25",
        "B_f": (f"reported as an interval, not a number: the 95% CI is "
                f"{out.get('B_f_ci')} and the N_c floor above is computed from "
                f"its CONSERVATIVE end. Across admissible functional forms the "
                f"parent task found bias@128 spanning a factor of five, so B_f "
                f"must never be quoted as a point value."),
        "what_B_f_does_NOT_transfer_across": (
            "lambda, at fixed L and zeta. Two cells with overlapping VIF differ "
            "in B_f by ~24x when only lambda moves."),
        "what_would_tighten_it": "one pilot population at the exact cell, which "
                                 "supplies tau^2 and a direct wall-clock, and "
                                 "R=8 pilots which supply VIF directly",
    }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--T", type=float, default=None, help="defaults to L")
    p.add_argument("--zeta", type=float, required=True)
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--dtau-mult", type=float, default=6.0)
    p.add_argument("--tau2", type=float, default=None,
                   help="within-population across-clone variance of CMI, from a pilot")
    p.add_argument("--target-sem", type=float, default=None)
    p.add_argument("--target-dlam", type=float, default=None)
    p.add_argument("--dD", type=float, default=None,
                   help="|d/dlam (CMI_L2 - CMI_L1)| at the crossing")
    p.add_argument("--budget-core-h", type=float, default=None)
    p.add_argument("--R-min", type=int, default=None)
    p.add_argument("--bias-tol", type=float, default=None,
                   help="absolute tolerance on the finite-N_c bias, in CMI "
                        "units. Defaults to half the target SEM when a CMI SEM "
                        "target is given, and is REQUIRED otherwise.")
    p.add_argument("--resample-scheme", default="systematic",
                   choices=("systematic", "multinomial"))
    a = p.parse_args(argv)
    res = plan(a.L, a.T if a.T is not None else float(a.L), a.zeta, a.lam,
               target_sem=a.target_sem, target_dlam=a.target_dlam,
               budget_core_h=a.budget_core_h, tau2=a.tau2, dD=a.dD,
               dtau_mult=a.dtau_mult, R_min=a.R_min, bias_tol=a.bias_tol,
               resample_scheme=a.resample_scheme)
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
