#!/usr/bin/env python3
"""The ONE place TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION's frozen criteria are
evaluated. Rules: ../analysis_spec.yaml. Nothing here is decided after the data
arrive.

    python3 analysis/lowlambda_analysis.py [--task-root DIR]

It builds the full 17-point CMI(lambda) curves at L = 32, 48, 64 by combining

    the 39 completed predecessor cells, lambda = 0.2332 .. 0.3532, from
    ../frozen_inputs/predecessor_nc1024_populations.csv          (REUSED)

with

    the 12 new cells, lambda = 0.1932 .. 0.2232, from
    ../lowlam{L32,L48,L64}/results/*.json                        (NEW)

and then re-runs the predecessor's curve-quality battery and crossing protocol
on the extended grid, plus the join test this task adds.

Every uncertainty is ACROSS INDEPENDENT POPULATIONS. Within-clone spread appears
only as VIF/N_eff and is never a standard error.

No smoothing. No interpolation replacing a measured point. No value-based
exclusion. No special fit across the join. The audit block below is written into
the results JSON so that an edit which changed any of that would have to change
the audit too.

Runs to completion with zero new results and degrades explicitly.
Contains no scheduler call. Reads only; writes only inside this task directory.
"""
import os, sys, csv, json, glob, math, argparse, textwrap, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TASK = os.path.abspath(os.path.join(HERE, os.pardir))

# ----- frozen constants, mirrored from analysis_spec.yaml -------------------
GRID = [round(0.1932 + 0.010 * i, 4) for i in range(17)]
OLD_GRID = [round(0.2332 + 0.010 * i, 4) for i in range(13)]
assert GRID[4:] == OLD_GRID
NEW_IDX = [0, 1, 2, 3]
NEW_LAMS = [GRID[i] for i in NEW_IDX]
JOIN_IDX = 4                     # first already-measured point, lambda = 0.2332
DLAM = 0.010
ZETA = 0.35
DTAU_MULT = 6.0
NC = 1024
LS = [32, 48, 64]
PAIRS = [(32, 48), (32, 64), (48, 64)]
ARMS = {32: "lowlamL32", 48: "lowlamL48", 64: "lowlamL64"}
B_BOOT = 10000
SEED = 20260903
BLOCK = 24                       # the matched-R block, unchanged
BLOCK_LABELS = "ABCDEFGH"
PRIMARY_BLOCK = 0                # block A, always

# J2 fits the five LOWEST already-measured points only, and never sees a new one
J2_FIT_IDX = [4, 5, 6, 7, 8]     # 0.2332 .. 0.2732

AUDIT = dict(smoothing_applied=False, value_based_exclusions=0,
             lambda_points_removed=0, special_join_fit=False,
             grid_extended_again=False)


# ===========================================================================
# loading
# ===========================================================================
def _cellkey(L, lam):
    return (int(L), round(float(lam), 6))


def load(task):
    """cells[(L, lam)] = dict(pops=..., within=..., ...), seed-ordered."""
    cells, suspect = {}, []
    nonok = 0

    def add(L, lam, rec):
        c = cells.setdefault(_cellkey(L, lam),
                             dict(pops=[], within=[], nonfin=0, clones=0,
                                  fallbacks=0, anc=[], wall=[], seeds=[],
                                  src=set()))
        for f, v in (("pops", rec["mean"]), ("within", rec["within"]),
                     ("anc", rec["anc"]), ("wall", rec["wall"]),
                     ("seeds", rec["seed"])):
            c[f].append(v)
        c["nonfin"] += rec["nonfin"]
        c["clones"] += NC
        c["fallbacks"] += rec["fallbacks"]
        c["src"].add(rec["src"])

    # ---- REUSED: the predecessor's 39 cells, frozen into this package -----
    fp = os.path.join(task, "frozen_inputs",
                      "predecessor_nc1024_populations.csv")
    n_frozen = 0
    if os.path.isfile(fp):
        for r in csv.DictReader(open(fp)):
            if r["status"] != "ok":
                nonok += 1
                continue
            # defensive: this file is frozen and hashed, but a hand-edit that
            # slipped in an off-design row must not be averaged into a curve.
            if (int(r["N_c"]) != NC or float(r["dtau_mult"]) != DTAU_MULT
                    or float(r["zeta"]) != ZETA
                    or r["resample_scheme"] != "systematic"):
                sys.exit("frozen input carries an off-design row: %s" % r["seed"])
            add(r["L"], r["lam"],
                dict(mean=float(r["cmi_weighted_mean"]),
                     within=float(r["cmi_within_var"]),
                     nonfin=int(r["n_nonfinite"]),
                     fallbacks=int(r["brentq_fallbacks"]),
                     anc=int(r["n_distinct_anc_final"]),
                     wall=float(r["wall_s"]), seed=int(r["seed"]),
                     src="frozen:predecessor"))
            n_frozen += 1

    # ---- NEW: this task's four low-lambda points -------------------------
    n_new = 0
    for L, arm in sorted(ARMS.items()):
        for p in sorted(glob.glob(os.path.join(task, arm, "results", "*.json"))):
            d = json.load(open(p))
            if d.get("status") not in (None, "ok"):
                nonok += 1
                continue
            if int(d["N_c"]) != NC or float(d["dtau_mult"]) != DTAU_MULT:
                sys.exit("%s: off-design N_c or dtau_mult" % p)
            add(d["L"], d["lam"],
                dict(mean=float(d["cmi_weighted_mean"]),
                     within=float(d["cmi_within_var"]),
                     nonfin=int(d.get("n_nonfinite", 0)),
                     fallbacks=int(d.get("brentq_fallbacks", 0)),
                     anc=int(d.get("n_distinct_anc_final", 0)),
                     wall=float(d.get("wall_s", float("nan"))),
                     seed=int(d["seed"]), src=arm))
            n_new += 1

    for k, c in cells.items():
        # SEED ORDER, fixed here and nowhere else. Every block cut downstream
        # is a slice of this ordering, so block membership depends only on the
        # seeds -- never on CMI, and never on filesystem read order.
        order = np.argsort(np.asarray(c["seeds"], dtype=np.int64), kind="stable")
        for f in ("pops", "within", "anc", "wall", "seeds"):
            c[f] = np.asarray(c[f], float if f != "seeds" else np.int64)[order]
        c["lam"] = float(k[1])
        c.update(_stats(c["pops"], c["within"]))
        c["n_blocks"] = c["R"] // BLOCK
        c["block"] = None
        if c["nonfin"] > 0.01 * c["clones"]:
            suspect.append(k)
    return cells, dict(n_frozen=n_frozen, n_new=n_new, nonok=nonok,
                       suspect=[list(s) for s in suspect])


def _stats(pops, within):
    """Across-population statistics. The ONLY error bar in this task."""
    R = int(pops.size)
    var = float(pops.var(ddof=1)) if R > 1 else float("nan")
    wm = float(np.mean(within)) if within.size else float("nan")
    return dict(R=R, mean=float(pops.mean()) if R else float("nan"), var=var,
                sem=math.sqrt(var / R) if R > 1 else float("nan"),
                vif=var * NC / wm if wm and wm > 0 else float("nan"),
                n_eff=wm / var if var and var > 0 else float("nan"))


def cell_block(c, k, size=BLOCK):
    """Block k: populations [k*size, (k+1)*size) in SEED order.

    Deterministic and OBSERVABLE-BLIND -- permuting the CMI values within a
    cell cannot change which population lands in which block. Returns None if
    the cell does not hold a full block k.
    """
    lo, hi = k * size, (k + 1) * size
    if hi > int(c["R"]):
        return None
    sub = dict(c)
    for f in ("pops", "within", "anc", "wall", "seeds"):
        sub[f] = c[f][lo:hi]
    sub.update(_stats(sub["pops"], sub["within"]))
    sub["block"] = k
    sub["block_label"] = BLOCK_LABELS[k]
    sub["parent_R"] = int(c["R"])
    # nonfin/clones/fallbacks stay PARENT-level: the exclusion accounting is a
    # property of the cell as run and is not re-scaled to a block.
    return sub


def curve(cells, L, block=PRIMARY_BLOCK, idx=None):
    """The curve at matched R = BLOCK over grid indices `idx`, or None."""
    idx = list(range(len(GRID))) if idx is None else idx
    ks = [_cellkey(L, GRID[i]) for i in idx]
    if any(k not in cells for k in ks):
        return None
    if block is None:
        if any(cells[k]["R"] < 2 for k in ks):
            return None
        return [cells[k] for k in ks]
    out = [cell_block(cells[k], block) for k in ks]
    return None if any(o is None for o in out) else out


def have(cells, L):
    return [i for i in range(len(GRID)) if _cellkey(L, GRID[i]) in cells]


# ===========================================================================
# bootstrap machinery -- resample INDEPENDENT POPULATIONS within each cell
# ===========================================================================
def boot_curves(cs, rng, B=B_BOOT):
    out = np.empty((B, len(cs)))
    for j, c in enumerate(cs):
        p = c["pops"]
        idx = rng.integers(0, p.size, size=(B, p.size))
        out[:, j] = p[idx].mean(axis=1)
    return out


# ===========================================================================
# curve quality -- identical statistics to the predecessor, on 17 points
# ===========================================================================
def curve_quality(cs, lams, rng):
    m = np.array([c["mean"] for c in cs])
    s = np.array([c["sem"] for c in cs])
    d = np.diff(m)
    sd = np.sqrt(s[:-1] ** 2 + s[1:] ** 2)
    r = np.abs(d) / sd
    q = m[2:] - 2 * m[1:-1] + m[:-2]
    sq = np.sqrt(s[:-2] ** 2 + 4 * s[1:-1] ** 2 + s[2:] ** 2)
    z = q / sq
    rough = float(np.mean(z ** 2))

    bc = boot_curves(cs, rng)
    bq = bc[:, 2:] - 2 * bc[:, 1:-1] + bc[:, :-2]
    brough = np.mean((bq / sq) ** 2, axis=1)

    # weighted quadratic YARDSTICK -- a comparator, never a replacement
    x = np.array(lams) - float(np.mean(lams))
    W = 1.0 / s ** 2
    A = np.vstack([np.ones_like(x), x, x ** 2]).T
    coef, _r, _rk, _sv = np.linalg.lstsq(A * np.sqrt(W)[:, None],
                                         m * np.sqrt(W), rcond=None)
    chi2 = float(np.sum(W * (A @ coef - m) ** 2))
    return dict(lams=list(lams), mean=m.tolist(), sem=s.tolist(),
                d=d.tolist(), sem_d=sd.tolist(), r=r.tolist(),
                r_median=float(np.median(r)), n_r_ge_2=int((r >= 2).sum()),
                n_increments=int(r.size),
                q=q.tolist(), sem_q=sq.tolist(), z_q=z.tolist(),
                roughness=rough,
                roughness_ci=[float(np.percentile(brough, 2.5)),
                              float(np.percentile(brough, 97.5))],
                quad_chi2=chi2, quad_chi2_dof=chi2 / (len(lams) - 3),
                quad_coef=coef.tolist())


def population_diagnostics(cs, lams):
    out = []
    for c, lam in zip(cs, lams):
        p = c["pops"]
        R = p.size
        # a FIXED permutation, seeded once, identical at every cell: the
        # split-half partition must not depend on the values it is testing
        perm = np.random.default_rng(SEED).permutation(R)
        a, b = p[perm[:R // 2]], p[perm[R // 2:2 * (R // 2)]]
        if a.size >= 2 and b.size >= 2:
            dsh = float(a.mean() - b.mean())
            ssh = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
            sh_ok = abs(dsh) <= 2.5 * ssh
        else:
            dsh, ssh, sh_ok = float("nan"), float("nan"), None
        sd = p.std(ddof=1)
        zz = np.abs(p - p.mean()) / sd if sd > 0 else np.zeros_like(p)
        zmax = float(zz.max())
        zthr = float(_z_for_R(R))
        i = int(np.argmax(zz))
        loo = np.delete(p, i).mean()
        shift = abs(loo - p.mean()) / c["sem"] if c["sem"] > 0 else float("inf")
        out.append(dict(lam=lam, R=R, new=bool(lam in NEW_LAMS),
                        split_half=dsh, split_half_sem=ssh,
                        split_half_ok=sh_ok, zmax=zmax, z_thr=zthr,
                        outlier_ok=zmax <= zthr,
                        loo_shift_sem=float(shift), loo_ok=shift <= 1.0))
    return out


def _z_for_R(R):
    """Two-sided 1 % expected-maximum threshold for R draws."""
    from math import erf, sqrt
    p = 1 - 0.01 / (2 * R)
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1 + erf(mid / sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ===========================================================================
# THE JOIN TEST -- J1, J2, J3 (analysis_spec.yaml: join_test)
# ===========================================================================
def _wls(x, y, w, deg):
    """Weighted least squares. Returns (coef, cov, chi2, dof)."""
    A = np.vstack([x ** k for k in range(deg + 1)]).T
    sw = np.sqrt(w)
    coef, _r, _rk, _sv = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    chi2 = float(np.sum(w * (A @ coef - y) ** 2))
    cov = np.linalg.inv((A * w[:, None]).T @ A)
    return coef, cov, chi2, len(x) - (deg + 1)


def join_tests(cs, rng):
    """The four new points join the thirteen old ones -- or they do not.

    Nothing here may remove a point, and no fit is applied ACROSS the join to
    smooth it. J2 in particular NEVER sees a new point: it is fitted on old
    points only and asked to predict.
    """
    m = np.array([c["mean"] for c in cs])
    s = np.array([c["sem"] for c in cs])
    lam = np.array(GRID)
    out = {}

    # ---- J1: is the join exceptional relative to the curve's OWN noise? ----
    q = m[2:] - 2 * m[1:-1] + m[:-2]
    sq = np.sqrt(s[:-2] ** 2 + 4 * s[1:-1] ** 2 + s[2:] ** 2)
    z = q / sq                                # z[j] is the triple centred at j+1
    # STRADDLING triples contain both a new and an old lambda: centres at grid
    # index 3 (0.2232) and 4 (0.2332), i.e. z indices 2 and 3.
    straddle = [2, 3]
    T_join = float(np.max(np.abs(z[straddle])))
    bc = boot_curves(cs, rng)
    bq = bc[:, 2:] - 2 * bc[:, 1:-1] + bc[:, :-2]
    Mstar = np.max(np.abs(bq / sq), axis=1)   # max over ALL interior triples
    p_join = float(np.mean(Mstar >= T_join))
    out["J1"] = dict(
        straddling_centres=[GRID[j + 1] for j in straddle],
        z_straddling=[float(z[j]) for j in straddle],
        T_join=T_join,
        null_max_p95=float(np.percentile(Mstar, 95)),
        p_join=p_join,
        verdict="PASS" if p_join >= 0.05 else "FAIL",
        note="T_join is compared against the bootstrap distribution of the "
             "maximum |z_q| over ALL interior triples of the same curve, so "
             "the join is judged against this curve's own worst roughness "
             "and not against zero.")

    # ---- J2: out-of-sample extrapolation from OLD points only --------------
    fi = J2_FIT_IDX
    x0 = float(np.mean(lam[fi]))
    xf = lam[fi] - x0
    wf = 1.0 / s[fi] ** 2
    coef, cov, chi2, dof = _wls(xf, m[fi], wf, 2)
    scale = max(1.0, chi2 / dof) if dof > 0 else 1.0   # conservative inflation
    rows = []
    for i in NEW_IDX:
        v = np.array([1.0, lam[i] - x0, (lam[i] - x0) ** 2])
        pred = float(v @ coef)
        var_pred = float(v @ cov @ v) * scale
        sig = math.sqrt(var_pred + s[i] ** 2)
        rows.append(dict(lam=GRID[i], measured=float(m[i]), sem=float(s[i]),
                         predicted=pred, sem_pred=math.sqrt(var_pred),
                         z=float((m[i] - pred) / sig)))
    zmax = max(abs(r["z"]) for r in rows)
    out["J2"] = dict(fit_lambdas=[GRID[i] for i in fi], fit_degree=2,
                     fit_chi2=chi2, fit_dof=dof, variance_scale=scale,
                     points=rows, max_abs_z=zmax,
                     verdict="PASS" if zmax <= 3.0 else "FAIL",
                     note="The fit never sees a new point. This is a "
                          "diagnostic of continuity; a FAIL licenses saying "
                          "the join is not smooth, never dropping a point.")

    # ---- J3: no step in the increment straddling the join ------------------
    d = np.diff(m)
    sd = np.sqrt(s[:-1] ** 2 + s[1:] ** 2)
    tgt = 3                                   # d[3] = 0.2232 -> 0.2332
    nb = [0, 1, 2, 4, 5, 6]                   # three increments each side
    xi = np.array(nb, float) - tgt
    wi = 1.0 / sd[nb] ** 2
    coef3, cov3, chi23, dof3 = _wls(xi, d[nb], wi, 1)
    sc3 = max(1.0, chi23 / dof3) if dof3 > 0 else 1.0
    v = np.array([1.0, 0.0])
    pred3 = float(v @ coef3)
    sig3 = math.sqrt(float(v @ cov3 @ v) * sc3 + sd[tgt] ** 2)
    res3 = float(d[tgt] - pred3)
    out["J3"] = dict(join_increment_lambda=[GRID[tgt], GRID[tgt + 1]],
                     d_measured=float(d[tgt]), sem_d=float(sd[tgt]),
                     d_predicted=pred3, sem_total=sig3,
                     residual=res3, z=res3 / sig3,
                     neighbour_increments=[GRID[i] for i in nb],
                     verdict="PASS" if abs(res3 / sig3) <= 3.0 else "FAIL")

    passed = all(out[k]["verdict"] == "PASS" for k in ("J1", "J2", "J3"))
    out["overall"] = "CONTINUOUS" if passed else "NOT CONTINUOUS ON ALL THREE"
    return out


# ===========================================================================
# the crossing protocol
# ===========================================================================
def _crossings_of(D, lams):
    """All raw sign changes with their interpolated lambdas."""
    out = []
    for i in range(len(D) - 1):
        if D[i] == 0.0:
            out.append((i, lams[i]))
        elif D[i] * D[i + 1] < 0:
            t = D[i] / (D[i] - D[i + 1])
            out.append((i, lams[i] + t * (lams[i + 1] - lams[i])))
    return out


def _core_crossing(cA, cB, lams, rng):
    """The shared machinery: raw, resolved, bootstrap. cA is the smaller L."""
    mA = np.array([c["mean"] for c in cA])
    sA = np.array([c["sem"] for c in cA])
    mB = np.array([c["mean"] for c in cB])
    sB = np.array([c["sem"] for c in cB])
    D = mB - mA
    sD = np.sqrt(sA ** 2 + sB ** 2)
    raw = _crossings_of(D.tolist(), lams)
    resolved = [(i, x) for (i, x) in raw
                if abs(D[i]) >= 2 * sD[i] and abs(D[i + 1]) >= 2 * sD[i + 1]]
    bD = boot_curves(cB, rng) - boot_curves(cA, rng)
    counts = collections.Counter()
    firsts = []
    for row in bD:
        cx = _crossings_of(row.tolist(), lams)
        counts[len(cx)] += 1
        if len(cx) == 1:
            firsts.append(cx[0][1])
    if firsts:
        ci = [float(np.percentile(firsts, 2.5)),
              float(np.percentile(firsts, 97.5))]
    else:
        ci = [float("nan"), float("nan")]
    return D, sD, raw, resolved, counts, firsts, ci


def crossing_analysis(cells, L1, L2, rng, label):
    """Full protocol on the 17-point grid, plus the pre-registered interiority
    test, which is the whole reason this task exists."""
    cA = curve(cells, L1)
    cB = curve(cells, L2)
    D, sD, raw, resolved, counts, firsts, ci = _core_crossing(cA, cB, GRID, rng)
    B = sum(counts.values())
    frac_one = counts[1] / B
    width = ci[1] - ci[0] if firsts else float("nan")
    boot_stable = bool(firsts) and width <= 2 * DLAM

    # jackknife over deleted lambda points -- a STABILITY DIAGNOSTIC ONLY
    jack = []
    for k in range(len(GRID)):
        idx = [j for j in range(len(GRID)) if j != k]
        cx = _crossings_of(D[idx].tolist(), [GRID[j] for j in idx])
        jack.append(dict(deleted=GRID[k], n=len(cx),
                         x=float(cx[0][1]) if len(cx) == 1 else None))
    jack_ok = (bool(firsts)
               and all(j["x"] is not None and ci[0] <= j["x"] <= ci[1]
                       for j in jack))

    # ---- I2: DELETE THE FIRST LAMBDA POINT. Pre-registered before any datum.
    # A crossing that needs the new lower endpoint has moved the boundary, not
    # become interior.
    sub = list(range(1, len(GRID)))
    cA1 = curve(cells, L1, idx=sub)
    cB1 = curve(cells, L2, idx=sub)
    lams1 = [GRID[i] for i in sub]
    D1, sD1, raw1, res1, cnt1, f1, ci1 = _core_crossing(
        cA1, cB1, lams1, np.random.default_rng(SEED + 1))
    drop_first = dict(n_raw=len(raw1), crossings=[float(x) for _, x in raw1],
                      boot_ci=ci1,
                      frac_exactly_one=cnt1[1] / max(sum(cnt1.values()), 1))
    if firsts:
        I2 = bool(raw1) and any(ci[0] <= x <= ci[1] for _, x in raw1)
    else:
        I2 = bool(raw1)

    # ---- I1 and I3
    I1 = bool(raw) and all(1 <= i <= len(GRID) - 3 for i, _ in raw)
    I3 = bool(firsts) and (ci[0] >= GRID[0] + DLAM / 2
                           and ci[1] <= GRID[-1] - DLAM / 2)
    endpoint = any(i == 0 or i == len(GRID) - 2
                   or x <= GRID[0] + DLAM / 2 or x >= GRID[-1] - DLAM / 2
                   for i, x in raw)

    # ---- outcome class, exactly as frozen in analysis_spec.yaml.
    # STILL_BOUNDARY is reachable ONLY when a raw crossing exists: it means the
    # locator moved with the boundary. When there is no raw crossing at all the
    # locator found nothing, and the only question left is whether the bootstrap
    # mass piles against the lower end (BELOW_GRID) or is merely scattered
    # (NONE). Routing "no crossing" to STILL_BOUNDARY would report a boundary
    # artefact where there is no locator to be an artefact of.
    if raw:
        cls = "INTERIOR" if (I1 and I2 and I3) else "STILL_BOUNDARY"
    elif firsts and ci[1] <= GRID[0] + 2 * DLAM:
        cls = "BELOW_GRID"
    else:
        cls = "NONE"

    # ---- split-half crossing stability: two disjoint R=12 halves, seed order
    halves = []
    for h in (0, 1):
        hA = [cell_block(c, h, size=BLOCK // 2) for c in cA]
        hB = [cell_block(c, h, size=BLOCK // 2) for c in cB]
        if any(x is None for x in hA + hB):
            halves.append(None)
            continue
        mh = np.array([c["mean"] for c in hB]) - np.array([c["mean"] for c in hA])
        cx = _crossings_of(mh.tolist(), GRID)
        halves.append(dict(half=BLOCK_LABELS[h], R=BLOCK // 2, n_raw=len(cx),
                           crossings=[float(x) for _, x in cx]))
    sh_ok = (all(h is not None for h in halves)
             and halves[0]["n_raw"] == 1 and halves[1]["n_raw"] == 1
             and abs(halves[0]["crossings"][0]
                     - halves[1]["crossings"][0]) <= 2 * DLAM)

    return dict(pair=label, D=D.tolist(), sem_D=sD.tolist(), lams=GRID,
                n_raw=len(raw), n_resolved=len(resolved),
                crossings=[float(x) for _, x in raw],
                crossing_intervals=[int(i) for i, _ in raw],
                resolved_crossings=[float(x) for _, x in resolved],
                boot_ci=ci, boot_width=width,
                boot_count_hist=dict((str(k), v) for k, v in sorted(counts.items())),
                frac_exactly_one=frac_one,
                unique=(len(raw) == 1 and frac_one >= 0.95),
                endpoint_induced=endpoint,
                stable_bootstrap=boot_stable,
                jackknife=jack, stable_point_deletion=jack_ok,
                drop_first_lambda=drop_first,
                I1_not_in_end_interval=I1, I2_survives_dropping_first=I2,
                I3_ci_clear_of_endpoints=I3,
                outcome_class=cls,
                split_half=halves, split_half_stable=sh_ok)


# ===========================================================================
# figures
# ===========================================================================
def figures(cells, res, figdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  figures SKIPPED: matplotlib unavailable (%s)" % e)
        return []
    os.makedirs(figdir, exist_ok=True)
    made = []
    COL = {32: "#1b6ca8", 48: "#c85200", 64: "#118050"}
    DCOL = {(32, 48): "#7b3fbf", (32, 64): "#b5006e", (48, 64): "#0f7d7d"}
    NEWLO, NEWHI = GRID[0] - 0.004, GRID[3] + 0.005

    def newband(ax, label=True):
        """The provenance marking, deliberately SUBTLE: the new points are the
        same measurement as the old ones and the figure must not imply two
        datasets. A faint band, no separate marker, no separate legend entry."""
        ax.axvspan(NEWLO, NEWHI, color="#000000", alpha=0.035, zorder=0, lw=0)
        if label:
            ax.text(0.5 * (NEWLO + NEWHI), 0.985, "new",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7, color="#999")

    # ---- FIGURE A : the full 17-point curves
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    newband(ax)
    any_ = False
    for L in LS:
        cs = curve(cells, L)
        if cs is None:
            idx = have(cells, L)
            if idx:
                ax.plot([], [], color=COL[L],
                        label="L = %d: INCOMPLETE (%d/17)" % (L, len(idx)))
            continue
        any_ = True
        ax.errorbar(GRID, [c["mean"] for c in cs], yerr=[c["sem"] for c in cs],
                    marker="o", ms=4, lw=1.2, capsize=2, color=COL[L],
                    label="L = %d (matched R = %d)" % (L, BLOCK))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("CMI")
    ax.set_title(r"A  —  CMI$(\lambda)$, $N_c=1024$, $\zeta=0.35$, $T=L$, "
                 r"17 measured points   (no smoothing)", fontsize=10)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    if not any_:
        ax.text(.5, .5, "no complete 17-point curve yet", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="#a00")
    fig.text(0.01, 0.01, "shaded: measured by this task; unshaded: measured by "
                         "TASK-2026-09-02-MOCK-PRODUCTION. Same sampler "
                         "(sha256 0a33c403...), same design - one dataset.",
             fontsize=6, color="#777")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    p = os.path.join(figdir, "figureA_cmi_17point.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    made.append(p)

    # ---- FIGURE B : cross-L differences over the full grid
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    newband(ax)
    got = False
    for (L1, L2) in PAIRS:
        x = res["crossings"].get("L%d-L%d" % (L1, L2))
        if not x:
            continue
        got = True
        ax.errorbar(GRID, x["D"], yerr=x["sem_D"], marker="o", ms=4, lw=1.2,
                    capsize=2, color=DCOL[(L1, L2)],
                    label=r"$I_{%d}-I_{%d}$  (%s)" % (L2, L1, x["outcome_class"]))
        for c in x["crossings"]:
            ax.axvline(c, color=DCOL[(L1, L2)], lw=.8, ls=":", alpha=.8)
    ax.axhline(0, color="k", lw=.9)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$I_{L_2}-I_{L_1}$")
    ax.set_title("B  —  cross-$L$ differences on the full 17-point grid "
                 "(dotted: raw crossings)", fontsize=10)
    ax.grid(alpha=.25)
    if got:
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "no complete pair yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#a00")
    fig.tight_layout()
    p = os.path.join(figdir, "figureB_crossL_differences.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    made.append(p)

    # ---- FIGURE C : the low-lambda zoom -- ARE the crossings bracketed?
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    newband(ax, label=False)
    lo, hi = GRID[0] - 0.004, GRID[8] + 0.004
    got = False
    for (L1, L2) in PAIRS:
        x = res["crossings"].get("L%d-L%d" % (L1, L2))
        if not x:
            continue
        got = True
        c = DCOL[(L1, L2)]
        ax.errorbar(GRID, x["D"], yerr=x["sem_D"], marker="o", ms=5, lw=1.3,
                    capsize=2.5, color=c,
                    label=r"$I_{%d}-I_{%d}$" % (L2, L1))
        cint = x["boot_ci"]
        if all(np.isfinite(cint)):
            ax.axvspan(cint[0], cint[1], color=c, alpha=.13, lw=0)
        for v in x["crossings"]:
            ax.axvline(v, color=c, lw=1.0, ls=":")
    ax.axhline(0, color="k", lw=.9)
    ax.axvline(GRID[0], color="#a00", lw=1.0, ls="--")
    ax.text(GRID[0], 0.02, " lower scan boundary", color="#a00", fontsize=7,
            rotation=90, va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$I_{L_2}-I_{L_1}$")
    ax.set_title("C  —  low-$\\lambda$ zoom: is the crossing bracketed?   "
                 "(shaded band: bootstrap 95 % interval)", fontsize=10)
    ax.grid(alpha=.25)
    if got:
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "no complete pair yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#a00")
    fig.tight_layout()
    p = os.path.join(figdir, "figureC_lowlambda_zoom.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    made.append(p)
    return made


# ===========================================================================
# main
# ===========================================================================
def hdr(t):
    print("\n" + "=" * 78)
    print("  " + t)
    print("=" * 78)


def _wrap(t, w):
    return textwrap.wrap(" ".join(str(t).split()), width=w) or [""]


def quality_predecessor_window(cs):
    """The 13 old points alone, recomputed from the frozen snapshot.

    A consistency check on the reuse, not a new result: it must reproduce the
    predecessor's published roughness / median r / chi2 per dof.
    """
    sub = cs[4:]
    m = np.array([c["mean"] for c in sub])
    s = np.array([c["sem"] for c in sub])
    d = np.diff(m)
    sd = np.sqrt(s[:-1] ** 2 + s[1:] ** 2)
    q = m[2:] - 2 * m[1:-1] + m[:-2]
    sq = np.sqrt(s[:-2] ** 2 + 4 * s[1:-1] ** 2 + s[2:] ** 2)
    x = np.array(OLD_GRID) - float(np.mean(OLD_GRID))
    W = 1.0 / s ** 2
    A = np.vstack([np.ones_like(x), x, x ** 2]).T
    coef, _r, _rk, _sv = np.linalg.lstsq(A * np.sqrt(W)[:, None],
                                         m * np.sqrt(W), rcond=None)
    chi2 = float(np.sum(W * (A @ coef - m) ** 2))
    return dict(roughness=float(np.mean((q / sq) ** 2)),
                r_median=float(np.median(np.abs(d) / sd)),
                quad_chi2_dof=chi2 / (len(OLD_GRID) - 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-root", default=DEFAULT_TASK,
                    help="package root (used by tools/smoke_test.py)")
    args = ap.parse_args()
    task = os.path.abspath(args.task_root)
    figdir = os.path.join(task, "analysis", "figures")

    rng = np.random.default_rng(SEED)
    cells, meta = load(task)
    res = dict(audit=AUDIT, loaded=meta, grid=GRID, new_lambdas=NEW_LAMS)

    print("=" * 78)
    print("  TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION - low-lambda extension")
    print("  parent: TASK-2026-09-02-MOCK-PRODUCTION (complete, READ ONLY)")
    print("  frozen rules: analysis_spec.yaml")
    print("=" * 78)
    print("  populations loaded: %d (%d reused from the predecessor, %d newly "
          "measured)" % (meta["n_new"] + meta["n_frozen"], meta["n_frozen"],
                         meta["n_new"]))
    print("  cells: %d of 51 (17 lambdas x 3 L)   runs with status != ok: %d"
          % (len(cells), meta["nonok"]))
    if meta["suspect"]:
        print("  ** SUSPECT cells (>1 %% non-finite clones): %s" % meta["suspect"])
    print("  every error bar below is ACROSS INDEPENDENT POPULATIONS; "
          "within-clone")
    print("  spread appears only as VIF/N_eff and is never a standard error.")
    print("  PRIMARY: matched R = %d per cell, block A in SEED order." % BLOCK)
    print("  The four new lambdas and the thirteen reused ones are ONE dataset:")
    print("  same sampler bytes, same design, same R. Nothing below treats them")
    print("  differently, and no fit is applied across the join.")

    # ---------------- A0. inventory --------------------------------------
    hdr("A0.  GRID INVENTORY")
    res["inventory"] = {}
    print("  %4s%8s%6s%8s%14s%18s"
          % ("L", "cells", "new", "reused", "R per cell", "full 17-pt curve"))
    for L in LS:
        idx = have(cells, L)
        nnew = sum(1 for i in idx if i in NEW_IDX)
        Rs = sorted(set(int(cells[_cellkey(L, GRID[i])]["R"]) for i in idx))
        full = curve(cells, L) is not None
        res["inventory"]["L=%d" % L] = dict(
            cells=len(idx), new=nnew, reused=len(idx) - nnew, R=Rs,
            complete=full,
            missing=[GRID[i] for i in range(len(GRID)) if i not in idx])
        print("  %4d%8d%6d%8d%14s%18s"
              % (L, len(idx), nnew, len(idx) - nnew, str(Rs),
                 "YES" if full else "no"))
    missing = sorted(set(GRID[i] for L in LS for i in range(len(GRID))
                         if i not in have(cells, L)))
    if missing:
        print("\n  MISSING lambdas at one or more L: %s"
              % ["%g" % m for m in missing])
        print("  The sections below that need a complete curve will say so")
        print("  rather than silently analysing a shorter grid.")

    # ---------------- A. the curves --------------------------------------
    hdr("A.  THE 17-POINT CMI(lambda) CURVES - PRIMARY, MATCHED R = %d" % BLOCK)
    res["curves"] = {}
    quality = {}
    for L in LS:
        cs = curve(cells, L)
        tag = "L=%d" % L
        if cs is None:
            print("\n  [%s]  INCOMPLETE - %d/17 grid points with a full R=%d "
                  "block" % (tag, len(have(cells, L)), BLOCK))
            continue
        qq = curve_quality(cs, GRID, rng)
        quality[L] = qq
        res["curves"][tag] = qq
        parent_R = [int(cells[_cellkey(L, l)]["R"]) for l in GRID]
        print("\n  [%s]   primary R = %d at every lambda (drawn from cells of "
              "R = %s)" % (tag, BLOCK, sorted(set(parent_R))))
        print("     lambda  src     R  fullR   mean CMI       SEM       VIF   "
              "|  d      SEM(d)      r")
        for i, (lam, c) in enumerate(zip(GRID, cs)):
            if i < len(GRID) - 1:
                tail = "| %+8.5f %8.5f %7.2f" % (qq["d"][i], qq["sem_d"][i],
                                                 qq["r"][i])
            else:
                tail = "|"
            src = "NEW " if i in NEW_IDX else "prev"
            print("     %6.4f  %s  %4d %6d %10.5f %9.5f %9.2f   %s"
                  % (lam, src, c["R"], parent_R[i], c["mean"], c["sem"],
                     c["vif"], tail))
        print("     median r = %.2f   increments with r>=2: %d/%d"
              % (qq["r_median"], qq["n_r_ge_2"], qq["n_increments"]))
        print("     roughness = mean (q/SEM q)^2 = %.3f  bootstrap 95%% CI "
              "[%.3f, %.3f]" % (qq["roughness"], qq["roughness_ci"][0],
                                qq["roughness_ci"][1]))
        print("     chi2/dof vs weighted quadratic YARDSTICK = %.3f"
              % qq["quad_chi2_dof"])
        print("       (a comparator for how much of the point-to-point")
        print("        structure the error bars already explain; the")
        print("        quadratic is never plotted in place of the points)")
        pq = quality_predecessor_window(cs)
        res["curves"][tag]["old_window_recheck"] = pq
        print("     the 13 OLD points alone, recomputed here: roughness %.3f, "
              "median r %.2f, chi2/dof %.3f"
              % (pq["roughness"], pq["r_median"], pq["quad_chi2_dof"]))
        print("       (compare TASK-2026-09-02-MOCK-PRODUCTION/"
              "MOCK_PRODUCTION_RESULTS.json -> curves)")

    # ---------------- B. population diagnostics --------------------------
    hdr("B.  PER-CELL POPULATION DIAGNOSTICS - PRIMARY, R = %d" % BLOCK)
    res["population_diagnostics"] = {}
    for L in LS:
        cs = curve(cells, L)
        if cs is None:
            print("  L=%d: incomplete" % L)
            continue
        pd = population_diagnostics(cs, GRID)
        res["population_diagnostics"]["L=%d" % L] = pd
        print("\n  [L=%d]   split-half |m_A-m_B| vs 2.5*s_AB ; z_max vs z_R ; "
              "leave-one-out shift in SEM" % L)
        for e in pd:
            flag = "".join(["" if e["split_half_ok"] else " SPLIT-HALF",
                            "" if e["outlier_ok"] else " OUTLIER",
                            "" if e["loo_ok"] else " LEAVE-ONE-OUT"])
            print("     %6.4f %s R=%3d  dsh=%+8.5f+-%.5f  zmax=%.2f/%.2f  "
                  "loo=%.2f  %s"
                  % (e["lam"], "NEW " if e["new"] else "prev", e["R"],
                     e["split_half"], e["split_half_sem"], e["zmax"],
                     e["z_thr"], e["loo_shift_sem"],
                     "ok" if not flag else "** FAIL:" + flag))

    # ---------------- C. the join test -----------------------------------
    hdr("C.  THE JOIN AT lambda = 0.2332  (J1, J2, J3)")
    print("  Do the four new points join the thirteen old ones continuously?")
    print("  No fit is applied ACROSS the join, and no point may be dropped")
    print("  because it fails a test here.")
    res["join"] = {}
    for L in LS:
        cs = curve(cells, L)
        if cs is None:
            print("\n  [L=%d] incomplete" % L)
            continue
        j = join_tests(cs, np.random.default_rng(SEED + L))
        res["join"]["L=%d" % L] = j
        print("\n  [L=%d]" % L)
        a = j["J1"]
        print("     J1 local roughness at the join")
        print("        straddling triples centred at %s, z_q = %s"
              % (["%g" % v for v in a["straddling_centres"]],
                 [round(v, 2) for v in a["z_straddling"]]))
        print("        T_join = %.2f  vs bootstrap max-|z| 95th percentile %.2f"
              % (a["T_join"], a["null_max_p95"]))
        print("        p_join = %.4f   -> %s" % (a["p_join"], a["verdict"]))
        b = j["J2"]
        print("     J2 out-of-sample extrapolation from the five lowest OLD "
              "points %s" % ["%g" % v for v in b["fit_lambdas"]])
        print("        fit chi2/dof = %.2f   variance inflation x%.2f"
              % (b["fit_chi2"] / max(b["fit_dof"], 1), b["variance_scale"]))
        for r in b["points"]:
            print("        lambda=%6.4f  measured %.5f+-%.5f   predicted "
                  "%.5f+-%.5f   z = %+.2f"
                  % (r["lam"], r["measured"], r["sem"], r["predicted"],
                     r["sem_pred"], r["z"]))
        print("        max |z| = %.2f (threshold 3)  -> %s"
              % (b["max_abs_z"], b["verdict"]))
        c3 = j["J3"]
        print("     J3 step in the increment straddling the join %g -> %g"
              % (c3["join_increment_lambda"][0], c3["join_increment_lambda"][1]))
        print("        d measured %+.5f+-%.5f   local trend %+.5f   z = %+.2f"
              "  -> %s" % (c3["d_measured"], c3["sem_d"], c3["d_predicted"],
                           c3["z"], c3["verdict"]))
        print("     -> %s" % j["overall"])

    # ---------------- D. crossings ---------------------------------------
    hdr("D.  CROSSING ANALYSIS ON THE 17-POINT GRID")
    print("  LOCATOR QUALITY ONLY. CMI is the locator. Nothing here is")
    print("  lambda_c(zeta) and nothing here may be promoted into a physical")
    print("  phase boundary, however stable.")
    print("  The falsification question: does extending the scan convert the")
    print("  predecessor's lower-boundary / endpoint-sensitive locator into a")
    print("  genuinely interior, bracketed, reproducible crossing?")
    res["crossings"] = {}
    for (L1, L2) in PAIRS:
        lab = "L%d-L%d" % (L1, L2)
        if curve(cells, L1) is None or curve(cells, L2) is None:
            print("\n  [%s] incomplete" % lab)
            continue
        x = crossing_analysis(cells, L1, L2, rng, lab)
        res["crossings"][lab] = x
        print("\n  [%s]  D = I_%d - I_%d" % (lab, L2, L1))
        print("     raw sign changes            %d" % x["n_raw"])
        print("     resolved (both |D|>=2 SEM)  %d" % x["n_resolved"])
        print("     crossing lambda(s)          %s"
              % [round(v, 5) for v in x["crossings"]])
        print("     bootstrap 95%% CI            [%.5f, %.5f]  width %.5f"
              % (x["boot_ci"][0], x["boot_ci"][1], x["boot_width"]))
        print("     bootstrap count histogram   %s   (never discarded)"
              % x["boot_count_hist"])
        print("     fraction exactly one        %.3f" % x["frac_exactly_one"])
        print("     unique                      %s" % x["unique"])
        print("     endpoint-induced            %s" % x["endpoint_induced"])
        print("     stable to bootstrap         %s  (needs width <= %g)"
              % (x["stable_bootstrap"], 2 * DLAM))
        print("     stable to deleting one lam  %s" % x["stable_point_deletion"])
        jn = [j for j in x["jackknife"] if j["n"] != 1]
        if jn:
            print("       deletions not leaving exactly one crossing: %s"
                  % [("%g" % j["deleted"], j["n"]) for j in jn])
        d1 = x["drop_first_lambda"]
        print("     I1  crossing not in an end interval        %s"
              % x["I1_not_in_end_interval"])
        print("     I2  survives dropping lambda=%g         %s   "
              "(16-point grid: %d raw at %s)"
              % (GRID[0], x["I2_survives_dropping_first"], d1["n_raw"],
                 [round(v, 5) for v in d1["crossings"]]))
        print("     I3  bootstrap CI clear of both ends        %s"
              % x["I3_ci_clear_of_endpoints"])
        print("     split-half (two disjoint R=12 halves)      %s"
              % x["split_half_stable"])
        for h in x["split_half"]:
            if h:
                print("        half %s: %d raw at %s"
                      % (h["half"], h["n_raw"],
                         [round(v, 5) for v in h["crossings"]]))
        print("     ==> OUTCOME CLASS: %s" % x["outcome_class"])

    # ---------------- E. pre-registered criteria -------------------------
    hdr("E.  PRE-REGISTERED CRITERIA  X1-X7  (SUCCESS_CRITERIA.md)")
    res["criteria"] = {}

    def emit(k, v, why):
        res["criteria"][k] = dict(verdict=v, reason=why)
        print("  %s  -> %s" % (k, v))
        for ln in _wrap(why, 70):
            print("        " + ln)

    complete = [L for L in LS if curve(cells, L) is not None]

    # X1 -- the twelve new cells are individually sound
    pdg = res["population_diagnostics"]
    if len(complete) < 3:
        emit("X1", "NOT EVALUATED", "not all three curves are complete")
    else:
        f_sh = sum(1 for v in pdg.values() for e in v
                   if e["new"] and not e["split_half_ok"])
        f_lo = sum(1 for v in pdg.values() for e in v
                   if e["new"] and not e["loo_ok"])
        f_ou = sum(1 for v in pdg.values() for e in v
                   if e["new"] and not e["outlier_ok"])
        if f_sh == 0 and f_lo == 0 and f_ou == 0:
            emit("X1", "SUPPORTED", "split-half, outlier and leave-one-out all "
                                    "pass at every one of the 12 new cells")
        elif f_lo > 0 or f_sh >= 2:
            emit("X1", "KILLED", "new-cell failures: split-half %d, "
                                 "leave-one-out %d, outlier %d"
                                 % (f_sh, f_lo, f_ou))
        else:
            emit("X1", "INCONCLUSIVE", "isolated new-cell failures: split-half "
                                       "%d, outlier %d" % (f_sh, f_ou))
    # X2 -- the extended curve is still statistically smooth
    if len(complete) < 3:
        emit("X2", "NOT EVALUATED", "not all three curves are complete")
    else:
        med = dict((L, quality[L]["r_median"]) for L in LS)
        cnt = dict((L, quality[L]["n_r_ge_2"]) for L in LS)
        ch = dict((L, round(quality[L]["quad_chi2_dof"], 3)) for L in LS)
        if all(2 <= med[L] <= 20 for L in LS) and all(cnt[L] >= 12 for L in LS):
            emit("X2", "SUPPORTED", "median standardized increment r %s, "
                                    "increments with r>=2 %s of 16, weighted "
                                    "quadratic chi2/dof %s" % (med, cnt, ch))
        elif any(med[L] < 2 for L in LS):
            emit("X2", "KILLED", "median r %s: the curve is not resolved point "
                                 "to point" % med)
        else:
            emit("X2", "INCONCLUSIVE", "median r %s, r>=2 counts %s"
                                       % (med, cnt))
    # X3 -- the join
    if not res["join"]:
        emit("X3", "NOT EVALUATED", "no complete curve to test the join on")
    else:
        v = dict((L, res["join"]["L=%d" % L]["overall"]) for L in complete)
        nfail = sum(1 for x in v.values() if x != "CONTINUOUS")
        if nfail == 0:
            emit("X3", "SUPPORTED", "J1, J2 and J3 all pass at every L: %s" % v)
        elif nfail >= 2:
            emit("X3", "KILLED", "the join is not continuous at %d of %d "
                                 "rungs: %s" % (nfail, len(v), v))
        else:
            emit("X3", "INCONCLUSIVE", "one rung fails a join test: %s" % v)
    # X4 -- THE POINT OF THE TASK: interiority
    xs = res["crossings"]
    if len(xs) < 3:
        emit("X4", "NOT EVALUATED", "the crossing set is incomplete")
    else:
        cls = dict((k, v["outcome_class"]) for k, v in xs.items())
        n_int = sum(1 for v in cls.values() if v == "INTERIOR")
        if n_int == 3:
            emit("X4", "SUPPORTED", "all three pairs give an INTERIOR, "
                                    "bracketed crossing: %s" % cls)
        elif n_int == 0:
            emit("X4", "KILLED", "extending the scan did NOT produce an "
                                 "interior crossing for any pair: %s. The "
                                 "locator remains boundary-driven or has moved "
                                 "below the new lower endpoint. This is a "
                                 "reportable negative result and does NOT "
                                 "license extending the grid again." % cls)
        else:
            emit("X4", "INCONCLUSIVE", "%d of 3 pairs interior: %s"
                                       % (n_int, cls))
    # X5 -- reproducibility of any crossing that exists
    if len(xs) < 3:
        emit("X5", "NOT EVALUATED", "the crossing set is incomplete")
    else:
        ok = dict((k, (v["stable_bootstrap"], v["stable_point_deletion"],
                       v["split_half_stable"])) for k, v in xs.items())
        n_all = sum(1 for v in ok.values() if all(v))
        withx = [k for k, v in xs.items() if v["n_raw"] >= 1]
        if not withx:
            emit("X5", "NOT EVALUATED",
                 "no pair has a raw crossing to reproduce")
        elif n_all == len(xs):
            emit("X5", "SUPPORTED", "bootstrap, point-deletion and split-half "
                                    "stability all hold: %s" % ok)
        else:
            emit("X5", "INCONCLUSIVE", "(bootstrap, point-deletion, "
                                       "split-half) per pair: %s" % ok)
    # X6 -- the standing analysis prohibitions
    ok6 = all(AUDIT[k] in (False, 0) for k in AUDIT)
    emit("X6", "SUPPORTED" if ok6 else "KILLED",
         "audit block %s: no smoothing, no interpolation replacing a measured "
         "point, no value-based exclusion, no lambda point removed, no special "
         "fit across the join, and the grid was not extended again." % AUDIT)
    # X7 -- the campaign was as cheap as it claimed
    walls = {}
    for L in LS:
        cs = curve(cells, L, block=None, idx=NEW_IDX)
        if cs is None:
            continue
        w = np.concatenate([c["wall"] for c in cs])
        walls[L] = dict(median=float(np.median(w)), max=float(np.max(w)),
                        core_h=float(np.sum(w) / 3600.0))
    res["measured_cost"] = walls
    if len(walls) < 3:
        emit("X7", "NOT EVALUATED", "the new arms have not all returned")
    else:
        tot = sum(v["core_h"] for v in walls.values())
        emit("X7", "SUPPORTED" if tot <= 85.83 else "INCONCLUSIVE",
             "measured %.1f core-hours against a predicted 61.31 (85.83 "
             "pessimistic); per-L median/max wall_s %s"
             % (tot, dict((k, (round(v["median"]), round(v["max"])))
                          for k, v in walls.items())))

    # ---------------- figures --------------------------------------------
    hdr("F.  FIGURES  (validation figures, NOT manuscript figures)")
    for p in figures(cells, res, figdir):
        print("  wrote %s" % os.path.relpath(p, task))

    hdr("WHAT THIS ANALYSIS MAY NOT SAY")
    print("  * No crossing above is lambda_c(zeta = 0.35) or a finite-size")
    print("    estimate of it. L = 32, 48, 64 are at or below the programme's")
    print("    own corpus floor. The output is LOCATOR QUALITY.")
    print("  * A smoothness statement is bound to zeta = 0.35, L <= 64,")
    print("    N_c = 1024 and this guided-cloning configuration. It implies")
    print("    NOTHING about N_c = 1024 at L = 96 or 128, nothing about lower")
    print("    zeta, and there is no global N_c(L, zeta) law here.")
    print("  * No phase-boundary law and no exponent is fitted anywhere.")
    print("  * If the crossing simply moved to lambda <= 0.1932 or stayed")
    print("    boundary-driven, that is the result. The grid is NOT extended")
    print("    again automatically.")

    out = os.path.join(task, "LOWLAMBDA_RESULTS.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("\n" + "=" * 78)
    print("  wrote %s" % out)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
