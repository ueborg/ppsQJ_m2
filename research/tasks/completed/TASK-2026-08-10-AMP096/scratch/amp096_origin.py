"""Probe 2: (a) can A=0.96 be reproduced from the OLD aggregates using the
May-2026 procedure (1/sqrt(L) extrapolation of Binder crossings, small-zeta fit)?
(b) true OBS-BLKMR-001 recompute using S_AB_mean (only present in the guided
prod/highL aggregates, NOT in agg_caseB_combined).
READ-ONLY."""
import pickle, os, itertools, json
import numpy as np
from collections import defaultdict

D = os.path.expanduser('~/Downloads/01_M1_Internship/Data')

def load(*paths):
    out = []
    for p in paths:
        o = pickle.load(open(os.path.join(D, p), 'rb'))
        out += list(o.values()) if isinstance(o, dict) else list(o)
    return out

def cross(c1, c2):
    c1 = np.array(sorted(c1)); c2 = np.array(sorted(c2))
    lo, hi = max(c1[0, 0], c2[0, 0]), min(c1[-1, 0], c2[-1, 0])
    if hi - lo < 1e-9: return None
    g = np.linspace(lo, hi, 400)
    d = np.interp(g, c1[:, 0], c1[:, 1]) - np.interp(g, c2[:, 0], c2[:, 1])
    s = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if len(s) == 0: return None
    i = s[-1]
    return float(g[i] - d[i] * (g[i + 1] - g[i]) / (d[i + 1] - d[i]))

def build(recs, valfn):
    by = defaultdict(list)
    for r in recs:
        try: v = valfn(r)
        except Exception: continue
        if v is None or not np.isfinite(v): continue
        lam = r.get('lam', r.get('lambda'))
        if lam is None: continue
        by[(round(float(r['zeta']), 4), int(r['L']))].append((float(lam), float(v)))
    return by

def lc_table(by, wide_only, power):
    """returns {zeta: (raw_median, extrap)}"""
    raw, ext = {}, {}
    for z in sorted({zz for zz, _ in by}):
        Ls = sorted(L for zz, L in by if zz == z)
        pts = []
        for L1, L2 in itertools.combinations(Ls, 2):
            if wide_only and L2 < 2 * L1: continue
            x = cross(by[(z, L1)], by[(z, L2)])
            if x is not None and x > 0.005:
                pts.append((np.sqrt(L1 * L2), x))
        if not pts: continue
        raw[z] = float(np.median([x for _, x in pts]))
        Ls_e = np.array([l for l, _ in pts]); xs = np.array([x for _, x in pts])
        if len(set(np.round(Ls_e, 3))) >= 2:
            ext[z] = float(np.polyfit(Ls_e ** (-power), xs, 1)[1])
    return raw, ext

def fits(d, wins, tag):
    rows = []
    for w in wins:
        zs = np.array(sorted(z for z in d if w[0] - 1e-9 <= z <= w[1] + 1e-9))
        if len(zs) < 3: continue
        y = np.clip(np.array([d[z] for z in zs]), 1e-6, 1 - 1e-6)
        r = y / (1 - y)
        pl = np.polyfit(np.log(zs), np.log(y), 1)
        pr = np.polyfit(np.log(zs), np.log(r), 1)
        rows.append((tag, w, len(zs), float(np.mean(y / np.sqrt(zs))), float(pl[0]),
                     float(np.exp(pl[1])), float(np.mean(r / np.sqrt(zs))),
                     float(pr[0]), float(np.exp(pr[1]))))
    return rows

WINS = [(0.02, 0.15), (0.02, 0.25), (0.02, 0.3), (0.05, 0.3), (0.02, 0.5), (0.02, 0.92)]

print("=== (a) ORIGIN PROBE: old aggregates, B_L_mean crossings ===")
print(f"{'dataset/proc':>44} {'window':>12} {'n':>3} | lam A@.5  phi   A_free | r A@.5  phi   A_free")
for name, files in [('ladder_fss_ready', ['old_cloning_data/ladder_fss_ready.pkl']),
                    ('clone_aggregate_dense_full', ['old_cloning_data/clone_aggregate_dense_full.pkl']),
                    ('dense_full+rescue(L128)', ['old_cloning_data/clone_aggregate_dense_full.pkl',
                                                 'old_cloning_data/clone_aggregate_rescue.pkl']),
                    ('clone_aggregate(2)', ['old_cloning_data/clone_aggregate(2).pkl'])]:
    recs = load(*files)
    by = build(recs, lambda r: r.get('B_L_mean'))
    for wide in (False, True):
        for power in (0.5, 1.0):
            raw, ext = lc_table(by, wide, power)
            tags = [(f"{name} raw med {'wide' if wide else 'allpairs'}", raw)] if power == 0.5 else []
            tags.append((f"{name} extrap L^-{power} {'wide' if wide else 'allpairs'}", ext))
            for tag, d in tags:
                for row in fits(d, WINS, tag):
                    t, w, n, a1, p1, af1, a2, p2, af2 = row
                    print(f"{t:>44} {str(w):>12} {n:>3} | {a1:6.3f} {p1:6.3f} {af1:6.3f} | {a2:6.3f} {p2:6.3f} {af2:6.3f}")

print("\n=== (b) TRUE OBS-BLKMR-001 vs OBS-BLPROD-001 on guided prod+highL ===")
recs = load('pps_aggregates/agg_pps_clone_guided_prod.pkl',
            'pps_aggregates/agg_pps_clone_guided_highL.pkl')
print("n =", len(recs))
for label, fn in [('OBS-BLPROD-001 stored B_L_mean', lambda r: r.get('B_L_mean')),
                  ('OBS-BLKMR-001 CMI_mean*S_AB_mean', lambda r: r['CMI_mean'] * r['S_AB_mean'])]:
    by = build(recs, fn)
    raw, ext = lc_table(by, True, 0.5)
    print(f"\n  {label}: lam_c(raw wide-pair median)")
    print("   " + "  ".join(f"{z}:{raw[z]:.4f}" for z in sorted(raw)))
    for row in fits(raw, [(0.05, 0.125), (0.05, 0.2), (0.05, 0.4), (0.05, 0.85), (0.25, 0.85)], label):
        t, w, n, a1, p1, af1, a2, p2, af2 = row
        print(f"   {str(w):>13} n={n:2d} | lam A@.5={a1:.3f} phi={p1:.3f} | r A@.5={a2:.3f} phi={p2:.3f} A_free={af2:.3f}")
