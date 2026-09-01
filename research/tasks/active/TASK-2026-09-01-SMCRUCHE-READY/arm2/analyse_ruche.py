#!/usr/bin/env python3
"""Analyse the SMCCERT Ruche results under the FROZEN rules in analysis_spec.yaml.

    python3 analyse_ruche.py [results_dir]

Pools ARM1 with the parent's existing A-P96 rows when they are present, because
ARM1's seeds continue that stream at the identical cell. Applies exactly the
decision rules frozen in analysis_spec.yaml; nothing here is tuned.
"""
import os, sys, json, glob, math
import numpy as np

BOOT, SEED = 4000, 20260831
MDE_B = 3.5
HERE = os.path.dirname(os.path.abspath(__file__))
res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")

rows = []
for p in sorted(glob.glob(os.path.join(res, "*.json"))):
    rows.append(json.load(open(p)))
# optionally fold in the parent's completed A-P96 / A-BUD rows at the same cell
for extra in sys.argv[2:]:
    for line in open(extra):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
print(f"loaded {len(rows)} runs from {res}" + (f" + {len(sys.argv)-2} extra file(s)" if len(sys.argv) > 2 else ""))

def key(r):
    return (int(r["L"]), float(r["T"]), float(r["zeta"]), round(float(r["lam"]), 6))

def pop(r):
    if "cmi_weighted_mean" in r:
        return float(r["cmi_weighted_mean"]), float(r["cmi_within_var"])
    v = np.array([np.nan if x is None else x for x in r["per_clone"]["CMI"]], float)
    v = v[np.isfinite(v)]
    return float(v.mean()), float(v.var(ddof=1))

cells = {}
for r in rows:
    if r.get("status") not in (None, "ok"):
        continue
    cells.setdefault(key(r), {}).setdefault(int(r["N_c"]), []).append(r)

rng = np.random.default_rng(SEED)
for k in sorted(cells):
    lad = {}
    for nc, rs in cells[k].items():
        m, s2 = zip(*(pop(r) for r in rs))
        if len(m) >= 6:
            lad[nc] = (np.array(m), np.array(s2))
    if len(lad) < 3:
        print(f"\n[L={k[0]} T={k[1]} z={k[2]} lam={k[3]}] only {sorted(lad)} - skipped")
        continue
    print(f"\n[L={k[0]} T={k[1]} z={k[2]} lam={k[3]}]")
    Ns = sorted(lad)
    for nc in Ns:
        m, s2 = lad[nc]
        V = float(np.var(m, ddof=1)); s2m = float(np.mean(s2))
        print(f"  N_c={nc:5d} R={m.size:4d} VIF={V*nc/s2m:8.2f} N_eff={s2m/V:7.2f} "
              f"Var={V:.4e} mean={m.mean():.5f} SEM={math.sqrt(V/m.size):.5f}")

    def windows(Ns):
        ws = [tuple(Ns)]
        for w in range(len(Ns) - 1, 2, -1):
            for i in range(len(Ns) - w + 1):
                t = tuple(Ns[i:i + w])
                if t not in ws:
                    ws.append(t)
        return ws

    print("  gamma = -dlogVar/dlogN_c  (>=3-window scan):")
    gres = {}
    for win in windows(Ns):
        lx = np.log([float(n) for n in win])
        g0 = float(-np.polyfit(lx, [math.log(max(np.var(lad[n][0], ddof=1), 1e-300)) for n in win], 1)[0])
        bs = []
        for _ in range(BOOT):
            ly = [math.log(max(np.var(lad[n][0][rng.integers(0, lad[n][0].size, lad[n][0].size)], ddof=1), 1e-300)) for n in win]
            bs.append(float(-np.polyfit(lx, ly, 1)[0]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        gres[win] = (g0, lo, hi)
        print(f"    {'+'.join(map(str, win)):>24}  gamma={g0:+.3f} CI=[{lo:+.3f},{hi:+.3f}] "
              f"width={hi-lo:.3f}{'  INSIDE [0.5,1.5]' if lo >= 0.5 and hi <= 1.5 else ''}")

    print("  bias fit  I(N_c) = I_inf + B/N_c:")
    for win in windows(Ns):
        x = np.array([1.0 / n for n in win])
        B0 = float(np.polyfit(x, [lad[n][0].mean() for n in win], 1)[0])
        bs = []
        for _ in range(BOOT):
            yb = [lad[n][0][rng.integers(0, lad[n][0].size, lad[n][0].size)].mean() for n in win]
            bs.append(float(np.polyfit(x, yb, 1)[0]))
        bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5]); se = bs.std(ddof=1)
        print(f"    {'+'.join(map(str, win)):>24}  B={B0:+8.3f} CI=[{lo:+8.3f},{hi:+8.3f}] "
              f"MDE|B|={2.80*se:.2f}{'' if 2.80*se <= MDE_B else '  (above the frozen 3.5)'}")

    # --- FROZEN VERDICTS -----------------------------------------------------
    full = tuple(Ns); ds = tuple(Ns[1:]); dl = tuple(Ns[:-1])
    g, lo, hi = gres[full]
    med_vif = float(np.median([np.var(lad[n][0], ddof=1) * n / np.mean(lad[n][1]) for n in Ns]))
    if not (lo >= 0.5 and hi <= 1.5):
        v = "INCONCLUSIVE (gamma CI not contained in [0.5,1.5]; INCONCLUSIVE dominates)"
    elif hi < 1.0 and gres[ds][2] < 1.0:
        v = "KILLED (gamma < 1 in the full AND drop-smallest windows)"
    elif lo <= 1.0 <= hi and (gres[ds][1] <= 1 <= gres[ds][2] or gres[dl][1] <= 1 <= gres[dl][2]):
        v = "SUPPORTED"
    else:
        v = "INCONCLUSIVE"
    gate = "" if med_vif >= 40 else "  ** median VIF < 40 -> UNTESTED AT HIGH VIF **"
    print(f"  median measured VIF = {med_vif:.1f}{gate}")
    print(f"  VERDICT (variance scaling, frozen rule): {v}")
