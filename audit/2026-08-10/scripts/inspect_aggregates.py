"""READ-ONLY inspection of aggregate pickles. Audit 2026-08-10. Writes nothing."""
import pickle, glob, os, sys
import numpy as np

def describe(path):
    print("="*70); print(os.path.basename(path), f"({os.path.getsize(path)/1e6:.2f} MB)")
    try:
        with open(path,'rb') as f: obj = pickle.load(f)
    except Exception as e:
        print("  LOAD FAILED:", e); return
    print("  type:", type(obj).__name__)
    recs = None
    if isinstance(obj, dict):
        ks = list(obj.keys())
        print(f"  dict, {len(ks)} keys; sample: {ks[:3]}")
        if ks and isinstance(obj[ks[0]], dict): recs = list(obj.values())
    elif isinstance(obj, list):
        print(f"  list, {len(obj)} entries")
        if obj and isinstance(obj[0], dict): recs = obj
    if recs is None:
        try:
            print("  columns:", list(obj.columns)); print("  nrows:", len(obj))
        except Exception: pass
        return
    print(f"  n_records: {len(recs)}")
    fields = sorted({k for r in recs[:400] for k in r})
    print("  fields:", fields)
    for p in ('L','zeta','lam','lambda','N_c','Nc','T','n_real','seed'):
        vals = [r[p] for r in recs if p in r and np.isscalar(r[p])]
        if vals:
            u = sorted(set(vals))
            print(f"    {p}: n={len(vals)} uniq={len(u)} " +
                  (f"vals={u}" if len(u)<=22 else f"range=[{min(u)},{max(u)}]"))

paths = sorted(glob.glob(os.path.expanduser('~/Downloads/01_M1_Internship/Data/pps_aggregates/*.pkl')))
for p in paths: describe(p)
