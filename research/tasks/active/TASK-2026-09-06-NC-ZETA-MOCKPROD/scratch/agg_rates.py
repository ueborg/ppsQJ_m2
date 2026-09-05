import json, glob
from collections import defaultdict
import statistics as st

files = glob.glob("research/tasks/**/results/*.json", recursive=True)
rows=[]
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    def get(dd, keys):
        found = {}
        def rec(x):
            if isinstance(x, dict):
                for k,v in x.items():
                    if k in keys and k not in found:
                        found[k]=v
                    rec(v)
            elif isinstance(x, list):
                for it in x:
                    rec(it)
        rec(dd)
        return found
    keys = ["wall_s","n_steps","N_c","Nc","L","lam","lambda","dtau_mult","zeta","T"]
    found = get(d, keys)
    if "wall_s" in found and "n_steps" in found and ("N_c" in found or "Nc" in found):
        Nc = found.get("N_c", found.get("Nc"))
        try:
            Nc = float(Nc); L=float(found.get("L")); n_steps=float(found["n_steps"]); wall=float(found["wall_s"])
            dtau = found.get("dtau_mult")
            dtau = float(dtau) if dtau is not None else None
            zeta = found.get("zeta")
            lam = found.get("lam", found.get("lambda"))
        except Exception:
            continue
        if n_steps<=0 or Nc<=0: continue
        rate_ms = wall/(Nc*n_steps)*1000.0
        rows.append(dict(f=f,Nc=Nc,L=L,n_steps=n_steps,wall=wall,dtau=dtau,zeta=zeta,lam=lam,rate_ms=rate_ms))

print("total rows", len(rows))
# filter dtau_mult==6
rows6 = [r for r in rows if r["dtau"]==6.0]
print("dtau==6 rows", len(rows6))

by = defaultdict(list)
for r in rows6:
    by[(r["L"], r["Nc"])].append(r["rate_ms"])

print("\n(L, Nc) -> n, median rate_ms/clone-window, min, max")
for k in sorted(by.keys()):
    v = by[k]
    print(k, len(v), round(st.median(v),4), round(min(v),4), round(max(v),4))
