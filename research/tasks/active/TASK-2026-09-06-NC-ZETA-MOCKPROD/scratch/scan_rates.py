import json, glob, os, sys

files = glob.glob("research/tasks/**/results/*.json", recursive=True)
files += glob.glob("results/**/*.json", recursive=True)
rows = []
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    def get(dd, keys):
        # search recursively for keys in dict
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
    keys = ["wall_s","n_steps","N_c","Nc","L","lam","lambda","dtau_mult","zeta","sec_per_clone_window","T"]
    found = get(d, keys)
    if "wall_s" in found and ("n_steps" in found) and ("N_c" in found or "Nc" in found):
        rows.append((f, found))

print(len(rows), "candidate rows with wall_s/n_steps/N_c")
for f, found in rows:
    print(f, found)
