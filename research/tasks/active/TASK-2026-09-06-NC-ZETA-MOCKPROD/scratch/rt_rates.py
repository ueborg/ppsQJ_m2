import glob,json,collections,statistics as st,math
ROOT="/Users/catlover1337/Documents/ppsQJ_m2"
out=collections.defaultdict(list); lamr=collections.defaultdict(list)
for p in glob.glob(ROOT+"/research/tasks/**/results/*.json",recursive=True):
    try: d=json.load(open(p))
    except Exception: continue
    if not isinstance(d,dict) or d.get("status")!="ok": continue
    if float(d.get("dtau_mult",0))!=6.0 or float(d["zeta"])!=0.35: continue
    n,N=int(d["n_steps"]),int(d["N_c"])
    r=float(d["wall_s"])/(N*n)*1000.0
    out[(int(d["L"]),N)].append(r); lamr[(int(d["L"]),N)].append((float(d["lam"]),r))
for k in sorted(out):
    v=out[k]
    print(k,"n=%d"%len(v),"med=%.3f"%st.median(v),"max=%.3f"%max(v),"p90=%.3f"%sorted(v)[int(0.9*(len(v)-1))],"min=%.3f"%min(v))
print("--- lambda at which max occurs, and lambda span")
for k in sorted(lamr):
    v=sorted(lamr[k]); mx=max(v,key=lambda t:t[1])
    lams=sorted(set(round(a,4) for a,_ in v))
    print(k,"lam_at_max=%.4f"%mx[0],"lam span %.4f..%.4f"%(lams[0],lams[-1]),"nlam=%d"%len(lams))
