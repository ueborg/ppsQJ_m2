#!/usr/bin/env python3
"""Analyse low-zeta horizon-crossing pilot.

Reports the actual failure metric: number of sign changes and bootstrap
probability of exactly one cross-L crossing. It never interprets GESS as a
success criterion.
"""
from __future__ import annotations
import os, json, argparse
from collections import defaultdict
import numpy as np


def load(root):
    rows=[]
    for d,_,fs in os.walk(root):
        for fn in fs:
            if not fn.endswith(".json"):
                continue
            try:
                r=json.load(open(os.path.join(d,fn)))
            except Exception:
                continue
            if r.get("status")=="ok" and "Tmult" in r:
                rows.append(r)
    return rows


def sign_changes(y):
    y=np.asarray(y,float)
    good=np.isfinite(y)
    y=y[good]
    if len(y)<2:
        return 0
    s=np.sign(y)
    nz=np.flatnonzero(s)
    if len(nz)==0:
        return 0
    for i in range(len(s)):
        if s[i]==0:
            s[i]=s[nz[np.argmin(np.abs(nz-i))]]
    return int(np.sum(s[:-1]*s[1:]<0))


def crossings(x,y):
    out=[]
    for i in range(len(y)-1):
        if not np.isfinite(y[i]) or not np.isfinite(y[i+1]):
            continue
        if y[i]==0:
            out.append(float(x[i]))
        elif y[i]*y[i+1]<0:
            out.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
    return out


def analyse_one(rows, mode, Tmult, obs, B, seed=7):
    rr=[r for r in rows if r["mode"]==mode and r["Tmult"]==Tmult]
    Ls=sorted(set(int(r["L"]) for r in rr))
    lams=sorted(set(float(r["lambda"]) for r in rr))
    if len(Ls)!=2 or len(lams)<3:
        return None
    L0,L1=Ls

    g=defaultdict(list)
    for r in rr:
        g[(int(r["L"]),float(r["lambda"]))].append(float(r[obs]))
    if any((L,lam) not in g for L in Ls for lam in lams):
        return None

    D=np.array([np.mean(g[(L1,lam)])-np.mean(g[(L0,lam)]) for lam in lams])
    hits=crossings(lams,D)
    central=hits[0] if len(hits)==1 else np.nan

    rng=np.random.default_rng(seed)
    bcross=[]
    nsign=[]
    for _ in range(B):
        Db=[]
        for lam in lams:
            a=np.asarray(g[(L0,lam)],float)
            b=np.asarray(g[(L1,lam)],float)
            ma=float(np.mean(rng.choice(a,size=len(a),replace=True)))
            mb=float(np.mean(rng.choice(b,size=len(b),replace=True)))
            Db.append(mb-ma)
        n=sign_changes(Db)
        nsign.append(n)
        h=crossings(lams,Db)
        if n==1 and len(h)==1:
            bcross.append(h[0])

    if bcross:
        q16,q50,q84=np.percentile(bcross,[16,50,84])
    else:
        q16=q50=q84=np.nan
    return dict(
        Ls=Ls,lams=lams,D=D,central=central,central_signs=sign_changes(D),
        p_unique=float(len(bcross)/B),
        q16=float(q16),q50=float(q50),q84=float(q84),
        mean_signs=float(np.mean(nsign)),
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dir",required=True)
    ap.add_argument("--bootstrap",type=int,default=1000)
    a=ap.parse_args()
    rows=load(a.dir)
    if not rows:
        raise SystemExit("no horizon-crossing records found")
    modes=sorted(set(r["mode"] for r in rows))
    Ts=sorted(set(float(r["Tmult"]) for r in rows))
    print("records",len(rows),"modes",modes,"Tmult",Ts)

    results={}
    for obs in ("CMI","CMI_tavg50","CMI_tavg75"):
        print("\n==",obs,"==")
        for mode in modes:
            for T in Ts:
                x=analyse_one(rows,mode,T,obs,a.bootstrap,
                              seed=1100+int(100*T)+sum(map(ord,mode+obs)))
                if x is None:
                    continue
                results[(mode,T,obs)]=x
                print(
                    "%-7s T=%g signs=%d Puniq=%.3f "
                    "lambda*=%.5f boot=%.5f [%.5f, %.5f] mean_signs=%.2f"
                    %(mode,T,x["central_signs"],x["p_unique"],x["central"],
                      x["q50"],x["q16"],x["q84"],x["mean_signs"])
                )

        if len(Ts)>=2:
            T0,T1=Ts[0],Ts[-1]
            print("  horizon drift",T0,"->",T1)
            for mode in modes:
                a0=results.get((mode,T0,obs)); a1=results.get((mode,T1,obs))
                if a0 and a1 and np.isfinite(a0["central"]) and np.isfinite(a1["central"]):
                    print("   %-7s d_lambda = %+0.5f"
                          %(mode,a1["central"]-a0["central"]))

    print("\nNever-mode note: self-normalized importance sampling targets the correct "
          "tilted measure but normalized observables have finite-N bias. "
          "Use its ESS and convergence with Nc before treating it as a reference.")


if __name__=="__main__":
    main()
