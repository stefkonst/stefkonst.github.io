import os
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data"); TR=os.path.join(DATA,"tracks")
"""Active vs unpowered wing at 8 Hz, 1200 V, in two views. LE line per frame, hinge = least-squares crossing of all LE lines,
stroke = angle of the LE about the hinge. Per-cycle peak to peak, harmonic fit, cycle-to-cycle spread."""
import sys,json,numpy as np
from track import run
from scipy.optimize import minimize_scalar
fs=200.
RUNS={"side":[("full_DEA_core_wing_8hz_1200V_side",dict(x0=400,x1=660)),("only_DEA_core_8hz_1200V_side",dict(x0=400,x1=660))],
      "diag":[("full_DEA_core_wing_8hz_1200V_diag_good",dict(x0=480,x1=610)),("only_DEA_core_8hz_1200V_diag",dict(x0=480,x1=640))]}
def lines(n):
    d=np.load(os.path.join(TR,n+".npz")); xs=d["xs"]; top=d["top"]; P=[]
    for k in range(len(top)):
        m=np.isfinite(top[k])
        P.append(np.polyfit(xs[m],top[k][m],1) if m.sum()>10 else (np.nan,np.nan))
    return np.array(P)
def hinge(P):
    g=np.isfinite(P[:,0]); s,c=P[g,0],P[g,1]
    # line: y = s x + c  ->  s x - y = -c ; least squares point
    A=np.c_[s,-np.ones_like(s)]/np.sqrt(1+s**2)[:,None]; b=-c/np.sqrt(1+s**2)
    return np.linalg.lstsq(A,b,rcond=None)[0]
def analyse(n,f=8.):
    P=lines(n); ang=np.degrees(np.arctan(P[:,0])); t=np.arange(len(ang))/fs; g=(t>0.6)&np.isfinite(ang)
    N=int(round(fs/f)); k0=int(0.6*fs)
    pp=np.array([np.nanmax(ang[i:i+N])-np.nanmin(ang[i:i+N]) for i in range(k0,len(ang)-N+1,N)])
    def fit(ff,H=8):
        X=np.column_stack([np.ones(g.sum())]+[fn(2*np.pi*h*ff*t[g]) for h in range(1,H+1) for fn in (np.cos,np.sin)])
        c=np.linalg.lstsq(X,ang[g],rcond=None)[0]; return c,np.sqrt(np.mean((ang[g]-X@c)**2))
    ff=minimize_scalar(lambda q:fit(q)[1],bounds=(7.9,8.1),method="bounded").x
    c,rms=fit(ff); amp=[float(np.hypot(c[2*h-1],c[2*h])) for h in range(1,9)]
    hx,hy=hinge(P)
    return dict(run=n,f=round(ff,3),pp_med=float(np.median(pp)),pp_sd=float(np.std(pp)),n_cyc=len(pp),pp_all=pp.round(2).tolist(),
                first_amp=amp[0],harm=[round(a,2) for a in amp],resid=float(rms),hinge=[float(hx),float(hy)],mean_ang=float(np.nanmean(ang[g])))
if __name__=="__main__":
    out={}
    for view,L in RUNS.items():
        for n,kw in L:
            if not os.path.exists(os.path.join(TR,n+".npz")): run(n,**kw)
            r=analyse(n); out[n]=r; print(view,n,"pp %.2f±%.2f (n=%d) 1st %.2f harm %s hinge %s"%(r["pp_med"],r["pp_sd"],r["n_cyc"],r["first_amp"],r["harm"][:4],np.round(r["hinge"])))
    json.dump(out,open(os.path.join(DATA,"pair.json"),"w"),indent=1)
