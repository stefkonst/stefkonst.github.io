import os
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data"); TR=os.path.join(DATA,"tracks")
"""Fundamental (and 2nd) harmonic of the stroke angle in every side run; fit a one-mode oscillator with a first-order force lag:
theta1(f) = K / sqrt((1-r^2)^2+(2 z r)^2) / sqrt(1+(2 pi f tau)^2), r=f/fn."""
import json,numpy as np
from scipy.optimize import minimize_scalar,least_squares
fs=200.
RUNS=[("full_DEA_core_wing_1hz_1200v_side",1,1200),("full_DEA_core_wing_5hz_1200v_side",5,1200),("full_DEA_core_wing_7hz_1200V_side",7,1200),
      ("full_DEA_core_wing_8hz_1200V_side",8,1200),("full_DEA_core_wing_9hz_1200V_side",9,1200),("full_DEA_core_wing_10hz_1200V_side",10,1200),
      ("full_DEA_core_wing_15hz_1200v_side",15,1200),("full_DEA_core_wing_1hz_1000V_side",1,1000),("full_DEA_core_wing_8hz_1400V_side",8,1400),
      ("only_DEA_core_8hz_1200V_side",8,1200)]
def harm(n,f):
    a=np.load(os.path.join(TR,n+".npz"))["ang"]; t=np.arange(len(a))/fs
    rest=np.nanmedian(a[:40]); on=np.argmax(np.abs(a-rest)>1.0)
    end=len(a)
    if "1400V_side" in n: end=int(2.15*fs)
    g=np.zeros(len(a),bool); g[on+int(0.5*fs):end]=True; g&=np.isfinite(a)
    H=6
    def fit(ff):
        X=np.column_stack([np.ones(g.sum())]+[fn(2*np.pi*h*ff*t[g]) for h in range(1,H+1) for fn in (np.cos,np.sin)])
        c=np.linalg.lstsq(X,a[g],rcond=None)[0]; return c,np.sqrt(np.mean((a[g]-X@c)**2))
    ff=minimize_scalar(lambda q:fit(q)[1],bounds=(f*0.99,f*1.01),method="bounded").x
    c,rms=fit(ff); A=[float(np.hypot(c[2*h-1],c[2*h])) for h in range(1,H+1)]
    # cycle-to-cycle spread of the fundamental: fit per 0.5 s block
    blocks=[]
    idx=np.where(g)[0]
    B=int(max(0.5,2.0/f)*fs)
    for b0 in range(idx[0],idx[-1]-B,B):
        m=np.zeros(len(a),bool); m[b0:b0+B]=True; m&=g
        X=np.column_stack([np.ones(m.sum()),np.cos(2*np.pi*ff*t[m]),np.sin(2*np.pi*ff*t[m])]); cc=np.linalg.lstsq(X,a[m],rcond=None)[0]; blocks.append(np.hypot(cc[1],cc[2]))
    return dict(run=n,f=float(ff),A=A,A1_sd=float(np.std(blocks)),n_blocks=len(blocks),mean=float(c[0]))
def model(p,f):
    K,fn,z,tau=p; r=f/fn
    return K/np.sqrt((1-r**2)**2+(2*z*r)**2)/np.sqrt(1+(2*np.pi*f*tau)**2)
if __name__=="__main__":
    R=[]
    for n,f,V in RUNS:
        h=harm(n,f); h["V"]=V; h["f_drive"]=f; R.append(h); print("%-40s %4.1f Hz %d V  A1=%.2f±%.2f  A2=%.2f  A2/A1=%.2f"%(n,h["f"],V,h["A"][0],h["A1_sd"],h["A"][1],h["A"][1]/h["A"][0]))
    sw=[r for r in R if r["V"]==1200 and r["run"].startswith("full")]
    f=np.array([r["f"] for r in sw]); y=np.array([r["A"][0] for r in sw]); s=np.array([max(r["A1_sd"],0.05*r["A"][0]) for r in sw])
    fits={}
    for name,zfix in (("zeta free",None),("zeta 0.07 (decay)",0.07)):
        if zfix is None:
            fun=lambda p:(model(p,f)-y)/s; p0=[4,8.5,0.1,0.03]; lb=[0.1,6,0.005,0]; ub=[50,12,1,0.5]
            r=least_squares(fun,p0,bounds=(lb,ub)); p=r.x
        else:
            fun=lambda q:(model([q[0],q[1],zfix,q[2]],f)-y)/s
            r=least_squares(fun,[4,8.5,0.03],bounds=([0.1,6,0],[50,12,0.5])); p=np.r_[r.x[0],r.x[1],zfix,r.x[2]]
        J=r.jac; cov=np.linalg.pinv(J.T@J)*np.sum(r.fun**2)/max(1,len(y)-len(r.x)); se=np.sqrt(np.diag(cov))
        fits[name]=dict(K=p[0],fn=p[1],zeta=p[2],tau=p[3],se=se.tolist(),chi2=float(np.sum(r.fun**2)),pred=model(p,f).tolist())
        print("   pred",np.round(model(p,f),2),"meas",np.round(y,2))
        print(name,"K=%.2f fn=%.2f zeta=%.3f tau=%.1f ms  se=%s chi2=%.1f"%(p[0],p[1],p[2],p[3]*1e3,np.round(se,3),fits[name]["chi2"]))
    # what fn shift reproduces the unpowered fundamental at 8 Hz, other parameters fixed?
    p=fits["zeta 0.07 (decay)"]; un=[r for r in R if r["run"].startswith("only")][0]["A"][0]; ac=[r for r in sw if r["f_drive"]==8][0]["A"][0]
    from scipy.optimize import brentq
    fn_un=brentq(lambda q: model([p["K"],q,p["zeta"],p["tau"]],8.0)-un, p["fn"], 12)
    print("active 8 Hz A1 %.2f, unpowered %.2f, ratio %.2f; same if fn moves %.2f -> %.2f Hz (+%.1f %%)"%(ac,un,ac/un,p["fn"],fn_un,100*(fn_un/p["fn"]-1)))
    json.dump(dict(runs=R,fits=fits,fn_unpowered_equiv=fn_un),open(os.path.join(DATA,"sweepfit.json"),"w"),indent=1)
