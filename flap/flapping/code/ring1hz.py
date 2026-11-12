"""Ringing after each release in the 1 Hz runs (drive near 0 V): from the first minimum after each fall, fit
quadratic drift + damped cosine over 0.3 s. Gives the small-amplitude natural frequency and damping."""
import os,json,numpy as np
from scipy.optimize import least_squares
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data"); TR=os.path.join(DATA,"tracks")
fs=200.
def rings(n):
    a=np.load(os.path.join(TR,n+".npz"))["ang"]; t=np.arange(len(a))/fs; da=np.gradient(a); res=[]
    for c in range(len(a)//200):
        k=c*200+int(np.nanargmin(da[c*200:(c+1)*200]))
        mins=[i for i in range(k,min(k+40,len(a)-1)) if a[i]<=a[i-1] and a[i]<=a[i+1]]
        if not mins: continue
        k1=mins[0]; k2=k1+60
        if k2>len(a): break
        tt=t[k1:k2]-t[k1]; y=a[k1:k2]; best=None
        for fn0 in np.arange(7,12.1,0.5):
            f=lambda p: p[0]+p[1]*tt+p[2]*tt**2+np.exp(-p[4]*2*np.pi*p[3]*tt)*(p[5]*np.cos(2*np.pi*p[3]*tt)+p[6]*np.sin(2*np.pi*p[3]*tt))-y
            r=least_squares(f,[y.mean(),0,0,fn0,0.07,-1,0],bounds=([-90,-500,-5000,5,0,-30,-30],[90,500,5000,15,0.6,30,30]))
            if best is None or r.cost<best.cost: best=r
        res.append(dict(t=float(t[k1]),f_n=float(best.x[3]),zeta=float(best.x[4]),A=float(np.hypot(best.x[5],best.x[6]))))
    return res
if __name__=="__main__":
    R={}
    for n in ["full_DEA_core_wing_1hz_1000V_side","full_DEA_core_wing_1hz_1200v_side"]:
        r=rings(n); F=np.array([x["f_n"] for x in r]); Z=np.array([x["zeta"] for x in r]); A=np.array([x["A"] for x in r])
        R[n]=dict(rings=r,f_n=float(np.median(F)),f_n_sd=float(F.std()),zeta=float(np.median(Z)),A=float(np.median(A)))
        print(n,"fn=%.2f (median, sd %.2f) zeta=%.3f amp %.2f deg"%(R[n]["f_n"],F.std(),R[n]["zeta"],R[n]["A"]))
    json.dump(R,open(os.path.join(DATA,"ring1hz.json"),"w"),indent=1)
