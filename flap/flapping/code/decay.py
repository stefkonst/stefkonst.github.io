import os
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data"); TR=os.path.join(DATA,"tracks")
import json,numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
fs=200.
def decay(n,t0,col="ang"):
    a=np.load(os.path.join(TR,n+".npz"))[col]; t=np.arange(len(a))/fs
    rest=np.nanmedian(a[-60:]); y=a-rest; k=int(t0*fs)
    seg=y[k:]; tt=t[k:]-t[k]; m=np.isfinite(seg)
    f=lambda x,A,z,w,p: A*np.exp(-z*w*x)*np.cos(w*np.sqrt(1-z*z)*x+p)
    best=None
    for w0 in 2*np.pi*np.arange(7,11,0.5):
        try:
            p,_=curve_fit(f,tt[m],seg[m],p0=(np.nanmax(abs(seg[:10])),0.07,w0,0),maxfev=20000)
            r=np.sqrt(np.mean((f(tt[m],*p)-seg[m])**2))
            if best is None or r<best[1]: best=(p,r)
        except Exception: pass
    p,r=best; A,z,w,ph=p; fd=w*np.sqrt(1-z*z)/2/np.pi
    # instantaneous frequency against amplitude: successive peaks and troughs (parabolic sub-sample)
    from scipy.signal import find_peaks
    def ext(z_):
        p_,_=find_peaks(z_,distance=8); o=[]
        for i in p_:
            if 1<=i<len(z_)-1:
                dd=(z_[i-1]-z_[i+1])/(2*(z_[i-1]-2*z_[i]+z_[i+1])); o.append((i+dd,z_[i]))
        return o
    E=np.array(sorted(ext(seg)+ext(-seg))); fa=[]
    for j in range(len(E)-2):
        amp=(E[j,1]+E[j+1,1])/2
        if amp>0.5: fa.append((float(amp),float(fs/(E[j+2,0]-E[j,0]))))
    return dict(run=n,t0=t0,f_n=float(w/2/np.pi),f_d=float(fd),zeta=float(abs(z)),rms=float(r),A0=float(abs(A)),f_vs_amp=fa)
if __name__=="__main__":
    R=[decay("full_DEA_core_wing_8hz_1400V_side",2.2+0.1),decay("only_DEA_core_8hz_1200V_diag",0.62)]
    for r in R: print(r)
    json.dump(R,open(os.path.join(DATA,"decays.json"),"w"),indent=1)
