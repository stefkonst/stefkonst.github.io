"""LE line per frame from dark-vs-background, then tip y and angle."""
import os,sys,json,numpy as np,cv2
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data"); TR=os.path.join(DATA,"tracks")

sys.path.insert(0,r"C:/Users/stefk/Documents/website/site/flap/first")
from track_views import Stack
from scipy.ndimage import uniform_filter1d
D=r"C:/Users/stefk/MIT Dropbox/SMRL/Shared/Stefania/full_tension/"
PX=0.152  # mm/px from ruler (8 cm over 528 px)
def run(n,x0=400,x1=660,y0=150,y1=480,thr=35,fps=200.):
    s=Stack(D+n+".avi"); N=len(s)
    idx=np.arange(0,N,max(1,N//200))
    bg=np.percentile(np.stack([np.asarray(s[i],np.float32) for i in idx]),97,axis=0)
    xs=np.arange(x0,x1,4)
    top=np.full((N,len(xs)),np.nan); bot=top.copy()
    for k in range(N):
        a=np.asarray(s[k],np.float32)
        d=bg[y0:y1][:,xs]-a[y0:y1][:,xs]
        d=uniform_filter1d(d,5,axis=0)
        m=d>thr
        has=m.any(0)
        t=np.argmax(m,0).astype(float); b=(m.shape[0]-1-np.argmax(m[::-1],0)).astype(float)
        t[~has]=np.nan; b[~has]=np.nan
        top[k]=t+y0; bot[k]=b+y0
    ok=np.isfinite(top)
    ang=np.full(N,np.nan); tip=np.full(N,np.nan)
    for k in range(N):
        g=ok[k]
        if g.sum()>10:
            p=np.polyfit(xs[g],top[k,g],1); ang[k]=np.degrees(np.arctan(p[0])); tip[k]=np.polyval(p,x1)
    np.savez_compressed(os.path.join(TR,n+".npz"),top=top.astype(np.float32),bot=bot.astype(np.float32),xs=xs,ang=ang,tip=tip,fps=fps)
    return ang,tip,bg
def stats(n,f,ang,tip,fps=200.):
    t=np.arange(len(tip))/fps; g=np.isfinite(tip)
    y=(tip-np.nanmean(tip))*PX; a=ang-np.nanmean(ang)
    def harm(z,ff):
        zz=z[g]; tt=t[g]; c=np.exp(-2j*np.pi*ff*tt); return 2*abs(np.mean(zz*c))
    # dominant freq
    zz=np.where(g,y,0); F=np.fft.rfftfreq(len(zz),1/fps); P=abs(np.fft.rfft(zz-zz.mean())); P[F<0.5]=0
    fd=F[np.argmax(P)]
    lo,hi=np.nanpercentile(y,[2,98]); alo,ahi=np.nanpercentile(a,[2,98])
    return dict(run=n,f_drive=f,f_dom=round(float(fd),2),tip_pp_mm=round(float(hi-lo),2),ang_pp_deg=round(float(ahi-alo),2),
                tip_1f_amp_mm=round(float(harm(y,f)),3),tip_2f_amp_mm=round(float(harm(y,2*f)),3),valid=round(float(g.mean()),2))
if __name__=="__main__":
    n=sys.argv[1]; f=float(sys.argv[2])
    ang,tip,bg=run(n)
    r=stats(n,f,ang,tip); print(json.dumps(r))
    open(os.path.join(DATA,"results.jsonl"),"a").write(json.dumps(r)+"\n")
