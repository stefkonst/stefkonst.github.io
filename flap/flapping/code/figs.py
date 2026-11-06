import json,glob,os,shutil,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"Arial","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
OUT=r"C:/Users/stefk/Documents/YEAR2/robot_flap"; os.makedirs(OUT+"/svg",exist_ok=True)
SC={"only_DEA_core_8hz_1200V_side":0.135}  # mm/px; others 0.152
R={}
for l in open("results.jsonl"): r=json.loads(l); R[r["run"]]=r
def load(n):
    d=np.load(n+".npz"); fs=200.; s=SC.get(n,0.152)
    a=d["ang"]; xs=d["xs"]; top,bot=d["top"],d["bot"]
    c=np.nanmean((bot-top)[:,-12:],1)*s   # projected chord near tip, mm
    return a,c,fs
def cyc(n,f):
    a,c,fs=load(n); t=np.arange(len(a))/fs; ph=(t*f)%1
    g=(t>0.5)&(t<2.1); b=np.linspace(0,1,26); i=np.digitize(ph[g],b)-1
    A=np.array([np.nanmean(a[g][i==k]) for k in range(25)]); C=np.array([np.nanmean(c[g][i==k]) for k in range(25)])
    sa=np.array([np.nanstd(a[g][i==k]) for k in range(25)])
    return (b[:-1]+b[1:])/2,A,C,sa
def cyc_pp(n,f):
    a,c,fs=load(n); N=int(round(fs/f)); k0=int(0.6*fs)
    pp=np.array([np.nanmax(a[i:i+N])-np.nanmin(a[i:i+N]) for i in range(k0,len(a)-N,N)])
    keep=pp>0.6*pp.max(); return float(np.median(pp[keep])),int(keep.sum())
rows=[]
for n,r in R.items():
    r["ang_pp_deg"],r["n_cycles"]=cyc_pp(n,r["f_drive"]); r["ang_pp_deg"]=round(r["ang_pp_deg"],2)
    a,c,fs=load(n); g=np.arange(len(a))>100
    lo,hi=np.nanpercentile(c[g],[2,98]); r["chord_pp_mm"]=round(float(hi-lo),2); r["chord_mean_mm"]=round(float(np.nanmean(c[g])),2)
    rows.append(r)
json.dump(rows,open(OUT+"/results.json","w"),indent=1)
sweep=sorted([r for r in rows if r["run"].startswith("full") and "1200" in r["run"].lower().replace("v","V")],key=lambda r:r["f_drive"])
fig,ax=plt.subplots(2,2,figsize=(7.2,5.6))
# a: frequency sweep
x=[r["f_drive"] for r in sweep]; y=[r["ang_pp_deg"] for r in sweep]
ax[0,0].plot(x,y,"o-",color="#750014",label="active wing + core, 1200 V")
p=R.get("only_DEA_core_8hz_1200V_side")
if p: ax[0,0].plot(8,p["ang_pp_deg"],"s",color="#5b7fbf",ms=7,label="passive wing, core only, 1200 V")
ax[0,0].axvspan(8,9,color="#d1e0ff",alpha=.6,lw=0,label="ringing after release, 8-9 Hz")
ax[0,0].set(xlabel="Drive frequency (Hz)",ylabel="Stroke, peak to peak (deg)",title="a  Frequency sweep"); ax[0,0].legend(frameon=False,fontsize=7)
# b: cycle at 8 Hz active vs passive
for n,col,lab in (("full_DEA_core_wing_8hz_1200V_side","#750014","active wing"),("only_DEA_core_8hz_1200V_side","#5b7fbf","passive wing")):
    if n in R:
        ph,A,C,sa=cyc(n,8); A=A-np.nanmean(A); sh=int(np.nanargmin(A))-6; A=np.roll(A,-sh); sa=np.roll(sa,-sh); m=np.isfinite(A); ax[0,1].plot(ph[m],A[m],color=col,label=lab); ax[0,1].fill_between(ph[m],(A-sa)[m],(A+sa)[m],color=col,alpha=.2,lw=0)
ax[0,1].set(xlabel="Cycle phase",ylabel="Stroke angle (deg, mean removed)",title="b  One cycle at 8 Hz, 1200 V "); ax[0,1].legend(frameon=False,fontsize=7)
# c: voltage
V=[(n,r) for n,r in R.items() if n.startswith("full")]
for f,col in ((1,"#444"),(8,"#750014")):
    pts=sorted([(float(n.lower().split("hz_")[1].split("v")[0]),r["ang_pp_deg"]) for n,r in V if r["f_drive"]==f and "side" in n])
    if pts: ax[1,0].plot(*zip(*pts),"o-",color=col,label=f"{f} Hz")
ax[1,0].set(ylim=(0,28),xlabel="Voltage (V)",ylabel="Stroke, peak to peak (deg)",title="c  Voltage"); ax[1,0].legend(frameon=False,fontsize=7)
# d: free decay after the 1400 V drive stops
from scipy.signal import find_peaks
a,c,fs=load("full_DEA_core_wing_8hz_1400V_side"); t=np.arange(len(a))/fs
a0=np.nanmedian(a[-150:]); y=a-a0; k=int(2.2*fs)
pk,_=find_peaks(y[k:],distance=8); pk=pk+k; pk=pk[y[pk]>0.3]
tp,yp=t[pk],y[pk]; fd=1/np.median(np.diff(tp)); dl=np.mean(np.log(yp[:-1]/yp[1:])); z=dl/np.sqrt(4*np.pi**2+dl**2)
ax[1,1].plot(t[k-60:k+200]-t[k],y[k-60:k+200],color="#750014"); ax[1,1].plot(tp-t[k],yp,"o",color="k",ms=3)
tt=np.linspace(0,1,100); ax[1,1].plot(tt+tp[0]-t[k],yp[0]*np.exp(-z*2*np.pi*fd*tt),"--",color="#5b7fbf",lw=1)
ax[1,1].set(xlabel="Time after drive stops (s)",ylabel="Stroke angle (deg, rest removed)",title=f"d  Free decay: {fd:.1f} Hz, damping ratio {z:.2f}")
json.dump(dict(f_damped=fd,zeta=z,peaks=yp.tolist()),open(OUT+"/decay.json","w"))
print("decay",fd,z,len(yp))
fig.tight_layout(); fig.savefig(OUT+"/R_flap.png",dpi=200); fig.savefig(OUT+"/svg/R_flap.svg")
for r in rows: print(r)
