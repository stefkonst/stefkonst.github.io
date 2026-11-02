"""Fishing line as a trim on the pre-stretched sheet, no batten, root edge held: full-resolution check at three speeds. Output: line_check.json."""
import json,math,os,sys
import numpy as np
sys.path.insert(0,r"C:\Users\stefk\Documents\website\site\flap\full_membrane")
import membrane as M
U,d=float(sys.argv[1]),float(sys.argv[2])
ln=1.05**-.5+.04
w=M.Wing(U=U,lam_d=(1.05,1.05),lam_n=(ln,ln),w_s=(13e-3,13e-3))
le=np.array(sorted(set(int(n) for n in w.le_nodes)));C=int(w.fib[0][-1])
fx=np.zeros(w.x0.shape[0],bool);fx[np.r_[le,np.array(w.tip,int),np.array(w.root,int)]]=True;fx[C]=True
x=w.x0.copy();x[C,1]+=d*1e-3
r=w.relax(0.,x_init=x,fixed=fx,fixed_x=x);w.build_aero(r["x"])
a0=w.relax(0.,x_init=r["x"],fixed=fx,fixed_x=r["x"],aero=True);a1=w.relax(1600.,x_init=a0["x"],fixed=fx,fixed_x=r["x"],aero=True)
s0,s1=w.summary(a0["x"]),w.summary(a1["x"],1600.)
f=-w.reactions(r["x"])[C,:2]*1e3
o=dict(U=U,d=d,CL0=a0["CL"],CL1=a1["CL"],dCL=a1["CL"]-a0["CL"],ok=[bool(r["ok"]),bool(a0["ok"]),bool(a1["ok"])],line_mN=f.tolist(),N2=[s0[0]["N2"],s0[1]["N2"]],wr1=[s1[0]["wrinkled"],s1[1]["wrinkled"]],cv=[s0[0]["N1_cv"],s0[1]["N1_cv"]],sag=[w.fibre_report(r["x"])[k]["sag"]*1e3 for k in (0,1)])
print(json.dumps(o),flush=True)
