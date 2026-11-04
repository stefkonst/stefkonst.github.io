import numpy as np, cv2, json
from scipy.ndimage import map_coordinates, gaussian_filter1d
S=r"C:\Users\stefk\AppData\Local\Temp\claude\c--Users-stefk-OneDrive-Documents-PlatformIO-Projects-CopSlide\c3f935dd-3a36-42e9-8d1a-98953bd40102\scratchpad\ft"
A=np.load(S+r"\A.npy",mmap_mode='r'); z=np.load(S+r"\ref.npz"); A0,H,L=z['A0'],z['H'],z['L']
PXMM=8.24
def track(P0,P1,stations,half,ws,step=1.0,along=8,search=14,I0=None):
    """shift normal to line P0->P1 at each station, per frame, with respect to the first 0.3 s mean; positive = toward n"""
    P0=np.array(P0,float);P1=np.array(P1,float);Ln=np.linalg.norm(P1-P0);e=(P1-P0)/Ln;n=np.array([e[1],-e[0]])
    w=np.arange(ws[0],ws[1]+.1,step); out=[]
    aa=np.arange(-along,along+.1,2.)
    pts=[]
    for si in stations:
        c=P0+si*e
        X=c[0]+np.outer(aa,e)[:,0][:,None]+w[None,:]*n[0]
        Y=c[1]+np.outer(aa,e)[:,1][:,None]+w[None,:]*n[1]
        pts.append((X,Y))
    u=np.zeros((A.shape[0],len(stations)))
    ref=[gaussian_filter1d(map_coordinates(A0,[Y,X],order=1).mean(0),1.) for X,Y in pts]
    for f in range(A.shape[0]):
        I=np.asarray(A[f],np.float32)
        for j,(X,Y) in enumerate(pts):
            p=gaussian_filter1d(map_coordinates(I,[Y,X],order=1).mean(0),1.)
            r=ref[j]
            sh=np.arange(-search,search+1)
            # SSD over overlap, remove mean
            c=[]
            for d in sh:
                a=p[max(0,d):len(p)+min(0,d)]; b=r[max(0,-d):len(r)+min(0,-d)]
                c.append(((a-a.mean()-(b-b.mean()))**2).mean())
            c=np.array(c);m=int(np.argmin(c));
            if 0<m<len(c)-1: dd=sh[m]+.5*(c[m-1]-c[m+1])/(c[m-1]-2*c[m]+c[m+1])
            else: dd=sh[m]
            u[f,j]=dd*step
    return u,e,n
# TE: 11 stations between beads
st=np.linspace(40,450,14)
uTE,eTE,nTE=track((65,372),(458,88),st,8,(-30,30),search=14)
np.savez(S+r"\te.npz",u=uTE,st=st,e=eTE,n=nTE)
print('TE shift px, mean frames [0:60]',np.round(uTE[:60].mean(0),2)); print('H',np.round(uTE[1990:].mean(0),2))
# LE straight
stL=np.linspace(20,320,7)
uLE,eL,nL=track((55,165),(385,42),stL,8,(-14,14),search=5)
np.savez(S+r"\le.npz",u=uLE,st=stL,e=eL,n=nL); print('LE std',np.round(uLE.std(0),3))
