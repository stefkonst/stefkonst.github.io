import sys,numpy as np,cv2
sys.path.insert(0,r"C:/Users/stefk/Documents/website/site/flap/first")
from track_views import Stack
D=r"C:/Users/stefk/MIT Dropbox/SMRL/Shared/Stefania/full_tension/"
out=[]
for n in sys.argv[1:]:
    d=np.load(n+".npz"); s=Stack(D+n+".avi"); a=d["ang"]
    ims=[]
    for k in (int(np.nanargmin(a[200:]))+200,int(np.nanargmax(a[200:]))+200):
        im=cv2.cvtColor(np.ascontiguousarray(s[k]),cv2.COLOR_GRAY2BGR)
        xs=d["xs"]; t=d["top"][k]
        for x,y in zip(xs,t):
            if np.isfinite(y): cv2.circle(im,(int(x),int(y)),2,(0,0,255),-1)
        cv2.putText(im,f"{n[:30]} f{k} {a[k]:.1f}deg",(10,30),0,0.7,(0,255,255),2)
        ims.append(im)
    out.append(np.hstack(ims))
cv2.imwrite("overlay.png",cv2.resize(np.vstack(out),None,fx=0.5,fy=0.5))
