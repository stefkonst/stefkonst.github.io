import sys,numpy as np,cv2
sys.path.insert(0,r"C:/Users/stefk/Documents/website/site/flap/first")
from track_views import Stack
D=r"C:/Users/stefk/MIT Dropbox/SMRL/Shared/Stefania/full_tension/"
n=sys.argv[1]; s=Stack(D+n+".avi"); N=len(s); print(n,N)
idx=np.arange(0,N,max(1,N//300))
A=np.stack([np.asarray(s[i],np.float32) for i in idx])
lo,hi=A.min(0),A.max(0)
out=np.c_[A[0],hi-lo]; out=np.clip(out,0,255).astype(np.uint8)
cv2.imwrite(n+"_range.png",out)
