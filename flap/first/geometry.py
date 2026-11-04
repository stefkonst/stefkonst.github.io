"""Wing outline and trailing edge shape from the video, in mm, from the ruler in the frame (8.24 px/mm).
Needs the frame stack: A.npy (2001, 480, 640), ref.npz. Run: python geometry.py <stack folder> -> data/geometry.json"""
import json, os, sys
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d, gaussian_filter
HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1]
z = np.load(os.path.join(D, "ref.npz"))
PX = 8.24
A0, H, L = z["A0"], z["H"], z["L"]
TIP, ROOT = np.array([458., 88.]), np.array([22., 410.])     # bead at the tip, knot at the root end of the roll
Ln = np.linalg.norm(TIP - ROOT)
e = (TIP - ROOT) / Ln
n = np.array([e[1], -e[0]])                                   # towards the leading edge
st = np.arange(0, Ln + 1, 4.0)
w = np.arange(-26, 34, 0.25)


def centre(I):
    """Roll position at each station: centroid of the deviation from a straight background, within the roll."""
    X = ROOT[0] + np.outer(st, e)[:, 0][:, None] + w[None, :] * n[0]
    Y = ROOT[1] + np.outer(st, e)[:, 1][:, None] + w[None, :] * n[1]
    P = gaussian_filter(map_coordinates(I, [Y, X], order=1), (2.5, 2))
    out, wid = [], []
    for p in P:
        # background from outer 6 px each side, straight line
        b = np.polyval(np.polyfit(np.r_[w[:24], w[-24:]], np.r_[p[:24], p[-24:]], 1), w)
        d = np.abs(p - b)
        d[d < 0.35 * d.max()] = 0
        c = (d * w).sum() / d.sum()
        idx = np.where(d > 0)[0]
        out.append(c)
        wid.append(w[idx[-1]] - w[idx[0]])
    return np.array(out), np.array(wid)


res = {}
for k, I in (("before", A0), ("high", H), ("low", L)):
    c, wd = centre(I)
    res[k] = dict(sag_mm=(gaussian_filter1d(c, 2) / PX).tolist(), width_mm=(wd / PX).tolist())
s = st / PX
pts = dict(LE_root=(55, 165), LE_knee=(385, 42), tip=(478, 82), TE_tip=(458, 88), TE_root=(65, 372), knot=(22, 410))
u = np.array([330., -123.]); u /= np.linalg.norm(u)
v = np.array([-u[1], u[0]])
lay = {k: [float((np.array(p) - pts["LE_root"]) @ u / PX), float((np.array(p) - pts["LE_root"]) @ v / PX)] for k, p in pts.items()}
out = dict(px_per_mm=PX, s_mm=s.tolist(), length_mm=float(Ln / PX), points_px=pts, points_mm_along_LE=lay, **res)
json.dump(out, open(os.path.join(HERE, "data", "geometry.json"), "w"), indent=1)
for k in res:
    a = np.array(res[k]["sag_mm"])
    i = (s > 15) & (s < 58)
    print(k, "max sag %.2f at %.1f mm, roll width median %.2f" % (a[i].max(), s[i][np.argmax(a[i])], np.median(np.array(res[k]["width_mm"])[i])))
print(lay, "knot to tip", Ln / PX)
print(np.round(np.array(res["before"]["sag_mm"])[::10], 2))
print(np.round(np.array(res["high"]["sag_mm"])[::10], 2))
print(np.round(s[::10], 1))
