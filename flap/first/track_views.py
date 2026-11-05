"""Second set of videos, 5 Nov: side, front and diagonal views. Tracks the trailing edge (roll) and leading edge across their own direction at 200 fps.
Run: python track_views.py <folder with the avi files> <output folder> -> <output>/<run>.npz, <output>/check_<run>.png"""
import glob, multiprocessing as mp, os, struct, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d, map_coordinates

# rough lines in the image, refined for each run
LINES = dict(front=dict(te=[(310, 468), (345, 448), (400, 425), (470, 400), (540, 376), (600, 357)], le=[(320, 309), (570, 308)], reach=26, half=16, search=12),
             side=dict(te=[(310, 365), (400, 365), (500, 365), (560, 364), (595, 360)], le=[(330, 342), (590, 354)], reach=4, half=9, search=20),
             side_tension=dict(te=[(310, 344), (400, 345), (500, 344), (580, 341)], le=[(350, 322), (500, 328), (590, 336)], reach=4, half=9, search=20),
             diag=dict(te=[(300, 378), (350, 367), (400, 360), (450, 355), (500, 348), (550, 340), (590, 333)], le=[(310, 298), (400, 301), (500, 303), (550, 306)], reach=5, half=13, search=10),
             diag_tension=dict(te=[(300, 373), (350, 367), (400, 362), (450, 355), (500, 349), (550, 343), (590, 338)], le=[(310, 302), (400, 308), (500, 312), (550, 315)], reach=5, half=13, search=10))
NT, NL = 16, 8


class Stack:
    """8-bit uncompressed AVI, frames found by walking the RIFF chunks."""

    def __init__(self, f):
        c = cv2.VideoCapture(f)
        self.h, self.w = int(c.get(4)), int(c.get(3))
        ok, a = c.read(); c.release()
        self.b = np.memmap(f, dtype=np.uint8, mode="r")
        sz = self.w * self.h
        off = []
        fh = open(f, "rb")

        def walk(start, end):
            pos = start
            while pos + 8 <= end:
                fh.seek(pos)
                cid = fh.read(4); size = struct.unpack("<I", fh.read(4))[0]
                if cid in (b"RIFF", b"LIST"):
                    fh.read(4)
                    walk(pos + 12, min(pos + 8 + size, end))
                elif cid[2:] in (b"db", b"dc") and size == sz:
                    off.append(pos + 8)
                pos += 8 + size + (size & 1)
        walk(0, len(self.b))
        fh.close()
        self.off = off
        g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        first = np.asarray(self[0], float)
        self.flip = np.abs(first - g).mean() > np.abs(first[::-1] - g).mean()

    def __len__(self):
        return len(self.off)

    def __getitem__(self, i):
        a = self.b[self.off[i]:self.off[i] + self.w * self.h].reshape(self.h, self.w)
        return a[::-1] if getattr(self, "flip", False) else a


def stations(line, n):
    """n points along a polyline, with the unit normal at each (pointing up in the image)."""
    P = np.array(line, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    q = np.linspace(0, s[-1], n)
    c = np.c_[np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])]
    e = np.gradient(c, axis=0)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return c, e, np.c_[e[:, 1], -e[:, 0]]


def refine(I, c, e, n, reach):
    """Move each station across the line to where the image varies most, then smooth along the line."""
    w = np.arange(-reach, reach + 1.)
    d = np.zeros(len(c))
    for j in range(len(c)):
        aa = np.arange(-8, 9, 2.)[:, None]
        X = c[j, 0] + aa * e[j, 0] + w[None, :] * n[j, 0]
        Y = c[j, 1] + aa * e[j, 1] + w[None, :] * n[j, 1]
        p = gaussian_filter1d(map_coordinates(I, [Y, X], order=1).mean(0), 1.)
        g = gaussian_filter1d(np.abs(np.gradient(p)), 2.)
        d[j] = w[np.argmax(g)]
    d = np.median(np.c_[d, np.roll(d, 1), np.roll(d, -1)], axis=1)
    return c + d[:, None] * n


def track(A, c, e, n, half, search):
    """Shift across the line, in px, of the profile within +-half of each station, against the mean of the first 40 frames. Positive along n."""
    w = np.arange(-half - search, half + search + 1.)
    aa = np.arange(-8, 9, 2.)[:, None]
    pts = [(c[j, 0] + aa * e[j, 0] + w[None, :] * n[j, 0], c[j, 1] + aa * e[j, 1] + w[None, :] * n[j, 1]) for j in range(len(c))]
    I0 = np.mean([np.asarray(A[i], np.float32) for i in range(40)], 0)
    L = 2 * half + 1
    ref = [gaussian_filter1d(map_coordinates(I0, [Y, X], order=1).mean(0), 1.)[search:search + L] for X, Y in pts]
    sh = np.arange(-search, search + 1)
    u = np.zeros((len(A), len(c)), np.float32)
    for f in range(len(A)):
        I = np.asarray(A[f], np.float32)
        for j, (X, Y) in enumerate(pts):
            p = gaussian_filter1d(map_coordinates(I, [Y, X], order=1).mean(0), 1.)
            b = ref[j] - ref[j].mean()
            m_ = []
            for d in sh:
                a = p[search + d:search + d + L]
                m_.append(((a - a.mean() - b) ** 2).mean())
            m_ = np.array(m_)
            m = int(np.argmin(m_))
            u[f, j] = sh[m] + (.5 * (m_[m - 1] - m_[m + 1]) / (m_[m - 1] - 2 * m_[m] + m_[m + 1]) if 0 < m < len(m_) - 1 else 0.)
    return u


def run(a):
    f, out = a
    nm = os.path.basename(f)[7:-4]
    view = next(v for v in ("front", "side", "diag") if v in nm)
    L = LINES[view + "_tension"] if "tension" in nm and view != "front" else LINES[view]
    A = Stack(f)
    I0 = np.mean([np.asarray(A[i], np.float32) for i in range(40)], 0)
    r = {}
    vis = cv2.cvtColor(np.clip(I0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for k, N, col, half in (("te", NT, (0, 200, 255), L["half"]), ("le", NL, (0, 255, 0), 8)):
        c, e, n = stations(L[k], N)
        c = refine(I0, c, e, n, L["reach"])
        u = track(A, c, e, n, half, L["search"] if k == "te" else 8)
        r[k + "_c"], r[k + "_n"], r[k + "_u"] = c, n, u
        for p in c:
            cv2.circle(vis, tuple(np.round(p).astype(int)), 3, col, 1, cv2.LINE_AA)
    np.savez_compressed(os.path.join(out, nm + ".npz"), **r)
    cv2.imwrite(os.path.join(out, "check_%s.png" % nm), vis)
    return nm


if __name__ == "__main__":
    D, O = sys.argv[1:3]
    os.makedirs(O, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(D, "*.avi")))
    if len(sys.argv) > 3:
        fs = [x for x in fs if sys.argv[3] in x]
    with mp.Pool(13) as pool:
        for nm in pool.imap_unordered(run, [(x, O) for x in fs]):
            print(nm, flush=True)
    print("DONE", flush=True)
