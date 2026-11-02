import cv2, numpy as np, sys, json
from scipy.ndimage import map_coordinates
D = sys.argv[1]; out = sys.argv[2]
def load(name, every=1):
    cap = cv2.VideoCapture(D + "/" + name); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        fr.append(cv2.GaussianBlur(255 - bh, (5, 5), 0).astype(np.float32))
    return np.array(fr)
A = load("5Hz_0kV.avi"); B = load("5Hz_900V.avi")
fps = len(A) / 5.0
th = np.radians(np.arange(-35, 80, 0.25)); r = np.arange(150, 330, 2.0)
def ray_score(img, px, py):
    xs = px + np.outer(np.cos(th), r); ys = py - np.outer(np.sin(th), r)
    v = map_coordinates(img, [ys, xs], order=1, mode="constant", cval=255)
    return v.mean(1)
# pivot: grid search maximising rod darkness over sample frames
samp = A[::37][:12]; best = None
for px in range(90, 130, 4):
    for py in range(310, 360, 4):
        s = sum(ray_score(f, px, py).min() for f in samp)
        if best is None or s < best[0]: best = (s, px, py)
_, px, py = best
for dx in []:
    for dy in []:
        s = sum(ray_score(f, px + dx, py + dy).min() for f in samp)
        if s < best[0]: best = (s, px + dx, py + dy)
_, px, py = best
print("pivot", px, py)
def pick(sc):
    # the LE rod is the uppermost strong dark line: highest angle among local minima close to the global minimum
    lo, med = sc.min(), np.median(sc)
    ok = [i for i in range(1, len(sc) - 1) if sc[i] <= sc[i - 1] and sc[i] <= sc[i + 1] and sc[i] < lo + 0.35 * (med - lo)]
    return th[max(ok)] if ok else th[np.argmin(sc)]
def angles(F, start=20, win=np.radians(5.0)):
    # temporal continuity: global detection at a reliable frame (mid-upstroke), then search +-5 deg around the previous angle
    out = np.zeros(len(F)); out[start] = th[np.argmin(ray_score(F[start], px, py))]
    for order in (range(start + 1, len(F)), range(start - 1, -1, -1)):
        prev = out[start]
        for i in order:
            sc = ray_score(F[i], px, py); m = np.abs(th - prev) <= win
            prev = th[m][np.argmin(sc[m])]; out[i] = prev
    return np.degrees(out)
chk = []
for i in range(0, 80, 10):
    im = cv2.cvtColor(A[i].astype(np.uint8), cv2.COLOR_GRAY2BGR); a = pick(ray_score(A[i], px, py))
    cv2.line(im, (int(px), int(py)), (int(px + 600 * np.cos(a)), int(py - 600 * np.sin(a))), (0, 0, 255), 2)
    chk.append(cv2.resize(im, (480, 300)))
cv2.imwrite(out + "/kin_check.png", np.vstack([np.hstack(chk[:4]), np.hstack(chk[4:])]))
phA, phB = angles(A), angles(B)
t = np.arange(len(A)) / fps
res = {}
for lab, ph in (("0kV", phA), ("900V", phB)):
    F = np.abs(np.fft.rfft(ph - ph.mean())); fq = np.fft.rfftfreq(len(ph), 1 / fps); f0 = fq[np.argmax(F[1:]) + 1]
    cyc = int(round(fps / f0)); n = len(ph) // cyc
    C = ph[:n * cyc].reshape(n, cyc)
    res[lab] = dict(f_Hz=float(f0), amp_pp_deg=float(np.mean(C.max(1) - C.min(1))), amp_pp_sd=float(np.std(C.max(1) - C.min(1))),
                    max_deg=float(C.max(1).mean()), min_deg=float(C.min(1).mean()), mean_deg=float(ph.mean()),
                    down_frac=None, cycles=n)
    # downstroke fraction: time from max to next min
    i_max = np.argmax(C, 1); i_min = np.argmin(C, 1)
    res[lab]["down_frac"] = float(np.mean(((i_min - i_max) % cyc) / cyc))
print(json.dumps(res, indent=1))
np.savetxt(out + "/phi.txt", np.c_[t, phA, phB], fmt="%.4f")
# shape test: phase-matched frames, membrane region mask = around the wing; compare within-0kV cycle scatter vs 0kV-vs-900V
cyc = int(round(fps / res["0kV"]["f_Hz"]))
def phase_frames(ph, F, target, rising):
    d = np.gradient(ph); idx = [i for i in range(2, len(ph) - 2) if abs(ph[i] - target) < 1.0 and (d[i] > 0) == rising]
    sel = []
    for i in idx:
        if not sel or i - sel[-1] > cyc // 2: sel.append(i)
    return [F[i] for i in sel]
rows = []
for target in (0.0, 20.0, 40.0):
    for rising in (True, False):
        fa, fb = phase_frames(phA, A, target, rising), phase_frames(phB, B, target, rising)
        if len(fa) < 4 or len(fb) < 4: continue
        ma, mb = np.mean(fa, 0), np.mean(fb, 0)
        within = np.mean([np.abs(f - ma).mean() for f in fa])
        half = np.abs(np.mean(fa[::2], 0) - np.mean(fa[1::2], 0)).mean()
        between = np.abs(ma - mb).mean()
        rows.append(dict(phi=target, stroke="up" if rising else "down", nA=len(fa), nB=len(fb), within_single=float(within), split_half_0kV=float(half), between_0kV_900V=float(between)))
        if target == 0.0 and not rising:
            cv2.imwrite(out + "/shape_0kV_vs_900V.png", np.hstack([ma, mb, np.clip(np.abs(ma - mb) * 8, 0, 255)]).astype(np.uint8)[::2, ::2])
print(json.dumps(rows, indent=1))
json.dump(dict(pivot=[px, py], kin=res, shape=rows), open(out + "/kin.json", "w"), indent=1)
