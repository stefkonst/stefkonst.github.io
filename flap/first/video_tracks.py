"""Collect the tracks from the frame stack into one small file and make the clips. Run: python video_tracks.py <stack folder> -> data/tracks.json, clips/*.mp4"""
import json, os, subprocess, sys
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1]
PX, FPS = 8.24, 200.0
A = np.load(os.path.join(D, "A.npy"), mmap_mode="r")
te, le, bd = np.load(os.path.join(D, "te.npz")), np.load(os.path.join(D, "le.npz")), np.load(os.path.join(D, "beads.npz"))
u = te["u"] / PX
t = np.arange(len(u)) / FPS
mid = u[:, 5:9].mean(1)
ms = gaussian_filter1d(mid, 3)
on = t[np.where(ms > 5 * mid[:60].std())[0][0]]
p, _ = find_peaks(ms, distance=140, prominence=.05)
q, _ = find_peaks(-ms, distance=140, prominence=.05)
k = t > 2
X = [np.ones(k.sum()), t[k] - 6]
for h in (1, 2, 3):
    X += [np.sin(2 * np.pi * h * t[k]), np.cos(2 * np.pi * h * t[k])]
c = np.linalg.lstsq(np.c_[tuple(X)], mid[k], rcond=None)[0]
ph = np.arange(0, 1, 0.025)
tp = t[p][1:].mean() % 1
phase = ((t - tp + .5) % 1) - .5
pa = [float(mid[(t > 2) & (np.abs(((phase - x + .5) % 1) - .5) < .0125)].mean()) for x in ph - .5]
U = gaussian_filter1d(u, 3, axis=0)
d = bd["tip"] - bd["root"]
L0 = np.array([466 - 22, 80 - 410.]); e = L0 / np.linalg.norm(L0)
ends = gaussian_filter1d(d @ e, 4) / PX
out = dict(fps=FPS, px_per_mm=PX, noise_mm=float(mid[:60].std()), onset_s=float(on), t_max=t[p].tolist(), mid_max=ms[p].tolist(), t_min=t[q].tolist(), mid_min=ms[q].tolist(), mean_mm=float(c[0]), slope_mm_s=float(c[1]),
           amp_1Hz=float(np.hypot(c[2], c[3])), amp_2Hz=float(np.hypot(c[4], c[5])), amp_3Hz=float(np.hypot(c[6], c[7])), phase_avg_x=(ph - .5).tolist(), phase_avg=pa,
           s_mm=(te["st"] / PX).tolist(), u_high=U[p[1:]].mean(0).tolist(), u_low=U[q[1:]].mean(0).tolist(), LE_std_mm=(le["u"].std(0) / PX).tolist(),
           ends_high=float(ends[300:].max()), ends_low=float(ends[300:].min()), ends_before=float(ends[:50].mean()), mid_t=t[::2].tolist(), mid=[round(float(x), 4) for x in mid[::2]])
json.dump(out, open(os.path.join(HERE, "data", "tracks.json"), "w"))
print({a: out[a] for a in ("noise_mm", "onset_s", "mean_mm", "slope_mm_s", "amp_1Hz", "amp_2Hz", "amp_3Hz", "ends_before", "ends_high", "ends_low")}); print(np.round(out["u_high"], 2)); print(np.round(out["u_low"], 2))
# clips
Cl = os.path.join(HERE, "clips")
def enc(frames, name, fps):
    h, w = frames[0].shape[:2]
    pr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow", "-movflags", "+faststart", os.path.join(Cl, name)], stdin=subprocess.PIPE)
    for f in frames:
        pr.stdin.write(f.tobytes())
    pr.stdin.close(); pr.wait()
# raw first 3 s, real time at 50 fps, so the first voltage rise is visible
enc([np.asarray(A[i]) for i in range(0, 600, 4)], "first_3s.mp4", 50)
# one cycle averaged over 8 cycles, removes noise; shown beside the difference from the state before voltage, x4
A0 = np.asarray(A[:60], np.float32).mean(0)
fr = []
for b in range(50):
    idx = [int((2.0 + tp % 1 - .5 + n + b / 50) * FPS) for n in range(8)]
    idx = [i + j for i in idx for j in range(-1, 3) if i + j < len(A)]
    fr.append(np.asarray(A[idx], np.float32).mean(0))
cyc = []
for I in fr:
    d = cv2.GaussianBlur(I - A0, (0, 0), .8)
    cyc.append(np.hstack([np.clip(I, 0, 255), np.clip(127 + 5 * d, 0, 255)]).astype(np.uint8))
enc(cyc * 3, "cycle_averaged.mp4", 25)
for n in os.listdir(Cl):
    print(n, os.path.getsize(os.path.join(Cl, n)) // 1024, "kB")
