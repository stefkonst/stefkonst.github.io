"""Actuated against non-actuated outlines on the video frame, for the 'b' panels of the full figures.
For each case: a faded frame, with the wing's outline (dark region against the background, smoothed) drawn at two
frames: at rest / at one end of the stroke (blue) and at peak actuation / the other end (amber or maroon).
Run: python overlays.py -> blocks/frames/ov_*.png"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FLAP = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(FLAP, "first")]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from track_views import Stack
VID = r"C:/Users/stefk/MIT Dropbox/SMRL/Shared/Stefania/full_tension/"
OUT = os.path.join(HERE, "blocks", "frames"); os.makedirs(OUT, exist_ok=True)
TR = lambda n: np.load(os.path.join(FLAP, "flapping", "data", "tracks", n + ".npz"))
BLUE, AMBER, MAROON = "#2563eb", "#f59e0b", "#9f1239"


def bg_of(s, idx):
    return np.percentile(np.stack([np.asarray(s[i], np.float32) for i in idx]), 97, axis=0)


def mask(s, k, bg, thr=30):
    from scipy.ndimage import label, binary_fill_holes, binary_closing
    d = gaussian_filter(bg - np.asarray(s[k], np.float32), 1.5)
    m = binary_closing(d > thr, iterations=2); m = binary_fill_holes(m)
    lab, n = label(m)
    if n:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        m = np.isin(lab, np.where(sizes > 600)[0])
    return m


def peak_frame(s, k0, k1, roi, rest):
    y0, y1, x0, x1 = roi; best, kb = -1, k0
    for k in range(k0, k1, 2):
        v = np.abs(np.asarray(s[k], np.float32)[y0:y1, x0:x1] - rest[y0:y1, x0:x1]).mean()
        if v > best: best, kb = v, k
    return kb


def draw(name, s, kA, kB, bg, crop, labels, cols, px_mm=0.152, bar=5, fname=None, kBG=None):
    y0, y1, x0, x1 = crop
    base = np.asarray(s[kA if kBG is None else kBG], float)[y0:y1, x0:x1]
    h, w = base.shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(base, cmap="gray", vmin=np.percentile(base, 1) - 40, vmax=np.percentile(base, 99.5) + 30, alpha=1.0)
    ax.imshow(np.ones_like(base), cmap="gray", vmin=0, vmax=1, alpha=0.35)
    for k, col in ((kA, cols[0]), (kB, cols[1])):
        m = mask(s, k, bg)[y0:y1, x0:x1].astype(float)
        ax.contour(gaussian_filter(m, 1.2), levels=[0.5], colors=[col], linewidths=1.6)
    for i, (lab, col) in enumerate(zip(labels, cols)):
        ax.text(0.03, 0.92 - i * 0.10, lab, color=col, fontsize=11, fontweight="bold", transform=ax.transAxes, family="Arial")
    L = bar / px_mm
    ax.plot([w * 0.72, w * 0.72 + L], [h * 0.92] * 2, color="k", lw=2); ax.text(w * 0.72 + L / 2, h * 0.88, f"{bar} mm", color="k", fontsize=10, ha="center", family="Arial")
    out = os.path.join(OUT, (fname or name) + ".png"); fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    # Fig. 2: bench, side, 1.8 kV, 1 Hz: rest against peak
    n = "2layer_1800V_1hz_side"; s = Stack(VID + n + ".avi")
    bg = bg_of(s, range(0, len(s), 10)); rest = np.asarray(s[20], np.float32)
    kp = peak_frame(s, 100, 600, (300, 430, 300, 620), rest)
    draw(n, s, 20, kp, bg, (250, 470, 260, 660), ["0 V", "1.8 kV, peak"], [BLUE, AMBER], bar=5, fname="ov_bench_side")
    # Fig. 3: bench, front, 1.8 kV: rest against peak
    n = "2layer_1800V_1hz_front_avi"; s = Stack(VID + n + ".avi")
    bg = bg_of(s, range(0, len(s), 10)); rest = np.asarray(s[20], np.float32)
    kp = peak_frame(s, 100, 600, (330, 500, 300, 620), rest)
    draw(n, s, 20, kp, bg, (260, 520, 260, 660), ["0 V", "1.8 kV, peak"], [BLUE, AMBER], bar=5, fname="ov_bench_front")
    # Figs. 4, 7, 8: robot, ends of the stroke (from the angle tracks)
    for n, crop, fname, cols, labs in (
            ("full_DEA_core_wing_8hz_1200V_side", (150, 480, 180, 760), "ov_robot_side", [BLUE, AMBER], ["top of stroke", "bottom of stroke"]),
            ("full_DEA_core_wing_8hz_1200V_diag_good", (130, 470, 200, 760), "ov_robot_diag", [BLUE, AMBER], ["top of stroke", "bottom of stroke"])):
        s = Stack(VID + n + ".avi"); a = TR(n)["ang"]; g = np.arange(len(a)) > 200
        kmin = int(np.nanargmin(np.where(g, a, np.inf))); kmax = int(np.nanargmax(np.where(g, a, -np.inf)))
        bg = bg_of(s, range(0, len(s), 7))
        draw(n, s, kmin, kmax, bg, crop, labs, cols, bar=10, fname=fname)
    # Fig. 7: bench, diagonal, 1.8 kV: rest against peak (the camber change)
    n = "2layer_1800V_1hz_diag_sine"; s = Stack(VID + n + ".avi")
    bg = bg_of(s, range(0, len(s), 10)); rest = np.asarray(s[20], np.float32)
    kp = peak_frame(s, 100, 700, (280, 460, 280, 640), rest)
    draw(n, s, 20, kp, bg, (220, 500, 240, 680), ["0 V", "1.8 kV, peak"], [BLUE, AMBER], bar=5, fname="ov_bench_diag")
    # powered against unpowered, side view, same scale: both at the bottom of the stroke
    for n, fname in (("full_DEA_core_wing_8hz_1200V_side", "ov_pw"), ("only_DEA_core_8hz_1200V_side", "ov_un")):
        s = Stack(VID + n + ".avi"); a = TR(n)["ang"]; g = np.arange(len(a)) > 200
        kmin = int(np.nanargmin(np.where(g, a, np.inf))); kmax = int(np.nanargmax(np.where(g, a, -np.inf)))
        bg = bg_of(s, range(0, len(s), 7)); px = 0.152 if "full" in n else 0.135
        draw(n, s, kmin, kmax, bg, (100, 500, 150, 790), ["top", "bottom"], [BLUE, AMBER if "full" in n else MAROON], px_mm=px, bar=10, fname=fname)
