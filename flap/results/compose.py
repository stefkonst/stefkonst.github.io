"""Full journal figures: schematic + video frames + data panels, one lettering sequence.
Top row: the schematic block (a) and a video frame (b); below: the data panels (lettered from c, or b, by the scripts).
Fig. 1 is built from the robot schematic, the three camera views, the motion range and the system diagram.
Inputs: blocks/png/*.png (render_blocks.py), fig/*.png (figures_*.py), videos in MIT Dropbox.
Run: python compose.py -> fig/Figure1.png ... Figure8.png (+ .pdf)"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FLAP = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.expanduser("~"), "Documents", "paper_tension", "common"), os.path.join(FLAP, "first")]
import style as ST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from track_views import Stack
ST.apply()
VID = r"C:/Users/stefk/MIT Dropbox/SMRL/Shared/Stefania/full_tension/"
BP = lambda n: os.path.join(HERE, "blocks", "png", n + ".png")
FP = lambda n: os.path.join(HERE, "fig", n + ".png")
W = ST.WIDTH_MM * ST.MM
CACHE = os.path.join(HERE, "blocks", "frames"); os.makedirs(CACHE, exist_ok=True)


def frame(name, k=0, crop=(60, 540, 0, 790), mode="frame", nmax=1000):
    """A video frame (or the motion range of a run), cached as PNG."""
    tag = f"{name}_{mode}_{k}.png"; cp = os.path.join(CACHE, tag)
    if os.path.exists(cp): return mpimg.imread(cp)
    s = Stack(VID + name + ".avi")
    if mode == "frame":
        im = np.asarray(s[k], float)
    else:
        A = np.stack([np.asarray(s[j], np.float32) for j in range(120, min(len(s), nmax), 3)]); im = A.max(0) - A.min(0)
    y0, y1, x0, x1 = crop; im = im[y0:y1, x0:x1]
    lo, hi = np.percentile(im, (0.5, 99.7)); im = np.clip((im - lo) / (hi - lo), 0, 1)
    plt.imsave(cp, im, cmap="gray" if mode == "frame" else "magma"); return mpimg.imread(cp)


def place(fig, img, rect, letter=None, title=None, scale_mm=None, px_mm=0.152, crop_w=None):
    ax = fig.add_axes(rect); ax.imshow(img); ax.set_axis_off()
    if letter: fig.text(rect[0] - 0.004, rect[1] + rect[3] + 0.004, letter, fontsize=9, fontweight="bold", va="bottom", ha="left")
    if title: ax.text(0.02, 0.96, title, transform=ax.transAxes, color="w", fontsize=6.5, va="top", family="Arial")
    if scale_mm:
        h, w = img.shape[:2]; L = scale_mm / px_mm
        ax.plot([w * 0.70, w * 0.70 + L], [h * 0.90] * 2, color="w", lw=1.6); ax.text(w * 0.70 + L / 2, h * 0.86, f"{scale_mm} mm", color="w", fontsize=5.5, ha="center", family="Arial")
    return ax


def compose(name, block, photo, data, photo_title=None, photo_scale=10, split=0.64, photo_px=0.152):
    """Top row: block (a) + photo (b), bottom: data figure (already lettered c...)."""
    B = mpimg.imread(BP(block)); D = mpimg.imread(FP(data))
    bw, bh = split, split * B.shape[0] / B.shape[1]                       # widths as fraction of figure width
    pw = 1 - split - 0.03; ph = pw * photo.shape[0] / photo.shape[1]
    top = max(bh, ph); dh = D.shape[0] / D.shape[1]
    H = W * (top + dh + 0.07)
    fig = plt.figure(figsize=(W, H))
    f = lambda v: v * W / H                                                # width units -> height fraction
    y_top = 1 - f(top) - f(0.035)
    place(fig, B, [0.0, y_top + f(top - bh) / 2, bw, f(bh)], "a")
    place(fig, photo, [split + 0.025, y_top + f(top - ph) / 2, pw, f(ph)], "b", photo_title, photo_scale, photo_px)
    axd = fig.add_axes([0.0, 0.0, 1.0, f(dh)]); axd.imshow(D); axd.set_axis_off()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig", f"{name}.{ext}"), dpi=300)
    plt.close(fig); print("wrote", name)


def compose_b(name, block, data, split=1.0):
    """Block (a) on top across the width, data figure (lettered from b) below."""
    B = mpimg.imread(BP(block)); D = mpimg.imread(FP(data))
    bh = B.shape[0] / B.shape[1] * 0.72; dh = D.shape[0] / D.shape[1]
    H = W * (bh + dh + 0.06); fig = plt.figure(figsize=(W, H)); f = lambda v: v * W / H
    place(fig, B, [0.14, 1 - f(bh) - f(0.035), 0.72, f(bh)], "a")
    axd = fig.add_axes([0.0, 0.0, 1.0, f(dh)]); axd.imshow(D); axd.set_axis_off()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig", f"{name}.{ext}"), dpi=300)
    plt.close(fig); print("wrote", name)


def figure1():
    B = mpimg.imread(BP("B1_robot")); S = mpimg.imread(BP("B1_system"))
    views = [frame("full_DEA_core_wing_1hz_1000V_side"), frame("full_DEA_core_wing_8hz_1200V_diag_good"), frame("full_DEA_core_wing_8hz_1400V")]
    rng = frame("full_DEA_core_wing_8hz_1200V_side", mode="range")
    bh = B.shape[0] / B.shape[1]; vh = (0.32) * views[0].shape[0] / views[0].shape[1]
    rw = 0.36; rh = rw * rng.shape[0] / rng.shape[1]; sw = 1 - rw - 0.03; sh = sw * S.shape[0] / S.shape[1]
    row3 = max(rh, sh)
    H = W * (bh + vh + row3 + 0.11); fig = plt.figure(figsize=(W, H)); f = lambda v: v * W / H
    y = 1 - f(bh) - f(0.035); place(fig, B, [0.0, y, 1.0, f(bh)], "a")
    y -= f(vh) + f(0.03)
    for i, (im, lab) in enumerate(zip(views, ("side", "diagonal", "top"))):
        place(fig, im, [0.01 + i * 0.335, y, 0.32, f(vh)], "bcd"[i], lab, 10)
    y -= f(row3) + f(0.03)
    place(fig, rng, [0.01, y + f(row3 - rh) / 2, rw, f(rh)], "e", "swept area, 8 Hz", 10)
    place(fig, S, [rw + 0.03, y + f(row3 - sh) / 2, sw, f(sh)], "f")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig", f"Figure1.{ext}"), dpi=300)
    plt.close(fig); print("wrote Figure1")


if __name__ == "__main__":
    figure1()
    bench_side = frame("2layer_1800V_1hz_side", 300)
    bench_front = frame("2layer_1800V_1hz_front_avi", 300)
    compose("Figure2", "B2_bench", bench_side, "Fig1_bench", "wing on the bench, side, 1.8 kV")
    compose("Figure3", "B3_mechanism", bench_front, "Fig2_mechanism", "rolled edge, front view")
    compose("Figure4", "B4_flapping", frame("full_DEA_core_wing_8hz_1200V_side", mode="range"), "Fig6_flapping", "swept area at resonance, 8 Hz")
    compose_b("Figure5", "B5_wing_state", "Fig8_cycle")
    compose_b("Figure6", "B6_model", "Fig11_model")
    compose("Figure7", "B7_air", frame("full_DEA_core_wing_8hz_1400V", 0), "Fig5_air", "the wing from above")
    compose("Figure8", "B8_flight", frame("full_DEA_core_wing_8hz_1200V_diag_good", mode="range"), "Fig7_flight", "swept area, diagonal, 8 Hz")
