"""Image figures built straight from the high-speed video (same style as Fig. 1 = Fig0_robot):
Fig8_cycle   one stroke cycle at 8 Hz, 1200 V, wing powered against unpowered: leading edge and lower outline at 12 phases
             over a faded frame, in mm from the hinge; stroke of every cycle; the mean cycle.
Fig9_sweep   swept area (brightest minus darkest pixel) at 1, 5, 7, 8, 9, 10 and 15 Hz, 1200 V, wing powered,
             and 8 Hz with the wing unpowered, same scale and colour range.
Run: python figures_video.py (needs the videos in MIT Dropbox SMRL/Shared/Stefania/full_tension)."""
import os, sys, json, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FLAP = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.expanduser("~"), "Documents", "paper_tension", "common"), os.path.join(FLAP, "first"), os.path.join(FLAP, "flapping", "code")]
import style as ST
import matplotlib.pyplot as plt
from matplotlib import cm
from track_views import Stack
VID = r"C:/Users/stefk/MIT Dropbox/SMRL/Shared/Stefania/full_tension/"
TR = lambda n: np.load(os.path.join(FLAP, "flapping", "data", "tracks", n + ".npz"))
PAIR = json.load(open(os.path.join(FLAP, "flapping", "data", "pair.json")))
OUT = [os.path.join(HERE, "fig"), os.path.join(os.path.expanduser("~"), "Documents", "YEAR2", "robot", "results")]
FS = 200.0
ST.apply()


def save(fig, name):
    fig.savefig(os.path.join(OUT[0], name + ".png"), dpi=300); fig.savefig(os.path.join(OUT[0], name + ".svg"))
    fig.savefig(os.path.join(OUT[0], name + ".ai"), format="pdf")
    os.makedirs(os.path.join(OUT[1], "svg"), exist_ok=True)
    shutil.copy2(os.path.join(OUT[0], name + ".png"), OUT[1]); shutil.copy2(os.path.join(OUT[0], name + ".svg"), os.path.join(OUT[1], "svg"))
    plt.close(fig)


# ---------------- Fig. 8: one cycle, powered against unpowered ----------------
runs = (("full_DEA_core_wing_8hz_1200V_side", 0.152, "wing powered"), ("only_DEA_core_8hz_1200V_side", 0.135, "wing unpowered"))
fig = plt.figure(figsize=(ST.WIDTH_MM * ST.MM, 118 * ST.MM))
cmap = cm.twilight_shifted
for i, (n, mm, lab) in enumerate(runs):
    d = TR(n); s = Stack(VID + n + ".avi")
    hx, hy = PAIR[n]["hinge"] if "full" in n else PAIR["full_DEA_core_wing_8hz_1200V_side"]["hinge"]
    # hinge for the unpowered view: where the leading edge meets the body in the first frame (the crossing of its
    # lines is far off because its leading edge bends); take the left end of the tracked edge, extrapolated 10 mm
    xs = d["xs"].astype(float); top, bot = d["top"], d["bot"]
    if "only" in n:
        hx, hy = 305.0, 237.0          # leading edge leaves the body, read from frame 10
    k0 = int(1.0 * FS); N = 25
    frames = list(range(k0, k0 + N, 2))
    ax = fig.add_axes([0.04 + i * 0.49, 0.42, 0.45, 0.54])
    bg = np.asarray(s[k0], float)
    ext = [(-hx) * mm, (bg.shape[1] - hx) * mm, (bg.shape[0] - hy) * mm, (-hy) * mm]
    ax.imshow(bg, cmap="gray", extent=ext, alpha=0.35, vmin=0, vmax=255)
    for j, k in enumerate(frames):
        c = cmap(j / len(frames))
        m = np.isfinite(top[k]); ax.plot((xs[m] - hx) * mm, (top[k][m] - hy) * mm, color=c, lw=1.0)
        m2 = np.isfinite(bot[k]); ax.plot((xs[m2] - hx) * mm, (bot[k][m2] - hy) * mm, color=c, lw=0.6, ls=":")
    ax.set(xlim=(-5, 72), ylim=(22, -22), xlabel="mm from the hinge", ylabel="mm" if i == 0 else None)
    ax.set_aspect("equal")
    ax.text(0.02, 0.97, "ab"[i] + "  " + lab, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    ST.note(ax, "solid: leading edge, dotted: lower outline\n12 phases of one 8 Hz cycle", xy=(0.02, 0.10), va="bottom")
# c: stroke of every cycle; d: the mean cycle
ax = fig.add_axes([0.08, 0.08, 0.36, 0.25])
for x0, (n, col, lab) in enumerate((("full_DEA_core_wing_8hz_1200V_side", ST.C[0], "powered"), ("only_DEA_core_8hz_1200V_side", ST.C[1], "unpowered"))):
    pp = np.array(PAIR[n]["pp_all"]); jit = (np.random.default_rng(1).random(len(pp)) - 0.5) * 0.35
    ax.plot(x0 + jit, pp, "o", ms=2.5, color=col, alpha=0.8)
    ax.errorbar([x0 + 0.32], [pp.mean()], yerr=[pp.std()], fmt="s", color="k", ms=3)
ax.set_xticks([0, 1]); ax.set_xticklabels(["wing powered", "wing unpowered"]); ax.set_xlim(-0.5, 1.6)
ax.set(ylabel="Stroke per cycle, peak to peak (deg)"); ax.minorticks_off(); ax.yaxis.set_minor_locator(plt.NullLocator())
ST.note(ax, "8 Hz, 1200 V, side view, 35 cycles each", xy=(0.03, 0.06), va="bottom")
ax.text(-0.22, 1.02, "c", transform=ax.transAxes, fontsize=9, fontweight="bold")
ax = fig.add_axes([0.58, 0.08, 0.36, 0.25])
for n, col, lab in (("full_DEA_core_wing_8hz_1200V_side", ST.C[0], "powered"), ("only_DEA_core_8hz_1200V_side", ST.C[1], "unpowered")):
    a = TR(n)["ang"]; t = np.arange(len(a)) / FS; f = PAIR[n]["f"]; g = (t > 0.6) & np.isfinite(a)
    ph = (t[g] * f) % 1; y = a[g] - np.nanmean(a[g]); b = np.linspace(0, 1, 26); idx = np.digitize(ph, b) - 1
    A = np.array([np.mean(y[idx == k]) for k in range(25)]); S = np.array([np.std(y[idx == k]) for k in range(25)])
    sh = int(np.argmin(A)); A = np.roll(A, -sh); S = np.roll(S, -sh); x = (b[:-1] + b[1:]) / 2
    ax.plot(x, A, color=col, label=lab); ax.fill_between(x, A - S, A + S, color=col, alpha=0.25, lw=0)
ax.set(xlabel="Phase from the lowest point", ylabel="Stroke angle (deg, mean removed)")
ax.legend(loc="upper left", fontsize=5.5)
ax.text(-0.22, 1.02, "d", transform=ax.transAxes, fontsize=9, fontweight="bold")
save(fig, "Fig8_cycle")

# ---------------- Fig. 9: swept area across the sweep ----------------
sweep = [("full_DEA_core_wing_1hz_1200v_side", "1 Hz", 0.152), ("full_DEA_core_wing_5hz_1200v_side", "5 Hz", 0.152), ("full_DEA_core_wing_7hz_1200V_side", "7 Hz", 0.152),
         ("full_DEA_core_wing_8hz_1200V_side", "8 Hz", 0.152), ("full_DEA_core_wing_9hz_1200V_side", "9 Hz", 0.152), ("full_DEA_core_wing_10hz_1200V_side", "10 Hz", 0.152),
         ("full_DEA_core_wing_15hz_1200v_side", "15 Hz", 0.152), ("only_DEA_core_8hz_1200V_side", "8 Hz, wing unpowered", 0.135)]
SW = {r["run"]: r for r in json.load(open(os.path.join(FLAP, "flapping", "data", "sweepfit.json")))["runs"]}
fig = plt.figure(figsize=(ST.WIDTH_MM * ST.MM, 78 * ST.MM))
import cv2
for i, (n, lab, mm) in enumerate(sweep):
    s = Stack(VID + n + ".avi"); N = len(s)
    k1 = N if "1hz" not in n else 600
    A = np.stack([np.asarray(s[k], np.float32) for k in range(int(0.6 * FS), k1, 2)])
    rg = A.max(0) - A.min(0)
    if mm != 0.152:   # bring the unpowered view to the same scale, hinge on hinge (305, 237) -> (323, 269)
        q = mm / 0.152; small = cv2.resize(rg, None, fx=q, fy=q)
        canvas = np.zeros_like(rg); dx, dy = int(round(323 - 305 * q)), int(round(269 - 237 * q))
        h, w = small.shape; canvas[dy:dy + h, dx:dx + w] = small[:600 - dy, :832 - dx]; rg = canvas
    r, c = divmod(i, 4)
    ax = fig.add_axes([0.005 + c * 0.249, 0.51 - r * 0.49, 0.245, 0.46])
    ax.imshow(rg[150:480, 230:760], cmap="magma", vmin=0, vmax=170)
    ax.set_axis_off()
    pp = SW[n]["A"][0] * 2 if n in SW else np.nan
    ax.text(6, 22, "%s  %s" % ("abcdefgh"[i], lab), color="w", fontsize=7, fontweight="bold")
    ax.text(6, 318, "stroke %.0f° (fundamental, pk-pk)" % pp, color="w", fontsize=5.5)
    if i == 0:
        ax.plot([440, 440 + 10 / 0.152], [300, 300], color="w", lw=2); ax.text(440 + 5 / 0.152, 290, "10 mm", color="w", fontsize=5.5, ha="center")
save(fig, "Fig9_sweep")
print("done")
