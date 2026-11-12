"""Schematics and model figures for B-cube (vector, drawn in matplotlib).
Fig10_schematic  a, the robot from the side (stroke plane): DEA core, lever, hinge, carbon leading edge, DEA skin,
                 rolled DEA edge tied to the body; b, the film stack and the rolled edge; c, block diagram of the system
                 and of the model.
Fig11_model      a, asymmetry of the stroke spring per run (harmonic balance); b, mean cycles, data and model; c,
                 harmonics, data and model; d, wing mode against edge tension (wing model), with the robot's fitted
                 mode and the force the roll loses at 1200 V.
Run after flapping/code/{harmonic_balance,edge_on_stroke,two_mode,wing_link}.py: python figures_model.py"""
import os, sys, json, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FLAP = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.expanduser("~"), "Documents", "paper_tension", "common")]
import style as ST
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle, FancyArrowPatch, Arc, Wedge
OUT = [os.path.join(HERE, "fig"), os.path.join(os.path.expanduser("~"), "Documents", "YEAR2", "robot", "results")]
FD = lambda n: json.load(open(os.path.join(FLAP, "flapping", "data", n)))
C = ST.C
BLUE, MAROON, INK, GREY, LIGHT = "#d1e0ff", "#750014", "#1a1a1a", "#7a7a7a", "#ececec"
SANS = {"family": "Arial"}


def save(fig, name):
    fig.savefig(os.path.join(OUT[0], name + ".png"), dpi=300); fig.savefig(os.path.join(OUT[0], name + ".svg"))
    fig.savefig(os.path.join(OUT[0], name + ".ai"), format="pdf")
    os.makedirs(os.path.join(OUT[1], "svg"), exist_ok=True)
    shutil.copy2(os.path.join(OUT[0], name + ".png"), OUT[1]); shutil.copy2(os.path.join(OUT[0], name + ".svg"), os.path.join(OUT[1], "svg"))
    plt.close(fig)


def lab(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 6.5); kw.setdefault("color", INK)
    ax.text(x, y, s, fontdict=SANS, **kw)


def arrow(ax, p0, p1, color=INK, lw=0.8, style="-|>", ms=6, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, shrinkA=0, shrinkB=0, **kw))


ST.apply()
# ======================= Fig. 10: schematic =======================
fig = plt.figure(figsize=(ST.WIDTH_MM * ST.MM, 205 * ST.MM))
# ---- a-c: the robot in three camera views ----
sys.path.insert(0, os.path.join(FLAP, "first"))
from track_views import Stack
VID = r"C:/Users/stefk/MIT Dropbox/SMRL/Shared/Stefania/full_tension/"
for i_, (n_, l_) in enumerate((("full_DEA_core_wing_1hz_1000V_side", "side"), ("full_DEA_core_wing_8hz_1200V_diag_good", "diagonal"), ("full_DEA_core_wing_8hz_1400V", "top"))):
    a_ = fig.add_axes([0.005 + i_ * 0.333, 0.7367, 0.325, 0.2583])
    try:
        im_ = np.asarray(Stack(VID + n_ + ".avi")[0], float)
    except Exception:
        im_ = np.zeros((600, 832))
    a_.imshow(im_[60:540, 0:790], cmap="gray"); a_.set_axis_off()
    a_.plot([600, 600 + 10 / 0.152], [445, 445], color="w", lw=2); a_.text(600 + 5 / 0.152, 430, "10 mm", color="w", fontsize=6, ha="center", fontdict=SANS)
    a_.text(10, 35, "abc"[i_], color="w", fontsize=9, fontweight="bold"); a_.text(45, 35, l_, color="w", fontsize=7, fontdict=SANS)
# ---- d: robot, side view ----
ax = fig.add_axes([0.0, 0.3439, 0.62, 0.3805]); ax.set_xlim(-34, 72); ax.set_ylim(-30, 30); ax.set_aspect("equal"); ax.axis("off")
ax.text(-33, 28, "d", fontsize=9, fontweight="bold")
ax.add_patch(FancyBboxPatch((-30, -9), 26, 14, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=INK, lw=0.9))   # frame
for yy, nm in ((0.5, "DEA core"), (-5.0, "spring core")):
    ax.add_patch(FancyBboxPatch((-27, yy - 1.6), 19, 3.2, boxstyle="round,pad=0.1,rounding_size=1.5", fc=BLUE, ec=MAROON, lw=0.8))
    for xx in np.linspace(-25, -10, 7): ax.plot([xx, xx], [yy - 1.5, yy + 1.5], color=MAROON, lw=0.4)
    lab(ax, -26.5, yy + 2.2, nm, fontsize=5.5, color=GREY)
hx, hy = 0.0, 0.0                                     # hinge
ax.plot([-8, -3], [0.5, 0.5], color=INK, lw=1.2); ax.plot([-3, hx], [0.5, hy], color=INK, lw=1.2)
ax.add_patch(Circle((hx, hy), 1.1, fc="white", ec=INK, lw=0.9)); ax.add_patch(Circle((hx, hy), 0.35, fc=INK, ec=INK))
lab(ax, -7.5, 2.0, "lever", fontsize=5.5, color=GREY); lab(ax, -1.0, -3.2, "hinge", fontsize=5.5, color=GREY)
R = 50.0; ax_y = -6.0                                  # anchor of the rolled edge on the body, off the stroke axis
for th, alpha, ls in ((-13, 0.25, "--"), (13, 0.25, "--"), (0, 1.0, "-")):
    t = np.radians(th); tip = (hx + R * np.cos(t), hy - R * np.sin(t))
    anc = (-3.0, ax_y)
    if ls == "-":
        mid = ((tip[0] + anc[0]) / 2, (tip[1] + anc[1]) / 2 - 1.8)
        xs = np.linspace(0, 1, 40); te = np.array([(1 - s) ** 2 * np.array(anc) + 2 * s * (1 - s) * np.array(mid) + s ** 2 * np.array(tip) for s in xs])
        ax.add_patch(Polygon(np.vstack([[hx, hy], tip, te[::-1]]), closed=True, fc=BLUE, ec="none", alpha=0.9))
        ax.plot(te[:, 0], te[:, 1], color=MAROON, lw=2.2, solid_capstyle="round")
        ax.plot([hx, tip[0]], [hy, tip[1]], color=INK, lw=2.0, solid_capstyle="round")
        ax.add_patch(Circle(anc, 0.7, fc=MAROON, ec=MAROON))
    else:
        ax.plot([hx, tip[0]], [hy, tip[1]], color=INK, lw=1.0, ls=ls, alpha=0.6)
ax.add_patch(Arc((hx, hy), 70, 70, theta1=-13, theta2=13, color=INK, lw=0.7))
arrow(ax, (35 * np.cos(np.radians(8)), 35 * np.sin(np.radians(8))), (35 * np.cos(np.radians(13)), 35 * np.sin(np.radians(13))), ms=5)
arrow(ax, (35 * np.cos(np.radians(-8)), 35 * np.sin(np.radians(-8))), (35 * np.cos(np.radians(-13)), 35 * np.sin(np.radians(-13))), ms=5)
lab(ax, 36.5, 7.0, r"stroke $\theta$", fontsize=6.5)
lab(ax, 40, 1.4, "carbon leading edge", fontsize=6)
lab(ax, 22, -2.6, "DEA skin", fontsize=6, color=INK)
lab(ax, 9, -13.8, r"rolled DEA edge, tension $T$", fontsize=6, color=MAROON)
lab(ax, -12.5, -11.5, "anchor", fontsize=5.5, color=MAROON)
ax.annotate("", xy=(-3.0, -0.2), xytext=(-3.0, ax_y + 0.8), arrowprops=dict(arrowstyle="<->", lw=0.6, color=GREY, shrinkA=0, shrinkB=0))
lab(ax, -2.4, -3.6, "e", fontsize=6, color=GREY, style="italic")
lab(ax, -29, -14.5, r"$V(t)$, sine squared, 0–1.4 kV", fontsize=6)
arrow(ax, (-22, -12.8), (-22, -9.6), ms=5)
ax.plot([5, 15], [-26, -26], color=INK, lw=1.2); lab(ax, 5.5, -25, "10 mm", fontsize=5.5)

# ---- b: film stack and rolled edge ----
ax = fig.add_axes([0.63, 0.3659, 0.36, 0.3439]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(0, 97, "e", fontsize=9, fontweight="bold")
layers = [("0.5 D", BLUE, 4), ("E", MAROON, 1.2), ("D", BLUE, 8), ("E", MAROON, 1.2), ("0.5 D", BLUE, 4)]
y = 58
for nm, col, h in layers:
    ax.add_patch(Rectangle((6, y), 48, h * 1.6, fc=col, ec="white", lw=0.3))
    lab(ax, 56, y + h * 0.8 - 1.6, nm, fontsize=5.5, color=GREY)
    y += h * 1.6
lab(ax, 6, 92, "skin: one active layer, about 60 µm", fontsize=6)
ax.annotate("", xy=(4, 58), xytext=(4, y), arrowprops=dict(arrowstyle="<->", lw=0.5, color=GREY))
lab(ax, 6, 47, "D dielectric (silicone)\nE electrode (carbon nanotubes)", fontsize=5.2, color=GREY)
# rolled edge: spiral
tt = np.linspace(0, 4.2 * np.pi, 600); rr = 2.6 + 2.2 * tt / (2 * np.pi)
cx, cy = 82, 72
ax.plot(cx + rr * np.cos(tt), cy + rr * np.sin(tt), color=BLUE, lw=3.2, solid_capstyle="round")
ax.plot(cx + rr * np.cos(tt), cy + rr * np.sin(tt), color=MAROON, lw=0.6)
lab(ax, 68, 92, "rolled edge", fontsize=6)
lab(ax, 68, 47, "23 mm strip, 20 mm\nactive, 2 active films;\ncarries the contacts", fontsize=5.2, color=GREY)
# voltage effect: tension drop
ax.add_patch(FancyBboxPatch((6, 8), 88, 34, boxstyle="round,pad=0.5,rounding_size=2", fc=LIGHT, ec="none"))
lab(ax, 9, 36, "at voltage", fontsize=6, fontweight="bold")
lab(ax, 9, 27, r"skin: Maxwell stress $p = \eta\,\varepsilon(V/H)^2$ thins and stretches it", fontsize=5.6)
lab(ax, 9, 19, r"edge: loses $\Delta F = \eta_f n_f \varepsilon (V/H)^2 H w_a$ of tension", fontsize=5.6)
lab(ax, 9, 11, r"$T \to 0$: the edge goes slack, its mode leaves the band", fontsize=5.6)

# ---- c: block diagram ----
ax = fig.add_axes([0.0, 0.0, 1.0, 0.3366]); ax.set_xlim(0, 180); ax.set_ylim(0, 66); ax.axis("off")
ax.text(1, 63, "f", fontsize=9, fontweight="bold")


def box(x, y, w, h, title, body, fc="white", ec=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=0.8))
    lab(ax, x + 1.5, y + h - 4.2, title, fontsize=6.4, fontweight="bold")
    lab(ax, x + 1.5, y + h - 6.0, body, fontsize=5.4, color="#333333", va="top")


box(2, 38, 30, 20, "Drive", "sine squared,\n0–1.4 kV, 1–15 Hz,\none signal for both")
box(42, 44, 40, 16, "DEA core", "force $\\propto V^2$\nlag $\\tau$ 30–50 ms", fc=BLUE)
box(92, 40, 44, 22, "Stroke mode", "$\\ddot\\theta + 2\\zeta\\omega\\dot\\theta + \\omega^2\\theta + \\beta\\theta^2$\n$= G\\,u(t) - \\mu\\,\\omega_w^2 q$\n$f_n$ 8.4 Hz, $\\zeta$ 0.07", fc="white")
box(42, 8, 40, 22, "Wing DEA", "skin + rolled edge\nedge tension $T_0 - \\Delta F(V)$\nrobot: $T_0 \\approx$ 4 mN", fc=BLUE)
box(92, 6, 44, 24, "Wing mode", "$\\ddot q + 2\\zeta_w\\omega_w\\dot q + \\omega_w^2 q = -\\ddot\\theta$\n$f_w(T)$: 17 Hz at 0 V,\n10 Hz when slack", fc="white")
box(146, 24, 32, 22, "Video", "leading-edge angle\n$\\theta + q$, 200 fps")
arrow(ax, (32.5, 50), (41.5, 52)); arrow(ax, (32.5, 44), (41.5, 20))
arrow(ax, (82.5, 52), (91.5, 52)); lab(ax, 84, 54, "$u(t)$", fontsize=5.5)
arrow(ax, (82.5, 19), (91.5, 19)); lab(ax, 83.2, 21, "$T$", fontsize=5.5)
arrow(ax, (108, 39.5), (108, 30.5), color=MAROON); lab(ax, 99.5, 34, r"$-\ddot\theta$", fontsize=5.8, color=MAROON)
arrow(ax, (120, 30.5), (120, 39.5), color=MAROON); lab(ax, 121.5, 34, r"$-\mu\omega_w^2 q$", fontsize=5.8, color=MAROON)
arrow(ax, (136.5, 50), (145.5, 39)); arrow(ax, (136.5, 18), (145.5, 30))
lab(ax, 2, 1.2, "Blue: dielectric elastomer actuators. Red arrows: coupling between the stroke and the wing mode, which the edge tension switches on (0 V) and off (powered).", fontsize=5.4, color=GREY)
save(fig, "Fig10_schematic")

# ======================= Fig. 11: measuring the stroke =======================
# b: frames of the unpowered run through the bottom of the stroke, with the top-silhouette track (red) and the tip;
# c: silhouette angle against tip angle over the same cycles, both runs; d: spring asymmetry per run from the tip;
# e: free decays (one mode).
import sys as _s
_s.path.insert(0, os.path.join(FLAP, "first")); _s.path.insert(0, os.path.join(FLAP, "flapping", "code"))
from track_views import Stack
import matplotlib.image as mpimg
VID = r"C:/Users/stefk/MIT Dropbox/SMRL/Shared/Stefania/full_tension/"
TRK = lambda n: np.load(os.path.join(FLAP, "flapping", "data", "tracks", n + ".npz"))
TIPd = lambda n: np.load(os.path.join(FLAP, "flapping", "data", "tracks", "tip_" + n + ".npz"))
HT = FD("hb_tip.json"); DEC = FD("decays.json"); TSd = FD("tip_stroke.json")
fig = plt.figure(figsize=(ST.WIDTH_MM * ST.MM, 150 * ST.MM))
# b: four frames
n = "only_DEA_core_8hz_1200V_side"; d = TRK(n); tp = TIPd(n)["tip"]; s = Stack(VID + n + ".avi")
k0 = 400 + int(np.nanargmax(tp[400:425, 1]))
for c_, k in enumerate((k0 - 4, k0, k0 + 3, k0 + 6)):
    a_ = fig.add_axes([0.01 + c_ * 0.2475, 0.70, 0.24, 0.27]); im = np.asarray(s[k], float)[170:420, 230:780]
    a_.imshow(im, cmap="gray"); a_.set_axis_off()
    xs = d["xs"]; top = d["top"][k]; m = np.isfinite(top)
    a_.plot(xs[m] - 230, top[m] - 170, color="#e11d48", lw=1.2); a_.plot(tp[k, 0] - 230, tp[k, 1] - 170, "o", ms=4, mfc="none", mec="#22c55e", mew=1.4)
    a_.text(0.03, 0.88, "%+d ms" % ((k - k0) * 5), transform=a_.transAxes, color="w", fontsize=6.5)
fig.text(0.005, 0.975, "b", fontsize=9, fontweight="bold")
fig.text(0.03, 0.975, "unpowered run through the bottom of the stroke: silhouette track (red) and tip (green)", fontsize=6.5)
# c: silhouette angle vs tip angle, mean cycle
ax = [fig.add_axes([0.09 + c_ * 0.33, 0.07, 0.24, 0.55]) for c_ in range(3)]
for i_, a_ in enumerate(ax): a_.text(-0.3, 1.04, "cde"[i_], transform=a_.transAxes, fontsize=9, fontweight="bold")
a = ax[0]
for nn, col in (("full_DEA_core_wing_8hz_1200V_side", C[0]), ("only_DEA_core_8hz_1200V_side", C[1])):
    sil = -TRK(nn)["ang"]; tip = np.array(TSd[nn]["theta"]); t = np.arange(len(sil)) / 200.0; g = (t > 0.9) & np.isfinite(sil) & np.isfinite(tip)
    ph = (t[g] * 8.0) % 1; b = np.linspace(0, 1, 26); idx = np.minimum(np.digitize(ph, b) - 1, 24)
    A_s = np.array([np.mean(sil[g][idx == k] - np.mean(sil[g])) for k in range(25)]); A_t = np.array([np.mean(tip[g][idx == k] - np.mean(tip[g])) for k in range(25)])
    sh = int(np.argmin(A_t)); A_s, A_t = np.roll(A_s, -sh), np.roll(A_t, -sh); x = (b[:-1] + b[1:]) / 2
    lab = "powered" if "full" in nn else "unpowered"
    a.plot(x, A_t, color=col, lw=1.2, label=lab + ", tip"); a.plot(x, A_s, color=col, lw=1.0, ls="--", label=lab + ", silhouette")
a.set(xlabel="Phase from the lowest point", ylabel="Stroke angle, up positive (deg)")
a.legend(loc="upper left", fontsize=4.8)
# d: asymmetry per run, from the tip
a = ax[1]
order = (("full_DEA_core_wing_7hz_1200V_side", "7 Hz"), ("full_DEA_core_wing_8hz_1200V_side", "8 Hz"), ("full_DEA_core_wing_8hz_1400V_side", "8 Hz\n1.4 kV"), ("only_DEA_core_8hz_1200V_side", "8 Hz\nunpow."))
for i_, (nn, lab) in enumerate(order):
    r = HT[nn]; a.bar(i_, r["asym"], yerr=r["asym_sd"], color=C[1] if "only" in nn else C[0], width=0.6, error_kw=dict(lw=0.6, capsize=1.5))
a.axhline(0, color="k", lw=0.5); a.set_xticks(range(4)); a.set_xticklabels([l for _, l in order], fontsize=6); a.minorticks_off()
a.set(ylabel=r"Spring asymmetry $\beta A_1/\omega^2$, tip")
# e: free decays
a = ax[2]
for q, col, lab in ((DEC[0]["f_vs_amp"], C[3], "after the 1.4 kV run"), (DEC[1]["f_vs_amp"], C[1], "unpowered session")):
    q = np.array([v for v in q if v[0] > 0.8]); a.plot(q[:, 0], q[:, 1], "o-", ms=2.5, lw=0.7, color=col, label=lab)
a.axhline(8.38, color=C[0], ls="--", lw=0.7); ST.note(a, "sweep fit", xy=(0.62, 0.30), color=C[0])
a.set(xlabel="Amplitude (deg)", ylabel="Free frequency (Hz)", ylim=(7, 11))
a.legend(loc="upper right", fontsize=4.8)

save(fig, "Fig11_model")
print("done")
