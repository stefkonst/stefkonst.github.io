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

# ======================= Fig. 11: model =======================
HB = FD("harmonic_balance.json"); TM = FD("two_mode.json"); WL = FD("wing_link.json"); EO = FD("edge_on_stroke.json")
fig = plt.figure(figsize=(ST.WIDTH_MM * ST.MM, 150 * ST.MM))
ax = [fig.add_axes([0.09 + c_ * 0.5, 0.57 - r_ * 0.49, 0.38, 0.38]) for r_ in (0, 1) for c_ in (0, 1)]
for i_, a_ in enumerate(ax): a_.text(-0.2, 1.04, "abcd"[i_], transform=a_.transAxes, fontsize=9, fontweight="bold")
a = ax[0]
runs = [("full_DEA_core_wing_7hz_1200V_side", "7 Hz"), ("full_DEA_core_wing_8hz_1200V_side", "8 Hz"), ("full_DEA_core_wing_8hz_1400V_side", "8 Hz\n1.4 kV"), ("only_DEA_core_8hz_1200V_side", "8 Hz\nunpow.")]
wn2 = (2 * np.pi * 8.38) ** 2
for i, (n, l) in enumerate(runs):
    r = HB[n]; A = r["A1_deg"] * np.pi / 180
    for j, (case, hatch) in enumerate((("fn 8.38 (sweep fit)", None), ("fn 8.95 (drift case)", "////"))):
        c = r["cases"][case]; val = c["beta_re"] * A / wn2; err = c["beta_re_sd"] * A / wn2
        a.bar(i + (j - 0.5) * 0.36, val, width=0.34, color=(C[1] if "only" in n else C[0]), alpha=1.0 if j == 0 else 0.45, hatch=hatch, edgecolor="white", lw=0.3, yerr=err, error_kw=dict(lw=0.6, capsize=1.2))
a.axhline(0, color="k", lw=0.5)
a.set_xticks(range(len(runs))); a.set_xticklabels([l for _, l in runs], fontsize=6); a.minorticks_off()
a.set(ylabel=r"Spring asymmetry $\beta A_1/\omega^2$")
ST.note(a, "hatched: the same if $f_n$ had\ndrifted to 8.95 Hz", xy=(0.04, 0.06), va="bottom")
a = ax[1]
x = np.arange(25) / 25
for key, mkey, col, lb in (("cycle_powered", "cycle_model_powered", C[0], "powered"), ("cycle_unpowered", "cycle_model_unpowered", C[1], "unpowered")):
    d = FD("stroke_model.json")[key] if os.path.exists(os.path.join(FLAP, "flapping", "data", "stroke_model.json")) else None
    data = np.array(d["data"]); sd = np.array(d["sd"])
    a.errorbar(x, data, yerr=sd, fmt="o", ms=2.2, color=col, lw=0.5, capsize=0, label=lb + ", video")
    a.plot(x, TM[mkey], color=col, lw=1.0)
a.set(xlabel="Phase from the lowest point", ylabel="Stroke angle (deg, mean removed)")
a.legend(loc="upper left", fontsize=5)
ST.note(a, "lines: model\n(powered: one mode;\n0 V: + wing mode)", xy=(0.40, 0.05), va="bottom")
a = ax[2]
hh = np.arange(3)
for k, (key_m, key_d, col, off) in enumerate((("model_powered", "data_powered", C[0], -0.2), ("model_unpowered", "data_unpowered", C[1], 0.2))):
    a.bar(hh + off - 0.08, TM[key_d], width=0.16, color=col, label=("powered" if k == 0 else "unpowered") + ", video")
    a.bar(hh + off + 0.08, TM[key_m], width=0.16, color="white", edgecolor=col, hatch="////", lw=0.6, label="model")
a.set_yscale("log"); a.set_ylim(0.05, 30)
a.set_xticks(hh); a.set_xticklabels(["8 Hz", "16 Hz", "24 Hz"]); a.minorticks_off()
a.set(ylabel="Harmonic amplitude (deg)")
a.legend(loc="upper right", fontsize=4.8, ncol=1)
a = ax[3]
T = np.array(WL["T"]) * 1e3; fe = np.array(WL["f_e"])
a.plot(T, fe, color=INK, lw=1.0)
lo, mid, hi = WL["T_robot_mN"]
a.axhspan(WL["f_w"] - WL["f_w_se"], WL["f_w"] + WL["f_w_se"], color=C[1], alpha=0.25, lw=0)
a.plot([mid], [WL["f_w"]], "o", color=C[1], ms=4)
a.axhline(16, color=GREY, ls=":", lw=0.7); a.axhline(8, color=GREY, ls=":", lw=0.7)
ST.note(a, "16 Hz (2nd harmonic)", xy=(0.62, (16 - 5) / 65 + 0.045)); ST.note(a, "8 Hz (stroke)", xy=(0.62, (8 - 5) / 65 + 0.045))
pk = [WL["dF"]["1200.0"][e]["peak_mN"] for e in ("0.17", "0.8")] if "1200.0" in WL["dF"] else [8, 38]
a.annotate("", xy=(0, WL["f_skin_only"] + 1.5), xytext=(mid, WL["f_w"] - 1.5), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C[0]))
ST.note(a, "1.2 kV: the roll loses\n%.0f–%.0f mN, edge slack" % tuple(pk), xy=(0.03, 0.36), color=C[0])
ST.note(a, "robot at 0 V: %.1f Hz\n→ T = %.1f mN" % (WL["f_w"], mid), xy=(0.38, 0.50), color=C[1])
a.axvspan(50, 96, color=GREY, alpha=0.15, lw=0); ST.note(a, "bench\nmount", xy=(0.80, 0.72))
a.set(xscale="symlog", xlabel="Edge tension $T$ (mN)", ylabel="Wing (edge) mode $f_w$ (Hz)", ylim=(5, 70))
a.set_xlim(0, 120)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
a.xaxis.set_major_locator(FixedLocator([0, 1, 3, 10, 30, 100])); a.xaxis.set_major_formatter(FixedFormatter(["0", "1", "3", "10", "30", "100"])); a.xaxis.set_minor_locator(NullLocator())
save(fig, "Fig11_model")
print("done")
