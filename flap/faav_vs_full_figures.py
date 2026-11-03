"""Figures for the FAAV wing with a tensioned trailing edge, against the same planform as a full membrane. Analytical model only (wing_analytical.py).
Run: python faav_vs_full_figures.py   (PNG in ../results, SVG in ../results/svg, numbers in faav_vs_full_numbers.json)"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\Users\stefk\Documents\website\site\flap")
import wing_analytical as W

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
BLUE, RED, GREEN, PURPLE, GRAY = "#0072BD", "#D95319", "#77AC30", "#7E2F8E", "#7F7F7F"
plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.grid": True, "grid.color": "#DDDDDD", "axes.axisbelow": True, "savefig.dpi": 200})

AREA = 84e-4
RHO = 1030.0
FLOW = dict(U=4.0, alpha_deg=10.0, area=AREA, Phi0_deg=30.0, f=1.0)
GEOM = dict(a=(0.06, 0.06), b=(0.07, 0.06), share=(0.55, 0.45), radius=(0.06, 0.12))
ISO, CYC = 8.6e-3, 3.2e-3


def disc(lam, r_in, r_out, dF=ISO, **kw):
    return dict(W.DISCRETE, lam_x=lam, lam_y=lam, dF=dF, s=0.010, r=(r_in, r_out), **dict(GEOM, **kw))


def ten(w, i, V):
    Hm, G = w["H_m"], 2 * w["C1"]
    ex, ey = w["lam_x"] - 1, w["lam_y"] - 1
    nx = 2 * G * Hm * (2 * ex + ey) + w["E_x"] * Hm * ex
    ny = 2 * G * Hm * (ex + 2 * ey) + w["E_y"] * Hm * ey + w["r"][i] * max(w["F_D"] - w["dF"] * (V / 1600.0) ** 2, 0.0) / w["s"]
    return nx, ny, Hm, G + 0.5 * (w["E_x"] + w["E_y"]) / 6


def full(n, eta):
    k = 41.1 / 70.0
    return dict(W.ROTATED, n_layers=(n, n), eta=eta, chord=0.070, x_batten=0.060, x_tip=0.120, tip_drop=0.010, theta_b=4.5 * k, theta_r=7.5 * k,
                a=(0.056, 0.061), b=(0.070, 0.027), share=(0.70, 0.30), radius=(0.06, 0.12))


def dL(w, t, **fl):
    return W.mean_delta(w, t, **dict(FLOW, **fl))


def L0(w, t, **fl):
    return float(W.cycle(w, t, V=0.0, **dict(FLOW, **fl))[1].mean())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RES, name + ".png"))
    fig.savefig(os.path.join(RES, "svg", name + ".svg"))
    plt.close(fig)


N = {}
# ---- a) the same wing with and without the line
rs = np.array([0.02, 0.05, 0.1, 0.15, 0.22, 0.3, 0.4, 0.5, 0.6, 0.7, 0.81, 0.9, 1.0])
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
a = ax[0]
a.axhspan(10.9 - 1.0, 10.9 + 1.0, color=GRAY, alpha=0.25, label="Old cycle mean per wing: 10.9 ± 1.0 mN (mostly drift)")
for lam, c in ((1.00, BLUE), (1.01, RED)):
    a.plot(rs, [dL(disc(lam, r, r), ten) for r in rs], "-", color=c, lw=2, label=f"Line tensions whole edge, $\\lambda$ = {lam:.2f}")
    a.plot(rs, [dL(disc(lam, r, 0.02), ten) for r in rs], "--", color=c, lw=2, label=f"Line reaches inboard panel only, $\\lambda$ = {lam:.2f}")
a.plot(rs, [dL(disc(1.01, r, r, dF=CYC), ten) for r in rs], ":", color=PURPLE, lw=2, label="Whole edge, cyclic DEA force (3.2 mN)")
a.set_xlabel("Load-path factor $r$ (0: free edge, 1: rigid anchor)")
a.set_ylabel(r"$\Delta L$ per wing in mN, 4 m/s, 10°, 1 Hz")
a.legend(fontsize=8.5, loc="upper left")
a.set_ylim(0, 46)
a.set_xlim(0, 1)
a.set_title("a) Discrete fibres, 12 × 7 cm", loc="left", fontsize=12)
b = ax[1]
meas = {2: (-15.4, 7, 57.6), 4: (-27.9, -3.2, 21.8), 6: (-35.4, 4.3, 30.2)}
pts = [(-5, 0), (0, 1), (10, 2)]
for U, c in ((2, GREEN), (4, PURPLE), (6, BLUE)):
    mod = [2 * dL(disc(1.01, 0.81, 0.81), ten, U=float(U), alpha_deg=float(p)) for p, _ in pts]
    b.plot(meas[U], mod, "o", color=c, ms=8, label=f"{U} m/s")
    N[f"trend_{U}"] = dict(model=mod, measured=meas[U], free=[dL(disc(1.01, 0.02, 0.02), ten, U=float(U), alpha_deg=float(p)) for p, _ in pts])
b.plot([-70, 65], [-70, 65], ":", color=GRAY)
b.set_xlabel(r"Old cycle-mean $\Delta L$ in mN, both wings")
b.set_ylabel(r"Model $\Delta L$ in mN, both wings")
b.set_xlim(-70, 65)
b.set_ylim(-110, 75)
b.legend(fontsize=10, loc="upper left")
b.set_title("b) Against the old result", loc="left", fontsize=12)
save(fig, "L_faav_line")

# ---- b) tension curve: the same physics for both wings
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
ns = np.linspace(0.05, 6.0, 240)
q = 0.5 * 1.225 * 16
c = dict(W.COMMON, lift_slope=3.71, U=4.0)
al = math.radians(6.0)


def cl_of_n(ny, b_=0.07, a_=0.06):
    def tension(w, i, V):
        return 0.5, ny, 180e-6, 20e3
    w = dict(a=(a_,), b=(b_,))
    return W.panel(w, 0, 0.0, q, al, c, tension)[1]


a = ax[0]
CL = np.array([cl_of_n(n) for n in ns])
a.plot(ns, CL, color=BLUE, lw=2)
_d, _f = disc(1.01, 0.81, 0.81), full(2, 0.4)
_arrows = (
    (ten(_d, 0, 0.0)[1], ten(_d, 0, 1600.0)[1], "Fibres, line, $r$ = 0.81", RED, 0.10),
    (W.rotation_tension(_f, 0, 0.0)[1], W.rotation_tension(_f, 0, 1600.0)[1], "Full, $n$ = 2, $\\eta$ = 0.4", PURPLE, 0.17),
)
for x0, x1, lab, col, dy in _arrows:
    N[f"ny {lab}"] = (x0, x1)
    a.annotate("", xy=(x1, np.interp(x1, ns, CL)), xytext=(x0, np.interp(x0, ns, CL)), arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
    a.text(max(x0, x1) + 0.15, np.interp(max(x0, x1), ns, CL) + dy, lab, color=col, fontsize=11)
a.set_xlabel(r"Chordwise tension $n_y$ in N/m")
a.set_ylabel(r"$C_L$ of the panel, 4 m/s, 6°")
a.set_title("a) Panel lift", loc="left", fontsize=12)
b = ax[1]
Ts = np.logspace(-3, 0.3, 80)
ka = 8 * Ts * 0.01 / 0.06 ** 2
b.semilogx(Ts, ka / (ka + 2.2), color=BLUE, lw=2, label="12 × 7 cm: $s$ = 10 mm, $L_s$ = 60 mm, $k_d$ = 2.2 N/m")
b.set_xlabel(r"Trailing-edge tension $T_c$ in N")
b.set_ylabel(r"$r = k_a / (k_a + k_d)$")
b.legend(fontsize=9.5, loc="upper left")
b.set_ylim(0, 1)
b.set_title("b) Load path", loc="left", fontsize=12)
save(fig, "L_tension_curve")
for T in (0.02, 0.05, 0.1, 0.3, 0.5):
    ka = 8 * T * 0.01 / 0.06 ** 2
    N[f"r_T{T}"] = ka / (ka + 2.2)

# ---- c) same planform, both wings
rows = []
for lab, hi, lo, m in (
    ("Fibres,\nfree", dL(disc(1.01, 0.02, 0.02), ten), dL(disc(1.01, 0.02, 0.02, dF=CYC), ten), None),
    ("Fibres,\nline in", dL(disc(1.01, 0.81, 0.02), ten), dL(disc(1.01, 0.81, 0.02, dF=CYC), ten), None),
    ("Fibres,\nline all", dL(disc(1.01, 0.81, 0.81), ten), dL(disc(1.01, 0.81, 0.81, dF=CYC), ten), None),
    ("Full,\n$n$ = 1", dL(full(1, 0.4), W.rotation_tension), dL(full(1, 0.15), W.rotation_tension), 1),
    ("Full,\n$n$ = 2", dL(full(2, 0.4), W.rotation_tension), dL(full(2, 0.15), W.rotation_tension), 2),
    ("Full,\n$n$ = 3", dL(full(3, 0.4), W.rotation_tension), dL(full(3, 0.15), W.rotation_tension), 3),
):
    if m is None:
        mass = (AREA * W.DISCRETE["H_m"] * RHO + 12 * 50e-3 * 12e-3 * 70e-6 * RHO) * 1e3
        msg = dict(membrane=AREA * W.DISCRETE["H_m"] * RHO * 1e3, dea=12 * 50e-3 * 12e-3 * 70e-6 * RHO * 1e3)
    else:
        mass = (AREA + 0.12 * 20e-3) * (m + 1) * 35e-6 * RHO * 1e3
        msg = dict(sheet=AREA * (m + 1) * 35e-6 * RHO * 1e3, edge=0.12 * 20e-3 * (m + 1) * 35e-6 * RHO * 1e3)
    rows.append((lab, hi, lo, mass))
    N[lab.replace("\n", " ")] = dict(hi=hi, lo=lo, mass_g=mass, **msg)
fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.1))
xs = np.arange(len(rows))
cols = [GRAY, RED, RED, BLUE, BLUE, BLUE]
labs = [r[0] for r in rows]
ax[0].bar(xs, [r[1] for r in rows], color=cols, alpha=0.35, width=0.7, label="DEA force high ($\\eta$ 0.4; 8.6 mN)")
ax[0].bar(xs, [r[2] for r in rows], color=cols, width=0.7, label="DEA force low ($\\eta$ 0.15; 3.2 mN)")
ax[0].set_ylabel(r"$\Delta L$ in mN, 4 m/s, 10°, 1 Hz")
ax[0].set_title("a) Lift change", loc="left", fontsize=12)
ax[0].legend(fontsize=8, loc="upper left")
ax[0].set_ylim(0, 40)
ax[1].bar(xs, [r[3] for r in rows], color=cols, width=0.7)
ax[1].set_ylabel("Film mass in g, one wing")
ax[1].set_title("b) Mass, film only", loc="left", fontsize=12)
ax[2].bar(xs, [r[1] / r[3] for r in rows], color=cols, alpha=0.35, width=0.7)
ax[2].bar(xs, [r[2] / r[3] for r in rows], color=cols, width=0.7)
ax[2].set_ylabel(r"$\Delta L$ per gram, mN/g")
ax[2].set_title("c) Lift change per mass", loc="left", fontsize=12)
for a in ax:
    a.set_xticks(xs)
    a.set_xticklabels(labs, fontsize=10)
    a.grid(axis="x", visible=False)
save(fig, "L_faav_vs_full")

# ---- numbers for the slides
for k, w, t in (("full_n2_eta04", full(2, 0.4), W.rotation_tension), ("full_n1_eta04", full(1, 0.4), W.rotation_tension), ("disc_line", disc(1.01, 0.81, 0.81), ten), ("disc_free", disc(1.01, 0.02, 0.02), ten)):
    N["L0_" + k] = L0(w, t)
    N["CL0_" + k] = L0(w, t) / (0.5 * 1.225 * 16 * AREA * 1e3)
    N["pct_" + k] = 100 * dL(w, t) / L0(w, t)
N["full_stretch"] = W.rotation_geometry(full(2, 0.4))[:3]
# gliding wing: 2 x (6 x 5 cm), 5 DEAs, no flap, with and without a line
gl = dict(a=(0.06,), b=(0.05,), share=(1.0,), radius=(0.03,))
for U in (2.0, 4.0, 6.0):
    for r, tag in ((0.02, "free"), (0.81, "line")):
        for a_ in (5.0, 10.0):
            w = dict(W.DISCRETE, lam_x=1.01, lam_y=1.01, dF=ISO, s=0.012, r=(r,), **gl)
            N[f"glide_{tag}_U{U:g}_a{a_:g}"] = W.solve(w, ten, U=U, alpha_deg=a_, area=30e-4)["L_mN"] - W.solve(w, ten, V=0.0, U=U, alpha_deg=a_, area=30e-4)["L_mN"]
json.dump(N, open(os.path.join(HERE, "faav_vs_full_numbers.json"), "w"), indent=1, default=float)
print(json.dumps(N, indent=0, default=lambda v: round(float(v), 3)))
