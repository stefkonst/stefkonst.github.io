"""Figures for the robot page (style of the SMS 2023 / CST 2024 papers, plotstyle.py). Every PNG gets an SVG beside it.
Run after edge_fe.py, calib_edge.py, robot_study.py: python figures_robot.py"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, FancyArrowPatch
import params as P
import edge_theory as ET
PS.apply()
C = PS.MATLAB
D = lambda n: os.path.join(HERE, "data", n)
FE = [json.loads(l) for l in open(D("edge_fe.jsonl"))]
FE = [r for r in FE if "res" in r]
CAL = json.load(open(D("edge_calib.json")))
S = json.load(open(D("robot_study.json")))
W = json.load(open(os.path.join(HERE, "..", "first", "data", "views.json")))
F = lambda n: os.path.join(HERE, n)
MEAS = sorted((v, W["%dV_1hz_side" % v]["fit_low"] + W["%dV_1hz_side" % v]["fit_amp"] - W["%dV_1hz_side" % v]["fit_low"]) for v in (1000, 1400, 1800))
MEAS_HI = [(v, W["%dV_1hz_side" % v]["fit_low"] + W["%dV_1hz_side" % v]["fit_amp"]) for v in (1000, 1400, 1800)]


def fe(edge, p, pull=None, eta_f=None, a=None, eta=0.15):
    for r in FE:
        if r["edge"] == edge and r["p"] == p and r["eta"] == eta and (pull is None or r["pull"] == pull) and (eta_f is None or r["eta_f"] == eta_f) and (a is None or r.get("a") == a):
            return r
    raise KeyError((edge, p, pull, eta_f, a))


def dz(r):
    z0 = r["res"][0]["z_te_mid"]
    return [q["V"] for q in r["res"]], [q["z_te_mid"] - z0 for q in r["res"]]


def dcl(r, i, ref=0):
    """Thin-airfoil lift change from the mid-span chord shape, state i against state ref: dCL = 2 int (dz/dy)(cos th - 1) dth."""
    def shape(q):
        a = np.array(q["chord"])
        return a[:, 0] * 1e-3, a[:, 1] * 1e-3
    y, z = shape(r["res"][i]); y0, z0 = shape(r["res"][ref])
    c = y.max()
    pz = np.polyfit(y / c, z - np.interp(y, y0, z0), 4)
    th = np.linspace(0, math.pi, 400)
    s = (1 - np.cos(th)) / 2
    dzdy = np.polyval(np.polyder(pz), s) / c
    return 2 * np.trapz(dzdy * (np.cos(th) - 1), th)


def eq(V, **kw):
    """Calibrated edge equation: drop change (mm, + up like the FE)."""
    a = ET.solve(V, N0=CAL["N0"], c=CAL["c"], **kw)
    b = ET.solve(0.0, N0=CAL["N0"], c=CAL["c"], **kw)
    return -(a["d"] - b["d"]) * CAL["qf"], a


# ---------- R1: the edge, as built and with other edges ----------
fig, ax = plt.subplots(2, 3, figsize=(13.2, 7.4))
a = ax[0, 0]
for ef, col in ((0.0, C[0]), (0.2, C[2]), (0.4, C[1]), (0.8, C[6])):
    V, z = dz(fe("roll", 0.0, 6.0, ef))
    a.plot(V, z, "-o", color=col, label=r"roll, $\eta_f$ %.1f" % ef)
V, z = dz(fe("roll", 0.0, 6.0, 0.0, eta=0.4)); a.plot(V, z, "--s", color=C[0], mfc="none", label=r"passive roll, skin $\eta$ 0.4")
a.plot([v for v, _ in MEAS_HI], [-h for _, h in MEAS_HI], "k*", ms=9, label="measured, side view (high)")
a.set(xlabel="Voltage $V$ in V", ylabel="Edge height change $\\Delta z$ in mm", title="Bench: own weight only")
a.legend(fontsize=7, loc="lower left")
a = ax[0, 1]
for p, ls in ((8.0, "-"), (15.0, "--")):
    for ef, col in ((0.0, C[0]), (0.4, C[1]), (0.8, C[6])):
        V, z = dz(fe("roll", p, 6.0, ef))
        a.plot(V, z, ls + "o", color=col, ms=3, label=r"$p$ %g Pa, $\eta_f$ %.1f" % (p, ef))
a.set(xlabel="Voltage $V$ in V", ylabel="Edge height change $\\Delta z$ in mm", title="With a lift-like pressure $p$")
a.legend(fontsize=7, ncol=2)
a = ax[0, 2]
for p, col in ((8.0, C[0]), (15.0, C[1])):
    for ef, mk in ((0.4, "o"), (0.8, "s")):
        r = fe("roll", p, 6.0, ef)
        a.plot([q["V"] for q in r["res"]], [dcl(r, i) for i in range(len(r["res"]))], "-" + mk, color=col, label=r"$p$ %g Pa, $\eta_f$ %.1f" % (p, ef))
    r = fe("roll", p, 6.0, 0.0)
    a.plot([q["V"] for q in r["res"]], [dcl(r, i) for i in range(len(r["res"]))], ":", color=col, label=r"$p$ %g Pa, passive" % p)
a.axhline(0, color="k", lw=0.6)
a.set(xlabel="Voltage $V$ in V", ylabel="Lift change $\\Delta C_L$ (thin aerofoil, mid span)", title="What it does to lift")
a.legend(fontsize=7)
# options with the calibrated equation, at a flight load
a = ax[1, 0]
opts = (("Roll, active\n(as built)", dict(edge="roll", eta_f=0.4)), ("Roll,\npassive", dict(edge="roll", eta_f=0.0)), ("TPU\n1 x 0.1 mm", dict(edge="TPU 1 x 0.1 mm")),
        ("Nylon\n0.10 mm", dict(edge="nylon 0.10 mm")), ("Inner half\nactive + nylon", dict(edge="roll", a=0.5, passive="nylon 0.10 mm", eta_f=0.4)))
Vs = np.linspace(0, 1800, 19)
for (lab, kw), col in zip(opts, (C[1], C[0], C[2], C[3], C[4])):
    a.plot(Vs, [eq(v, T0=0.043, p=8.0, **kw)[0] for v in Vs], "-", color=col, lw=1.6, label=lab.replace("\n", " "))
a.set(xlabel="Voltage $V$ in V", ylabel="Edge height change $\\Delta z$ in mm", title="Edge options, $p$ 8 Pa, 43 mN (edge equation)")
a.legend(fontsize=7)
a = ax[1, 1]
names = list(P.CORDS)
mass = []
for n in names:
    c = P.CORDS[n]
    mass.append(c["rho"] * c["A"] * P.EDGE_L * 1e6 if "DEA" not in n else P.ROLL_MASS * 1e6)
a.barh(range(len(names)), mass, color=[C[1]] + [C[3]] * 2 + [C[2]] * 2)
for i, m in enumerate(mass):
    a.text(m * 1.15, i, "%.1f mg" % m, va="center", fontsize=8)
a.set_xscale("log"); a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=8)
a.set(xlabel="Edge mass in mg (one wing)", title="Mass", xlim=(0.1, 600))
a = ax[1, 2]
err = []
for n in names:
    c = P.CORDS[n]
    EA = c["E"] * c["A"] if "DEA" not in n else 6 * P.C1 * 2.16 * c["A"]
    err.append(EA / P.EDGE_L * 0.1e-3 * 1e3)
a.barh(range(len(names)), err, color=[C[1]] + [C[3]] * 2 + [C[2]] * 2)
for i, m in enumerate(err):
    a.text(m * 1.15, i, "%.1f mN" % m, va="center", fontsize=8)
a.axvline(43, color="k", ls="--", lw=0.8); a.text(48, 4.3, "edge tension", fontsize=8)
a.set_xscale("log"); a.set_yticks(range(len(names))); a.set_yticklabels([""] * len(names))
a.set(xlabel="Tension error for 0.1 mm of length error, in mN", title="How hard to set", xlim=(0.1, 3000))
PS.finish(ax)
fig.tight_layout(); fig.savefig(F("R_edge.png")); plt.close(fig)

# ---------- R2: only part of the edge active ----------
fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))
a = ax[0]
aa = np.linspace(0.05, 1, 96)
EAr = 6 * P.C1 * 2.16 * P.ROLL_W * P.FILM_T
for lab, EAp, col in (("rest: roll, no electrode", EAr, C[0]), ("rest: TPU 1 x 0.1 mm", P.CORDS["TPU 1 x 0.1 mm"]["E"] * P.CORDS["TPU 1 x 0.1 mm"]["A"], C[2]), ("rest: nylon 0.10 mm", P.CORDS["nylon 0.10 mm"]["E"] * P.CORDS["nylon 0.10 mm"]["A"], C[3])):
    ka, kp = EAr / (aa * P.EDGE_L), EAp / np.maximum((1 - aa) * P.EDGE_L, 1e-9)
    a.plot(aa * 100, kp / (ka + kp), color=col, lw=1.6, label=lab)
a.set(xlabel="Active share of the edge, from the body, in %", ylabel="Tension lost / force the active part loses", title="Series rule", ylim=(0, 1.05))
a.legend(fontsize=8)
a = ax[1]
for p, col in ((0.0, C[0]), (15.0, C[1])):
    for pull, ef, mk in ((6.0, 0.4, "o"), (9.0, 0.8, "s")):
        pts = [(0.0, 0.0)] + [(x, dz(fe("part", p, pull, ef, a=x))[1][-1]) for x in (0.25, 0.5, 0.75)] + [(1.0, dz(fe("roll", p, pull, ef))[1][-1])]
        a.plot([x * 100 for x, _ in pts], [y for _, y in pts], "-" + mk, color=col, label=r"FE, $p$ %g Pa, $\eta_f$ %.1f" % (p, ef))
    a.plot(aa * 100, [eq(1800, T0=0.029, p=p, eta_f=0.4, a=min(x, 0.999))[0] for x in aa], ":", color=col, label="edge equation, $p$ %g Pa" % p)
a.axhline(0, color="k", lw=0.6)
a.set(xlabel="Active share of the edge, from the body, in %", ylabel="Edge height change at 1800 V in mm", title="Same roll, part with electrodes")
a.legend(fontsize=7)
a = ax[2]
for lab, kw, col in (("rest: roll, no electrode", dict(), C[0]), ("rest: TPU", dict(passive="TPU 1 x 0.1 mm"), C[2]), ("rest: nylon", dict(passive="nylon 0.10 mm"), C[3])):
    a.plot(aa * 100, [eq(1800, T0=0.043, p=8.0, eta_f=0.4, a=min(x, 0.999), **kw)[0] for x in aa], color=col, lw=1.6, label=lab)
a.set(xlabel="Active share of the edge, from the body, in %", ylabel="Edge height change at 1800 V in mm", title="Roll + cord, $p$ 8 Pa, 43 mN (equation)")
a.legend(fontsize=8)
PS.finish(ax)
fig.tight_layout(); fig.savefig(F("R_part.png")); plt.close(fig)

# ---------- R3: thickness ----------
T = S["thickness"]
t = np.array(T["t_um"])
fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9))
a = ax[0]
for key, col, ls in (("2 electrodes, 46 V/um", C[0], "-"), ("2 electrodes, 60 V/um", C[0], "--"), ("4 electrodes, 46 V/um", C[1], "-"), ("4 electrodes, 60 V/um", C[1], "--")):
    a.plot(t, np.array(T[key]["V"]) / 1e3, ls, color=col, label=key.replace("um", "µm"))
for v, lab in ((1.8, "used, 1.8 kV"), (2.5, "2.5 kV"), (4.0, "4 kV")):
    a.axhline(v, color="0.5", lw=0.7, ls=":"); a.text(22, v + 0.1, lab, fontsize=7)
a.set(xlabel="Film thickness, all layers, in µm", ylabel="Drive voltage in kV", title="Voltage sets the layer", ylim=(0, 8))
a.legend(fontsize=7)
a = ax[1]
for key, col, ls in (("2 electrodes, 46 V/um", C[0], "-"), ("2 electrodes, 60 V/um", C[0], "--"), ("4 electrodes, 46 V/um", C[1], "-"), ("4 electrodes, 60 V/um", C[1], "--")):
    a.plot(t, T[key]["dF_mN"], ls, color=col, label=key.replace("um", "µm"))
a.axhspan(0.5 * T["T_needed_mN"], T["T_needed_mN"], color=C[2], alpha=0.15)
a.text(25, T["T_needed_mN"] * 0.72, "half to all of the\ntension a flight edge needs", fontsize=7)
a.set(xlabel="Film thickness, all layers, in µm", ylabel=r"Force the roll loses, $\eta_f$ 0.4, in mN", title="Authority of the edge")
a.legend(fontsize=7)
a = ax[2]
a.plot(t, T["wing_mg"], color=C[0], lw=1.6)
a.axhline(273, color="k", ls=":", lw=0.7); a.text(25, 280, "measured, 273 mg", fontsize=7)
a.set(xlabel="Film thickness, all layers, in µm", ylabel="Wing mass in mg", title="Mass (2 electrodes)")
a = ax[3]
a.plot(t, T["f_res_35"], color=C[1], lw=1.6)
a.axhspan(18, 25, color=C[2], alpha=0.15); a.text(25, 21, "flapping band", fontsize=7)
a.set(xlabel="Film thickness, all layers, in µm", ylabel="Flapping resonance, $r$ 3.5 mm, in Hz", title="Resonance with the measured cores")
for a in ax:
    a.axvspan(45, 90, color="0.85", alpha=0.5, zorder=0)
PS.finish(ax)
fig.tight_layout(); fig.savefig(F("R_thickness.png")); plt.close(fig)

# ---------- R4: resonance ----------
R = S["resonance"]; E = S["edge_mode"]; TT = S["te_transfer"]
fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9))
a = ax[0]
for k, col in zip(R["f"], (C[1], C[0])):
    a.plot(R["r_mm"], R["f"][k], color=col, lw=1.6, label=k)
a.axhspan(18, 25, color=C[2], alpha=0.15)
a.set(xlabel="Lever arm $r$ in mm", ylabel="Flapping resonance $f_r$ in Hz", title="Wing on two cores")
a.legend(fontsize=8)
a = ax[1]
for (k, v), col in zip(E["f"].items(), (C[1], C[6], C[3], C[0])):
    a.plot(E["V"], v, color=col, lw=1.6, label=k.replace("eta_f", r"$\eta_f$"))
a.axhspan(18, 25, color=C[2], alpha=0.15); a.text(50, 21, "flapping band", fontsize=7)
a.plot([1800], [15], "k*", ms=10, label="bump in the 1800 V sweep")
a.set(xlabel="Voltage $V$ in V", ylabel="First edge mode in Hz", title="Voltage tunes the edge", ylim=(0, 60))
a.legend(fontsize=7)
a = ax[2]
for (k, v), col, ls in zip(TT["cases"].items(), (PS.OFF, PS.OFF, PS.ON, PS.ON), ("--", "-", "--", "-")):
    a.plot(TT["f"], v["amp"], ls, color=col, label=k)
a.axvspan(18, 25, color=C[2], alpha=0.15)
a.set(xlabel="Frequency $f$ in Hz", ylabel="Edge motion / leading-edge motion", title="Edge relative to the stroke", yscale="log", ylim=(0.01, 30))
a.legend(fontsize=7)
a = ax[3]
for (k, v), col, ls in zip(TT["cases"].items(), (PS.OFF, PS.OFF, PS.ON, PS.ON), ("--", "-", "--", "-")):
    a.plot(TT["f"], v["ph"], ls, color=col, label=k)
a.axvspan(18, 25, color=C[2], alpha=0.15)
a.set(xlabel="Frequency $f$ in Hz", ylabel="Edge lag in deg", title="Phase", ylim=(0, 185))
PS.finish(ax)
fig.tight_layout(); fig.savefig(F("R_resonance.png")); plt.close(fig)

# ---------- R5: can it fly ----------
Fl = S["flight"]; MC = S["mc"]; N = S["need"]
fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9))
a = ax[0]
U = np.array(Fl["U"])
a.fill_between(U, Fl["P_req_mW"]["optimistic"], Fl["P_req_mW"]["pessimistic"], color=C[0], alpha=0.15, label="required, range")
a.plot(U, Fl["P_req_mW"]["model"], color=C[0], lw=1.8, label="required, middle")
for (k, v), col in zip(Fl["P_avail_mW_25Hz"].items(), (C[1], C[3], C[4])):
    a.axhline(v, color=col, ls="--", lw=1.2, label="cores, %s" % k)
a.axvspan(0, Fl["U_min"], color="0.85", alpha=0.6); a.text(2.1, 150, "below stall\nspeed", fontsize=7)
a.set(xlabel="Flight speed $U$ in m/s", ylabel="Power in mW", title="Level flight, %.2f g, 25 Hz" % Fl["mass_g"], ylim=(0, 200), xlim=(2, 8))
a.legend(fontsize=6.5, loc="upper right")
a = ax[1]
edges = np.linspace(0, 2, 41)
a.bar(edges[:-1], MC["hist"], width=edges[1] - edges[0], align="edge", color=C[0], alpha=0.8)
a.axvline(1, color="k", lw=1)
a.text(1.05, max(MC["hist"]) * 0.8, "flies: %d %%" % round(100 * MC["p_fly"]), fontsize=8)
a.set(xlabel="Power available / required", ylabel="Draws", title="20,000 draws over the unknowns")
a = ax[2]
a.plot(MC["e_cyc"], MC["margin"], ".", ms=1.5, color=C[0], alpha=0.4)
a.axhline(1, color="k", lw=1)
for k, e in P.E_CYCLE.items():
    a.axvline(e, color="0.5", ls=":", lw=0.8)
a.set(xlabel="Core energy per cycle in J/kg", ylabel="Power available / required", title="What decides it")
a = ax[3]
a.plot(np.array(N["core_frac"]) * 100, N["e_needed"], color=C[1], lw=1.6)
for (k, e), col in zip(P.E_CYCLE.items(), (C[1], C[3], C[4])):
    a.axhline(e, color=col, ls="--", lw=1, label=k)
a.axvline(100 * 1.131 / 1.88, color="k", ls=":", lw=0.8); a.text(61, 3.3, "now", fontsize=8)
a.set(xlabel="Cores as a share of robot mass in %", ylabel="Energy per cycle needed in J/kg", title="What the cores must give, 25 Hz", ylim=(0, 4))
a.legend(fontsize=7)
PS.finish(ax)
fig.tight_layout(); fig.savefig(F("R_flight.png")); plt.close(fig)

# ---------- R6: the robot, to scale ----------
G = json.load(open(os.path.join(HERE, "..", "first", "data", "geometry.json")))["points_mm_along_LE"]
fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
a = ax[0]
for sgn in (1, -1):
    xo = 6.0
    pts = [(xo + G["LE_root"][0], 0), (xo + G["LE_knee"][0], 0), (xo + G["tip"][0], G["tip"][1]), (xo + G["TE_tip"][0], G["TE_tip"][1]), (xo + G["TE_root"][0], G["TE_root"][1])]
    pts = [(sgn * x, -y) for x, y in pts]
    a.add_patch(Polygon(pts, closed=True, fc="#d1e0ff", ec="k", lw=0.8))
    a.plot([sgn * xo, sgn * (xo + G["LE_knee"][0])], [0, 0], color="k", lw=3)
    a.plot([sgn * (xo + G["TE_root"][0]), sgn * (xo + G["TE_tip"][0])], [-G["TE_root"][1], -G["TE_tip"][1]], color="#750014", lw=4)
    a.plot([sgn * (xo + G["TE_root"][0]), sgn * 2], [-G["TE_root"][1], -24], color="0.3", lw=0.8)
a.add_patch(Rectangle((-6, -30), 12, 34, fc="0.85", ec="k"))
for sgn in (1, -1):
    a.add_patch(Rectangle((sgn * 3 - 1.5 - (0 if sgn > 0 else 0), -26), 3, 20, fc="#750014", ec="k", alpha=0.7))
a.text(0, -28.5, "body", ha="center", fontsize=8)
a.text(30, 3, "carbon leading edge", fontsize=8)
a.text(22, -22, "rolled DEA edge", fontsize=8, color="#750014")
a.text(-60, -17, "DEA skin", fontsize=8)
a.text(-5, -4, "cores", fontsize=8, color="#750014", rotation=90)
a.annotate("", (63, 7), (-63, 7), arrowprops=dict(arrowstyle="<->", lw=0.8)); a.text(0, 8.5, "%.0f mm tip to tip" % (2 * (6 + G["tip"][0])), ha="center", fontsize=8)
a.set_xlim(-66, 66); a.set_ylim(-34, 14); a.set_aspect("equal"); a.set_xlabel("mm"); a.set_title("From above, to scale")
a = ax[1]
a.add_patch(Rectangle((-6, -8), 12, 16, fc="0.85", ec="k"))
for sgn in (1, -1):
    ang = math.radians(20)
    a.plot([sgn * 6, sgn * (6 + 52 * math.cos(ang))], [0, 52 * math.sin(ang)], color="k", lw=2.5, alpha=0.35)
    a.plot([sgn * 6, sgn * (6 + 52 * math.cos(ang))], [0, -52 * math.sin(ang)], color="k", lw=2.5, alpha=0.35)
    a.plot([sgn * 6, sgn * 58], [0, 0], color="k", lw=2.5)
    a.add_patch(Circle((sgn * 6, 0), 1.0, fc="w", ec="k"))
    a.plot([sgn * 6, sgn * 6], [-2.5, 2.5], color="k", lw=1.2)
    a.add_patch(Rectangle((sgn * 2.5 - 1.5, 2.5), 3, 4.5, fc="#750014", alpha=0.7))
    a.add_patch(Rectangle((sgn * 2.5 - 1.5, -7), 3, 4.5, fc="#750014", alpha=0.7))
a.text(0, 10, "core pair per wing,\nlever r 2.2–2.8 mm", ha="center", fontsize=8)
a.text(40, 21, "±20°", fontsize=9)
a.set_xlim(-62, 62); a.set_ylim(-24, 24); a.set_aspect("equal"); a.set_xlabel("mm"); a.set_title("From the front: stroke")
for a in ax:
    a.grid(False)
fig.tight_layout(); fig.savefig(F("R_robot.png")); plt.close(fig)

# ---------- numbers for the page ----------
out = {}
for p in (8.0, 15.0):
    for ef in (0.0, 0.4, 0.8):
        r = fe("roll", p, 6.0, ef)
        out["dCL p%g ef%.1f" % (p, ef)] = [round(dcl(r, i), 4) for i in range(len(r["res"]))]
for lab, kw in opts:
    out["eq p8 " + lab.replace("\n", " ")] = round(eq(1800, T0=0.043, p=8.0, **kw)[0], 3)
json.dump(out, open(D("figures_numbers.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
