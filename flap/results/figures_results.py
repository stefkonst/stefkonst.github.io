"""Main figures, in the paper's style (npj, 180 mm, square panels; paper_tension/common/style.py).
Fig. 1 bench, Fig. 2 mechanism, Fig. 3 edge design, Fig. 4 robot. Writes PNG, SVG, AI and single panels here and in YEAR2/robot/results.
Run after flap/robot/*: python figures_results.py"""
import json, math, os, sys, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
FLAP = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.expanduser("~"), "Documents", "paper_tension", "common"), os.path.join(FLAP, "robot")]
import style as ST
import matplotlib.pyplot as plt
import params as P
import edge_theory as ET
C, INK2 = ST.C, ST.INK2
RD = lambda n: os.path.join(FLAP, "robot", "data", n)
W = json.load(open(os.path.join(FLAP, "first", "data", "views.json")))
FE = [r for r in map(json.loads, open(RD("edge_fe.jsonl"))) if "res" in r]
CAL = json.load(open(RD("edge_calib.json")))
S = json.load(open(RD("robot_study.json")))
NUM = json.load(open(RD("figures_numbers.json")))
OUTS = [os.path.join(HERE, "fig"), os.path.join(os.path.expanduser("~"), "Documents", "YEAR2", "robot", "results")]
for o in OUTS:
    os.makedirs(o, exist_ok=True)
PX, FPS = 6.14, 200.0


def save(fig, name):
    ST.save(fig, OUTS[0], name)
    for f in os.listdir(OUTS[0]):
        if f.startswith(name) and f.endswith(".png"):
            shutil.copy2(os.path.join(OUTS[0], f), OUTS[1])
    os.makedirs(os.path.join(OUTS[1], "svg"), exist_ok=True)
    for f in os.listdir(OUTS[0]):
        if f.startswith(name) and f.endswith(".svg"):
            shutil.copy2(os.path.join(OUTS[0], f), os.path.join(OUTS[1], "svg"))


def fe(edge, p, pull=None, ef=None, a=None, eta=0.15):
    return next(r for r in FE if r["edge"] == edge and r["p"] == p and r["eta"] == eta and (pull is None or r["pull"] == pull) and (ef is None or r["eta_f"] == ef) and (a is None or r.get("a") == a))


def dz(r):
    return [q["V"] for q in r["res"]], [q["z_te_mid"] - r["res"][0]["z_te_mid"] for q in r["res"]]


def eq(V, **kw):
    a = ET.solve(V, N0=CAL["N0"], c=CAL["c"], **kw); b = ET.solve(0.0, N0=CAL["N0"], c=CAL["c"], **kw)
    return -(a["d"] - b["d"]) * CAL["qf"]


# ---------------- Fig. 1: the wing on the bench ----------------
ST.LETTER_START = 2
fig, ax = ST.make_fig(4)
a = ax[0]
for nm, col, lab in (("1800V_1hz_side", C[1], "out of plane (side)"), ("1800V_1hz_front_avi", C[0], "in plane (front)")):
    z = np.load(os.path.join(HERE, "data", nm + ".npz"))
    r = W[nm]
    u = (-1 if r["view"] != "front" else 1) * z["te_u"][:, r["station"]] / PX
    t = np.arange(len(u)) / FPS
    u = u - u[t < .15].mean()
    m = t < 4
    a.plot(t[m], u[m], color=col, lw=0.8, label=lab)
a.set(xlabel="Time (s)", ylabel="Trailing-edge motion (mm)", ylim=(-0.1, 2.5))
a.legend(loc="upper left", ncol=1)
ST.note(a, "1,800 V, 1 Hz", xy=(0.96, 0.06), ha="right", va="bottom")
a = ax[1]
V = np.array([1000, 1400, 1800.0])
hi = np.array([W["%dV_1hz_side" % v]["fit_low"] + W["%dV_1hz_side" % v]["fit_amp"] for v in V])
a.loglog(V, hi, "o", color=C[1], label="measured, side view")
k = np.polyfit(np.log(V), np.log(hi), 1)
vv = np.linspace(900, 2000, 20)
a.loglog(vv, np.exp(np.polyval(k, vv * 0 + np.log(vv))), "-", color=INK2, lw=0.8, label=r"$\propto V^{%.1f}$" % k[0])
a.set(xlabel="Voltage (V)", ylabel="Out-of-plane high (mm)")
a.set_xticks([1000, 1400, 1800]); a.set_xticklabels(["1,000", "1,400", "1,800"]); a.minorticks_off()
a.legend(loc="lower right")
a = ax[2]
for ten, col, lab in ((False, C[1], "as built"), (True, C[0], "more tension")):
    q = sorted((r for r in W.values() if r["view"] == "diag" and r["tension"] == ten), key=lambda r: r["f"])
    a.loglog([r["f"] for r in q], [r["A1_mm"] for r in q], "o-", color=col, label=lab)
a.set(xlabel="Frequency (Hz)", ylabel="Swing amplitude (mm)")
a.set_xticks([1, 2, 5, 10, 20]); a.set_xticklabels(["1", "2", "5", "10", "20"])
a.legend(loc="upper right")
a = ax[3]
lab = ["cut", "measured"]
L = [P.ROLL_CUT * 1e3, np.mean(P.ROLL_LEN) * 1e3]
a.bar([0, 1], L, color=[C[0], C[1]], width=0.55, yerr=[0, (P.ROLL_LEN[1] - P.ROLL_LEN[0]) * 1e3 / 2])
a.set_xticks([0, 1]); a.set_xticklabels(lab); a.minorticks_off()
a.set(ylabel="Roll length (mm)", ylim=(0, 75))
ST.note(a, "stretch 1.16–1.22\ntension 50–96 mN", xy=(0.04, 0.96))
save(fig, "Fig1_bench")

# ---------------- Fig. 2: the edge goes slack ----------------
ST.LETTER_START = 2
fig, ax = ST.make_fig(4)
a = ax[0]
for ef, col in ((0.0, C[0]), (0.4, C[1]), (0.8, C[3])):
    v, z = dz(fe("roll", 0.0, 6.0, ef)); a.plot(v, z, "o-", color=col, label=r"$\eta_f$ = %.1f" % ef)
a.plot(V, -hi, "k*", ms=6, label="measured")
a.set(xlabel="Voltage (V)", ylabel=r"Edge height change (mm)")
a.legend(loc="lower left")
a = ax[1]
fe_, eq_ = np.array(CAL["fe"]), np.array(CAL["eq"])
a.plot(fe_, eq_, ".", color=C[0], ms=3)
lim = [min(fe_.min(), eq_.min()) - .2, max(fe_.max(), eq_.max()) + .2]
a.plot(lim, lim, "-", color=INK2, lw=0.6)
a.set(xlabel="Finite elements (mm)", ylabel="Edge equation (mm)", xlim=lim, ylim=lim)
ST.note(a, "160 points, r = %.2f\nrms %.2f mm" % (CAL["corr"], CAL["rms_mm"]))
a = ax[2]
E = S["edge_mode"]
for (kk, v), col in zip(E["f"].items(), (C[1], C[3], C[2], C[0])):
    a.plot(E["V"], v, color=col, label=kk.replace("eta_f", r"$\eta_f$").replace("roll, ", ""))
a.axhspan(18, 25, color=LIGHT if (LIGHT := ST.LIGHT) else "0.9", alpha=0.6, lw=0)
a.plot([1800], [15], "k*", ms=6)
a.set(xlabel="Voltage (V)", ylabel="Edge mode (Hz)", ylim=(0, 60))
a.legend(loc="lower left", fontsize=5.5)
a = ax[3]
for p, col in ((8.0, C[0]), (15.0, C[1])):
    for ef, ls in ((0.0, ":"), (0.4, "-"), (0.8, "--")):
        a.plot([0, 500, 1000, 1400, 1800], NUM["dCL p%g ef%.1f" % (p, ef)], ls, color=col, label=r"%g Pa, $\eta_f$ %.1f" % (p, ef))
a.axhline(0, color=INK2, lw=0.5)
a.set(xlabel="Voltage (V)", ylabel=r"$\Delta C_L$ (thin aerofoil)")
a.legend(loc="lower left", fontsize=5)
save(fig, "Fig2_mechanism")

# ---------------- Fig. 3: designing the edge ----------------
ST.LETTER_START = 0
fig, ax = ST.make_fig(4)
a = ax[0]
Vs = np.linspace(0, 1800, 19)
for (lab, kw), col in zip((("active roll", dict(edge="roll", eta_f=0.4)), ("passive roll", dict(edge="roll", eta_f=0.0)), ("TPU", dict(edge="TPU 1 x 0.1 mm")), ("nylon", dict(edge="nylon 0.10 mm")), ("half roll + nylon", dict(edge="roll", a=0.5, passive="nylon 0.10 mm", eta_f=0.4))), (C[1], C[0], C[2], C[3], C[4])):
    a.plot(Vs, [eq(v, T0=0.043, p=8.0, **kw) for v in Vs], color=col, label=lab)
a.set(xlabel="Voltage (V)", ylabel="Edge height change (mm)")
a.legend(loc="upper left", fontsize=5.5)
a = ax[1]
aa = np.linspace(0.05, 1, 96)
EAr = 6 * P.C1 * 2.16 * P.ROLL_W * P.FILM_T
for lab, EAp, col in (("unpowered roll", EAr, C[0]), ("TPU", P.CORDS["TPU 1 x 0.1 mm"]["E"] * P.CORDS["TPU 1 x 0.1 mm"]["A"], C[2]), ("nylon", P.CORDS["nylon 0.10 mm"]["E"] * P.CORDS["nylon 0.10 mm"]["A"], C[3])):
    ka, kp = EAr / (aa * P.EDGE_L), EAp / np.maximum((1 - aa) * P.EDGE_L, 1e-9)
    a.plot(aa * 100, kp / (ka + kp), color=col, label=lab)
for x in (0.25, 0.5, 0.75):
    pass
a.set(xlabel="Active share of the edge (%)", ylabel=r"$\Delta T / \Delta F$", ylim=(0, 1.05))
a.legend(loc="lower right", title="rest of the edge", title_fontsize=5.5)
a = ax[2]
for p, col in ((0.0, C[0]), (15.0, C[1])):
    pts = [(0, 0)] + [(x * 100, abs(dz(fe("part", p, 6.0, 0.4, a=x))[1][-1])) for x in (0.25, 0.5, 0.75)] + [(100, abs(dz(fe("roll", p, 6.0, 0.4))[1][-1]))]
    a.plot(*zip(*pts), "o-", color=col, label="%g Pa" % p)
a.plot([0, 100], [0, abs(dz(fe("roll", 15.0, 6.0, 0.4))[1][-1])], ":", color=INK2, lw=0.6)
a.set(xlabel="Active share, unpowered rest (%)", ylabel="Edge motion at 1,800 V (mm)")
a.legend(loc="upper left")
a = ax[3]
T = S["thickness"]; t = np.array(T["t_um"])
a.plot(t, T["2 electrodes, 60 V/um"]["dF_mN"], color=C[1], label="60 V µm$^{-1}$")
a.plot(t, T["2 electrodes, 46 V/um"]["dF_mN"], color=C[0], label="46 V µm$^{-1}$")
a.axhspan(0.5 * T["T_needed_mN"], T["T_needed_mN"], color=ST.LIGHT, alpha=0.6, lw=0)
a.axvspan(45, 90, color=C[2], alpha=0.12, lw=0)
a.set(xlabel="Film thickness (µm)", ylabel="Force the roll loses (mN)", xlim=(20, 200))
a.legend(loc="upper left")
save(fig, "Fig3_edge_design")

# ---------------- Fig. 4: the robot ----------------
ST.LETTER_START = 0
fig, ax = ST.make_fig(4)
R = S["resonance"]
a = ax[0]
a.plot(R["r_mm"], R["f"]["wing in air"], color=C[1])
a.axhspan(18, 25, color=ST.LIGHT, alpha=0.6, lw=0)
a.set(xlabel="Lever arm (mm)", ylabel="Stroke resonance (Hz)")
a = ax[1]
TT = S["te_transfer"]
for (kk, v), col, ls in zip(TT["cases"].items(), (C[0], C[0], C[1], C[1]), (":", "-", ":", "-")):
    a.semilogy(TT["f"], v["amp"], ls, color=col, label=kk)
a.axvspan(18, 25, color=ST.LIGHT, alpha=0.6, lw=0)
a.set(xlabel="Frequency (Hz)", ylabel="Edge / leading-edge motion", ylim=(0.01, 30))
a.legend(loc="lower right", fontsize=5)
a = ax[2]
Fl = S["flight"]; U = np.array(Fl["U"])
a.fill_between(U, Fl["P_req_mW"]["optimistic"], Fl["P_req_mW"]["pessimistic"], color=C[0], alpha=0.15, lw=0)
a.plot(U, Fl["P_req_mW"]["model"], color=C[0], label="required")
for (kk, v), col in zip(Fl["P_avail_mW_25Hz"].items(), (C[1], C[3], C[4])):
    a.axhline(v, color=col, ls="--", lw=0.8, label="%g J kg$^{-1}$" % P.E_CYCLE[kk])
a.axvspan(2, Fl["U_min"], color="0.9", lw=0)
a.set(xlabel="Speed (m s$^{-1}$)", ylabel="Power (mW)", xlim=(2, 8), ylim=(0, 150))
a.legend(loc="upper left", fontsize=5.5)
a = ax[3]
MC = S["mc"]
a.plot(MC["e_cyc"], MC["margin"], ".", ms=1, color=C[0], alpha=0.35)
a.axhline(1, color="k", lw=0.6)
a.set(xlabel="Energy per cycle (J kg$^{-1}$)", ylabel="Power available / required")
ST.note(a, "%d %% of 20,000 draws $\\geq$ 1" % round(100 * MC["p_fly"]), xy=(0.04, 0.96))
save(fig, "Fig4_robot")

# ---------------- Fig. 5: the skin, the edge and fibres in air (flap/air) ----------------
AD = lambda n: os.path.join(FLAP, "air", "data", n)
if os.path.exists(AD("fe_flow.jsonl")):
    FL = [r for r in map(json.loads, open(AD("fe_flow.jsonl"))) if "res" in r]
    LAG = [r for r in map(json.loads, open(AD("dyn_lag.jsonl"))) if "L_m" in r]
    DFL = [r for r in map(json.loads, open(AD("dyn_lagflight.jsonl"))) if "L_m" in r] + [r for r in map(json.loads, open(AD("dyn_flight.jsonl"))) if "L_m" in r]
    flw = lambda d, U, al: next(r for r in FL if r["design"] == d and r["U"] == U and r["alpha"] == al)
    ST.LETTER_START = 2
    fig, ax = ST.make_fig(4)
    a = ax[0]
    for d, col, lab in (("skin", C[0], r"skin, $\eta_s$ 0.15"), ("skin strong", C[4], r"skin, $\eta_s$ 0.4"), ("edge", C[1], r"edge, $\eta_f$ 0.4"), ("fibres", C[3], "4 fibres, 0.15"), ("fibres, eta 0.4", "#9fd36a", "4 fibres, 0.4")):
        r = flw(d, 4.0, 9.0)
        q = [x for x in r["res"] if x.get("CL") is not None and x["ok"]]
        a.plot([x["V"] / 1e3 for x in q], [x["CL"] - r["res"][0]["CL"] for x in q], "o-", ms=2.5, color=col, label=lab)
    a.axhline(0, color="k", lw=0.5)
    a.set(xlabel="Voltage (kV)", ylabel=r"$\Delta C_L$ (4 m s$^{-1}$, 9°)")
    a.legend(loc="upper left", fontsize=5)
    a = ax[1]
    for d, col in (("skin", C[0]), ("skin strong", C[4])):
        for U, mk in ((3.0, "s"), (4.0, "o"), (6.0, "^")):
            r = next((r for r in FL if r["design"] == d and r["U"] == U and r["alpha"] in (9.0, 6.0) and (U != 4.0 or r["alpha"] == 9.0)), None)
            ct = [x["Nc"] / (0.6 * U ** 2 * 0.016) for x in r["res"]]; cl = [x["CL"] for x in r["res"]]
            a.plot(ct, cl, mk + "-", ms=2.5, color=col)
    a.axvline(1.73, color="0.5", lw=0.6)
    ST.note(a, "2D divergence", xy=(0.03, 0.96))
    a.set(xscale="log", xlabel=r"$C_T = N_c/(qc)$", ylabel=r"$C_L$")
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    a.xaxis.set_major_locator(FixedLocator([1.5, 2, 3, 4, 6])); a.xaxis.set_major_formatter(FixedFormatter(["1.5", "2", "3", "4", "6"])); a.xaxis.set_minor_locator(NullLocator())
    ST.note(a, "▲ 6, ● 4, ■ 3 m s$^{-1}$", xy=(0.45, 0.10))
    a = ax[2]
    mf = sorted([(W[k]["f"], W[k]["A1_mm"]) for k in W if k.startswith("1800V") and k.endswith("diag_sine")])
    a.loglog([m[0] for m in mf], [m[1] / mf[0][1] for m in mf], "ks", ms=3, label="measured")
    for tau, col in ((0.02, C[0]), (0.05, C[1]), (0.1, C[3])):
        rr = sorted([r for r in LAG if r["tau"] == tau], key=lambda r: r["f_v"])
        a.loglog([r["f_v"] for r in rr], [r["w_te_a1"] / rr[0]["w_te_a1"] for r in rr], "-", color=col, label=r"$\tau$ = %g ms" % (tau * 1e3))
    a.set(xlabel="Drive frequency (Hz)", ylabel="Edge motion / 1 Hz value")
    a.legend(loc="lower left", fontsize=5)
    a = ax[3]
    g = lambda d, m, tau: next(r for r in DFL if r["design"] == d and r["mode"] == m and r.get("alpha_deg") == 10.0 and r.get("U", 4.0) == 4.0 and r.get("f", 25.0) == 25.0 and r.get("tau", 0.0) == tau)
    off = g("passive", "off", 0.0)
    for d, col, lab in (("skin strong", C[4], "skin 0.4"), ("edge", C[1], "edge 0.4"), ("both", C[3], "both")):
        rs = [off, g(d, "sin", 0.05), g(d, "on", 0.05)]
        a.plot([r["T_m"] * 1e3 for r in rs], [r["L_m"] * 1e3 for r in rs], "o-", ms=3, color=col, label=lab)
    a.axhline(18.4 / 2, color="k", ls="--", lw=0.6)
    ST.note(a, "0 V → half → full", xy=(0.04, 0.06))
    a.set(xlabel="Thrust per wing (mN)", ylabel="Lift per wing (mN)")
    a.legend(loc="upper right", fontsize=5)
    save(fig, "Fig5_air")
print("saved", os.listdir(OUTS[0]))
