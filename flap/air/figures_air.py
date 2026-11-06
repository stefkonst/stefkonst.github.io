"""Figures for the air page (style of the SMS 2023 / CST 2024 papers, plotstyle.py; every PNG gets an SVG).
Run after fe_air.py bench, fe_air.py flow, flap_dyn.py bench, flap_dyn.py flight: python figures_air.py -> A*.png/svg, data/air_numbers.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "plagiopatagiales"), os.path.join(HERE, "..", "robot")]
import plotstyle as PS
import matplotlib.pyplot as plt
import params as P
PS.apply()
C = PS.MATLAB
D = lambda n: os.path.join(HERE, "data", n)
F = lambda n: os.path.join(HERE, n)
load = lambda n: [r for r in (json.loads(l) for l in open(D(n))) if "res" in r or "L_m" in r] if os.path.exists(D(n)) else []
BENCH, FLOW = load("fe_bench.jsonl"), load("fe_flow.jsonl")
DB, DF = load("dyn_bench.jsonl"), load("dyn_flight.jsonl")
DLAG, DLF = load("dyn_lag.jsonl"), load("dyn_lagflight.jsonl")
SEC = json.load(open(D("section_curve.json")))
W = json.load(open(os.path.join(HERE, "..", "first", "data", "views.json")))
NUM = {}
COL = {"passive": "0.45", "skin": C[0], "skin strong": "#003f7f", "edge": C[1], "both": C[3], "both strong": "#4b1a55",
       "fibres": C[4], "fibres, low stretch": "#3d6b10", "fibres, eta 0.4": "#9fd36a"}
LAB = {"passive": "passive skin and edge", "skin": r"active skin, $\eta_s$ 0.15", "skin strong": r"active skin, $\eta_s$ 0.4", "edge": r"active edge, $\eta_f$ 0.4",
       "both": r"both (as built), 0.15 / 0.4", "both strong": "both, 0.4 / 0.8", "fibres": r"4 fibres, $\lambda$ 1.10, $\eta$ 0.15",
       "fibres, low stretch": r"4 fibres, $\lambda$ 1.03", "fibres, eta 0.4": r"4 fibres, $\lambda$ 1.10, $\eta$ 0.4"}
MAXW = lambda V, eta: eta * 1 * P.EPS * (V / 30e-6) ** 2 * 30e-6       # Maxwell resultant of the skin, N/m (1 active layer)
Q = lambda U: 0.5 * 1.2 * U ** 2


def bench(d, p):
    for r in BENCH:
        if r["design"] == d and r["p"] == p:
            return r


def flow(d, U, a):
    for r in FLOW:
        if r["design"] == d and r["U"] == U and r["alpha"] == a:
            return r


def col(r, key):
    return np.array([q["V"] for q in r["res"] if key in q]), np.array([q[key] for q in r["res"] if key in q]), np.array([q.get("ok", False) for q in r["res"] if key in q])


def dcl_chord(prof, ref):
    """Thin aerofoil lift change from the mid-span chord, state against ref: 2 int dz/dy (cos th - 1) dth (z up, y aft)."""
    a, b = np.array(prof) * 1e-3, np.array(ref) * 1e-3
    y = a[:, 0]; c = y.max() - y.min()
    pz = np.polyfit((y - y.min()) / c, a[:, 1] - np.interp(y, b[:, 0], b[:, 1]), 4)
    th = np.linspace(0, math.pi, 400)
    return 2 * np.trapz(np.polyval(np.polyder(pz), (1 - np.cos(th)) / 2) / c * (np.cos(th) - 1), th)


# =========================================================== A1. where the skin's pull goes (bench)
fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.0))
a = ax[0, 0]
for d in ("passive", "skin", "skin strong", "edge", "both", "both strong"):
    r = bench(d, 0.0)
    V, Nc, _ = col(r, "Nc")
    a.plot(V, Nc, "o-", color=COL[d], label=LAB[d])
Vf = np.linspace(0, 1800, 50)
N0 = bench("passive", 0.0)["res"][0]["Nc"]
for eta, ls in ((0.15, "--"), (0.4, ":")):
    a.plot(Vf, np.maximum(N0 - MAXW(Vf, eta), 0), ls, color="k", lw=0.9, label=r"if all of $P$ became tension, $\eta_s$ %.2g" % eta)
a.set(xlabel="Voltage $V$ in V", ylabel="Skin tension across the chord $N_c$ in N/m", ylim=(0, 0.6))
a.legend(fontsize=6.3, loc="lower left")
NUM["Nc0"] = N0
NUM["Nc_1800"] = {d: bench(d, 0.0)["res"][-1]["Nc"] for d in ("skin", "skin strong", "edge", "both", "both strong")}
NUM["Ns0"] = bench("passive", 0.0)["res"][0]["Ns"]
NUM["Ns_1800"] = {d: bench(d, 0.0)["res"][-1]["Ns"] for d in ("skin", "skin strong", "edge", "both", "both strong")}
NUM["P_1800"] = {"0.15": MAXW(1800, 0.15), "0.4": MAXW(1800, 0.4)}
NUM["retained_chord"] = {e: (N0 - NUM["Nc_1800"][d]) / MAXW(1800, float(e)) for e, d in (("0.15", "skin"), ("0.4", "skin strong"))}
NUM["retained_span"] = {e: (NUM["Ns0"] - NUM["Ns_1800"][d]) / MAXW(1800, float(e)) for e, d in (("0.15", "skin"), ("0.4", "skin strong"))}

a = ax[0, 1]
for d in ("passive", "skin", "skin strong", "edge", "both", "both strong", "fibres", "fibres, eta 0.4"):
    r = bench(d, 0.0)
    V, s, _ = col(r, "sag")
    a.plot(V, s - s[0], "o-", color=COL[d], label=LAB[d])
a.axhline(0, color="k", lw=0.6)
a.text(1790, -1.85, "edge moves aft, straighter", fontsize=7.5, ha="right"); a.text(60, 1.25, "edge bows towards the LE", fontsize=7.5)
a.set(xlabel="Voltage $V$ in V", ylabel="Change of the in-plane bow $\\Delta s$ in mm", ylim=(-1.9, 1.4))
a.legend(fontsize=6, loc="lower left", ncol=1, bbox_to_anchor=(0, 0.06))
NUM["ds_1800"] = {d: bench(d, 0.0)["res"][-1]["sag"] - bench(d, 0.0)["res"][0]["sag"] for d in ("skin", "skin strong", "edge", "both", "fibres", "fibres, eta 0.4")}

a = ax[1, 0]
ps = (0.0, 4.0, 8.0, 15.0)
for d in ("skin", "skin strong", "edge", "both", "fibres", "fibres, eta 0.4"):
    dz = [bench(d, p)["res"][-1]["z_te_mid"] - bench(d, p)["res"][0]["z_te_mid"] for p in ps]
    a.plot(ps, dz, "o-", color=COL[d], label=LAB[d])
a.axvspan(8, 15, color="0.9", zorder=0); a.text(8.4, -1.6, "flight\nload", fontsize=7.5)
a.axvline(P.RHO_F * 60e-6 * 9.81, color="k", ls=":", lw=0.8); a.text(0.8, 2.2, "bench: own weight only, 0.65 Pa", fontsize=7.5)
a.set(xlabel="Pressure on the wing $p$ in Pa (up, like lift)", ylabel="Edge height change at 1800 V $\\Delta z$ in mm")
a.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0, 0.93))

a = ax[1, 1]
for d in ("skin", "skin strong", "edge", "both"):
    for p, ls in ((0.0, "-"), (15.0, "--")):
        r = bench(d, p)
        a.plot([q["V"] for q in r["res"]], [dcl_chord(q["chord"], r["res"][0]["chord"]) for q in r["res"]], "o" + ls, color=COL[d], ms=3,
               label=(LAB[d] if p == 15.0 else None))
a.plot([], [], "k-", label="0 Pa (bench)"); a.plot([], [], "k--", label="15 Pa")
a.axhline(0, color="k", lw=0.6)
a.set(xlabel="Voltage $V$ in V", ylabel="Thin-aerofoil $\\Delta C_L$ from the mid-span chord")
a.legend(fontsize=6.3, loc="lower left")
NUM["dCL_bench_15Pa"] = {d: dcl_chord(bench(d, 15.0)["res"][-1]["chord"], bench(d, 15.0)["res"][0]["chord"]) for d in ("skin", "skin strong", "edge", "both")}
PS.finish(ax); fig.tight_layout(); fig.savefig(F("A_skin.png")); plt.close(fig)

# =========================================================== A2. lift in flow
if FLOW:
    fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.0))
    al = (3.0, 6.0, 9.0, 12.0)
    a = ax[0, 0]
    pas = [flow("passive", 4.0, x) for x in al]
    a.plot(al, [r["res"][0]["CL"] for r in pas if r], "k-o", label="0 V (all designs)")
    for d, Vq in (("skin", 1800.0), ("skin strong", 1800.0), ("fibres", 1800.0), ("fibres, eta 0.4", 1800.0), ("edge", 1400.0)):
        xs, ys = [], []
        for x in al:
            r = flow(d, 4.0, x)
            q = next((q for q in r["res"] if q["V"] == Vq), None) if r else None
            if q and q.get("ok") and q.get("CL") is not None:
                xs.append(x); ys.append(q["CL"])
        if xs:
            a.plot(xs, ys, "o-", color=COL[d], label=LAB[d] + ", %d V" % Vq)
    a.plot(np.array(al), 2 * math.pi * np.radians(al) * 7.4 / 9.4, "k:", lw=0.8, label=r"flat plate, $AR$ 7.4")
    a.axhspan(1.1, 2.2, color="0.92", zorder=0); a.text(3.1, 1.75, "likely stalled at Re 4000", fontsize=7.5)
    a.set(xlabel=r"Angle of attack $\alpha$ in °", ylabel=r"Wing lift coefficient $C_L$ ($U$ = 4 m/s)", ylim=(0, 2.0))
    a.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0, 0.83))

    a = ax[0, 1]
    for d in ("skin", "skin strong", "edge", "both", "both strong", "fibres", "fibres, low stretch", "fibres, eta 0.4"):
        r = flow(d, 4.0, 9.0)
        if not r:
            continue
        V = np.array([q["V"] for q in r["res"] if q.get("CL") is not None]); cl = np.array([q["CL"] for q in r["res"] if q.get("CL") is not None]); ok = np.array([q["ok"] for q in r["res"] if q.get("CL") is not None])
        a.plot(V[ok], cl[ok] - cl[0], "o-", color=COL[d], label=LAB[d])
        if (~ok).any():
            a.plot(V[~ok], cl[~ok] - cl[0], "o", mfc="w", color=COL[d])
    a.axhline(0, color="k", lw=0.6)
    a.set(xlabel="Voltage $V$ in V", ylabel=r"$\Delta C_L$ ($U$ = 4 m/s, $\alpha$ = 9°)")
    a.text(0.03, 0.04, "open: not settled (slack edge); off the axis: 1.5", transform=a.transAxes, fontsize=7)
    a.set_ylim(-0.35, 0.5)
    a.legend(fontsize=6, loc="upper left")
    NUM["dCL_flow"] = {}
    for r in FLOW:
        last = r["res"][-1]
        if last.get("CL") is not None:
            NUM["dCL_flow"]["%s | U %g a %g" % (r["design"], r["U"], r["alpha"])] = dict(CL0=r["res"][0]["CL"], dCL=last["CL"] - r["res"][0]["CL"], ok=last["ok"],
                                                                                      dCL_1400=next((q["CL"] - r["res"][0]["CL"] for q in r["res"] if q["V"] == 1400.0 and q.get("CL") is not None), None),
                                                                                      ok_1400=next((q["ok"] for q in r["res"] if q["V"] == 1400.0), None), S_cm2=r["res"][0]["S_cm2"])

    a = ax[1, 0]
    for k, ls in (("6.0", "--"), ("9.0", "-"), ("12.0", ":")):
        a.plot(SEC[k]["CT"], SEC[k]["CL"], ls, color="k", lw=0.9, label=r"2D membrane, $\alpha$ %s°" % k[:-2])
    a.axvline(SEC["eig"], color="0.5", lw=0.8); a.text(SEC["eig"] * 1.04, 2.85, "divergence", fontsize=7.5, color="0.4")
    for U, mk in ((3.0, "s"), (4.0, "o"), (6.0, "^")):
        for d in ("skin", "skin strong"):
            r = flow(d, U, 9.0) or flow(d, U, 6.0)
            if not r:
                continue
            q = Q(U) * 0.016
            ct = [x["Nc"] / q for x in r["res"]]; cl = [x["CL"] for x in r["res"]]
            a.plot(ct, cl, mk + "-", color=COL[d], ms=4, label="%s, %g m/s, %g°" % (LAB[d], U, r["alpha"]))
            a.annotate("", xy=(ct[-1], cl[-1]), xytext=(ct[0], cl[0]), arrowprops=dict(arrowstyle="->", color=COL[d], lw=0.8))
    a.text(12, 0.42, "the 3D wing sits lower\n(finite span, wrinkles)", fontsize=7.5)
    a.set(xscale="log", xlabel=r"Tension coefficient $C_T = N_c/(q\,c)$", ylabel="$C_L$", xlim=(1.2, 40), ylim=(0.3, 3.0))
    a.legend(fontsize=5.6, loc="upper right")

    a = ax[1, 1]
    for d, off in (("skin strong", 0.0), ("edge", 0.0), ("fibres, eta 0.4", 0.0)):
        r = flow(d, 4.0, 9.0) if d != "edge" else (flow(d, 3.0, 9.0))
        if not r:
            continue
        for q, ls in ((r["res"][0], ":"), (r["res"][-1], "-")):
            if "chord" not in q:
                continue
            pr = np.array(q["chord"]); yy = pr[:, 0] - pr[0, 0]
            pz = np.polyfit(yy, pr[:, 1], 4); yf = np.linspace(0, yy.max(), 60)
            a.plot(yy, pr[:, 1], ".", color=COL[d], ms=2, alpha=0.5)
            a.plot(yf, np.polyval(pz, yf), ls, color=COL[d], label="%s, %s%s" % (LAB[d], "0 V" if q["V"] == 0 else "1800 V", "" if d != "edge" else " (3 m/s)"))
    a.set(xlabel="Chord position from the LE $y$ in mm", ylabel="Height at mid span $z$ in mm ($\\alpha$ = 9°)")
    a.legend(fontsize=6, loc="upper left")
    PS.finish(ax); fig.tight_layout(); fig.savefig(F("A_lift.png")); plt.close(fig)

# =========================================================== A2b. a slack edge has no steady state in flow
if FLOW:
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    a = ax[0]
    pts = []
    for r in FLOW:
        if r["design"] not in ("edge", "both", "both strong"):
            continue
        for q in r["res"]:
            if "T_mN" not in q:
                continue
            pts.append((Q(r["U"]), q["T_mN"], q["ok"], r["U"], r["design"], q["V"]))
    for U, mk in ((3.0, "s"), (4.0, "o"), (6.0, "^")):
        for ok in (True, False):
            s_ = [p_ for p_ in pts if p_[3] == U and p_[2] == ok]
            a.plot([p_[5] for p_ in s_], [p_[1] for p_ in s_], mk, color="k" if ok else "r", mfc=("k" if ok else "w"), ms=4.5, ls="none",
                   label="%g m/s, %s" % (U, "settled" if ok else "no steady state"))
    a.set(xlabel="Voltage $V$ in V", ylabel="Edge tension $T$ in mN (last iterate)", ylim=(-1, 32))
    a.legend(fontsize=6.3, loc="upper right", ncol=1)
    a = ax[1]
    for ok in (True, False):
        s_ = [p_ for p_ in pts if p_[2] == ok]
        a.plot([p_[0] for p_ in s_], [p_[1] for p_ in s_], "o", color="k" if ok else "r", mfc=("k" if ok else "w"), ms=4.5, ls="none", label="settled" if ok else "no steady state")
    # lowest settled tension above which every case settles, per speed
    lim = {}
    for U in (3.0, 4.0, 6.0):
        bad = [p_[1] for p_ in pts if p_[3] == U and not p_[2]]
        lim[U] = max(bad) if bad else 0.0
    qq = np.array([Q(u) for u in lim]); tt = np.array(list(lim.values()))
    k_ = np.polyfit(qq, tt, 1)
    qf = np.linspace(0, 25, 50)
    a.plot(qf, np.maximum(np.polyval(k_, qf), 0), "r--", lw=0.9, label=r"$T_{min} \approx %.1f\,(q - %.1f)$ mN" % (k_[0], -k_[1] / k_[0]))
    a.set(xlabel="Dynamic pressure $q$ in Pa", ylabel="Edge tension $T$ in mN", xlim=(0, 25), ylim=(-1, 32))
    a.legend(fontsize=6.5, loc="upper left")
    NUM["T_min"] = dict(per_U={str(u): v for u, v in lim.items()}, slope=k_[0], q0=-k_[1] / k_[0])
    PS.finish(ax); fig.tight_layout(); fig.savefig(F("A_slack.png")); plt.close(fig)

# =========================================================== A3. three ways to make a wing active
fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.8))
a = ax[0]
names = ["skin", "skin strong", "edge", "both", "fibres", "fibres, low stretch", "fibres, eta 0.4"]
lab3 = ["skin\n0.15", "skin\n0.4", "edge\n0.4", "both\n(built)", "fibres\n1.10", "fibres\n1.03", "fibres\n0.4"]
vals, oks = [], []
for d in names:
    r = flow(d, 4.0, 9.0)
    if r and r["res"][-1].get("CL") is not None:
        vals.append(r["res"][-1]["CL"] - r["res"][0]["CL"]); oks.append(r["res"][-1]["ok"])
    else:
        vals.append(np.nan); oks.append(False)
x = np.arange(len(names))
a.bar(x, vals, color=[COL[d] for d in names], edgecolor="k", lw=0.5, hatch=None)
for i, (v, ok) in enumerate(zip(vals, oks)):
    if np.isfinite(v):
        a.text(i, v + (0.01 if v >= 0 else -0.04), "%+.2f%s" % (v, "" if ok else "*"), ha="center", fontsize=7)
a.axhline(0, color="k", lw=0.6)
a.set_xticks(x); a.set_xticklabels(lab3, fontsize=7)
a.set(ylabel=r"$\Delta C_L$ at 1800 V (4 m/s, 9°)")
a.text(0.02, 0.92, "* not settled", transform=a.transAxes, fontsize=7)
NUM["bars_dCL"] = dict(zip(names, vals))

a = ax[1]
A_f = 12e-3 * 60e-6
lam = 1.10
F_ideal = A_f * 2 * (lam - lam ** -2) * 42842 * 1e3
fb = bench("fibres", 0.0)["res"]; fb4 = bench("fibres, eta 0.4", 0.0)["res"]
dF_ideal = lambda eta: eta * 2 * P.EPS * (1800 / 30e-6) ** 2 * 30e-6 * 10e-3 * 1e3
bars = [("pre-tension\nas cut", F_ideal, "0.6"), ("held by\nthe edge", fb[0]["F_fib_mN"], C[4]),
        ("drop, ideal\n$\\eta$ 0.15", dF_ideal(0.15), "0.6"), ("drop in\nthe wing", fb[0]["F_fib_mN"] - fb[-1]["F_fib_mN"], C[4]),
        ("drop, ideal\n$\\eta$ 0.4", dF_ideal(0.4), "0.6"), ("drop in\nthe wing", fb4[0]["F_fib_mN"] - fb4[-1]["F_fib_mN"], "#9fd36a")]
a.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], edgecolor="k", lw=0.5)
for i, b in enumerate(bars):
    a.text(i, b[1] + 0.3, "%.1f" % b[1], ha="center", fontsize=7)
a.set_xticks(range(len(bars))); a.set_xticklabels([b[0] for b in bars], fontsize=6.5)
a.set(ylabel="Force per chordwise fibre in mN")
NUM["fibre"] = dict(F_ideal=F_ideal, F_held=fb[0]["F_fib_mN"], dF_ideal_015=dF_ideal(0.15), dF_wing_015=fb[0]["F_fib_mN"] - fb[-1]["F_fib_mN"], dF_ideal_04=dF_ideal(0.4), dF_wing_04=fb4[0]["F_fib_mN"] - fb4[-1]["F_fib_mN"],
                    mass_mg=4 * 1100 * 12e-3 * 60e-6 * 16e-3 * 1e6)

a = ax[2]
meas = [("earlier wing\n(FAAV, 1.6 kV,\nchord 70 mm)", 0.3 / 70 * 100, (0.2 / 70 * 100, 0.4 / 70 * 100), "0.6"),
        ("this wing\n(1.8 kV,\nchord 16 mm)", 1.84 / 16 * 100, None, C[3])]
for i, (lb, v, rng, cc) in enumerate(meas):
    a.bar(i, v, color=cc, edgecolor="k", lw=0.5)
    if rng:
        a.errorbar(i, v, yerr=[[v - rng[0]], [rng[1] - v]], color="k", capsize=3, lw=0.8)
    a.text(i, v + 0.4, "%.1f %%" % v if v > 1 else "%.2f-%.2f %%" % rng, ha="center", fontsize=7.5)
a.set_xticks(range(2)); a.set_xticklabels([m[0] for m in meas], fontsize=7)
a.set(ylabel="Measured edge motion / chord in %", ylim=(0, 14))
PS.finish(ax); fig.tight_layout(); fig.savefig(F("A_compare.png")); plt.close(fig)

# =========================================================== A4. the time model against the bench
if DB:
    fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.8))
    a = ax[0]
    mf = sorted([(W[k]["f"], W[k]["A1_mm"]) for k in W if k.startswith("1800V") and k.endswith("diag_sine")])
    m1 = dict(mf)[1.0]
    a.plot([m[0] for m in mf], [m[1] / m1 for m in mf], "ks", ms=5, label="measured (diagonal view)", zorder=5)
    for zt, ls in ((0.03, ":"), (0.1, "--")):
        rr = sorted([r for r in DB if r["design"] == "both" and r.get("drive") == "sine" and r.get("zeta") == zt], key=lambda r: r["f_v"])
        r1 = next((r for r in rr if r["f_v"] == 1.0), None)
        if r1:
            a.plot([r["f_v"] for r in rr], [r["w_te_a1"] / r1["w_te_a1"] for r in rr], ls, marker="o", ms=2.5, color="0.45", label=r"model, no lag, $\zeta$ %.2g" % zt)
    for tau, cc in ((0.02, C[5]), (0.05, C[1]), (0.1, C[6])):
        rr = sorted([r for r in DLAG if r["tau"] == tau], key=lambda r: r["f_v"])
        if rr:
            a.plot([r["f_v"] for r in rr], [r["w_te_a1"] / rr[0]["w_te_a1"] for r in rr], "-", marker="o", ms=3, color=cc, label=r"model + force lag $\tau$ = %g ms" % (tau * 1e3))
            NUM.setdefault("lag_fit", {})[str(tau)] = {str(r["f_v"]): r["w_te_a1"] / rr[0]["w_te_a1"] for r in rr}
    NUM["meas_a1_ratio"] = {str(m[0]): m[1] / m1 for m in mf}
    a.set(xscale="log", yscale="log", xlabel="Drive frequency $f$ in Hz", ylabel="Edge motion at $f$, over its 1 Hz value")
    a.legend(fontsize=6, loc="lower left")
    a = ax[1]
    mh = [(v, W["%dV_1hz_side" % v]["fit_low"] + W["%dV_1hz_side" % v]["fit_amp"]) for v in (1000, 1400, 1800)]
    a.plot([m[0] for m in mh], [m[1] for m in mh], "ks", ms=5, label="measured, cycle high (side)", zorder=5)
    for d in ("both", "both strong", "edge"):
        rr = sorted([r for r in DB if r["design"] == d and r.get("drive") == "sin2" and r.get("V") in (1000.0, 1400.0, 1800.0)], key=lambda r: r["V"])
        if rr:
            a.plot([r["V"] for r in rr], [(r["w_te_lo"] - r["w_te_hi"]) * 1e3 for r in rr], "o-", color=COL[d], label="model, " + LAB[d])
            NUM.setdefault("bench_V", {})[d] = {str(r["V"]): (r["w_te_lo"] - r["w_te_hi"]) * 1e3 for r in rr}
    a.set(xlabel="Voltage $V$ in V (1 Hz)", ylabel="Edge drop over the cycle in mm")
    a.legend(fontsize=6.5, loc="upper left")
    a = ax[2]
    r = next((r for r in DB if r["design"] == "both" and r.get("drive") == "sin2" and r.get("V") == 1800.0), None)
    if r:
        s = r["series"]; t = np.array(s["t"]); t = t - t[0]
        a.plot(t, np.array(s["w_te"]) * 1e3, color=C[3], label="edge height, mm")
        a.plot(t, np.array(s["F"]) * 1e3 / 10, color=C[1], label="edge tension / 10, mN")
        a.plot(t, np.array(s["v"]), "k:", lw=0.8, label=r"$(V/V_{max})^2$")
    a.set(xlabel="Time in one 1 Hz cycle in s", ylabel="Model, as built, 1800 V, no lag")
    a.legend(fontsize=6.5, loc="lower left")
    PS.finish(ax); fig.tight_layout(); fig.savefig(F("A_bench_dyn.png")); plt.close(fig)

# =========================================================== A5. flapping at 25 Hz, with the fitted lag
if DF and DLF:
    def dyn(d, mode, src=DF, U=4.0, a=10.0, f=25.0, tau=0.0):
        for r in src:
            if r["design"] == d and r["mode"] == mode and r.get("U", 4.0) == U and r.get("alpha_deg") == a and r.get("f", 25.0) == f and r.get("tau", 0.0) == tau:
                return r
    fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.0))
    des = ["skin strong", "edge", "both", "both strong"]
    lev = [0.0, 0.5, 1.0]
    for k, (key, yl) in enumerate((("L_m", "Mean lift per wing in mN"), ("T_m", "Mean thrust per wing in mN"), ("P_m", "Aerodynamic power per wing in mW"))):
        a = ax.flat[k]
        for d in des:
            rs = [dyn("passive", "off"), dyn(d, "sin", DLF, tau=0.05), dyn(d, "on", DLF, tau=0.05)]
            a.plot(lev, [r[key] * 1e3 for r in rs], "o-", color=COL[d], label=LAB[d])
            for alt in ("down", "up"):
                ra = dyn(d, alt, DLF, tau=0.05)
                a.plot([0.5], [ra[key] * 1e3], "x", color=COL[d], ms=5)
        a.set_xticks(lev); a.set_xticklabels(["0 V", "half force\n(any timing)", "1800 V\nheld"])
        a.set(ylabel=yl)
        if k == 0:
            a.axhline(18.4 / 2, color="k", ls="--", lw=0.8); a.text(0.02, 18.4 / 2 + 0.25, "half the robot weight", fontsize=7)
            a.plot([], [], "kx", label="on in the down- or upstroke")
            a.legend(fontsize=6.3, loc="lower left")
        if k == 1:
            a.axhline(0, color="k", lw=0.6)
    a = ax[1, 1]
    for d, ls in (("both", "-"), ("both strong", "--")):
        for m, cc in (("off", PS.OFF), ("on", PS.ON)):
            rr = sorted([r for r in DF if r["design"] == d and r["mode"] == m and r.get("alpha_deg") == 10.0 and r.get("U", 4.0) == 4.0 and r.get("tau", 0.0) == 0.0], key=lambda r: r.get("f", 25.0))
            if d == "both strong" and m == "off":
                continue
            a.plot([r.get("f", 25.0) for r in rr], [r["twist_amp"] for r in rr], ls, marker="o", color=cc, label="0 V" if m == "off" else LAB[d] + ", 1800 V held")
    a.axvspan(20, 25, color="0.9", zorder=0)
    a.set(xlabel="Flapping frequency $f$ in Hz", ylabel="Twist of the mid-span chord, amplitude, in °")
    a.legend(fontsize=6.3, loc="upper left")
    PS.finish(ax); fig.tight_layout(); fig.savefig(F("A_flap.png")); plt.close(fig)
    NUM["flap"] = {"%s | %s | U %g a %g f %g tau %g" % (r["design"], r["mode"], r.get("U", 4.0), r.get("alpha_deg", 10), r.get("f", 25.0), r.get("tau", 0.0)):
                   dict(L=r["L_m"] * 1e3, T=r["T_m"] * 1e3, P=r["P_m"] * 1e3, twist=r["twist_amp"], w_te=r["w_te_amp"] * 1e3, F=r["F_m"] * 1e3, Nc=r["Nc_m"], unstable=r["unstable"]) for r in DF + DLF}

json.dump(NUM, open(D("air_numbers.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in NUM.items() if k not in ("dCL_flow", "flap", "bench_a1", "lag_fit")}, indent=1, default=float))
