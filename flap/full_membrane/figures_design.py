"""Figures of the design page, from validation/active_*.jsonl (active_study.py). Style: ../plagiopatagiales/plotstyle.py.
Run: python figures_design.py -> validation/D_*.png and .svg, validation/design_summary.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
PS.apply()
V = os.path.join(HERE, "validation")
load = lambda s: [dict({"N": 2}, **json.loads(l)) for l in open(os.path.join(V, f"active_{s}.jsonl"))]
SW, EX, PI, CH, LE = load("sweep"), load("explore"), load("pick"), load("check"), load("lowEta")
ETA = [r for r in LE if r["N"] == 2 and r.get("active", True) and abs(r["we"][0] - .005) < 1e-6] + [r for r in CH if r["tag"] == "eta" and r["N"] == 2 and r["eta"] > 0.5]
ok = lambda r: bool(r.get("ok0") and r.get("ok1"))
CF, CD, C3 = PS.MATLAB[0], PS.MATLAB[1], PS.MATLAB[2]
D = next(r for r in PI if r["N"] == 2 and r["theta_r"] == 7.5 and r["lam_n"] == 1.05)
D3 = next(r for r in PI if r["N"] == 3 and r["theta_r"] == 7.5 and r["lam_n"] == 1.05)
tag = lambda t: [r for r in CH if r["tag"] == t and "error" not in r]
EPS, H, C1, q = 8.854e-12 * 2.8, 35e-6, 42842.0, 0.5 * 1.225 * 16


def save(fig, n):
    fig.tight_layout(); fig.savefig(os.path.join(V, n + ".png")); plt.close(fig)


def pts(a, X, Y, R, color, label=None, m="o"):
    """Settled states filled; states that did not settle hollow."""
    g, b = [r for r in R if ok(r)], [r for r in R if not ok(r) and "dL_mN" in r]
    a.plot([X(r) for r in g], [Y(r) for r in g], m, color=color, ms=4, label=label)
    a.plot([X(r) for r in b], [Y(r) for r in b], m, mfc="none", color=color, ms=4, lw=0)


def curve(a, X, Y, R, color, label=None, m="o-", ls=None):
    R = sorted(R, key=X)
    a.plot([X(r) for r in R], [Y(r) for r in R], m, color=color, ms=4, label=label)
    b = [r for r in R if not ok(r)]
    a.plot([X(r) for r in b], [Y(r) for r in b], "o", mfc="white", color=color, ms=5, lw=0)


# D1: layers
lay = {(r["N"], r.get("active", True)): r for r in tag("layers")}
lay[(2, True)], lay[(3, True)] = D, lay.get((3, True), D3)
Ns = [1, 2, 3, 4]
act, pas = [lay[(n, True)] for n in Ns], [lay[(n, False)] for n in Ns]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
a = ax[0]
a.plot(Ns, [r["dL_mN"] for r in act], "o-", color=CF, label="Active fibre"); a.plot(Ns, [r["dL_mN"] for r in pas], "s--", color=CD, label="Passive fibre")
a.set(xlabel="Active layers $n$", ylabel=r"Lift change $\Delta L$ in mN", xticks=Ns); a.legend()
a = ax[1]
a.plot(Ns, [r["V1"]["CL"] for r in act], "o-", color=PS.ON, label="1.6 kV: sheet slack"); a.plot(Ns, [r["V0"]["CL"] for r in act], "o-", color=PS.OFF, label="0 V: sheet taut")
a.fill_between(Ns, [r["V0"]["CL"] for r in act], [r["V1"]["CL"] for r in act], color="0.92")
a.axhspan(0.9, 1.3, facecolor="none", edgecolor="0.6", hatch="///", lw=0); a.text(1.05, 0.915, "above about 0.9: upper bound", fontsize=7, color="0.3", backgroundcolor="white")
a.set(xlabel="Active layers $n$", ylabel="$C_L$", xticks=Ns, ylim=(0.4, 1.3)); a.legend(fontsize=7, loc="lower left")
a = ax[2]
a.plot(Ns, [r["dL_mN"] / r["mass_mg"] * 1e3 for r in act], "o-", color=CF, label=r"$\Delta L$ per g of film")
a.set(xlabel="Active layers $n$", ylabel=r"$\Delta L$ / mass in mN/g", xticks=Ns)
b = a.twinx(); b.plot(Ns, [r["F_line_mN"] for r in act], "s--", color=CD); b.set_ylabel("Line force in mN", color=CD)
PS.finish(ax); save(fig, "D_layers")

# D2: the fibre
ALL = [r for r in SW + EX + PI if "dL_mN" in r and r.get("U", 4.0) == 4.0]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
a = ax[0]
Vs = np.linspace(0, 2000, 41)
lam = D["V0"]["p1"]["lam"]
F0 = 2 * C1 * 0.020 * 3 * H * (lam - lam ** -2) * 1e3
dF = lambda v, eta_f: eta_f * 3 * EPS * v ** 2 / H * 0.005 * 1e3
a.plot(Vs / 1e3, F0 - dF(Vs, 0.4), "-", color=CD, label=r"Closed form, $\eta_f$ = 0.4, panel 2")
a.fill_between(Vs / 1e3, F0 - dF(Vs, 0.6), F0 - dF(Vs, 0.2), color=CD, alpha=0.15, label=r"$\eta_f$ = 0.2 to 0.6")
ru = D_r = next((r for r in CH if r["tag"] == "design"), None)
if ru:
    a.plot([0] + [p["V"] / 1e3 for p in ru["ramp_up"]], [ru["V0"]["p1"]["F_mN"]] + [p["F"][1] for p in ru["ramp_up"]], "o", color=CD, ms=3.5, label="Finite elements, panel 2")
    a.plot([0] + [p["V"] / 1e3 for p in ru["ramp_up"]], [ru["V0"]["p0"]["F_mN"]] + [p["F"][0] for p in ru["ramp_up"]], "s", color=CF, ms=3.5, label="Finite elements, panel 1")
a.axhline(0, color="k", lw=0.6); a.set(xlabel="Voltage $V$ in kV", ylabel="Fibre force $F$ in mN", ylim=(0, None)); a.legend(fontsize=6.5)
a = ax[1]
for n, c in zip(Ns, PS.MATLAB):
    pts(a, lambda r: r["V1"]["p1"]["F_mN"], lambda r: r["dL_mN"], [r for r in ALL if r["N"] == n and r.get("xb", .06) == .06 and r.get("theta", 5.0) in (4.5, 5.0) and "V1" in r and r["dL_mN"] > -40], c, f"$n$ = {n}")
a.axvspan(0, 10, color="0.9", zorder=0); a.text(5, 24, "does not\nsettle", ha="center", fontsize=7, color="0.3")
a.set(xlabel="Fibre force at 1.6 kV, panel 2, in mN", ylabel=r"$\Delta L$ in mN", ylim=(-5, 27)); a.legend(fontsize=7, loc="lower right")
a = ax[2]
w = np.array([(r[s]["zte_mm"], r[s]["CL_no_te"] - r[s]["CL"]) for r in ALL if ok(r) and "V1" in r for s in ("V0", "V1")])
k = float(w[:, 0] @ w[:, 1] / (w[:, 0] @ w[:, 0]))
a.plot(w[:, 0], w[:, 1], "o", ms=2.5, color=CF); x_ = np.array([0, w[:, 0].max()])
a.plot(x_, k * x_, "k-", lw=0.8, label=rf"$\Delta C_L$ = {k:.2f} per mm")
a.plot(x_, 4.39 / 41.1 * x_, "k:", lw=0.8, label=r"$a_L / c$ = 0.11 per mm")
a.set(xlabel="Mean trailing-edge rise in mm", ylabel=r"$C_L$ lost to the rise"); a.legend(fontsize=8)
PS.finish(ax); save(fig, "D_fibre")

# D3: tilt, balance, chord stretch
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
a = ax[0]
for n, c in ((2, CF), (3, CD)):
    R = [r for r in EX if r["tag"] == "tilt" and r["N"] == n]
    base = next(r for r in SW if r["tag"] == "wswe" and r["N"] == n and r["ws"][0] == .02 and abs(r["we"][0] - .005) < 1e-6)
    curve(a, lambda r: r.get("theta", 5.0), lambda r: r["dL_mN"], R + [base], c, f"$n$ = {n}")
a.axvline(4.5, color="k", ls=":", lw=0.8); a.text(4.55, 11, "chosen", fontsize=8); a.legend(fontsize=8)
a.set(xlabel=r"Batten tilt $\theta_b$ in °, $\theta_r$ = 1.77 $\theta_b$", ylabel=r"$\Delta L$ in mN")
a = ax[1]
for th, ln, c, st, lb in ((5.0, 1.04, "0.6", "s-", r"$\theta_b$ = 5°, $\lambda_n$ = 1.04"), (4.5, 1.05, CF, "o-", r"$\theta_b$ = 4.5°, $\lambda_n$ = 1.05"), (4.5, 1.04, CD, "o--", r"$\theta_b$ = 4.5°, $\lambda_n$ = 1.04")):
    R = [r for r in EX if r["tag"] == "balance" and r["N"] == 2] + [next(r for r in SW if r["tag"] == "wswe" and r["N"] == 2 and r["ws"][0] == .02 and abs(r["we"][0] - .005) < 1e-6)] if th == 5.0 else [r for r in PI if r["N"] == 2 and r["lam_n"] == ln]
    R = sorted(R, key=lambda r: r.get("theta_r", 8.85))
    a.plot([r.get("theta_r", 8.85) for r in R], [r["M_batten_mNm"] for r in R], st, color=c, ms=4, label=lb)
a.axhline(0, color="k", lw=0.6); a.set(xlabel=r"Root edge tilt $\theta_r$ in °", ylabel="Moment on the batten in mN m"); a.legend(fontsize=7)
a = ax[2]
R = sorted([r for r in EX if r["tag"] == "lam_n" and r["N"] == 2] + [next(r for r in SW if r["tag"] == "wswe" and r["N"] == 2 and r["ws"][0] == .02 and abs(r["we"][0] - .005) < 1e-6)], key=lambda r: r.get("lam_n", 1.04))
curve(a, lambda r: r.get("lam_n", 1.04), lambda r: r["dL_mN"], R, CF)
a.set(xlabel=r"Chord stretch $\lambda_n$", ylabel=r"$\Delta L$ in mN", ylim=(0, 18))
b = a.twinx(); b.plot([r.get("lam_n", 1.04) for r in R], [r["V0"]["p0"]["wr"] * 100 for r in R], "s--", color=CD, ms=4); b.set_ylabel("Panel 1 wrinkled at 0 V in %", color=CD); b.set_ylim(0, 80)
PS.finish(ax); save(fig, "D_shape_choice")

# D4: roll and electrode
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
a = ax[0]
curve(a, lambda r: (r["ws"][0] * 1e3 if "ws" in r else 20.0), lambda r: r["dL_mN"], tag("ws") + [D], CF, "Active band 5 mm")
a.axvline(20.8, color="k", ls=":", lw=0.8); a.text(20.3, 4, "coater limit,\npanel 1", fontsize=7, ha="right")
a.set(xlabel="Roll width as cut $w_s$ in mm", ylabel=r"$\Delta L$ in mN", ylim=(0, 18)); a.legend(fontsize=7)
a = ax[1]
curve(a, lambda r: r["we"][0] * 1e3, lambda r: r["dL_mN"], tag("we") + [D], CF)
pv = lay[(2, False)]["dL_mN"]; a.axhline(pv, color=CD, ls="--", lw=0.9); a.text(2.6, pv + 0.25, "Passive fibre", color=CD, fontsize=8)
a.set(xlabel="Active band $w_e$ in mm", ylabel=r"$\Delta L$ in mN", ylim=(0, 18))
a = ax[2]
we = np.linspace(0, 20, 41)
for eta_f, st in ((0.3, "--"), (0.4, "-"), (0.6, ":")):
    a.plot(we, F0 - dF(1600.0, eta_f) * we / 5.0, st, color=CD, label=rf"$\eta_f$ = {eta_f}")
a.axhspan(0, 10, color="0.9", zorder=0); a.axvline(5, color="k", ls=":", lw=0.8); a.set(xlabel="Active band $w_e$ in mm", ylabel="Fibre force at 1.6 kV, panel 2, in mN", ylim=(0, F0 * 1.05), xlim=(0, 20)); a.legend(fontsize=8)
PS.finish(ax); save(fig, "D_roll")

# D5: tolerances
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
a = ax[0]
curve(a, lambda r: r.get("rot", 1.0) * 100, lambda r: r["dL_mN"], tag("rot") + [D], CF); a.set(xlabel="Rotation, % of the way to vertical", ylabel=r"$\Delta L$ in mN", ylim=(0, 18))
a = ax[1]
curve(a, lambda r: r.get("err_b", 0.0), lambda r: r["dL_mN"], [r for r in tag("err") if "err_b" in r] + [D], CF, "Batten")
curve(a, lambda r: r.get("err_r", 0.0), lambda r: r["dL_mN"], [r for r in tag("err") if "err_r" in r] + [D], CD, "Root rod")
a.set(xlabel="Rod past vertical (+) or short (−), in °", ylabel=r"$\Delta L$ in mN", ylim=(0, 18)); a.legend(fontsize=8)
a = ax[2]
curve(a, lambda r: r.get("H", 35e-6) * 1e6, lambda r: r["dL_mN"], tag("H") + [D], CF); a.set(xlabel="Dielectric thickness $H$ in µm", ylabel=r"$\Delta L$ in mN", ylim=(0, None))
a = ax[3]
curve(a, lambda r: r.get("eta_f", 0.4), lambda r: r["dL_mN"], tag("eta_f") + [D], CF); a.set(xlabel=r"$\eta_f$ of the roll", ylabel=r"$\Delta L$ in mN", ylim=(0, 18))
PS.finish(ax); save(fig, "D_tolerance")

# D6: what to expect
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
a = ax[0]
if ru:
    v_ = [0] + [p["V"] / 1e3 for p in ru["ramp_up"]]
    c_ = [ru["V0"]["CL"]] + [p["CL"] for p in ru["ramp_up"]]
    a.plot(v_, c_, "o-", color=PS.ON, label="Voltage up")
    a.plot([p["V"] / 1e3 for p in ru["ramp_down"]], [p["CL"] for p in ru["ramp_down"]], "s", mfc="none", color=PS.OFF, label="Voltage down")
    a.plot(np.array(v_), c_[0] + (c_[-1] - c_[0]) * (np.array(v_) / 1.6) ** 2, "k:", label=r"$V^2$")
a.set(xlabel="Voltage $V$ in kV", ylabel=r"$C_L$ at 4 m/s, 6°"); a.legend(fontsize=7)
a = ax[1]
curve(a, lambda r: r.get("U", 4.0), lambda r: r["dL_mN"], tag("U") + [D], CF); a.set(xlabel="Flow speed $U$ in m/s", ylabel=r"$\Delta L$ in mN")
a = ax[2]
curve(a, lambda r: r.get("eta", 0.5), lambda r: r["dL_mN"], ETA + [D], CF, "$n$ = 2")
curve(a, lambda r: r.get("eta", 0.5), lambda r: r["dL_mN"], [r for r in LE if not r.get("active", True)] + [lay[(2, False)]], CD, "Passive fibre", "s--")
a.legend(fontsize=7)
a.set(xlabel=r"$\eta$ of the sheet", ylabel=r"$\Delta L$ in mN")
a = ax[3]
for al_, c, lb in ((6.0, CF, "6°"), (8.0, CD, "8°")):
    r = next((q for q in tag("flap") if q["U"] == 4.0 and q["f"] == 2.0 and q["alpha"] == al_), None)
    if r is None:
        continue
    X = np.r_[r["phases"], 360.0]; d = np.array(r["L1"]) - np.array(r["L0"])
    a.plot(X, np.r_[d, d[0]], "-", color=c, label=f"Mean pitch {lb}")
    bad = ~(np.array(r["ok0"]) & np.array(r["ok1"]))
    a.plot(np.array(r["phases"])[bad], d[bad], "o", mfc="white", color=c, ms=4)
a.axvspan(0, 180, color="0.93", zorder=0); a.set(xlabel=r"Flap phase $\varphi$ in °", ylabel=r"$\Delta L$ in mN", xlim=(0, 360)); a.legend(fontsize=7)
PS.finish(ax); save(fig, "D_expect")

# D7: the shape
if ru and "shape" in ru:
    tri, x0, x1 = np.array(ru["shape"]["tri"]), np.array(ru["shape"]["x0"]) * 1e3, np.array(ru["shape"]["x1"]) * 1e3
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    vm = max(x0[:, 2].max(), x1[:, 2].max())
    for a, x_, t in ((ax[0], x0, "0 V"), (ax[1], x1, "1.6 kV")):
        c = a.tripcolor(Triangulation(x_[:, 0], x_[:, 1], tri), x_[:, 2], shading="gouraud", cmap="viridis", vmin=0, vmax=vm, rasterized=True)
        a.set_aspect("equal"); a.invert_yaxis(); a.set(xlabel="$x$ in mm", ylabel="$y$ in mm", title=t); fig.colorbar(c, ax=a, label="$w$ in mm", shrink=0.8)
    a = ax[2]
    ys = np.linspace(6.4, 47.5, 120)
    for x_, col, lb in ((x0, PS.OFF, "0 V"), (x1, PS.ON, "1.6 kV")):
        ip = LinearTriInterpolator(Triangulation(x_[:, 0], x_[:, 1], tri), x_[:, 2])
        a.plot(ys, ip(np.full_like(ys, 32.0), ys), "-", color=col, label=lb + ", $x$ = 32 mm")
        a.plot(ys, ip(np.full_like(ys, 72.0), ys), "--", color=col, label=lb + ", $x$ = 72 mm")
    a.set(xlabel="$y$ in mm, LE to TE", ylabel="$w$ in mm"); a.legend(fontsize=7)
    PS.finish(ax); save(fig, "D_map")

# D8: accuracy
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
a = ax[0]
curve(a, lambda r: r.get("h", 2e-3) * 1e3, lambda r: r["dL_mN"], tag("mesh") + [D], CF); a.set(xlabel="Mesh size $h$ in mm", ylabel=r"$\Delta L$ in mN", ylim=(0, 18))
a = ax[1]
curve(a, lambda r: r.get("na_x", 36), lambda r: r["dL_mN"], tag("lattice") + [D], CF); a.set(xlabel="Vortex lattice panels along the span", ylabel=r"$\Delta L$ in mN", ylim=(0, 18))
a = ax[2]
el = tag("elastic")
lb = [f"{r['rod_od'] * 1e3:g}/{r['rod_id'] * 1e3:g} mm LE\n{r['batten_d'] * 1e3:g} mm batten" for r in el]
a.barh(range(len(el)), [r["dL_mN"] for r in el], color=CF); a.axvline(D["dL_mN"], color="k", ls=":", lw=0.8)
a.set_yticks(range(len(el))); a.set_yticklabels(lb, fontsize=6); a.set(xlabel=r"$\Delta L$ with real rods, in mN")
a = ax[3]
bins = [0, 5, 10, 12, 15, 20, 30, 80]
xb, fr = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    q_ = [r for r in SW + EX + PI + CH if "V1" in r and lo <= r["V1"]["p1"]["F_mN"] < hi and r["tag"] != "elastic"]
    if q_:
        xb.append(0.5 * (lo + hi)); fr.append(100 * np.mean([ok(r) for r in q_]))
a.plot(xb, fr, "o-", color=CF); a.axvspan(0, 10, color="0.9", zorder=0)
a.set(xlabel="Fibre force at 1.6 kV, panel 2, in mN", ylabel="Cases that settle in %", ylim=(-5, 105), xlim=(0, 60))
PS.finish(ax); save(fig, "D_accuracy")

# summary
num = dict(design={k: D[k] for k in ("dL_mN", "L0_mN", "F_line_mN", "mass_mg", "area_cm2", "M_batten_mNm", "M_batten_flow0", "M_batten_flow1")}, CL=[D["V0"]["CL"], D["V1"]["CL"]], F0=[D["V0"]["p0"]["F_mN"], D["V0"]["p1"]["F_mN"]], F1=[D["V1"]["p0"]["F_mN"], D["V1"]["p1"]["F_mN"]], lam=[D["V0"]["p0"]["lam"], D["V0"]["p1"]["lam"]], wr0=[D["V0"]["p0"]["wr"], D["V0"]["p1"]["wr"]], wr1=[D["V1"]["p0"]["wr"], D["V1"]["p1"]["wr"]], zmax=[D["V0"]["zmax_mm"], D["V1"]["zmax_mm"]], zte=[D["V0"]["zte_mm"], D["V1"]["zte_mm"]], bow=[D["ns"]["p0"]["bow_mm"], D["ns"]["p1"]["bow_mm"]], k_te=k, three=dict(dL=D3["dL_mN"], CL=[D3["V0"]["CL"], D3["V1"]["CL"]], F_line=D3["F_line_mN"], mass=D3["mass_mg"]),
           F0_closed=F0, dF_closed=dF(1600.0, 0.4))
num["flap"] = [dict(U=r.get("U"), f=r.get("f"), alpha=r.get("alpha"), mean=float(np.mean(np.array(r["L1"]) - np.array(r["L0"]))), L0=float(np.mean(r["L0"])), lo=float(np.min(np.array(r["L1"]) - np.array(r["L0"]))), hi=float(np.max(np.array(r["L1"]) - np.array(r["L0"]))), ok=int(np.sum(r["ok0"]) + np.sum(r["ok1"]))) for r in tag("flap")]
json.dump(num, open(os.path.join(V, "design_summary.json"), "w"), indent=1, default=float)


# results as a table, so they can be placed like a figure
def table(name, head, rows, widths, h=None):
    fig, a = plt.subplots(figsize=(9.5, h or 0.32 * (len(rows) + 1) + 0.1))
    a.axis("off")
    t = a.table(cellText=rows, colLabels=head, colWidths=widths, loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.35)
    for (i, j), c in t.get_celld().items():
        c.set_linewidth(0.4); c.set_edgecolor("0.7")
        if i == 0:
            c.set_text_props(weight="bold")
    fig.savefig(os.path.join(V, name + ".png"), bbox_inches="tight"); plt.close(fig)


fmt = lambda r: f"{r['dL_mN']:.1f}" + ("" if ok(r) else "*")
table("D_results_design", ["Choice", "Value"], [
    ["Active layers", "2: half D, E, D, E, D, E, half D. 105 um"], ["Tilt as cut, batten and root edge", "4.5 deg, 7.5 deg"], ["Chord stretch", "1.05: 39.1 mm as cut, 41.1 mm on the rods"],
    ["Rolled strip; active band", "20 mm; 5 mm next to the sheet"], ["Batten; rods", "x = 60 mm; LE 3/2 mm tube, batten and root rod 0.8 mm"], ["Panel widths, coater limit 60 mm", "59.1 mm, 51.8 mm"],
    ["Line force; fibre stretch", f"{D['F_line_mN']:.0f} mN; {D['V0']['p0']['lam']:.3f}, {D['V0']['p1']['lam']:.3f}"], ["Fibre force, 0 V to 1.6 kV", f"{D['V0']['p0']['F_mN']:.0f}, {D['V0']['p1']['F_mN']:.0f} to {D['V1']['p0']['F_mN']:.0f}, {D['V1']['p1']['F_mN']:.0f} mN"],
    ["C_L, 4 m/s, 6 deg, 0 V to 1.6 kV", f"{D['V0']['CL']:.2f} to {D['V1']['CL']:.2f}"], ["Lift change, finite elements", f"{D['dL_mN']:.1f} mN, about +/- 1.5 from the lattice"], ["Film mass; panel 1 wrinkled at 0 V", f"{D['mass_mg']:.0f} mg; {D['V0']['p0']['wr'] * 100:.0f} %"]], [0.36, 0.64])
rows = [["Layers n = 1, 2, 3, 4, active", ", ".join(fmt(r) for r in act)], ["Layers n = 1, 2, 3, 4, passive", ", ".join(fmt(r) for r in pas)]]
ser = lambda t, lb, f: [lb, ", ".join(f"{f(r):g}: {fmt(r)}" for r in sorted([q for q in tag(t) if q.get("N", 2) == 2] + [D], key=f))]
for t_, lb_, f_ in (("U", "Speed, m/s", lambda r: r.get("U", 4.0)), ("alpha", "Pitch, deg", lambda r: r.get("alpha", 6.0)), ("eta", "Sheet eta", lambda r: r.get("eta", 0.5)), ("eta_f", "Roll eta_f", lambda r: r.get("eta_f", 0.4)), ("V", "Voltage, V", lambda r: r.get("V", 1600.0)), ("H", "Dielectric H, um", lambda r: round(r.get("H", 35e-6) * 1e6, 1)), ("ws", "Strip width, mm", lambda r: round(r["ws"][0] * 1e3, 1)), ("we", "Active band, mm", lambda r: round(r["we"][0] * 1e3, 1)), ("rot", "Rotation, %", lambda r: r.get("rot", 1.0) * 100), ("mesh", "Mesh, mm", lambda r: r.get("h", 2e-3) * 1e3), ("lattice", "Lattice, span panels", lambda r: r.get("na_x", 36))):
    rows.append(ser(t_, lb_, f_))
rows[[q_[0] for q_ in rows].index("Sheet eta")][1] = ", ".join(f"{r.get('eta', 0.5):g}: {fmt(r)}" for r in sorted(ETA + [D], key=lambda r: r.get("eta", 0.5)))
table("D_results_sensitivity", ["Change from the design", "Lift change in mN. * did not settle"], rows, [0.27, 0.73])
print(json.dumps(num, indent=0, default=float)[:1500])
