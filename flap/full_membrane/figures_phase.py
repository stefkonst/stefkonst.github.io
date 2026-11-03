"""Phase-dependent actuation, from the saved stroke (validation/active_check.jsonl, 4 m/s, 2 Hz, 8 deg) with no new FE runs.
The fully active fibre is an estimate: F = 0 at 1.6 kV, and the edge law z = k L / (F + b), fitted to validation/active_films.jsonl, gives its edge rise.
Run: python figures_phase.py -> validation/D_phase.png and .svg, validation/phase_summary.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
V = os.path.join(HERE, "validation")
load = lambda s: [json.loads(l) for l in open(os.path.join(V, f"active_{s}.jsonl"))]
R = next(r for r in load("check") if r["tag"] == "flap" and r["U"] == 4.0 and r["f"] == 2.0 and r["alpha"] == 8.0)
ph, L0, L1 = np.array(R["phases"]), np.array(R["L0"]), np.array(R["L1"])
ok = np.array(R["ok0"]) & np.array(R["ok1"])
F_b = float(np.mean(R["F1"]))                                 # fibre force at voltage in this stroke

# edge law from the settled two-film cases: 1/z is linear in F, and lift falls with z
Fi = [r for r in load("films") if r["tag"] == "films" and r["N"] == 2 and r.get("ok1")]
Fm = np.array([.5 * (r["V1"]["p0"]["F_mN"] + r["V1"]["p1"]["F_mN"]) for r in Fi])
Z = np.array([r["V1"]["zte_mm"] for r in Fi])
Lv = np.array([r["L0_mN"] + r["dL_mN"] for r in Fi])
m_, c_ = np.polyfit(Fm, Lv / Z, 1)
k, b = 1 / m_, c_ / m_                                        # z = k L / (F + b)
s = -np.polyfit(Z, Lv, 1)[0]                                  # mN of lift per mm of edge rise
g = s * k
L_s = L1 * (1 + g / (F_b + b))
L_hi = L_s / (1 + g / b)                                      # self-consistent: less lift, so less edge rise
L_lo = L1 - s * k * L1 * (1 / b - 1 / (F_b + b))              # linear: edge rise taken at the full lift
Lf = L_hi

# net forward force, an estimate: lift tilted by the stroke, less induced drag (profile drag left out)
A, om, U, S_, AR, r_bar = math.radians(30), 2 * math.pi * R["f"], R["U"], R["area_cm2"] * 1e-4 if "area_cm2" in R else 31.9e-4, 3.8, .040
qS = .5 * 1.225 * U ** 2 * S_ * 1e3
tilt = A * om * r_bar / U * np.sin(np.radians(ph))
fwd = lambda L: L * tilt - L ** 2 / (qS * math.pi * AR)
down = ph < 180
sched = {"Always on": np.ones_like(down), "Downstroke only": down, "Upstroke only": ~down}


def cycle(Lon):
    out = {}
    for n, w in sched.items():
        L = np.where(w, Lon, L0)
        out[n] = dict(L=float(L.mean()), pct=float(100 * (L.mean() / L0.mean() - 1)), fwd=float(fwd(L).mean()), lo=float(L.min()), hi=float(L.max()))
    return out


CB, CF, CL_, CH_ = cycle(L1), cycle(Lf), cycle(L_lo), cycle(L_hi)
base_fwd = float(fwd(L0).mean())

fig = plt.figure(figsize=(10, 6.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25])
a = fig.add_subplot(gs[0, 0])                                 # the schedules
t = np.linspace(0, 360, 361)
a.plot(t, 30 * np.cos(np.radians(t)), color="k")
a.axvspan(0, 180, color="0.92", lw=0)
a.text(90, 34, "Downstroke", ha="center", fontsize=8); a.text(270, 34, "Upstroke", ha="center", fontsize=8)
for i, (n, w) in enumerate(sched.items()):
    y = -44 - 11 * i
    a.plot([0, 360], [y, y], color="0.6", lw=.8)
    q = np.where(w)[0]
    a.plot([ph[q[0]], 360 if q[-1] == len(ph) - 1 else ph[q[-1]] + 15], [y, y], color=PS.ON, lw=4, solid_capstyle="butt")
    a.text(366, y, n, va="center", fontsize=8)
a.set(xlabel=r"Stroke phase $\phi$ in °", ylabel=r"Stroke angle in °", xlim=(0, 360), ylim=(-72, 42), xticks=range(0, 361, 90), yticks=[-30, 0, 30])
a.text(180, -38.5, "1.6 kV on (red)", ha="center", fontsize=8)

a = fig.add_subplot(gs[0, 1])                                 # lift through the stroke
a.axvspan(0, 180, color="0.92", lw=0)
a.plot(ph, L0, "o-", color=PS.OFF, ms=3, label="0 V")
a.plot(ph, L1, "o-", color=PS.ON, ms=3, label=f"1.6 kV, band: fibre {F_b:.0f} mN")
a.plot(ph[~ok], L1[~ok], "o", mfc="white", color=PS.ON, ms=5)
a.fill_between(ph, L_lo, L_hi, color=PS.MATLAB[3], alpha=.22, lw=0)
a.plot(ph, Lf, "--", color=PS.MATLAB[3], label="1.6 kV, all active: fibre slack (estimate)")
a.set(xlabel=r"Stroke phase $\phi$ in °", ylabel=r"Lift of one wing $L$ in mN", xlim=(0, 345), xticks=range(0, 361, 90), ylim=(0, 62))
a.legend(loc="upper right", fontsize=7.5)

names = list(sched)
x = np.arange(3)
a = fig.add_subplot(gs[1, 0])                                 # cycle-mean lift, %
a.bar(x - .19, [CB[n]["pct"] for n in names], .36, color=PS.ON, label="Band")
a.bar(x + .19, [CF[n]["pct"] for n in names], .36, color=PS.MATLAB[3], label="All active (estimate)", yerr=[[CF[n]["pct"] - CL_[n]["pct"] for n in names], [CH_[n]["pct"] - CF[n]["pct"] for n in names]], capsize=3, ecolor="k", error_kw=dict(lw=.8))
a.axhline(0, color="k", lw=.8)
for i, n in enumerate(names):
    a.text(i - .19, CB[n]["pct"] + 2, f"{CB[n]['pct']:+.0f} %", ha="center", fontsize=8)
    a.text(i + .19, max(CH_[n]["pct"], 0) + 2, f"{CF[n]['pct']:+.0f} %", ha="center", fontsize=8)
a.set(ylabel=r"Cycle-mean lift above 0 V in %", xticks=x, xticklabels=names, ylim=(-70, 80))
a.legend(loc="upper right", fontsize=7.5)

a = fig.add_subplot(gs[1, 1])                                 # net forward force
a.bar(x - .19, [CB[n]["fwd"] - base_fwd for n in names], .36, color=PS.ON, label="Band")
a.bar(x + .19, [CF[n]["fwd"] - base_fwd for n in names], .36, color=PS.MATLAB[3], label="All active (estimate)")
a.axhline(0, color="k", lw=.8)
for i, n in enumerate(names):
    for dx, C in ((-.19, CB), (.19, CF)):
        v = C[n]["fwd"] - base_fwd
        a.text(i + dx, v + (.07 if v >= 0 else -.07), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
a.set(ylabel="Change in cycle-mean forward force in mN\n(estimate: tilted lift less induced drag)", xticks=x, xticklabels=names, ylim=(-2.9, .8))
a.legend(loc="lower right", fontsize=7.5)
PS.finish(fig.axes)
fig.tight_layout(); fig.savefig(os.path.join(V, "D_phase.png")); plt.close(fig)

Vc = lambda frac: 1600 * math.sqrt(2 * 42842 * .1347 / (.4 * 8.854e-12 * 2.8 * (1600 / 35e-6) ** 2 * frac))
out = dict(k=float(k), b=float(b), s_mN_per_mm=float(s), F_band=F_b, L0_mean=float(L0.mean()), settled_band=int(ok.sum()), base_fwd=base_fwd, band=CB, full=CF, full_low=CL_, full_high=CH_, Vc_all_films=Vc(1.0), Vc_three_electrodes=Vc(2 / 3), L1_mean=float(L1.mean()), Lf_mean=float(Lf.mean()))
json.dump(out, open(os.path.join(V, "phase_summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
