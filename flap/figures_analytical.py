"""Figures of the one-model comparison, all from wing_analytical.py. Style: plagiopatagiales/plotstyle.py. Run: python figures_analytical.py -> model/M*.png, model/numbers.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "plagiopatagiales"))
import plotstyle as PS
import matplotlib.pyplot as plt
import wing_analytical as W
PS.apply()
OUT = os.path.join(HERE, "model"); os.makedirs(OUT, exist_ok=True)
FE = os.path.join(HERE, "full_membrane", "validation")
FU, DI = W.FULL, W.DISCRETE
DF = dict(DI, T_c=0.0)
CF, CD = PS.MATLAB[0], PS.MATLAB[1]
LAB = dict(full="Full membrane", disc=f"Discrete DEAs, trailing-edge tension {DI['T_c']:g} N", free="Discrete DEAs, free trailing edge")
num = {}


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name + ".png")); plt.close(fig)


# M1: over the stroke
ph = W.cycle(FU, W.full_tension)[0]
X = np.r_[ph, 360.0]
close = lambda y: np.r_[y, y[0]]
d_f, d_d, d_0 = W.delta_cycle(FU, W.full_tension), W.delta_cycle(DI, W.discrete_tension), W.delta_cycle(DF, W.discrete_tension)
L0f, L1f = W.cycle(FU, W.full_tension, V=0.0)[1], W.cycle(FU, W.full_tension)[1]
L0d, L1d = W.cycle(DI, W.discrete_tension, V=0.0)[1], W.cycle(DI, W.discrete_tension)[1]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
a = ax[0]
a.axvspan(0, 180, color="0.93", zorder=0)
a.axvspan(240, 300, facecolor="none", edgecolor="0.55", hatch="///", lw=0)
a.plot(X, close(L0f), color=PS.OFF, label="Full membrane, 0 V"); a.plot(X, close(L1f), color=PS.ON, label="Full membrane, 1.6 kV")
a.plot(X, close(L0d), "--", color=PS.OFF, label="Discrete, 0 V"); a.plot(X, close(L1d), "--", color=PS.ON, label="Discrete, 1.6 kV")
a.set(xlabel=r"Flap phase $\varphi$ in °", ylabel="Lift $L$ in mN, one wing", xlim=(0, 360)); a.legend(fontsize=7)
a = ax[1]
a.axvspan(0, 180, color="0.93", zorder=0)
a.axvspan(240, 300, facecolor="none", edgecolor="0.55", hatch="///", lw=0)
a.text(270, 0.5 * (d_f.max() + d_d.max()), "not\nreliable", ha="center", fontsize=7, color="0.3", backgroundcolor="white")
a.plot(X, close(d_f), color=CF, label=LAB["full"]); a.plot(X, close(d_d), color=CD, label=LAB["disc"]); a.plot(X, close(d_0), "k-.", label=LAB["free"])
a.set(xlabel=r"Flap phase $\varphi$ in °", ylabel=r"Lift change $\Delta L$ in mN", xlim=(0, 360)); a.legend(fontsize=7, loc="lower left")
PS.finish(ax); save(fig, "M1_stroke")
num["stroke_mean_mN"] = dict(full=W.mean_delta(FU, W.full_tension), disc=W.mean_delta(DI, W.discrete_tension), free=0.0)
num["stroke_range_mN"] = dict(full=[float(d_f.min()), float(d_f.max())], disc=[float(d_d.min()), float(d_d.max())])
num["mean_L0_mN"] = dict(full=float(L0f.mean()), disc=float(L0d.mean()))
num["L_90_270_mN"] = dict(full_90=float(d_f[6]), full_270=float(d_f[18]), disc_90=float(d_d[6]), disc_270=float(d_d[18]))

# M2: speed
Us = np.linspace(2, 7, 21)
f_u, d_u = [W.mean_delta(FU, W.full_tension, U=u) for u in Us], [W.mean_delta(DI, W.discrete_tension, U=u) for u in Us]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
a = ax[0]
a.axvspan(5, 7, color="0.93", zorder=0); a.text(6, 0.3, "over-predicts", ha="center", fontsize=8, color="0.35")
a.plot(Us, f_u, color=CF, label=LAB["full"]); a.plot(Us, d_u, color=CD, label=LAB["disc"]); a.plot(Us, 0 * Us, "k-.", label=LAB["free"])
a.set(xlabel="Flow speed $U$ in m/s", ylabel=r"Cycle-mean $\Delta L$ in mN"); a.legend(fontsize=7)
a = ax[1]
a.plot(Us, np.array(f_u) / np.maximum(d_u, 1e-3), color="k")
a.set(xlabel="Flow speed $U$ in m/s", ylabel=r"$\Delta L$ full / $\Delta L$ discrete", yscale="log")
PS.finish(ax); save(fig, "M2_speed")
num["speed_mN"] = {f"{u:g}": dict(full=float(np.interp(u, Us, f_u)), disc=float(np.interp(u, Us, d_u))) for u in (3.0, 4.0, 5.0, 6.0)}

# M3: eta
et = np.linspace(0.05, 0.8, 16)
f_e = [W.mean_delta(dict(FU, eta=e), W.full_tension) for e in et]
d_e = [W.mean_delta(dict(DI, dF=DI["dF"] * e / 0.4), W.discrete_tension) for e in et]
fig, a = plt.subplots(figsize=(5.2, 3.4))
a.plot(et, f_e, color=CF, label=LAB["full"]); a.plot(et, d_e, color=CD, label=LAB["disc"])
a.set(xlabel=r"Actuator efficiency $\eta$", ylabel=r"Cycle-mean $\Delta L$ in mN"); a.legend(fontsize=7)
PS.finish(a); save(fig, "M3_eta")
num["eta_mN"] = {f"{e:g}": dict(full=float(np.interp(e, et, f_e)), disc=float(np.interp(e, et, d_e))) for e in (0.15, 0.24, 0.4, 0.7)}

# M4: voltage
Vs = np.linspace(0, 1800, 37)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
c_f = [W.solve(FU, W.full_tension, V=v)["CL"] for v in Vs]
c_d = [W.solve(DI, W.discrete_tension, V=v)["CL"] for v in Vs]
a = ax[0]
a.plot(Vs / 1e3, c_f, color=CF, label=LAB["full"]); a.plot(Vs / 1e3, c_d, color=CD, label=LAB["disc"])
a.plot(Vs / 1e3, c_f[0] + (np.interp(1600, Vs, c_f) - c_f[0]) * (Vs / 1600) ** 2, ":", color=CF, label=r"$V^2$ through 1.6 kV")
a.set(xlabel="Voltage $V$ in kV", ylabel=r"$C_L$ ($U$ = 4 m/s, $\alpha$ = 6°)"); a.legend(fontsize=7)
a = ax[1]
vv = [W.full_tension(FU, 0, v) for v in Vs]
a.plot(Vs / 1e3, [q[0] for q in vv], color=CF, label=r"Panel 1, $n_x$"); a.plot(Vs / 1e3, [q[1] for q in vv], "--", color=CF, label=r"Panel 1, $n_y$")
a.plot(Vs / 1e3, [W.discrete_tension(DI, 0, v)[1] for v in Vs], color=CD, label=r"Discrete, $n_y$")
a.set(xlabel="Voltage $V$ in kV", ylabel=r"Resultant in N/m"); a.legend(fontsize=7)
PS.finish(ax); save(fig, "M4_voltage")
num["voltage"] = dict(full_n_0=vv[0][:2], full_n_16=W.full_tension(FU, 0, 1600.0)[:2], disc_ny_0=W.discrete_tension(DI, 0, 0.0)[1], disc_ny_16=W.discrete_tension(DI, 0, 1600.0)[1])

# M5: pre-stretch
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
lam = np.linspace(1.0, 1.05, 11)
a = ax[0]
d_l = [W.mean_delta(dict(DI, lam_x=l, lam_y=l), W.discrete_tension) for l in lam]
a.plot(lam, d_l, color=CD); a.set(xlabel=r"Membrane pre-stretch $\lambda_{pre}$, discrete wing", ylabel=r"Cycle-mean $\Delta L$ in mN")
a = ax[1]
ld = np.linspace(1.03, 1.14, 12)
d_ld = [W.mean_delta(dict(FU, lam_d=(l, l), lam_n=(l ** -0.5 + 0.04, l ** -0.5 + 0.04)), W.full_tension) for l in ld]
a.plot(ld, d_ld, color=CF)
lims = [W.fibre_force(dict(FU, eta_f=e), 0, 1600.0)[1] for e in (0.3, 0.4)]
num["fibre_force_mN"] = {f"{l:g}": W.fibre_force(dict(FU, lam_d=(l, l)), 0, 1600.0)[0] * 1e3 for l in (1.05, 1.08, 1.1)}
a.axvspan(1.03, lims[0], color="0.9", zorder=0); a.axvline(lims[1], color="k", ls=":", lw=0.8)
a.text(lims[1] + 0.002, min(d_ld), r"fibre limit, $\eta_f$ = 0.4", fontsize=7)
a.set(xlabel=r"Stretch along the fibre $\lambda_d$, full membrane", ylabel=r"Cycle-mean $\Delta L$ in mN")
PS.finish(ax); save(fig, "M5_stretch")
num["prestretch_disc_mN"] = {f"{l:g}": float(np.interp(l, lam, d_l)) for l in (1.0, 1.01, 1.02, 1.03, 1.05)}
num["stretch_full_mN"] = {f"{l:g}": float(np.interp(l, ld, d_ld)) for l in (1.04, 1.05, 1.06, 1.08, 1.1, 1.12)}
num["fibre_limit"] = dict(eta_f_03=lims[0], eta_f_04=lims[1])

# M6: trailing-edge tension
Tc = np.logspace(-2, 1, 25)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
a = ax[0]
a.semilogx(Tc, [W.load_path(dict(DI, T_c=t)) for t in Tc], color="k")
a.set(xlabel="Trailing-edge tension $T_c$ in N", ylabel=r"Load-path factor $r$")
a = ax[1]
d_t = [W.mean_delta(dict(DI, T_c=t), W.discrete_tension) for t in Tc]
d_t1 = [W.mean_delta(dict(DI, T_c=t, lam_x=1.01, lam_y=1.01), W.discrete_tension) for t in Tc]
a.semilogx(Tc, d_t, color=CD, label=r"$\lambda_{pre}$ = 1.03"); a.semilogx(Tc, d_t1, "--", color=CD, label=r"$\lambda_{pre}$ = 1.01")
a.set(xlabel="Trailing-edge tension $T_c$ in N", ylabel=r"Cycle-mean $\Delta L$ in mN"); a.legend()
PS.finish(ax); save(fig, "M6_tendon")
num["tendon"] = {f"{t:g}": dict(r=W.load_path(dict(DI, T_c=t)), dL=float(np.interp(math.log(t), np.log(Tc), d_t)), dL_101=float(np.interp(math.log(t), np.log(Tc), d_t1))) for t in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)}

# M7: layers, against the FE
ns = np.arange(1, 7)
ref = dict(n_layers=(1, 1))
E4 = dict(FU, eta=0.4)                                            # the finite-element runs used eta = 0.4 and a passive fibre
f_n = [W.solve(dict(E4, n_layers=(int(n), int(n)), lam_d=(1.05, 1.05), lam_n=(1.016, 1.016)), W.full_tension)["L_mN"] - W.solve(dict(E4, n_layers=(int(n), int(n)), lam_d=(1.05, 1.05), lam_n=(1.016, 1.016)), W.full_tension, V=0.0)["L_mN"] for n in ns]
R = [r for r in json.load(open(os.path.join(FE, "opt_layers.json"))) if r.get("U") == 4.0 and r.get("ok0") and r.get("ok1")]
fe_b = [max([r for r in R if r["na"][0] == n], key=lambda r: r["dL_mN"]) for n in ns]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
a = ax[0]
a.plot(ns, f_n, "-", color=CF, label=r"This model, $\lambda_d$ = 1.05"); a.plot(ns, [r["dL_mN"] for r in fe_b], "o", color="k", ms=4, label="Finite elements, best stretch")
a.set(xlabel="Active layers", ylabel=r"$\Delta L$ at 6° in mN"); a.legend(fontsize=7)
b = ax[1]
A = []
for r in R:
    n, l, d = r["na"][0], r["lam_d"][0], r["dn"][0]
    w = dict(E4, n_layers=(n, n), lam_d=(l, l), lam_n=(l ** -0.5 + d, l ** -0.5 + d))
    p, q = W.solve(w, W.full_tension, V=0.0), W.solve(w, W.full_tension)
    A.append((r["dL_mN"], q["L_mN"] - p["L_mN"], r["CL0"], p["CL"], r["CL1"], q["CL"]))
A = np.array(A)
b.plot(A[:, 0], A[:, 1], "o", color=CF, ms=3.5); m_ = max(A[:, 0].max(), A[:, 1].max()); b.plot([0, m_], [0, m_], "k-", lw=0.8)
b.set(xlabel=r"$\Delta L$, finite elements, in mN", ylabel=r"$\Delta L$, this model, in mN")
PS.finish(ax); save(fig, "M7_check")
num["check"] = dict(n=len(A), dL_rms=float(np.sqrt(np.mean((A[:, 1] - A[:, 0]) ** 2))), dL_corr=float(np.corrcoef(A[:, 0], A[:, 1])[0, 1]), CL0_rms=float(np.sqrt(np.mean((A[:, 3] - A[:, 2]) ** 2))), CL1_rms=float(np.sqrt(np.mean((A[:, 5] - A[:, 4]) ** 2))))
num["layers_model_mN"] = {int(n): float(v) for n, v in zip(ns, f_n)}
# reference checks quoted on the page
one = dict(E4, n_layers=(1, 1), lam_d=(1.05, 1.05), lam_n=(1.016, 1.016))
num["refs"] = dict(one_layer_cycle=W.mean_delta(one, W.full_tension),
                   disc_rigid={f"{u:g}": W.mean_delta(dict(DI, T_c=1e9), W.discrete_tension, U=u) for u in (3.0, 4.0, 6.0)},
                   CL_full={f"{u:g}": [W.solve(one, W.full_tension, V=0.0, U=u)["CL"], W.solve(one, W.full_tension, U=u)["CL"]] for u in (3.0, 4.0, 6.0)}, K_CAM=W.K_CAM, CAMBER=W.CAMBER, r_05=W.load_path(DI))
json.dump(num, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)
print(json.dumps(num, indent=0, default=float))
