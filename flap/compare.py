"""Comparison of the model with the measurements already available (sparse; see the caveats printed with each case).
Writes validation/C_comparison.png and validation/comparison.json.

C1  DEA silicone and efficiency: P7670 Yeoh fit and eta from the bench bundle (fit_p7670.py) -- calibration, not validation.
C2  Bench tension drop of the DEA bundle at lambda = 1.12 (quasi-static cycles) vs model with eta from the two bench methods.
C3  Static (gliding) wind-tunnel campaigns: predicted lift change at 1.6 kV vs the measured detection floor.
C4  Flapping (FAAV, 1 Hz, n = 1 wing, effect not yet confirmed): measured dCL/CL0 vs model, sign and magnitude.
Run: python compare.py
"""
import json, math, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M
import plotstyle as PS
import matplotlib.pyplot as plt

PS.apply()
OUT = os.path.join(HERE, "validation")
PT = r"C:/Users/stefk/Documents/paper_tension"
rep = {}

# ---------------- C2: bench bundle ----------------
n, w, H, Hp, eps_r = 5, 10e-3, 30e-6, 30e-6, 2.8
A0, chi = n * w * (H + Hp), H / (H + Hp)
lam = 1 + 3.8 / 31.0
I1 = lam ** 2 + 2 / lam
C = M.DEA_YEOH
F_pass = A0 * 2 * (lam - lam ** -2) * (C[0] + 2 * C[1] * (I1 - 3) + 3 * C[2] * (I1 - 3) ** 2)
cyc = pd.read_csv(os.path.join(PT, "fig2_dea_actuator/results/dea_cyclic_summary.csv"))
Vb = np.linspace(0, 1800, 50)
ideal = chi * M.EPS0 * eps_r * (Vb / H) ** 2 * A0
model_rel = {eta: -100 * np.minimum(eta * ideal / F_pass, 1.0) for eta in (0.40, 0.15)}
v16 = cyc[cyc.V == 1600].iloc[0]
rep["C2"] = dict(F_passive_bundle_mN=F_pass * 1e3, measured_dT_1600_pct=[float(v16.dT_rel_lo_pct), float(v16.dT_rel_mid_pct), float(v16.dT_rel_hi_pct)],
                 model_dT_1600_eta040_pct=float(-100 * 0.40 * chi * M.EPS0 * eps_r * (1600 / H) ** 2 * A0 / F_pass),
                 model_dT_1600_eta015_pct=float(-100 * 0.15 * chi * M.EPS0 * eps_r * (1600 / H) ** 2 * A0 / F_pass))

# ---------------- C3: gliding (static wing) ----------------
gl = pd.read_csv(os.path.join(PT, "fig4_gliding/results/gliding_summary.csv"))
geo = dict(span=0.13, chord=0.06, t_mem=250e-6, n_dea_chord=4, nx=16, ny=8)
c3 = []
for mat, ds in (("aniso", "aniso"), ("DS20", "DS")):
    for U in (2.0, 4.0, 6.0):
        floor = gl[(gl.dataset == "may") & (gl.membrane == ds) & (gl.U == U)]
        fl = float(floor.floor_mN.iloc[0]) if len(floor) else float("nan")
        for cfg_lab, cfg in (("bat armwing", {}), ("rigid frame", dict(le_mode="rigid", clamp_tip=True))):
            for a in (5.0, 10.0):
                r = M.run(dict(geo, material=mat, U=U, alpha_deg=a, **cfg))
                c3.append(dict(membrane=ds, U=U, alpha=a, config=cfg_lab, dL_pred_mN=r["d"]["L_mN"], L0_mN=r["off"]["L_mN"],
                               floor_mN=fl, converged=bool(r["off"]["converged"] and r["on"]["converged"])))
rep["C3"] = c3

# ---------------- C4: flapping (FAAV) ----------------
fl4 = pd.read_csv(os.path.join(PT, "fig6_mechanism/results/camber_change_1p6kV.csv"))
fl4 = fl4[fl4.aoa.isin([-5, 0, 10])]
c4 = []
for row in fl4.itertuples():
    r = M.run(dict(span=0.20, chord=0.06, t_mem=250e-6, n_dea_chord=10, nx=24, ny=8, U=float(row.U), alpha_deg=float(row.aoa),
                   le_mode="rigid", clamp_tip=True))           # supports of the FAAV test wing unknown: rigid frame
    CL0m, dCLm = r["off"]["CL"], r["d"]["CL"]
    c4.append(dict(U=float(row.U), alpha=float(row.aoa), meas_CL0=float(row.CL0), meas_dCL=float(row.dCL), meas_dCL_sem=float(row.dCL_sem),
                   model_CL0=CL0m, model_dCL=dCLm, meas_ratio=float(row.dCL / row.CL0) if abs(row.CL0) > 1e-3 else float("nan"),
                   model_ratio=dCLm / CL0m if abs(CL0m) > 1e-6 else float("nan"),
                   converged=bool(r["off"]["converged"] and r["on"]["converged"])))
rep["C4"] = c4

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
ax[0].fill_between(cyc.V, cyc.dT_rel_lo_pct, cyc.dT_rel_hi_pct, color="0.85", label="measured (range)")
ax[0].plot(cyc.V, cyc.dT_rel_mid_pct, "o", color="k", ms=3.5, label="measured (mid)")
ax[0].plot(Vb, model_rel[0.40], "-", color=PS.MATLAB[0], label=r"model, $\eta$ = 0.40 (isometric)")
ax[0].plot(Vb, model_rel[0.15], "--", color=PS.MATLAB[1], label=r"model, $\eta$ = 0.15 (cycles)")
ax[0].set(xlabel=r"Voltage $V$ in V", ylabel=r"Bundle force change at $\lambda$ = 1.12 in %"); ax[0].legend(fontsize=6.5)
d3 = pd.DataFrame(c3)
for k, (ds, colr) in enumerate((("aniso", PS.MATLAB[0]), ("DS", PS.MATLAB[1]))):
    sub = d3[(d3.membrane == ds) & (d3.config == "rigid frame") & (d3.alpha == 10) & d3.converged]
    sub5 = d3[(d3.membrane == ds) & (d3.config == "rigid frame") & (d3.alpha == 5) & d3.converged]
    ax[1].plot(sub.U + 0.08 * k, sub.dL_pred_mN.abs(), "o-", color=colr, label=f"model {ds}, α = 10°")
    ax[1].plot(sub5.U + 0.08 * k, sub5.dL_pred_mN.abs(), "o:", color=colr, label=f"model {ds}, α = 5°")
    fl = gl[(gl.dataset == "may") & (gl.membrane == ds)]
    ax[1].plot(fl.U + 0.08 * k, fl.floor_mN, "v", color=colr, ms=7, mfc="none", label=f"measured floor {ds}")
ax[1].set(xlabel=r"Flow speed $U$ in m/s", ylabel=r"$|\Delta L|$ at 1.6 kV in mN"); ax[1].set_yscale("log"); ax[1].legend(fontsize=6)
d4 = pd.DataFrame(c4)
d4 = d4[d4.converged]
for U, mk in zip((2.0, 4.0, 6.0), "osd"):
    sub = d4[d4.U == U]
    ax[2].errorbar(sub.model_dCL, sub.meas_dCL, yerr=sub.meas_dCL_sem, fmt=mk, ms=5, capsize=2, label=f"U = {U:g} m/s")
    for rr in sub.itertuples():
        ax[2].annotate(f"{rr.alpha:g}°", (rr.model_dCL, rr.meas_dCL), textcoords="offset points", xytext=(4, 3), fontsize=7)
lim = max(0.05, float(np.nanmax(np.abs(np.r_[d4.meas_dCL, d4.model_dCL]))) * 1.1)
ax[2].plot([-lim, lim], [-lim, lim], "k-", lw=0.6, label="1:1")
ax[2].axhline(0, color="0.6", lw=0.5); ax[2].axvline(0, color="0.6", lw=0.5)
ax[2].set(xlabel=r"Model $\Delta C_L$", ylabel=r"Measured $\Delta C_L$ (flapping, 1 Hz)"); ax[2].legend(fontsize=6.5)
PS.finish(fig.axes)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "C_comparison.png"))
def _clean(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


with open(os.path.join(OUT, "comparison.json"), "w") as f:
    json.dump(_clean(json.loads(json.dumps(rep, default=float))), f, indent=1)
print(json.dumps(rep, indent=1, default=float))
