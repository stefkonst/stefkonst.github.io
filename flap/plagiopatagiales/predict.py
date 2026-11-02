"""Predictions of experimentally measurable quantities and design trends for the default wing (model.DEFAULTS).
Writes validation/P1_quasistatic.png, P2_flapping.png, P3_design.png and validation/predictions.csv
(long format: figure, curve, x, quantity, value) and validation/predictions.json (summary numbers).
Measurable with: camber -> side camera / laser line at mid-span; lift, vertical force, thrust -> load cell;
DEA strain -> markers on one DEA; rod tip -> camera from above; chord tension -> clamp force on the bench (U = 0).
Run: python predict.py
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M
import plotstyle as PS
import matplotlib.pyplot as plt

PS.apply()
OUT = os.path.join(HERE, "validation")
rows, summary = [], {}


def keep(fig, sw, qtys):
    for lab, rec in sw["curves"].items():
        for q in qtys:
            for xv, yv in zip(sw["values"], rec[q]):
                rows.append(dict(figure=fig, curve=lab, x=xv, quantity=q, value=yv))


def save(fig, name, axes=None):
    PS.finish(list(axes) if axes is not None else fig.axes)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


Q = ["L_mN", "F_vert_mN", "F_thrust_mN", "camber_mid_pct", "camber_mean_pct", "dea_strain_pct", "T_chord_total",
     "rod_tip_dy_mm", "rod_tip_dz_mm", "te_z_mid_mm"]
Vs = [0, 400, 800, 1000, 1200, 1400, 1600, 1800]

# ---------------- P1 quasi-static (no flapping) ----------------
sv = M.sweep({}, "voltage", Vs, aoas=[0, 3, 6, 9, 12])
sa = M.sweep({}, "aoa", list(range(-8, 13, 2)), voltages=[0, 1600])
sb = M.sweep(dict(U=0.0), "voltage", Vs, aoas=[0])
for n_, sw in (("P1_voltage", sv), ("P1_pitch", sa), ("P1_bench_U0", sb)):
    keep(n_, sw, Q)
fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.2))
cm = plt.cm.viridis(np.linspace(0, 0.85, len(sv["curves"])))
for (lab, r), c in zip(sv["curves"].items(), cm):
    ax[0, 0].plot(Vs, r["camber_mid_pct"], "o-", color=c, label=lab.replace("pitch", r"$\alpha$ ="))
    ax[0, 1].plot(Vs, np.array(r["L_mN"]) - r["L_mN"][0], "o-", color=c)
    ax[0, 2].plot(Vs, r["dea_strain_pct"], "o-", color=c)
ax[0, 0].set(xlabel=r"Voltage $V$ in V", ylabel=r"Span-averaged camber $h/c$ in %")
ax[0, 1].set(xlabel=r"Voltage $V$ in V", ylabel=r"Lift change $\Delta L$ in mN")
ax[0, 2].set(xlabel=r"Voltage $V$ in V", ylabel=r"DEA strain in %")
ax[0, 0].legend(fontsize=7)
for lab, r in sa["curves"].items():
    on = lab.endswith("1600 V")
    ax[1, 0].plot(sa["values"], r["L_mN"], "o-", color=PS.ON if on else PS.OFF, label="1.6 kV" if on else "0 V")
ax[1, 0].set(xlabel=r"Pitch angle $\alpha$ in °", ylabel=r"Lift $L$ in mN"); ax[1, 0].legend(fontsize=7)
r = sv["curves"]["pitch 6 deg"]
ax[1, 1].plot(Vs, r["rod_tip_dy_mm"], "o-", color=PS.MATLAB[0], label="in-plane (towards TE)")
ax[1, 1].plot(Vs, r["rod_tip_dz_mm"], "s-", color=PS.MATLAB[1], label="out-of-plane")
ax[1, 1].set(xlabel=r"Voltage $V$ in V", ylabel=r"LE rod tip displacement in mm"); ax[1, 1].legend(fontsize=7)
rb = sb["curves"]["pitch 0 deg"]
ax[1, 2].plot(Vs, 100 * (np.array(rb["T_chord_total"]) / rb["T_chord_total"][0] - 1), "o-", color="k")
ax[1, 2].set(xlabel=r"Voltage $V$ in V", ylabel=r"Chord tension change (U = 0) in %")
save(fig, "P1_quasistatic.png")
summary["dL_1600_pitch6_mN"] = sv["curves"]["pitch 6 deg"]["L_mN"][Vs.index(1600)] - sv["curves"]["pitch 6 deg"]["L_mN"][0]
summary["L0_pitch6_mN"] = sv["curves"]["pitch 6 deg"]["L_mN"][0]
summary["dT_bench_1600_pct"] = 100 * (rb["T_chord_total"][Vs.index(1600)] / rb["T_chord_total"][0] - 1)

# ---------------- P2 flapping (quasi-steady), DEA schedules ----------------
ph = list(range(0, 361, 30))
ss = M.sweep(dict(flap_amp_deg=30, flap_f=3), "schedule", ph, voltages=[1600], aoas=[6])
sp = M.sweep(dict(flap_amp_deg=30, flap_f=3), "phase", ph, voltages=[1600], aoas=[0, 6, 12])
keep("P2_schedule", ss, Q); keep("P2_phase", sp, Q)
fig, ax = plt.subplots(2, 2, figsize=(8.5, 6.2))
cols = {"DEA off": PS.OFF, "1600 V, always on": PS.ON, "1600 V, downstroke only": PS.MATLAB[4], "1600 V, upstroke only": PS.MATLAB[3]}
for lab, r in ss["curves"].items():
    ax[0, 0].plot(ph, r["F_vert_mN"], "o-", color=cols[lab], label=lab.replace("1600 V, ", "1.6 kV, "))
    ax[0, 1].plot(ph, r["F_thrust_mN"], "o-", color=cols[lab])
for x_ in (0, 1):
    ax[0, x_].axvspan(0, 180, color="0.9", zorder=0)
ax[0, 0].set(xlabel=r"Flap phase $\varphi$ in ° (shaded: downstroke)", ylabel=r"Vertical force $F_z$ in mN"); ax[0, 0].legend(fontsize=6)
ax[0, 1].set(xlabel=r"Flap phase $\varphi$ in °", ylabel=r"Thrust $F_x$ in mN")
for (lab, r), c in zip(sp["curves"].items(), PS.MATLAB):
    ax[1, 0].plot(ph, r["dea_strain_pct"], "o-", color=c, label=lab.replace("pitch", r"$\alpha$ =").replace(", 1600 V", ""))
ax[1, 0].set(xlabel=r"Flap phase $\varphi$ in °", ylabel=r"DEA strain at 1.6 kV in %"); ax[1, 0].legend(fontsize=7)
base_m = ss["curves"]["DEA off"]["cycle_mean"]
labs = [l for l in ss["curves"] if l != "DEA off"]
dFz = [ss["curves"][l]["cycle_mean"]["F_vert_mN"] - base_m["F_vert_mN"] for l in labs]
dFx = [ss["curves"][l]["cycle_mean"]["F_thrust_mN"] - base_m["F_thrust_mN"] for l in labs]
xx = np.arange(len(labs))
ax[1, 1].bar(xx - 0.18, dFz, 0.36, color=PS.MATLAB[0], label=r"$\Delta\bar F_z$")
ax[1, 1].bar(xx + 0.18, dFx, 0.36, color=PS.MATLAB[1], label=r"$\Delta\bar F_x$")
ax[1, 1].set_xticks(xx, [l.replace("1600 V, ", "") for l in labs], fontsize=8)
ax[1, 1].set(ylabel=r"Cycle-mean change in mN"); ax[1, 1].legend(fontsize=7); ax[1, 1].axhline(0, color="k", lw=0.6)
save(fig, "P2_flapping.png")
summary["cycle_mean_change"] = {l: dict(dFz_mN=a, dFx_mN=b) for l, a, b in zip(labs, dFz, dFx)}

# ---------------- P3 design trends ----------------
def dL(**kw):
    r = M.run(kw)
    return r["d"]["L_mN"], r["off"]["L_mN"], r["m_dea_g"], bool(r["off"]["converged"] and r["on"]["converged"])


des = {}
des["n_dea"] = [(n,) + dL(n_dea_chord=n) for n in (2, 3, 4, 6, 8)]
des["pre_dea"] = [(v,) + dL(pre_dea=v) for v in (1.04, 1.07, 1.10, 1.15, 1.20)]
des["pre_mem"] = [(v,) + dL(pre_span=v, pre_chord=v) for v in (1.0, 1.02, 1.03, 1.05, 1.08)]
combos = []
for dea_lab, dkw in (("chord DEAs", dict(n_dea_chord=4, n_dea_span=0)), ("span DEAs", dict(n_dea_chord=0, n_dea_span=3))):
    for rf_lab, rkw in (("no strips", {}), ("chord strips", dict(n_rf_chord=3)), ("span strips", dict(n_rf_span=2))):
        combos.append((dea_lab, rf_lab) + dL(**dkw, **rkw))
shapes = []
for sh_lab, skw in (("bat armwing (rod + 2 digits)", {}), ("bat armwing, 1 digit (tip)", dict(digits="1.0")),
                    ("tapered 0.5, rod + tip digit", dict(taper=0.5, digits="1.0")),
                    ("rectangle, rigid frame", dict(le_mode="rigid", clamp_tip=True))):
    shapes.append((sh_lab,) + dL(**skw))
fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.2))
for k, (key, xl) in enumerate((("n_dea", r"Number of chordwise DEAs $n$"), ("pre_dea", r"DEA pre-stretch $\lambda_{pre,DEA}$"),
                                ("pre_mem", r"Membrane pre-stretch $\lambda_{pre}$"))):
    a = ax.flat[k]
    xv = [d_[0] for d_ in des[key]]
    a.plot(xv, [d_[1] for d_ in des[key]], "o-", color=PS.MATLAB[0], label=r"$\Delta L$")
    a2 = a.twinx()
    a2.plot(xv, [d_[1] / d_[3] if d_[3] else np.nan for d_ in des[key]], "s--", color=PS.MATLAB[1], label=r"$\Delta L$ per DEA mass")
    a.set(xlabel=xl, ylabel=r"Lift change $\Delta L$ at 1.6 kV in mN")
    a2.set_ylabel(r"$\Delta L$ / DEA mass in mN/g", color=PS.MATLAB[1])
labs = [f"{c[0]}\n{c[1]}" for c in combos]
ax[1, 0].bar(range(len(combos)), [c[2] for c in combos], color=[PS.MATLAB[0]] * 3 + [PS.MATLAB[1]] * 3)
ax[1, 0].set_xticks(range(len(combos)), labs, fontsize=6.5, rotation=35, ha="right"); ax[1, 0].set_ylabel(r"$\Delta L$ at 1.6 kV in mN")
ax[1, 1].bar(range(len(shapes)), [sh[1] for sh in shapes], color=PS.MATLAB[2])
ax[1, 1].set_xticks(range(len(shapes)), [sh[0].replace(", ", ",\n").replace(" (", "\n(") for sh in shapes], fontsize=6.5)
ax[1, 1].set_ylabel(r"$\Delta L$ at 1.6 kV in mN")
ax[1, 2].bar(range(len(shapes)), [sh[2] for sh in shapes], color=PS.MATLAB[5])
ax[1, 2].set_xticks(range(len(shapes)), [sh[0].replace(", ", ",\n").replace(" (", "\n(") for sh in shapes], fontsize=6.5)
ax[1, 2].set_ylabel(r"Lift $L$ (DEA off) in mN")
save(fig, "P3_design.png", axes=ax.flat)
summary["design"] = dict(n_dea=des["n_dea"], pre_dea=des["pre_dea"], pre_mem=des["pre_mem"], combos=combos, shapes=shapes)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "predictions.csv"), index=False)
with open(os.path.join(OUT, "predictions.json"), "w") as f:
    json.dump(summary, f, indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
