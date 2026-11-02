"""Observations only (no model): what the bench and the videos show about the DEAs in the wings.
Inputs are derived files (the raw videos stay local; scripts in video/ produced the derived files):
  paper_tension/fig2_dea_actuator/results/isometric_force_drop.csv     bench, isometric 1 Hz pulses, 5-DEA bundle
  derived/static1Hz_region_signal.txt   FAAV wing, no flow, 1 Hz drive: mean |I - I_mean| in the tip and inboard panels
  derived/flap_phi_0kV_900V.txt         FAAV flapping videos: stroke angle of the LE rod (projected), 0 kV and 900 V
  derived/flap_kinematics.json          cycle statistics and phase-matched shape differences
  derived/gliding_video_strain.csv      gliding wing, no flow: tracked displacement / L0 (analyze_strain.py, Bat/videos)
Camera: 400 fps, 2001 frames (5 s). Run: python observations.py -> validation/O_observations.png, observations.json
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotstyle as PS
import matplotlib.pyplot as plt

PS.apply()
DER = os.path.join(HERE, "derived")
PT = r"C:/Users/stefk/Documents/paper_tension"
rep = {}
fig, ax = plt.subplots(2, 3, figsize=(11, 6.6))

# a) bench: isometric force drop vs V at lambda 1.00 and 1.12 -- V^2 law and saturation when the pre-tension is used up
iso = pd.read_csv(os.path.join(PT, "fig2_dea_actuator/results/isometric_force_drop.csv"))
for lam, col in ((1.00, PS.MATLAB[0]), (1.12, PS.MATLAB[1])):
    d = iso[np.isclose(iso.stretch, lam)]
    low = d[d.V <= 1000]
    k = float(np.sum(low.dF_N * low.V ** 2) / np.sum(low.V ** 4))
    ax[0, 0].errorbar(d.V, d.dF_N * 1e3, yerr=d.dF_sd * 1e3, fmt="o", color=col, capsize=2, label=rf"measured, $\lambda$ = {lam:.2f}")
    vv = np.linspace(0, 1800, 50)
    ax[0, 0].plot(vv, k * vv ** 2 * 1e3, "--", color=col, lw=0.8, label=r"$\propto V^2$ (fit $V \leq$ 1 kV)")
    rep[f"bench_lambda_{lam:.2f}"] = dict(V=d.V.tolist(), dF_mN=(d.dF_N * 1e3).tolist(), V2_fit_at_1600_mN=k * 1600 ** 2 * 1e3)
ax[0, 0].set(xlabel=r"Voltage $V$ in V", ylabel=r"Bundle force drop $\Delta F$ in mN (5 DEAs)"); ax[0, 0].legend(fontsize=6.5)

# b) FAAV static 1 Hz drive: spectrum of the tip and inboard panels
sig = np.loadtxt(os.path.join(DER, "static1Hz_region_signal.txt"))
t, tip, inb = sig[:, 0], sig[:, 1], sig[:, 2]
fs = 1 / np.mean(np.diff(t))
for y, lab, col in ((tip, "tip panel (outboard of digit)", PS.MATLAB[1]), (inb, "inboard panel", PS.MATLAB[0])):
    F = np.abs(np.fft.rfft(y - y.mean())) / len(y) * 2
    f = np.fft.rfftfreq(len(y), 1 / fs)
    ax[0, 1].plot(f, F, color=col, label=lab)
    rep["static_" + lab.split()[0]] = dict(amp_at_2Hz=float(F[np.argmin(abs(f - 2))]), amp_at_1Hz=float(F[np.argmin(abs(f - 1))]))
ax[0, 1].axvline(1, color="0.5", ls=":", lw=0.8); ax[0, 1].axvline(2, color="0.5", ls="--", lw=0.8)
ax[0, 1].text(2.1, ax[0, 1].get_ylim()[1] * 0.85, r"2$f_{drive}$ ($\propto V^2$)", fontsize=7)
ax[0, 1].set(xlim=(0, 6), xlabel=r"Frequency in Hz (drive 1 Hz)", ylabel="Image change amplitude in grey levels"); ax[0, 1].legend(fontsize=7)

# c) flapping: phase-averaged stroke angle, 0 kV vs 900 V
phi = np.loadtxt(os.path.join(DER, "flap_phi_0kV_900V.txt"))
kin = json.load(open(os.path.join(DER, "flap_kinematics.json")))
cyc = int(round(fs / kin["kin"]["0kV"]["f_Hz"]))
means = []
for j in range(2):
    y = phi[:, 1 + j]; n = len(y) // cyc
    C = y[:n * cyc].reshape(n, cyc); means.append((C.mean(0), C.std(0), n))
# the two recordings are not synchronised: shift the 900 V cycle to the best match of the 0 kV cycle (circular)
shift = int(np.argmax([np.dot(means[0][0], np.roll(means[1][0], -k)) for k in range(cyc)]))
ph = np.arange(cyc) / cyc * 360
for j, (lab, col) in enumerate((("0 kV", PS.OFF), ("900 V (phase-aligned)", PS.ON))):
    m, s, n = means[j]
    if j == 1:
        m, s = np.roll(m, -shift), np.roll(s, -shift)
    ax[0, 2].plot(ph, m, color=col, label=f"{lab}, mean of {n} cycles")
    ax[0, 2].fill_between(ph, m - s, m + s, color=col, alpha=0.15, lw=0)
ax[0, 2].axhspan(-40, -15, color="0.85", zorder=0)
ax[0, 2].text(5, -33, "rod over body shadow: not tracked reliably", fontsize=6.5)
rep["flap_cycle_rms_0kV_vs_900V_deg"] = float(np.sqrt(np.mean((means[0][0] - np.roll(means[1][0], -shift))[means[0][0] > -15] ** 2)))
ax[0, 2].set(xlabel=r"Cycle phase in °", ylabel=r"Projected stroke angle $\Phi$ in °", ylim=(-40, 45)); ax[0, 2].legend(fontsize=6.5, loc="upper right")
rep["flap_kinematics"] = kin["kin"]

# d) flapping: phase-matched shape difference, 0 kV vs 900 V, against the cycle-to-cycle scatter at 0 kV
sh = kin["shape"]
lab = [f"Φ = {r['phi']:.0f}°\n{r['stroke']}" for r in sh]
xx = np.arange(len(sh))
ax[1, 0].bar(xx - 0.27, [r["within_single"] for r in sh], 0.27, color="0.6", label="single cycle vs 0 kV mean")
ax[1, 0].bar(xx, [r["split_half_0kV"] for r in sh], 0.27, color=PS.MATLAB[0], label="0 kV: odd vs even cycles")
ax[1, 0].bar(xx + 0.27, [r["between_0kV_900V"] for r in sh], 0.27, color=PS.MATLAB[1], label="0 kV mean vs 900 V mean")
ax[1, 0].set_xticks(xx); ax[1, 0].set_xticklabels(lab, fontsize=7)
ax[1, 0].set(ylabel="Mean |image difference| in grey levels"); ax[1, 0].legend(fontsize=6.5)

# e) gliding wing, no flow: tracked displacement / L0 vs voltage (analyze_strain.py)
gs = pd.read_csv(os.path.join(DER, "gliding_video_strain.csv"))
for cfg, col, lab_ in (("ani", PS.MATLAB[1], "anisotropic membrane"), ("iso", PS.MATLAB[0], "isotropic membrane")):
    d = gs[gs.config == cfg]
    ax[1, 1].errorbar(d.voltage, d.mean_strain * 100, yerr=d.std_strain * 100, fmt="o-", color=col, capsize=2, label=lab_)
    v0 = d.voltage.iloc[0]; s0 = d.mean_strain.iloc[0]
    vv = np.linspace(v0, 1800, 30)
    ax[1, 1].plot(vv, s0 * (vv / v0) ** 2 * 100, ":", color=col, lw=0.8, label=r"$\propto V^2$ from first point")
ax[1, 1].set(xlabel=r"Voltage $V$ in V", ylabel=r"Tracked displacement / $L_0$ in %", ylim=(0, 2)); ax[1, 1].legend(fontsize=6.5)
rep["gliding_video"] = gs[["config", "voltage", "mean_strain", "std_strain"]].to_dict("list")

# f) summary of what each observation implies (text panel)
ax[1, 2].axis("off")
txt = ("Observations (no model):\n"
       "a) bench: at λ = 1.00 the drop saturates near 30 mN\n    above 1.4 kV (pre-tension used up); at 1.12 it follows V².\n"
       "b) FAAV, no flow: response at 2 Hz (V²); tip panel only.\n    Optical flow: tip ≤ 0.2 mm (p95), inboard ≤ 0.05 mm.\n"
       "c) flapping: 0 kV and 900 V strokes identical; frequency\n    2.0 Hz if the clip lasts 5 s (labelled 5 Hz).\n"
       "d) 900 V shape change not larger than cycle scatter.\n"
       "e) gliding wing: 0.5-1 % at 1-1.8 kV, iso flat in V,\n    ani saturates from 1.4 kV; resolution a few pixels.\n"
       "Video stills: DEAs bowed, wrinkles parallel to DEAs.")
ax[1, 2].text(0, 1, txt, va="top", fontsize=8.2, family="serif")
PS.finish([a for a in ax.flat][:5])
fig.tight_layout()
fig.savefig(os.path.join(HERE, "validation", "O_observations.png"))
json.dump(rep, open(os.path.join(HERE, "validation", "observations.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in rep.items() if k.startswith(("bench", "static"))}, indent=1, default=float))
