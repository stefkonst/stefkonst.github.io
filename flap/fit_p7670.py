"""Yeoh fit of the DEA silicone (Elastosil P7670) from the rolled-DEA bench data, and the electromechanical
efficiency eta from the same bundle.

Data: paper_tension/fig2_dea_actuator/results/cyclic_loading_mean.csv (bundle of n = 5 rolled DEAs, L0 = 31 mm,
strip width w = 10 mm, layers 15 + 30 + 15 um, loading branches of cycles 2-4, 2 mm/s) and
isometric_force_drop.csv (1 Hz pulses at lambda = 1.00 and 1.12).

Passive (0 V) uniaxial force of the bundle, incompressible Yeoh, reference section A0 = n w (H + Hp):
    F(lambda) = A0 * 2 (lambda - lambda^-2) * dW/dI1,   I1 = lambda^2 + 2/lambda
Active force drop (equation of state of the ideal DEA, chi = H/(H+Hp)):
    dF = eta * chi * eps0 eps_r (V/H)^2 * A0
Run: python fit_p7670.py   (prints the constants pasted into model.py and writes validation/F_p7670.png)
"""
import os
import numpy as np, pandas as pd
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = r"C:/Users/stefk/Documents/paper_tension/fig2_dea_actuator/results"
n, w, H, Hp, eps_r = 5, 10e-3, 30e-6, 30e-6, 2.8
A0, chi, EPS0 = n * w * (H + Hp), H / (H + Hp), 8.854e-12

d = pd.read_csv(os.path.join(RES, "cyclic_loading_mean.csv")).dropna(subset=["F_0V_N"])
lam, F0 = d["stretch"].to_numpy(), d["F_0V_N"].to_numpy()
I1 = lam ** 2 + 2 / lam
k = 2 * (lam - lam ** -2) * A0
B = np.c_[k, k * 2 * (I1 - 3), k * 3 * (I1 - 3) ** 2]
C, _ = nnls(B, F0)
rms = float(np.sqrt(np.mean((B @ C - F0) ** 2)))

# eta from the quasi-static cycles (force difference at equal stretch) and from the isometric pulses
ideal = lambda V: chi * EPS0 * eps_r * (V / H) ** 2 * A0
eta_cyc = {}
for V in (1000, 1200, 1400, 1600):
    col = f"F_{V}V_N"
    if col in d:
        m = np.isfinite(d[col])
        dF = np.interp(1.12, lam[m], F0[m] - d[col].to_numpy()[m])
        eta_cyc[V] = float(dF / ideal(V))
iso = pd.read_csv(os.path.join(RES, "isometric_force_drop.csv"))
iso112 = iso[iso.stretch == 1.12]
eta_iso = {int(r.V): float(r.dF_N / ideal(r.V)) for r in iso112.itertuples() if r.V >= 1000}

print("P7670 Yeoh (Pa):", [round(c, 1) for c in C], " rms (mN):", round(rms * 1e3, 3), " range:", lam.min(), "-", lam.max())
print("eta, quasi-static cycles at lambda = 1.12:", {k_: round(v, 3) for k_, v in eta_cyc.items()})
print("eta, isometric 1 Hz pulses at lambda = 1.12:", {k_: round(v, 3) for k_, v in eta_iso.items()})

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 10})
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
ll = np.linspace(1, 1.25, 200); i1 = ll ** 2 + 2 / ll; kk = 2 * (ll - ll ** -2) * A0
ax[0].plot(lam, F0 * 1e3, "o", ms=3, mfc="none", color="#0072BD", label="measured, 0 V (5 rolled DEAs)")
ax[0].plot(ll, (kk * (C[0] + 2 * C[1] * (i1 - 3) + 3 * C[2] * (i1 - 3) ** 2)) * 1e3, "-", color="#D95319", label="Yeoh fit")
ax[0].set(xlabel=r"Stretch $\lambda$", ylabel=r"Force $F$ in mN")
ax[0].legend(fontsize=8)
Vs = np.array(sorted(set(eta_cyc) | set(eta_iso)))
ax[1].plot(list(eta_iso), list(eta_iso.values()), "s", color="#0072BD", label="isometric pulses (1 Hz)")
ax[1].plot(list(eta_cyc), list(eta_cyc.values()), "o", color="#D95319", label="quasi-static cycles (2 mm/s)")
ax[1].set(xlabel=r"Voltage $V$ in V", ylabel=r"Efficiency $\eta$", ylim=(0, 0.6))
ax[1].legend(fontsize=8)
for a, lab in zip(ax, "ab"):
    a.grid(True, which="major", color="0.85"); a.minorticks_on(); a.grid(True, which="minor", color="0.93", lw=0.5)
    a.text(-0.16, 1.02, lab + ")", transform=a.transAxes, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "validation", "F_p7670.png"), dpi=150)
