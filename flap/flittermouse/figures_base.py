"""Figures for the actuator and lattice sections. Style: ../plagiopatagiales/plotstyle.py (voltage off blue, on red).
Run: python figures_base.py -> figures/*.png and .svg, data/energy_per_cycle.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
FIG, DATA = os.path.join(HERE, "figures"), os.path.join(HERE, "data")
os.makedirs(FIG, exist_ok=True)
A, B = PS.MATLAB[0], PS.MATLAB[1]
L0, MASS = 31e-3, 99.51e-6
DF_ISO = 44.7e-3                        # N, 1.6 kV, isometric at stretch 1.12, bundle a
DF_CYC = 17.2e-3                        # N, 1.6 kV, same bundle, between slow loading curves
E_CHEN = 500.0 / 400.0


def save(fig, n):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, n + ".png"))
    plt.close(fig)


c = np.load(os.path.join(DATA, "actuator_curves.npz"))
g = c["grid"]
bench = json.load(open(os.path.join(DATA, "actuator_bench.json")))

# hysteresis of the large loop, scaled to small loops: a viscoelastic loss goes with amplitude squared
ok = g <= 1.22
W_load = np.trapezoid(c["a_load_0"][ok], g[ok]) * L0 / MASS
frac = {b: bench["hysteresis_fraction_of_loading_work"][b]["0"] for b in "ab"}
s_big = (g[ok].max() - g[ok].min()) / 2
loss_big = {b: frac[b] * W_load for b in "ab"}
s = np.linspace(0.005, 0.12, 116)
x = s * L0
loss = lambda amp, b="a": loss_big[b] * (amp / s_big) ** 2
branch = 0.38 * (s / 0.05) ** 2         # the gap between the large-loop branches over +-5 %: an upper bound for a small loop
gross = lambda dF, kind: (np.pi / 2 if kind == "sine" else 2.0) * dF * x / MASS
rows = dict(W_load_J_kg=W_load, s_big=s_big, loss_big=loss_big, frac=frac)
for name, dF in (("iso", DF_ISO), ("cyc", DF_CYC)):
    for kind in ("sine", "square"):
        for tag, ls in (("minor", loss(s)), ("branch", branch)):
            net = gross(dF, kind) - ls
            i = int(np.argmax(net))
            rows[f"{name}_{kind}_{tag}"] = dict(best_net_J_kg=float(net[i]), at_strain=float(s[i]), at5=float(np.interp(0.05, s, net)))
json.dump(rows, open(os.path.join(DATA, "energy_per_cycle.json"), "w"), indent=1)
print(json.dumps(rows, indent=0)[:1400])

fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.5))
ax[0].plot(g, c["a_load_0"] * 1e3, color=PS.OFF, label="0 V, loading")
ax[0].plot(g, c["a_unload_0"] * 1e3, color=PS.OFF, ls="--", label="0 V, unloading")
ax[0].plot(g, c["a_load_1600"] * 1e3, color=PS.ON, label="1.6 kV, loading")
ax[0].plot(g, c["a_unload_1600"] * 1e3, color=PS.ON, ls="--", label="1.6 kV, unloading")
ax[0].set(xlabel=r"Stretch $\lambda$", ylabel=r"Force $F$ in mN")
ax[0].legend(loc="upper left")
ax[1].plot(s * 100, gross(DF_ISO, "sine"), color=A, ls=":", label="Work, sine, isometric force")
ax[1].plot(s * 100, gross(DF_ISO, "sine") - loss(s), color=A, label="Net, small-loop loss")
ax[1].plot(s * 100, gross(DF_ISO, "sine") - branch, color=A, ls="--", label="Net, branch gap as loss")
ax[1].plot(s * 100, gross(DF_CYC, "sine") - loss(s), color=B, label="Net, slow-cycle force")
ax[1].axhline(E_CHEN, color="0.3", lw=1)
ax[1].text(0.5, E_CHEN + 0.05, "Ren, Kim et al. 2022", fontsize=8)
ax[1].set(xlabel=r"Strain amplitude $s$ in %", ylabel="Energy per cycle in J/kg", ylim=(-0.5, 3))
ax[1].legend(loc="upper left", fontsize=7)
st = bench["steps"]
Vv = sorted({q["V"] for q in st})
for lab, key, col in (("Stretch 1.12", "38mm", A), ("Stretch 1.00", "0mm", B)):
    y = [[q["t1090_ms"] for q in st if q["V"] == v and key in q["file"]] for v in Vv]
    xx = [v for v, q in zip(Vv, y) if q]
    ax[2].plot(np.array(xx) / 1e3, [np.median(q) for q in y if q], "o-", color=col, label=lab)
ax[2].axhline(20, color="0.3", lw=1)
ax[2].text(0.82, 21.5, "half stroke at 25 Hz", fontsize=8)
ax[2].axhspan(130, 190, color="0.88")
ax[2].text(0.82, 145, "vocal cord motion, 1.8 kV", fontsize=8)
ax[2].set(xlabel=r"Voltage $V$ in kV", ylabel=r"Force fall, 10-90 %, in ms", ylim=(0, 200))
ax[2].legend(loc="center right")
PS.finish(ax)
save(fig, "A_actuator")

u = json.load(open(os.path.join(DATA, "uvlm_check.json")))
fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.5))
ar = [r["AR"] for r in u["steady"]]
ax[0].plot(ar, [r["helmbold"] for r in u["steady"]], "s--", color="0.4", label="Helmbold")
ax[0].plot(ar, [r["slope"] for r in u["steady"]], "o-", color=A, label="Lattice")
ax[0].set(xlabel=r"Aspect ratio $AR$", ylabel=r"Lift slope $C_{L\alpha}$ per rad")
ax[0].legend()
ax[1].plot(ar, [r["CDi"] for r in u["steady"]], "s--", color="0.4", label=r"$C_L^2/(\pi AR)$")
ax[1].plot(ar, [r["CD"] for r in u["steady"]], "o-", color=A, label="Lattice")
ax[1].set(xlabel=r"Aspect ratio $AR$", ylabel=r"Induced drag $C_D$ at 5°")
ax[1].legend()
k = [r["k"] for r in u["plunge"]]
ax[2].plot(k, [r["CT"] / r["CT_th"] for r in u["plunge"]], "o-", color=A, label="Thrust / Garrick")
ax[2].plot(k, [r["CP"] / r["CP_th"] for r in u["plunge"]], "s-", color=B, label="Power / Garrick")
ax[2].plot(k, [r["L_amp"] / r["L_amp_th"] for r in u["plunge"]], "^-", color=PS.MATLAB[3], label="Lift / Theodorsen")
ax[2].axhline(1, color="0.3", lw=1)
ax[2].axvspan(0.2, 0.7, color="0.9")
ax[2].set(xlabel=r"Reduced frequency $k=\omega c/2U$", ylabel="Lattice / theory", xscale="log")
ax[2].legend()
PS.finish(ax)
save(fig, "L_lattice")
print("done")
