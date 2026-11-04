"""Ribs in flapping, timing flipped, bat kinematics. Run: python figures_flip.py -> figures/R_ribs.png, F_flip.png, K_bat.png (and .svg)."""
import json, os, sys
import statistics as st
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
FIG = os.path.join(HERE, "figures")
load = lambda t: [r for r in (json.loads(l) for l in open(os.path.join(HERE, "data", f"coupled_{t}.jsonl"))) if "error" not in r and not r.get("unstable")]
KEY = lambda r: (r.get("U", 4.0), r["f"], r["Phi_deg"], r["alpha_deg"])
A_, B_, C_ = PS.MATLAB[0], PS.MATLAB[1], PS.MATLAB[3]
ETA, MF = 0.8, 0.2e-3


def best(rows, e, s=1.0, extra=0.0):
    b = None
    for r in rows:
        T = s * r["T"] + (1 - s) * r["T_ns"]
        if T < 0 or r["P"] <= 0:
            continue
        m = 2 * (r["P"] / (e * r["f"] * ETA) + r["mass_mg"] * 1e-6 + extra) + MF
        v = 2 * r["L"] / (m * 9.81)
        b = v if b is None else max(b, v)
    return b


# ---------- ribs
R = load("r")
W = ["plain", "one rib", "one stiff rib", "two stiff ribs", "four stiff ribs"]
lab = ["Plain", "1 rib\n0.5×0.1", "1 rib\n0.5×0.2", "2 ribs\n0.5×0.2", "4 ribs\n0.5×0.2"]
plain = {KEY(r): r for r in R if r["wing"] == "plain" and r["mode"] == "off"}
fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.4))
x = np.arange(len(W))
for q, nm, c, dx in (("L", "Lift", A_, -.24), ("T", "Thrust", B_, 0.0), ("P", "Air power", C_, .24)):
    y, lo, hi = [], [], []
    for w in W:
        v = [r[q] / plain[KEY(r)][q] * 100 for r in R if r["wing"] == w and r["mode"] == "off" and abs(plain[KEY(r)][q]) > 1e-3]
        y.append(st.median(v))
    ax[0].bar(x + dx, y, width=0.22, color=c, label=nm)
ax[0].axhline(100, color="0.3", lw=1)
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=8)
ax[0].set(ylabel="Share of the plain wing, %", ylim=(0, 125), title="0 V, median of six flight points")
ax[0].legend(fontsize=8, ncol=3, loc="upper center")
for e, c, nm in ((0.8, A_, "0.8 J/kg"), (1.25, B_, "1.25 J/kg"), (2.1, C_, "2.1 J/kg")):
    ax[1].plot(x, [best([r for r in R if r["wing"] == w and r["mode"] == "off"], e) for w in W], "o-", color=c, ms=5, label=nm)
ax[1].axhline(1, color=PS.ON, lw=1.2)
ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=8)
ax[1].set(ylabel=r"Best $L/W$, full suction", ylim=(0, 1.6), title="Closure, six flight points")
ax[1].legend(fontsize=8)
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "R_ribs.png"))
plt.close(fig)

# ---------- timing
F = load("f")
off = {(KEY(r), r.get("cn_cap")): r for r in F if r["mode"] == "off"}
D = (-0.4, -0.2, 0.0, 0.2, 0.4)
fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=True)
for i, tau in enumerate((0.0, 0.010)):
    for m, c, nm in (("down", PS.ON, "Soft down, stiff up"), ("up", PS.OFF, "Stiff down, soft up")):
        for cap, ls in ((1.2, "-"), (1.6, "--")):
            med, lo, hi = [], [], []
            for d in D:
                g = [(r["L"] - off[(KEY(r), cap)]["L"]) * 1e3 for r in F if r["mode"] == m and r.get("tau", 0) == tau and r.get("cn_cap") == cap and r.get("cn_cap_soft") == d]
                med.append(st.median(g)); lo.append(min(g)); hi.append(max(g))
            ax[i].plot(D, med, "o" + ls, color=c, ms=4, label=f"{nm}, $c_{{n,max}}$ {cap}")
            if cap == 1.6:
                ax[i].fill_between(D, lo, hi, color=c, alpha=0.1, lw=0)
    ax[i].axhline(0, color="0.3", lw=1)
    ax[i].axvline(0, color="0.8", lw=1)
    ax[i].set(xlabel=r"Stall limit change when soft, $\Delta c_{n,max}$", title=f"Response lag {tau * 1e3:.0f} ms")
ax[0].set_ylabel(r"$\Delta L$ against 0 V, mN per wing")
ax[0].legend(fontsize=7)
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "F_flip.png"))
plt.close(fig)

# ---------- bat
C = load("c")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
cfg = [(0.0, 0.0, "None"), (0.0, 20.0, "Tilt 20°"), (0.2, 0.0, "Fold 20 %"), (0.4, 0.0, "Fold 40 %"), (0.4, 20.0, "Fold 40 %,\ntilt 20°")]
x = np.arange(len(cfg))
B5 = [r for r in C if r["b"] == 0.05 and r["wing"] == "plain" and r["mode"] == "off"]
base = {KEY(r): r for r in B5 if r["fold"] == 0 and r["tilt_deg"] == 0}
for q, nm, c, dx in (("L", "Lift", A_, -.24), ("T", "Thrust", B_, 0.0), ("P", "Air power", C_, .24)):
    y = [st.median(r[q] / base[KEY(r)][q] * 100 for r in B5 if r["fold"] == fo and r["tilt_deg"] == ti and abs(base[KEY(r)][q]) > 1e-3 and (q != "T" or base[KEY(r)]["T"] > 5e-4)) for fo, ti, _ in cfg]
    ax[0].bar(x + dx, y, width=0.22, color=c, label=nm)
ax[0].axhline(100, color="0.3", lw=1)
ax[0].set_xticks(x); ax[0].set_xticklabels([c[2] for c in cfg], fontsize=7.5)
ax[0].set(ylabel="Share of no fold, no tilt, %", title="5 × 2.5 cm, 0 V", ylim=(0, 160))
ax[0].legend(fontsize=8, ncol=3, loc="upper center")
FO = (0.0, 0.2, 0.4)
for m, c, nm in (("off", PS.OFF, "0 V"), ("up", PS.ON, "Stiff down, soft up")):
    y = [st.median(r["L"] / base[KEY(r)]["L"] * 100 for r in C if r["b"] == 0.05 and r["wing"] == "plain" and r["mode"] == m and r["fold"] == fo and r["tilt_deg"] == 0) for fo in FO]
    ax[1].plot(FO, y, "o-", color=c, ms=5, label=nm)
ax[1].axhline(100, color="0.3", lw=1)
ax[1].set(xlabel="Span fold on the upstroke", ylabel="Lift, share of no fold at 0 V, %", title="Fold removes the timing penalty")
ax[1].legend(fontsize=8)
for (b, ls, nm) in ((0.05, "-", "5×2.5"), (0.07, "--", "7×3")):
    for e, c in ((1.25, B_),):
        y = []
        for fo in FO:
            S = [r for r in C if r["b"] == b and r["wing"] == "one stiff rib" and r["fold"] == fo and r["mode"] == "off"]
            y.append(best(S, e, 1.0, 30e-6 if fo else 0.0))
        ax[2].plot(FO, y, "o" + ls, color=A_, ms=5, label=f"{nm}, full suction")
        y = []
        for fo in FO:
            S = [r for r in C if r["b"] == b and r["wing"] == "one stiff rib" and r["fold"] == fo and r["mode"] == "off"]
            y.append(best(S, e, 0.5, 30e-6 if fo else 0.0))
        ax[2].plot(FO, [np.nan if v is None else v for v in y], "s" + ls, color=C_, ms=5, label=f"{nm}, half suction")
ax[2].axhline(1, color=PS.ON, lw=1.2)
ax[2].set(xlabel="Span fold on the upstroke", ylabel=r"Best $L/W$, 1.25 J/kg, one rib", ylim=(0, 1.5), title="Closure")
ax[2].legend(fontsize=7.5)
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "K_bat.png"))
plt.close(fig)
