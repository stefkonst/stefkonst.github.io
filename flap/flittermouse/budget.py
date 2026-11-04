"""W4, W8, W9. Drive at the chosen flight conditions with coupled-model air power, Monte Carlo on closure, rig resolution, on-board budget.
Run: python budget.py -> data/budget.json, figures/B_budget.png and .svg"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import drive
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
load = lambda s: [json.loads(l) for l in open(os.path.join(HERE, "data", f"timing_{s}.jsonl"))]
F, C = load("f"), load("c")
out = {}


def find(R, **k):
    return next(r for r in R if all(r.get(q) == v for q, v in k.items()))


def ca_for(P, Phi_deg, f):
    Om = np.radians(Phi_deg) * 2 * np.pi * f
    return P / (np.mean(np.abs(np.cos(np.linspace(0, 2 * np.pi, 1001))) ** 3) * Om ** 3)


# ---------- W4: inertia of the 299 mg wing, with added air, about the root
m_w = 0.299e-3
I_w = (m_w + 30e-6) * 0.05 ** 2 / 3
rows = []
for name, r_ in (("25 Hz, 20°, 16°, capped", find(F, mode="down", cn_cap=1.6, U=4.0, f=25.0, Phi_deg=20.0, alpha_deg=16.0)),
                 ("25 Hz, 30°, 8°, attached", find(C, mode="down", U=4.0, f=25.0, Phi_deg=30.0, alpha_deg=8.0)),
                 ("15 Hz, 30°, 8°, attached", find(C, mode="down", U=4.0, f=15.0, Phi_deg=30.0, alpha_deg=8.0))):
    P, f, Phi = r_["P"], r_["f"], r_["Phi_deg"]
    for m_core, sig, lab in ((0.5e-3, drive.SIG_ISO, "iso"), (1.0e-3, drive.SIG_ISO, "iso"), (1.0e-3, drive.SIG_CYC, "cyc")):
        # lever from a strain amplitude of 5 %
        r = 0.05 * 0.020 / np.radians(Phi)
        a = drive.simulate(m_core=m_core, r=r, I=I_w, f=f, sig=sig, c_a=ca_for(P, Phi, f), tau_mean=2 * r_["L"] * 0.03 / 2)
        rows.append(dict(case=name, P_need_mW=P * 1e3, m_core_mg=m_core * 1e6, force=lab, lever_mm=r * 1e3, Phi_target=Phi, **{k: a[k] for k in ("Phi_deg", "strain_amp", "F_min_mN", "F_max_mN", "slack", "spring_share", "P_aero_mW", "P_h_mW", "kt_s")}))
        print(name, int(m_core * 1e6), lab, "need %.1f mW, reach Phi %.1f of %.0f, P_air %.1f, loss %.1f, Fmin %.0f slack %s spring %.2f" % (P * 1e3, a["Phi_deg"], Phi, a["P_aero_mW"], a["P_h_mW"], a["F_min_mN"], a["slack"], a["spring_share"]))
out["drive"] = rows

# ---------- W9: Monte Carlo over the unknowns, on the coupled map (capped and attached)
rng = np.random.default_rng(4)
MAP = [r for r in F + C if r["mode"] == "down" and "error" not in r]
L = np.array([r["L"] for r in MAP]); T = np.array([r["T"] for r in MAP]); Tn = np.array([r["T_ns"] for r in MAP]); Pw = np.array([r["P"] for r in MAP]); ff = np.array([r["f"] for r in MAP])
cap = np.array([r.get("cn_cap") or 9.0 for r in MAP])
N = 20000
res = np.zeros(N)
win = np.zeros((N, 3))
samples = dict(e=np.exp(rng.uniform(np.log(0.25), np.log(1.5), N)), eta_tr=rng.uniform(0.5, 0.9, N), suction=rng.uniform(0, 1, N), m_w=rng.uniform(0.24e-3, 0.33e-3, N), m_f=rng.uniform(0.1e-3, 0.3e-3, N), flow=rng.integers(0, 3, N))
flows = (9.0, 1.6, 1.2)
for i in range(N):
    s = samples
    mask = cap == flows[s["flow"][i]]
    Th = s["suction"][i] * T + (1 - s["suction"][i]) * Tn
    m_c = Pw / (s["e"][i] * ff * s["eta_tr"][i])
    LW = 2 * L / ((2 * (m_c + s["m_w"][i]) + s["m_f"][i]) * 9.81)
    LW = np.where(mask & (Th >= 0), LW, 0)
    res[i] = LW.max()
out["mc"] = dict(N=N, median=float(np.median(res)), p10=float(np.percentile(res, 10)), p90=float(np.percentile(res, 90)), p_gt1=float((res >= 1).mean()), p_gt_07=float((res >= 0.7).mean()), no_trim=float((res == 0).mean()))
corr = {k: float(np.corrcoef(np.log(samples[k]) if k == "e" else samples[k], res)[0, 1]) for k in ("e", "eta_tr", "suction", "m_w", "m_f")}
corr["flow"] = float(np.corrcoef(samples["flow"], res)[0, 1])
out["mc"]["corr"] = corr
print(out["mc"])
# what e and suction are needed
ee = np.exp(np.linspace(np.log(0.25), np.log(3.0), 40))
needs = {}
for fl, nm in zip(flows, ("attached", "cap 1.6", "cap 1.2")):
    for s_ in (1.0, 0.5, 0.0):
        mask = cap == fl
        Th = s_ * T + (1 - s_) * Tn
        cur = []
        for e in ee:
            m_c = Pw / (e * ff * 0.8)
            LW = 2 * L / ((2 * (m_c + 0.299e-3) + 0.2e-3) * 9.81)
            LW = np.where(mask & (Th >= 0), LW, 0)
            cur.append(float(LW.max()))
        needs[f"{nm}|{s_}"] = cur
out["needs"] = dict(e=ee.tolist(), curves=needs)

# ---------- rig resolution: per-wing lift effect against noise, cycle averaged
effects = dict(lift_mN=8.0, timing_gain_mN=0.5, steady_dL_mN=3.3)
rig = []
for lab, sigma in (("Earlier rig, within-condition scatter", 40.0), ("Target rig, cycle mean scatter", 3.0), ("Light rig, cycle mean scatter", 1.0)):
    n = (2 * 2.8 * sigma / effects["timing_gain_mN"]) ** 2 / 2 * 2    # paired on/off, 2.8 = 80 % power at 5 %
    rig.append(dict(rig=lab, sigma_mN=sigma, cycles_each=float(n), seconds_at_25Hz=float(2 * n / 25)))
out["rig"] = rig
print(rig)

# ---------- on-board budget, order of magnitude
P_el = dict(no_recovery_W=2 * 0.5 * 9.4e-9 * 1600 ** 2 * 25, recovery_50pct_W=2 * 0.5 * 9.4e-9 * 1600 ** 2 * 25 * 0.5)
batt = 200.0 * 3600                                      # J/kg, LiPo order of magnitude
out["onboard"] = dict(P_el, minutes_per_100mg_batt=0.1e-3 * batt / P_el["no_recovery_W"] / 60, note="2 channels, 1.6 kV, 9.4 nF, 25 Hz")
print(out["onboard"])
json.dump(out, open(os.path.join(HERE, "data", "budget.json"), "w"), indent=1, default=float)

# ---------- figure
fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.5))
ax[0].hist(res, bins=np.arange(0, 2.0, 0.05), color=PS.MATLAB[0], edgecolor="white", lw=0.4)
ax[0].axvline(1, color=PS.ON, lw=1.4)
ax[0].text(1.02, ax[0].get_ylim()[1] * 0.9, "flight", color=PS.ON, fontsize=9)
ax[0].set(xlabel=r"Best lift / weight, $L/W$", ylabel="Samples")
sty = {"attached": PS.MATLAB[0], "cap 1.6": PS.MATLAB[1], "cap 1.2": PS.MATLAB[3]}
lab_s = {1.0: "-", 0.5: "--", 0.0: ":"}
for fl, c in sty.items():
    for s_, ls in lab_s.items():
        ax[1].semilogx(ee, needs[f"{fl}|{s_}"], ls, color=c, label=f"{fl}, suction {s_:g}" if s_ == 1.0 else None)
ax[1].axhline(1, color=PS.ON, lw=1.2)
ax[1].axvline(0.8, color="0.4", lw=1)
ax[1].axvline(1.25, color="0.4", lw=1)
ax[1].text(0.81, 0.04, "bench", fontsize=8, rotation=90)
ax[1].text(1.27, 0.04, "published", fontsize=8, rotation=90)
ax[1].set(xlabel=r"Energy per cycle $e$ in J/kg", ylabel=r"Best $L/W$", ylim=(0, 2))
ax[1].legend(fontsize=7, loc="upper left")
ks = ["e", "flow", "suction", "eta_tr", "m_w", "m_f"]
names = [r"$e$", "Flow model", "Suction", "Transmission", "Wing mass", "Frame mass"]
ax[2].barh(range(len(ks)), [corr[k] for k in ks], color=PS.MATLAB[0])
ax[2].set_yticks(range(len(ks)), names)
ax[2].invert_yaxis()
ax[2].axvline(0, color="0.3", lw=1)
ax[2].set(xlabel=r"Correlation with best $L/W$")
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "B_budget.png"))
