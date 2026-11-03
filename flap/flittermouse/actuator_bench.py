"""W1. Measured actuator: bundle of 5 rolled fibres (paper_tension fig. 2 data, texture analyser).
Roll: L0 31 mm, strip 10 mm, layers 15 + 30 + 15 um. Questions: work per cycle per kg from the
measured curves (no eta, no assumed strain), hysteresis, and how fast force follows voltage."""
import glob, json, os
import numpy as np
from scipy.signal import find_peaks, savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\stefk\Documents\paper_tension\fig2_dea_actuator\data\DEAs"
L0, N, W, T, RHO = 31e-3, 5, 10e-3, 60e-6, 1070.0
A = N * W * T
MASS = A * L0 * RHO
G = 9.81e-3


def load(name):
    d = np.loadtxt(os.path.join(SRC, name), skiprows=4)
    F = (d[:, 0] - d[:10, 0].mean()) * G
    x = np.abs(d[:, 1] - d[:10, 1].mean()) * 1e-3
    return x / L0 + 1, F, d[:, 2]


def cycles(lam, F):
    pk, _ = find_peaks(lam, prominence=0.05)
    va, _ = find_peaks(-lam, prominence=0.05)
    out = []
    for p in pk:
        v1, v2 = va[va < p], va[va > p]
        if len(v1) and len(v2):
            out.append((v1[-1], p, v2[0]))
    return out


grid = np.linspace(1.03, 1.22, 96)
res = dict(geom=dict(A_mm2=A * 1e6, mass_mg=MASS * 1e6))
load_c, unload_c, hyst = {}, {}, {}
for bundle, tags in (("a", {0: "0000V1", 400: "0400V1", 800: "0800V1", 1000: "1000V1", 1200: "1200V1", 1400: "1400V1", 1600: "1600V1"}),
                     ("b", {0: "0000V1", 400: "0400V1", 800: "0800V3", 1000: "1000V1", 1200: "1200V1", 1400: "1400V1", 1600: "1600V1"})):
    LC, UC, H, sp = {}, {}, {}, []
    for v, t in tags.items():
        fn = f"5fibres_{bundle}_{t}.tab"
        if not os.path.exists(os.path.join(SRC, fn)):
            continue
        lam, F, tm = load(fn)
        cy = cycles(lam, F)[1:4]
        ll, uu, hh = [], [], []
        for a, b, c in cy:
            ll.append(np.interp(grid, lam[a:b], F[a:b]))
            uu.append(np.interp(grid, lam[b:c][::-1], F[b:c][::-1]))
            mask = (grid > lam[a:b].min()) & (grid < lam[a:b].max())
            hh.append(np.trapezoid((ll[-1] - uu[-1])[mask], grid[mask]) * L0 / np.trapezoid(ll[-1][mask], grid[mask]) / L0)
            sp.append((lam[b] - lam[a]) * L0 / (tm[b] - tm[a]))
        if ll:
            LC[v], UC[v], H[v] = np.mean(ll, 0), np.mean(uu, 0), float(np.mean(hh))
    load_c[bundle], unload_c[bundle], hyst[bundle] = LC, UC, H
    res[f"speed_mm_s_{bundle}"] = float(np.mean(sp) * 1e3)

# work per cycle: stretch with voltage on (low force), relax with voltage off (high force).
rows = []
for bundle in load_c:
    LC, UC = load_c[bundle], unload_c[bundle]
    for v in LC:
        if v == 0:
            continue
        for lo, hi in ((1.05, 1.15), (1.05, 1.20), (1.08, 1.12), (1.10, 1.20)):
            m = (grid >= lo) & (grid <= hi)
            ideal = np.trapezoid((LC[0] - LC[v])[m], grid[m]) * L0       # same branch both ways: no hysteresis
            real = np.trapezoid((UC[0] - LC[v])[m], grid[m]) * L0        # off on the way back, on on the way out
            rows.append(dict(bundle=bundle, V=v, lo=lo, hi=hi, W_same_branch_J_kg=ideal / MASS, W_loop_J_kg=real / MASS,
                             dF_mN=float(np.mean((LC[0] - LC[v])[m]) * 1e3)))
res["work"] = rows
res["hysteresis_fraction_of_loading_work"] = hyst
Fref = load_c["a"][0]
res["k_bundle_N_m"] = float(np.polyfit(grid[(grid > 1.05) & (grid < 1.15)] * L0, Fref[(grid > 1.05) & (grid < 1.15)], 1)[0])
res["E_kPa"] = res["k_bundle_N_m"] * L0 / A / 1e3

# force response in the 1 Hz isometric tests
steps = []
for fn in sorted(glob.glob(os.path.join(SRC, "5fibres_a_*V_38mm*.tab")) + glob.glob(os.path.join(SRC, "5fibres_a_*V_0mm*.tab"))):
    d = np.loadtxt(fn, skiprows=4)
    F, t = d[:, 0] * G, d[:, 2]
    fs = 1 / np.median(np.diff(t))
    v = int(os.path.basename(fn).split("_")[2][:4])
    if v < 800:
        continue
    Fs = savgol_filter(F, 5, 2)
    dF = np.gradient(Fs)
    drops, _ = find_peaks(-dF, distance=int(fs * 0.6), prominence=np.percentile(np.abs(dF), 99) * 0.3)
    for k in drops[1:5]:
        a, b = k - int(0.12 * fs), k + int(0.45 * fs)
        if a < 0 or b >= len(Fs):
            continue
        hi, lo = Fs[a:k - int(0.03 * fs)].mean(), Fs[k + int(0.3 * fs):b].mean()
        if hi - lo < 3e-3:
            continue
        s = (hi - Fs[a:b]) / (hi - lo)
        i10, i90 = np.argmax(s > 0.1), np.argmax(s > 0.9)
        ov = (Fs[a:b].min() - lo) / (hi - lo)
        steps.append(dict(file=os.path.basename(fn), V=v, fs=float(fs), t1090_ms=float((i90 - i10) / fs * 1e3),
                          drop_mN=float((hi - lo) * 1e3), creep_after_frac=float((Fs[k + int(0.1 * fs)] - lo) / (hi - lo))))
res["steps"] = steps
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
json.dump(res, open(os.path.join(HERE, "data", "actuator_bench.json"), "w"), indent=1)
np.savez(os.path.join(HERE, "data", "actuator_curves.npz"), grid=grid,
         **{f"{b}_load_{v}": c for b in load_c for v, c in load_c[b].items()},
         **{f"{b}_unload_{v}": c for b in unload_c for v, c in unload_c[b].items()})

print(res["geom"], "k", res["k_bundle_N_m"], "E kPa", res["E_kPa"], "speed", res["speed_mm_s_a"], res.get("speed_mm_s_b"))
print("hyst", {b: {k: round(v, 3) for k, v in h.items()} for b, h in hyst.items()})
for r in rows:
    if r["V"] in (1200, 1400, 1600) and r["lo"] in (1.05, 1.08):
        print(r["bundle"], r["V"], r["lo"], r["hi"], "same %.3f loop %.3f J/kg dF %.1f mN" % (r["W_same_branch_J_kg"], r["W_loop_J_kg"], r["dF_mN"]))
import collections
g = collections.defaultdict(list)
for s in steps:
    g[(s["V"], "38" in s["file"])].append(s)
for k, v in sorted(g.items()):
    print(k, "n", len(v), "fs", v[0]["fs"], "t10-90 ms %.0f" % np.median([x["t1090_ms"] for x in v]), "drop mN %.1f" % np.median([x["drop_mN"] for x in v]), "creep %.2f" % np.median([x["creep_after_frac"] for x in v]))
